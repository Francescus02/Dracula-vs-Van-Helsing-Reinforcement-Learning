"""
Rich, information-set-compliant observation features for 'Dracula vs Van Helsing'.

Design principles
-----------------
1. LEAK-FREE: every quantity is a deterministic function of what `player` can legally
   observe (own rack, public flags, discard pile, revealed cards, public history).
   Hidden opponent cards / deck order are never read (see test_no_leak in test_obs.py).
2. RANK-CANONICAL FRAME: cards are described by (color RANK, value) instead of absolute
   color. The game is symmetric under color permutations, so an MLP no longer has to
   learn 24 different "trump" interactions (color_onehot x ranking_onehot).
3. EXACT TACTICAL QUANTITIES: the same objects V2-V4 compute internally (unseen-card
   Bayesian win probabilities, sequential district resolution, lethal windows) are given
   to the network as inputs, plus one-step look-ahead values aligned to the action space.

All outputs lie in [-1, 1].
"""
import numpy as np

_IDX = np.arange(32)
_VAL = (_IDX % 8 + 1).astype(np.int64)
_COL = (_IDX // 8).astype(np.int64)
_CARD_DIM = 13  # rank one-hot (4) + value one-hot (8) + strength/16 (1)

PHASE_TURN_START, PHASE_CHOOSE_DISCARD, PHASE_E1, PHASE_E3, PHASE_E4, PHASE_E6, PHASE_E7 = range(7)
_SWAP_PAIRS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


# ----------------------------------------------------------------------------
# Ranking-dependent tables
# ----------------------------------------------------------------------------
class _Tables:
    __slots__ = ("rank", "key", "strength", "beats", "card_feat", "ridx")

    def __init__(self, color_ranking):
        cr = np.asarray(color_ranking, dtype=np.int64)
        rank_of_col = np.empty(4, dtype=np.int64)
        rank_of_col[cr] = np.arange(4)
        self.rank = rank_of_col[_COL]
        # total order that exactly reproduces env._compare_cards
        self.key = (self.rank == 0) * 100 + _VAL * 10 + (3 - self.rank)
        self.strength = np.where(self.rank == 0, 8.0 + _VAL, _VAL - self.rank * 0.1)
        self.beats = self.key[:, None] > self.key[None, :]
        t = np.zeros((32, _CARD_DIM), dtype=np.float32)
        t[_IDX, self.rank] = 1.0
        t[_IDX, 4 + _VAL - 1] = 1.0
        t[:, 12] = self.strength / 16.0
        self.card_feat = t
        self.ridx = self.rank * 8 + (_VAL - 1)  # canonical index in rank frame


def _unseen_weights(tab, n_disc):
    """Opponent retention bias (same functional form as V2-V4's Bayesian model)."""
    sp = min(1.0, n_disc / 10.0)
    return np.where(tab.rank == 0, 1.0 + sp * 1.8, np.maximum(0.25, 1.0 + sp * ((_VAL - 4.5) / 4.0)))


def _pwin_table(tab, unseen, w):
    wu = w * unseen
    tot = wu.sum()
    if tot <= 0:
        return np.full(32, 0.5)
    return (tab.beats * wu[None, :]).sum(1) / tot


# ----------------------------------------------------------------------------
# District probabilities & sequential round DP
# ----------------------------------------------------------------------------
def _p_my(rack, opp_known, opp_cards, exposed, tab, P32, is_final, unseen_sixes):
    p = np.empty(5)
    for d in range(5):
        c = int(rack[d])
        if opp_known[d]:
            p[d] = 1.0 if tab.key[c] > tab.key[opp_cards[d]] else 0.0
        else:
            q = P32[c]
            if exposed[d] and not is_final:
                if q < 0.96:
                    q *= 0.80
                if tab.strength[c] >= 10.0 and unseen_sixes > 0:
                    q -= 0.08 * unseen_sixes
            p[d] = min(1.0, max(0.0, q))
    return p


def _round_dp(p_my, player, vamps, hp, round_num):
    """Exact sequential resolution D1->D5 under district independence."""
    p_drac = p_my if player == 0 else 1.0 - p_my
    alive = np.zeros(13)
    alive[int(hp)] = 1.0
    p_conv = p_kill = e_hp = e_vamp_w = e_vamp = 0.0
    reach = np.zeros(5)
    for d in range(5):
        m = alive.sum()
        reach[d] = m
        p = p_drac[d]
        win, lose = alive * p, alive * (1.0 - p)
        e_hp += m * (1.0 - p)
        e_vamp += m * p
        e_vamp_w += m * p * (1.0 + 0.5 * vamps[d])
        nxt = np.zeros(13)
        nxt[:-1] = lose[1:]
        p_kill += nxt[0]
        nxt[0] = 0.0
        if vamps[d] >= 3:
            p_conv += win.sum()
        else:
            nxt += win
        alive = nxt
    p_surv = alive.sum()
    p_drac_game = p_conv + (p_surv if round_num == 5 else 0.0)
    if player == 0:
        me_win, me_loss = p_drac_game, p_kill
    else:
        me_win, me_loss = p_kill, p_drac_game
    hp_crit = 1.0 + max(0.0, 6.0 - hp) / 3.0
    u_drac = (p_drac_game - p_kill) + 0.06 * (e_vamp_w - hp_crit * e_hp)
    u_me = u_drac if player == 0 else -u_drac
    return dict(reach=reach, me_win=me_win, me_loss=me_loss, p_surv=p_surv, e_hp=e_hp,
                e_vamp=e_vamp, p_conv=p_conv, p_kill=p_kill, u=u_me)


# ----------------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------------
def build_features(env, player):
    opp = 1 - player
    tab = _Tables(env.color_ranking)
    rack = env.racks[player].astype(np.int64)
    n_disc = len(env.discard_pile)
    is_final = bool(env.is_final_turn)
    vamps = env.vampires.astype(np.int64)
    hp = int(env.dracula_hp)
    rnd = int(env.round_num)
    phase = int(env.phase)

    opp_known = env.revealed[opp] | env.known_by_opp[opp]
    opp_cards = np.where(opp_known, env.racks[opp], -1).astype(np.int64)
    exposed = env.revealed[player] | env.known_by_opp[player]

    # ---- seen / unseen (same definition as V2 `_get_unseen_cards`) ----
    seen = np.zeros(32, dtype=bool)
    seen[rack] = True
    if n_disc:
        seen[np.asarray(env.discard_pile, dtype=np.int64)] = True
    is_own = (player == env.current_player)
    drawn = int(env.drawn_card) if (is_own and phase == PHASE_CHOOSE_DISCARD and env.drawn_card != -1) else -1
    if drawn != -1:
        seen[drawn] = True
    seen[opp_cards[opp_known]] = True
    top_c = -1
    if env.top_deck_revealed and len(env.deck) > 0:
        top_c = int(env.deck[0])
        seen[top_c] = True
    unseen = ~seen
    n_unseen = int(unseen.sum())
    w = _unseen_weights(tab, n_disc)
    P32 = _pwin_table(tab, unseen.astype(np.float64), w)
    unseen_sixes = 0 if is_final else int(np.sum(unseen & (_VAL == 6)))

    f = []  # list of float32 blocks

    def card_block(c):
        return tab.card_feat[c] if c >= 0 else np.zeros(_CARD_DIM, dtype=np.float32)

    # 1. own rack, rank frame (65)
    for s in range(5):
        f.append(tab.card_feat[rack[s]])
    # 2. opponent known cards, rank frame (75)
    for s in range(5):
        if opp_known[s]:
            f.append(np.array([float(env.revealed[opp, s]), float(not env.revealed[opp, s])], dtype=np.float32))
            f.append(tab.card_feat[opp_cards[s]])
        else:
            f.append(np.zeros(2 + _CARD_DIM, dtype=np.float32))
    # 3. drawn / revealed top / last discard (39)
    f.append(card_block(drawn))
    f.append(card_block(top_c))
    f.append(card_block(int(env.discard_pile[-1]) if n_disc else -1))

    # 4. rank-frame memory (96): discarded, seen, discarded-by-opponent
    disc_rel = np.zeros(32, dtype=np.float32)
    opp_disc_rel = np.zeros(32, dtype=np.float32)
    if n_disc:
        dp = np.asarray(env.discard_pile, dtype=np.int64)
        disc_rel[tab.ridx[dp]] = 1.0
        owners = env.discard_owner[dp]
        opp_disc_rel[tab.ridx[dp[owners == opp]]] = 1.0
    seen_rel = np.zeros(32, dtype=np.float32)
    seen_rel[tab.ridx[seen]] = 1.0
    f += [disc_rel, seen_rel, opp_disc_rel]

    # 5. unseen statistics + belief about a hidden opponent card (16)
    uv = np.array([np.sum(unseen & (_VAL == v)) for v in range(1, 9)], dtype=np.float32) / 4.0
    ur = np.array([np.sum(unseen & (tab.rank == r)) for r in range(4)], dtype=np.float32) / 8.0
    wu = w * unseen
    wsum = wu.sum()
    if wsum > 0:
        exp_str = float((wu * tab.strength).sum() / wsum) / 16.0
        p_trump = float(wu[tab.rank == 0].sum() / wsum)
        p_high = float(wu[_VAL >= 7].sum() / wsum)
    else:
        exp_str, p_trump, p_high = 0.5, 0.25, 0.25
    f += [uv, ur, np.array([n_unseen / 32.0, exp_str, p_trump, p_high], dtype=np.float32)]

    # 6. per-district block (5 x 16 = 80)
    p_my = _p_my(rack, opp_known, opp_cards, exposed, tab, P32, is_final, unseen_sixes)
    for d in range(5):
        c = rack[d]
        v_oh = np.zeros(4, dtype=np.float32)
        v_oh[min(3, vamps[d])] = 1.0
        if opp_known[d]:
            oc = opp_cards[d]
            kw = float(tab.key[c] > tab.key[oc])
            margin = (tab.strength[c] - tab.strength[oc]) / 16.0
            kl, kwin, mar = 1.0 - kw, kw, margin
        else:
            kl = kwin = mar = 0.0
        n_beat = float(np.sum(unseen & (tab.key > tab.key[c]))) / 32.0
        f.append(v_oh)
        f.append(np.array([
            d / 4.0,
            float(opp_known[d]), kwin, kl, mar,
            P32[c], p_my[d], float(exposed[d]),
            float(P32[c] >= 0.999), n_beat,
            env.slot_change_count[opp, d] / 4.0, env.slot_change_count[player, d] / 4.0,
        ], dtype=np.float32))

    # 7. sequential round DP (reachability, lethal/kill probabilities) (16)
    st = _round_dp(p_my, player, vamps, hp, rnd)
    n_lethal = float(np.sum(vamps >= 3)) / 5.0
    n_two = float(np.sum(vamps == 2)) / 5.0
    f.append(st["reach"].astype(np.float32))
    f.append(np.array([
        st["me_win"], st["me_loss"], st["p_surv"], st["e_hp"] / 5.0, st["e_vamp"] / 5.0,
        st["p_conv"], st["p_kill"], np.tanh(3.0 * st["u"]), n_lethal, n_two,
        float(np.sum(p_my)) / 5.0,
    ], dtype=np.float32))

    # 8. global context (29)
    hp_b = np.zeros(4, dtype=np.float32)
    hp_b[0 if hp <= 3 else 1 if hp <= 6 else 2 if hp <= 9 else 3] = 1.0
    rem_d = 5 * (5 - rnd + 1)
    mv = np.zeros(4, dtype=np.float32)
    mv[min(3, int(vamps.max()))] = 1.0
    caller = np.zeros(3, dtype=np.float32)
    if not env.end_of_round_called:
        caller[0] = 1.0
    elif env.caller_player == player:
        caller[1] = 1.0
    else:
        caller[2] = 1.0
    legal = env.get_legal_actions().astype(np.float32) if player == env.current_player else np.zeros(10, np.float32)
    f += [hp_b, mv, caller, legal]
    f.append(np.array([
        rem_d / 25.0, min(1.0, hp / rem_d), float(vamps.sum()) / 20.0,
        float(env.round_starter == player),
        env.turns_taken[player] / 10.0, env.turns_taken[opp] / 10.0,
        float(n_disc >= 6), float(len(env.deck) <= 3),
    ], dtype=np.float32))

    # 9. action-aligned one-step look-ahead values (10 + 10 + 48 + 6 + 5 = 79)
    act_val = np.zeros(10, dtype=np.float32)
    best = np.zeros(10, dtype=np.float32)
    eff_oh = np.zeros((6, 8), dtype=np.float32)
    disc_val = np.zeros(6, dtype=np.float32)
    e6 = np.zeros(5, dtype=np.float32)

    def value_of(new_rack, new_exposed, tb=tab, P=P32):
        pm = _p_my(new_rack, opp_known, opp_cards, new_exposed, tb, P, is_final, unseen_sixes)
        return _round_dp(pm, player, vamps, hp, rnd)["u"]

    if player == env.current_player and not env.done:
        v_now = st["u"]
        legal_b = env.get_legal_actions().astype(bool)
        vals = np.full(10, -np.inf)
        if phase == PHASE_TURN_START:
            vals[0] = v_now
            if legal_b[1]:
                vals[1] = v_now
        elif phase == PHASE_CHOOSE_DISCARD and drawn != -1:
            for a in range(6):
                if not legal_b[a]:
                    continue
                if a == 0:
                    vals[0] = v_now
                    dcard = drawn
                else:
                    nr = rack.copy()
                    nr[a - 1] = drawn
                    ne = exposed.copy()
                    ne[a - 1] = bool(env.drawn_card_was_revealed)
                    vals[a] = value_of(nr, ne)
                    dcard = int(rack[a - 1])
                eff_oh[a, (dcard % 8)] = 1.0
                disc_val[a] = ((dcard % 8) + 1) / 8.0
        elif phase == PHASE_E4:
            for a, (i, j) in enumerate(_SWAP_PAIRS):
                nr = rack.copy()
                nr[i], nr[j] = rack[j], rack[i]
                ne = exposed.copy()
                ne[i], ne[j] = exposed[j], exposed[i]
                vals[a] = value_of(nr, ne)
        elif phase == PHASE_E7:
            cr = np.asarray(env.color_ranking).copy()
            for a in (1, 2, 3):
                cr2 = cr.copy()
                cr2[0], cr2[a] = cr2[a], cr2[0]
                tb2 = _Tables(cr2)
                w2 = _unseen_weights(tb2, n_disc)
                vals[a] = value_of(rack, exposed, tb2, _pwin_table(tb2, unseen.astype(np.float64), w2))
        elif phase == PHASE_E6:
            for d in range(5):
                if opp_known[d]:
                    gain = (tab.strength[opp_cards[d]] - tab.strength[rack[d]]) / 16.0
                else:
                    gain = (exp_str * 16.0 - tab.strength[rack[d]]) / 16.0
                e6[d] = gain
        fin = np.isfinite(vals)
        if fin.any() and phase in (PHASE_CHOOSE_DISCARD, PHASE_E4, PHASE_E7, PHASE_TURN_START):
            ref = v_now
            act_val[fin] = np.tanh(3.0 * (vals[fin] - ref)).astype(np.float32)
            mx = vals[fin].max()
            best[fin & (vals >= mx - 1e-9)] = 1.0
    f += [act_val, best, eff_oh.ravel(), disc_val, e6]

    out = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in f])
    np.clip(out, -1.0, 1.0, out=out)
    return out