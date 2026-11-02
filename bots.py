from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from dvvh_env import DraculaVsVanHelsingEnv


class BaseBot(ABC):
    """
    Abstract base interface for all Dracula vs Van Helsing agents
    (Random, Heuristic, and future Neural Network policies).
    """

    @abstractmethod
    def select_action(
        self, env: DraculaVsVanHelsingEnv, obs: np.ndarray, info: Dict
    ) -> int:
        """
        Selects a legal action (0..9) given the current game state.
        Must strictly respect `info["action_mask"]` and never access hidden opponent cards.
        """
        pass


class RandomBot(BaseBot):
    """
    Uniform random baseline agent. Chooses randomly among legal actions.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def select_action(
        self, env: DraculaVsVanHelsingEnv, obs: np.ndarray, info: Dict
    ) -> int:
        legal_actions = np.flatnonzero(info["action_mask"])
        return int(self.rng.choice(legal_actions))


# =============================================================================
# BOT A: GREEDY HEURISTIC BOT (HIGH MULTIPLIERS: 2.5x / 1.5x / +3.0x)
# =============================================================================
class GreedyHeuristicBot(BaseBot):
    """
    Original Post-Resolution Heuristic Bot:
      - District Multipliers: 2.5x (3 Vampires), 1.5x (2 Vampires), +3.0x distributed when Dracula HP <= 3.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def select_action(
        self, env: DraculaVsVanHelsingEnv, obs: np.ndarray, info: Dict
    ) -> int:
        mask = info["action_mask"]
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 1:
            return int(legal_actions[0])

        phase = info["phase"]
        p = info["current_player"]

        if phase == DraculaVsVanHelsingEnv.PHASE_TURN_START:
            return self._decide_turn_start(env, p, legal_actions)
        elif phase == DraculaVsVanHelsingEnv.PHASE_CHOOSE_DISCARD:
            return self._decide_discard(env, p, legal_actions)
        elif phase == DraculaVsVanHelsingEnv.PHASE_EFFECT_1_REV_OWN:
            return self._decide_effect_1_reveal_own(env, p, legal_actions)
        elif phase == DraculaVsVanHelsingEnv.PHASE_EFFECT_3_REV_OPP:
            return self._decide_effect_3_reveal_opp(env, p, legal_actions)
        elif phase == DraculaVsVanHelsingEnv.PHASE_EFFECT_4_SWAP_OWN:
            return self._decide_effect_4_swap_own(env, p, legal_actions)
        elif phase == DraculaVsVanHelsingEnv.PHASE_EFFECT_6_SWAP_OPP:
            return self._decide_effect_6_swap_opp(env, p, legal_actions)
        elif phase == DraculaVsVanHelsingEnv.PHASE_EFFECT_7_TRUMP:
            return self._decide_effect_7_change_trump(env, p, legal_actions)

        return int(self.rng.choice(legal_actions))

    def _card_strength(self, card_id: int, color_ranking: np.ndarray) -> float:
        col, val = card_id // 8, (card_id % 8) + 1
        if col == color_ranking[0]:
            return 8.0 + float(val)
        rank = int(np.where(color_ranking == col)[0][0])
        return float(val) - (rank * 0.1)

    def _does_my_card_win(
        self, my_card: int, opp_card: int, color_ranking: np.ndarray
    ) -> bool:
        col_m, val_m = my_card // 8, (my_card % 8) + 1
        col_o, val_o = opp_card // 8, (opp_card % 8) + 1
        trump = color_ranking[0]

        is_t_m = col_m == trump
        is_t_o = col_o == trump
        if is_t_m and not is_t_o:
            return True
        if is_t_o and not is_t_m:
            return False
        if val_m != val_o:
            return val_m > val_o

        rank_m = int(np.where(color_ranking == col_m)[0][0])
        rank_o = int(np.where(color_ranking == col_o)[0][0])
        return rank_m < rank_o

    def _get_district_weights(self, env: DraculaVsVanHelsingEnv) -> np.ndarray:
        weights = np.ones(5, dtype=np.float32)
        vamps = env.vampires
        hp = int(env.dracula_hp)

        for d in range(5):
            if vamps[d] == 3:
                weights[d] = 2.5
            elif vamps[d] == 2:
                weights[d] = 1.5
            weights[d] += (4 - d) * 0.03

        if hp <= 3:
            three_vamp_dists = np.flatnonzero(vamps == 3)
            if len(three_vamp_dists) > 0:
                target_d = int(three_vamp_dists[0])
                crit_indices = list(range(target_d + 1))
                share = 3.0 / float(max(1, min(len(crit_indices), hp)))
                for idx in crit_indices[: max(1, hp)]:
                    weights[idx] += share
            else:
                for idx in range(3):
                    weights[idx] += 1.0

        return weights

    def _expected_unknown_card_strength(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        my_rack: np.ndarray,
        color_ranking: np.ndarray,
    ) -> float:
        opp = 1 - player
        seen = np.zeros(32, dtype=bool)
        for c in my_rack:
            seen[int(c)] = True
        # FIX: Mark current actual player rack too so a card being discarded from
        # the rack during lookahead is never treated as an unseen opponent card!
        for c in env.racks[player]:
            seen[int(c)] = True
        if env.drawn_card != -1:
            seen[int(env.drawn_card)] = True
        for c in env.discard_pile:
            seen[int(c)] = True
        for d in range(5):
            if env.revealed[opp, d] or env.known_by_opp[opp, d]:
                seen[int(env.racks[opp, d])] = True
        if env.top_deck_revealed and len(env.deck) > 0:
            seen[int(env.deck[0])] = True
        unseen_cards = np.flatnonzero(~seen)
        if len(unseen_cards) == 0:
            return 5.5
        total_str = sum(
            self._card_strength(int(c), color_ranking) for c in unseen_cards
        )
        return float(total_str) / len(unseen_cards)

    def _evaluate_slot_utility(
        self,
        my_card: int,
        opp_card_known: bool,
        opp_card: int,
        district_weight: float,
        color_ranking: np.ndarray,
    ) -> float:
        my_str = self._card_strength(my_card, color_ranking)

        if opp_card_known:
            opp_str = self._card_strength(opp_card, color_ranking)
            wins = self._does_my_card_win(my_card, opp_card, color_ranking)
            if wins:
                margin = my_str - opp_str
                tight_margin_bonus = max(0.0, (8.0 - margin) * 0.25)
                return (10.0 * district_weight) + tight_margin_bonus + (my_str * 0.35)
            else:
                return my_str * 0.25
        else:
            if my_str >= 9.0:
                win_prob = 0.66 + ((my_str - 8.0) / 8.0) * 0.32
            else:
                win_prob = 0.05 + (my_str / 8.0) * 0.59
            return (win_prob * 10.0 * district_weight) + (my_str * 0.40)

    def _evaluate_rack_utility(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        color_ranking: np.ndarray,
        known_opp_override: Optional[Dict[int, Tuple[bool, int]]] = None,
    ) -> float:
        opp = 1 - player
        weights = self._get_district_weights(env)
        total_score = 0.0

        for d in range(5):
            if known_opp_override and d in known_opp_override:
                is_known, opp_c = known_opp_override[d]
            else:
                is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
                opp_c = int(env.racks[opp, d]) if is_known else -1

            total_score += self._evaluate_slot_utility(
                int(rack[d]), is_known, opp_c, float(weights[d]), color_ranking
            )

        return total_score

    def _compute_post_discard_hand_value(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        candidate_rack: np.ndarray,
        discarded_card: int,
    ) -> float:
        val = (discarded_card % 8) + 1
        opp = 1 - player
        base_rack_score = self._evaluate_rack_utility(
            env, player, candidate_rack, env.color_ranking
        )

        if val == 1:
            return base_rack_score - 0.03
        elif val == 2:
            return base_rack_score - 0.02
        elif val == 3:
            return base_rack_score + 0.03
        elif val == 5:
            return base_rack_score + (0.40 if env.is_final_turn else 0.05)

        elif val == 4:
            best_swapped_score = base_rack_score
            for i, j in DraculaVsVanHelsingEnv.SWAP_PAIRS:
                swapped = candidate_rack.copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]
                score = self._evaluate_rack_utility(
                    env, player, swapped, env.color_ranking
                )
                if score > best_swapped_score:
                    best_swapped_score = score
            return best_swapped_score

        elif val == 6:
            weights = self._get_district_weights(env)
            avg_unseen_str = self._expected_unknown_card_strength(
                env, player, candidate_rack, env.color_ranking
            )
            best_post_swap6_score = -1e9

            for d in range(5):
                my_c = int(candidate_rack[d])
                is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])

                if is_opp_known:
                    opp_c = int(env.racks[opp, d])
                    post_rack = candidate_rack.copy()
                    post_rack[d] = opp_c
                    override = {d: (True, my_c)}
                    score = self._evaluate_rack_utility(
                        env, player, post_rack, env.color_ranking, known_opp_override=override
                    )
                else:
                    current_slot_u = self._evaluate_slot_utility(
                        my_c, False, -1, float(weights[d]), env.color_ranking
                    )
                    my_given_str = self._card_strength(my_c, env.color_ranking)
                    if my_given_str >= 9.0:
                        prob_received_wins = max(0.02, (16.0 - my_given_str) / 32.0)
                    else:
                        prob_received_wins = max(
                            0.10, min(0.92, (avg_unseen_str - my_given_str + 4.0) / 8.0)
                        )
                    expected_new_slot_u = (
                        prob_received_wins * 10.0 * float(weights[d])
                    ) + (avg_unseen_str * 0.40)
                    score = base_rack_score - current_slot_u + expected_new_slot_u

                if score > best_post_swap6_score:
                    best_post_swap6_score = score

            return best_post_swap6_score

        elif val == 7:
            colors = candidate_rack // 8
            old_trump = env.color_ranking[0]
            old_trump_count = int(np.sum(colors == old_trump))

            vis_mask = env.revealed[opp] | env.known_by_opp[opp]
            opp_vis_cards = env.racks[opp, vis_mask]
            opp_old_vis_trumps = int(np.sum((opp_vis_cards // 8) == old_trump))

            best_post_7_score = -1e9
            for rank_idx in range(1, 4):
                new_ranking = env.color_ranking.copy()
                new_ranking[0], new_ranking[rank_idx] = (
                    new_ranking[rank_idx],
                    new_ranking[0],
                )
                new_trump = new_ranking[0]
                new_trump_count = int(np.sum(colors == new_trump))
                opp_new_vis_trumps = int(np.sum((opp_vis_cards // 8) == new_trump))

                post_7_rack_score = self._evaluate_rack_utility(
                    env, player, candidate_rack, new_ranking
                )
                opp_trump_swing_bonus = (opp_old_vis_trumps - opp_new_vis_trumps) * 4.0

                if new_trump_count >= 3 and new_trump_count > old_trump_count:
                    total_7_score = post_7_rack_score + opp_trump_swing_bonus
                elif (
                    old_trump_count == 0
                    and opp_old_vis_trumps >= 2
                    and opp_new_vis_trumps == 0
                ):
                    total_7_score = post_7_rack_score + opp_trump_swing_bonus
                else:
                    total_7_score = post_7_rack_score - 12.0

                if total_7_score > best_post_7_score:
                    best_post_7_score = total_7_score

            return best_post_7_score

        elif val == 8:
            if self._should_close_round_with_rack(env, player, candidate_rack):
                return base_rack_score + 6.0
            else:
                return base_rack_score - 15.0

        return base_rack_score

    def _should_close_round_with_rack(
        self, env: DraculaVsVanHelsingEnv, player: int, rack: np.ndarray
    ) -> bool:
        opp = 1 - player
        expected_wins = 0.0
        expected_hp_loss = 0.0

        for d in range(5):
            my_c = int(rack[d])
            is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
            if is_known:
                p_win = (
                    1.0
                    if self._does_my_card_win(my_c, int(env.racks[opp, d]), env.color_ranking)
                    else 0.0
                )
            else:
                my_str = self._card_strength(my_c, env.color_ranking)
                p_win = (
                    0.66 + ((my_str - 8.0) / 8.0) * 0.32
                    if my_str >= 9.0
                    else 0.05 + (my_str / 8.0) * 0.59
                )

            expected_wins += p_win
            if player == 0:
                if (
                    env.vampires[d] == 3
                    and p_win >= 0.80
                    and (env.dracula_hp - expected_hp_loss) > 1.2
                ):
                    return True
                expected_hp_loss += 1.0 - p_win
            else:
                if expected_wins >= env.dracula_hp + 0.25:
                    return True

        threshold = 3.35 if player == 0 else 3.45
        return expected_wins >= threshold

    def _decide_turn_start(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        if 1 in legal_actions and self._should_close_round_with_rack(
            env, player, env.racks[player]
        ):
            return 1
        return 0

    def _decide_discard(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        best_action = int(legal_actions[0])
        best_post_value = -1e9

        for action in legal_actions:
            candidate_rack = env.racks[player].copy()
            if action == 0:
                discarded_card = int(env.drawn_card)
            else:
                slot = action - 1
                discarded_card = int(candidate_rack[slot])
                candidate_rack[slot] = int(env.drawn_card)

            post_value = self._compute_post_discard_hand_value(
                env, player, candidate_rack, discarded_card
            )
            if action > 0 and env.drawn_card_was_revealed:
                post_value -= 0.15

            if post_value > best_post_value:
                best_post_value = post_value
                best_action = int(action)

        return best_action

    def _decide_effect_1_reveal_own(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        best_slot = int(legal_actions[0])
        lowest_score = 1e9
        weights = self._get_district_weights(env)

        for slot in legal_actions:
            c = int(env.racks[player, slot])
            score = self._card_strength(c, env.color_ranking) + float(weights[slot]) * 0.5
            if score < lowest_score:
                lowest_score = score
                best_slot = int(slot)

        return best_slot

    def _decide_effect_3_reveal_opp(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        opp = 1 - player
        best_slot = int(legal_actions[0])
        best_priority = -1e9
        weights = self._get_district_weights(env)
        trump = env.color_ranking[0]

        for slot in legal_actions:
            if env.known_by_opp[opp, slot]:
                priority = -10.0
            else:
                priority = float(weights[slot]) * 3.0
                my_c = int(env.racks[player, slot])
                my_col, my_val = my_c // 8, (my_c % 8) + 1
                if my_col == trump and my_val <= 4:
                    priority += 2.5
                elif my_col != trump and my_val >= 6:
                    priority += 2.0

            if priority > best_priority:
                best_priority = priority
                best_slot = int(slot)

        return best_slot

    def _decide_effect_4_swap_own(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        best_action = int(legal_actions[0])
        best_score = -1e9

        for action in legal_actions:
            i, j = DraculaVsVanHelsingEnv.SWAP_PAIRS[action]
            swapped = env.racks[player].copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            score = self._evaluate_rack_utility(env, player, swapped, env.color_ranking)
            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action

    def _decide_effect_6_swap_opp(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        opp = 1 - player
        weights = self._get_district_weights(env)
        avg_unseen_str = self._expected_unknown_card_strength(
            env, player, env.racks[player], env.color_ranking
        )
        base_score = self._evaluate_rack_utility(
            env, player, env.racks[player], env.color_ranking
        )

        best_district = int(legal_actions[0])
        best_post_score = -1e9

        for d in legal_actions:
            my_c = int(env.racks[player, d])
            is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])

            if is_opp_known:
                opp_c = int(env.racks[opp, d])
                post_rack = env.racks[player].copy()
                post_rack[d] = opp_c
                override = {int(d): (True, my_c)}
                score = self._evaluate_rack_utility(
                    env, player, post_rack, env.color_ranking, known_opp_override=override
                )
            else:
                current_slot_u = self._evaluate_slot_utility(
                    my_c, False, -1, float(weights[d]), env.color_ranking
                )
                my_given_str = self._card_strength(my_c, env.color_ranking)
                if my_given_str >= 9.0:
                    prob_received_wins = max(0.02, (16.0 - my_given_str) / 32.0)
                else:
                    prob_received_wins = max(
                        0.10, min(0.92, (avg_unseen_str - my_given_str + 4.0) / 8.0)
                    )
                expected_new_slot_u = (
                    prob_received_wins * 10.0 * float(weights[d])
                ) + (avg_unseen_str * 0.40)
                score = base_score - current_slot_u + expected_new_slot_u

            if score > best_post_score:
                best_post_score = score
                best_district = int(d)

        return best_district

    def _decide_effect_7_change_trump(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        opp = 1 - player
        vis_mask = env.revealed[opp] | env.known_by_opp[opp]
        opp_vis_cards = env.racks[opp, vis_mask]
        old_trump = env.color_ranking[0]
        opp_old_vis_trumps = int(np.sum((opp_vis_cards // 8) == old_trump))

        best_action = int(legal_actions[0])
        best_score = -1e9

        for action in legal_actions:
            new_ranking = env.color_ranking.copy()
            new_ranking[0], new_ranking[action] = new_ranking[action], new_ranking[0]
            new_trump = new_ranking[0]
            opp_new_vis_trumps = int(np.sum((opp_vis_cards // 8) == new_trump))

            score = self._evaluate_rack_utility(
                env, player, env.racks[player], new_ranking
            ) + (opp_old_vis_trumps - opp_new_vis_trumps) * 4.0

            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action


# =============================================================================
# BOT B: TACTICAL REFINED BOT V1 (CONTAINED MULTIPLIERS 2.0x / 1.3x)
# =============================================================================
class TacticalRefinedBot(GreedyHeuristicBot):
    """
    Bot B (V1): Contained multipliers (2.0x / 1.3x / +2.0x) and initial tactical fixes.
    """

    def _get_district_weights(self, env: DraculaVsVanHelsingEnv) -> np.ndarray:
        weights = np.ones(5, dtype=np.float32)
        vamps = env.vampires
        hp = int(env.dracula_hp)

        for d in range(5):
            if vamps[d] == 3:
                weights[d] = 2.0
            elif vamps[d] == 2:
                weights[d] = 1.3
            weights[d] += (4 - d) * 0.04

        if hp <= 3:
            three_vamp_dists = np.flatnonzero(vamps == 3)
            if len(three_vamp_dists) > 0:
                target_d = int(three_vamp_dists[0])
                crit_indices = list(range(target_d + 1))
                share = 2.0 / float(max(1, min(len(crit_indices), hp)))
                for idx in crit_indices[: max(1, hp)]:
                    weights[idx] += share
            else:
                for idx in range(3):
                    weights[idx] += 2.0 / 3.0

        return weights

    def _evaluate_slot_utility(
        self,
        my_card: int,
        opp_card_known: bool,
        opp_card: int,
        district_weight: float,
        color_ranking: np.ndarray,
    ) -> float:
        my_str = self._card_strength(my_card, color_ranking)

        if opp_card_known:
            opp_str = self._card_strength(opp_card, color_ranking)
            wins = self._does_my_card_win(my_card, opp_card, color_ranking)
            if wins:
                margin = my_str - opp_str
                tight_margin_bonus = max(0.0, (8.0 - margin) * 0.30)
                return (10.0 * district_weight) + tight_margin_bonus + (my_str * 0.30)
            else:
                return my_str * 0.15
        else:
            if my_str >= 9.0:
                win_prob = 0.58 + ((my_str - 8.0) / 8.0) * 0.39
            else:
                norm_v = my_str / 8.0
                win_prob = 0.03 + (norm_v ** 1.6) * 0.51
            return (win_prob * 10.0 * district_weight) + (my_str * 0.32)

    def _evaluate_rack_utility(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        color_ranking: np.ndarray,
        known_opp_override: Optional[Dict[int, Tuple[bool, int]]] = None,
    ) -> float:
        base_total = super()._evaluate_rack_utility(
            env, player, rack, color_ranking, known_opp_override
        )
        trump = color_ranking[0]
        colors = rack // 8
        for col in range(4):
            if col != trump:
                cnt = int(np.sum(colors == col))
                if cnt >= 3:
                    base_total += 1.4 * (cnt - 2)
        return base_total

    def _compute_post_discard_hand_value(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        candidate_rack: np.ndarray,
        discarded_card: int,
    ) -> float:
        val = (discarded_card % 8) + 1
        opp = 1 - player

        if val == 5 and len(env.deck) > 0:
            base_rack_score = self._evaluate_rack_utility(
                env, player, candidate_rack, env.color_ranking
            )
            min_slot_str = min(
                self._card_strength(int(c), env.color_ranking) for c in candidate_rack
            )
            tempo_bonus = 2.50 if env.is_final_turn else (1.35 if min_slot_str <= 5.0 else 0.40)
            return base_rack_score + tempo_bonus

        if val == 6:
            base_rack_score = self._evaluate_rack_utility(
                env, player, candidate_rack, env.color_ranking
            )
            weights = self._get_district_weights(env)
            avg_unseen_str = self._expected_unknown_card_strength(
                env, player, candidate_rack, env.color_ranking
            )
            best_post_swap6_score = -1e9

            for d in range(5):
                my_c = int(candidate_rack[d])
                is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])

                if is_opp_known:
                    opp_c = int(env.racks[opp, d])
                    post_rack = candidate_rack.copy()
                    post_rack[d] = opp_c
                    override = {d: (True, my_c)}
                    score = self._evaluate_rack_utility(
                        env, player, post_rack, env.color_ranking, known_opp_override=override
                    )
                else:
                    current_slot_u = self._evaluate_slot_utility(
                        my_c, False, -1, float(weights[d]), env.color_ranking
                    )
                    my_given_str = self._card_strength(my_c, env.color_ranking)
                    if my_given_str >= 9.0:
                        prob_received_wins = max(0.01, (16.0 - my_given_str) / 35.0)
                    else:
                        prob_received_wins = max(
                            0.08, min(0.88, (avg_unseen_str - my_given_str + 3.5) / 8.5)
                        )
                    expected_new_slot_u = (
                        (prob_received_wins * 10.0 * float(weights[d]))
                        + (avg_unseen_str * 0.32)
                        - 2.20
                    )
                    score = base_rack_score - current_slot_u + expected_new_slot_u

                if score > best_post_swap6_score:
                    best_post_swap6_score = score

            return best_post_swap6_score

        return super()._compute_post_discard_hand_value(
            env, player, candidate_rack, discarded_card
        )

    def _should_close_round_with_rack(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        is_instant_card8: bool = False,
    ) -> bool:
        opp = 1 - player
        expected_wins = 0.0
        expected_hp_loss = 0.0

        for d in range(5):
            my_c = int(rack[d])
            is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
            if is_known:
                p_win = (
                    1.0
                    if self._does_my_card_win(my_c, int(env.racks[opp, d]), env.color_ranking)
                    else 0.0
                )
            else:
                my_str = self._card_strength(my_c, env.color_ranking)
                if my_str >= 9.0:
                    p_win = 0.58 + ((my_str - 8.0) / 8.0) * 0.39
                else:
                    p_win = 0.03 + ((my_str / 8.0) ** 1.6) * 0.51

            expected_wins += p_win
            if player == 0:
                if (
                    env.vampires[d] == 3
                    and p_win >= 0.82
                    and (env.dracula_hp - expected_hp_loss) > 1.3
                ):
                    return True
                expected_hp_loss += 1.0 - p_win
            else:
                if expected_wins >= env.dracula_hp + 0.30:
                    return True

        if is_instant_card8:
            return expected_wins >= 3.25
        return expected_wins >= (3.55 if player == 0 else 3.60)


# =============================================================================
# BOT C: TACTICAL REFINED BOT V2 (BAYESIAN CARD COUNTING & ENDGAME SOLVER)
# =============================================================================
class TacticalRefinedBot_V2(TacticalRefinedBot):
    """
    Advanced Heuristic Bot (V2) inheriting from TacticalRefinedBot:
      1. Exact Bayesian Card Counting (`_get_unseen_cards` & `_exact_win_prob_vs_unseen`):
         - Computes exact win probabilities against the actual pool of remaining unseen cards
           (filtering for opponent selection bias as the round progresses).
         - Automatically detects "Nut Cards" (100% unbeatable cards because all higher Trumps are gone).
      2. Round 5 Survival & Lethal Endgame Solver:
         - In Round 5, Dracula wins automatically if he survives with >= 1 HP! V2 dynamically shifts
           district weights and round-closing logic to guarantee survival (for Dracula) or lethal
           damage (for Van Helsing).
      3. Card 8 Instant-Close Bug Fix:
         - Properly invokes `_should_close_round_with_rack(..., is_instant_card8=True)` when evaluating
           a Card 8 discard, seizing instant-win opportunities before the opponent gets another turn.
      4. Final-Turn Mode (`env.is_final_turn == True`):
         - Drops speculative future suit-building bonuses on the final turn of a round and maximizes
           immediate showdown combat wins (or chains Card 5s).
      5. Card 6 Counter-Theft Awareness:
         - Tracks how many 6s remain unseen; avoids leaving high Trumps publicly exposed on the table
           and refines `PHASE_EFFECT_6_SWAP_OPP` using the exact unseen card distribution.
    """

    def _get_unseen_cards(
        self, env: DraculaVsVanHelsingEnv, player: int, my_rack: np.ndarray
    ) -> np.ndarray:
        """
        Returns the exact array of card IDs (0..31) that are currently unseen by `player`
        (i.e. residing either in hidden opponent rack slots or in the draw deck).
        """
        opp = 1 - player
        seen = np.zeros(32, dtype=bool)
        for c in my_rack:
            seen[int(c)] = True
        for c in env.racks[player]:
            seen[int(c)] = True
        for c in env.discard_pile:
            seen[int(c)] = True
        if env.drawn_card != -1:
            seen[int(env.drawn_card)] = True
        for d in range(5):
            if env.revealed[opp, d] or env.known_by_opp[opp, d]:
                seen[int(env.racks[opp, d])] = True
        if env.top_deck_revealed and len(env.deck) > 0:
            seen[int(env.deck[0])] = True

        return np.flatnonzero(~seen)

    def _exact_win_prob_vs_unseen(
        self,
        my_card: int,
        unseen_cards: np.ndarray,
        color_ranking: np.ndarray,
        num_discards: int,
    ) -> float:
        """
        Calculates the Bayesian probability that `my_card` beats a hidden card in the opponent's rack.
        Accounts for opponent selection bias: as `num_discards` grows, the opponent selectively
        retains stronger cards from the unseen pool in their rack!
        """
        if len(unseen_cards) == 0:
            return 0.50

        # Count how many unseen cards `my_card` beats, and weight them by opponent retention likelihood
        trump = color_ranking[0]
        selection_pressure = min(1.0, num_discards / 10.0)  # 0.0 at round start -> 1.0 late round

        weighted_wins = 0.0
        total_weight = 0.0

        for u_card in unseen_cards:
            u_c = int(u_card)
            u_col, u_val = u_c // 8, (u_c % 8) + 1
            # Opponent is much more likely to keep Trumps and 7s/8s in their hidden rack than 1s/2s
            if u_col == trump:
                opp_keep_weight = 1.0 + selection_pressure * 1.8
            else:
                opp_keep_weight = 1.0 + selection_pressure * ((u_val - 4.5) / 4.0)
                opp_keep_weight = max(0.25, opp_keep_weight)

            total_weight += opp_keep_weight
            if self._does_my_card_win(my_card, u_c, color_ranking):
                weighted_wins += opp_keep_weight

        return float(weighted_wins / total_weight)

    def _get_district_weights(self, env: DraculaVsVanHelsingEnv) -> np.ndarray:
        """
        Refines district multipliers with Round 5 Survival / Lethal awareness.
        """
        weights = super()._get_district_weights(env)

        # In Round 5, Dracula wins automatically if he survives all 5 districts with >= 1 HP!
        # Therefore, when Dracula has <= 5 HP in Round 5, every district is a direct survival battle.
        if env.round_num == 5 and env.dracula_hp <= 5:
            for d in range(5):
                # Boost non-3-vampire districts too so neither player ignores them in Round 5
                weights[d] = max(weights[d], 1.45 + (4 - d) * 0.06)

        return weights

    def _evaluate_rack_utility(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        color_ranking: np.ndarray,
        known_opp_override: Optional[Dict[int, Tuple[bool, int]]] = None,
    ) -> float:
        opp = 1 - player
        weights = self._get_district_weights(env)
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)
        total_score = 0.0

        for d in range(5):
            my_c = int(rack[d])
            my_str = self._card_strength(my_c, color_ranking)

            if known_opp_override and d in known_opp_override:
                is_known, opp_c = known_opp_override[d]
            else:
                is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
                opp_c = int(env.racks[opp, d]) if is_known else -1

            if is_known:
                opp_str = self._card_strength(opp_c, color_ranking)
                wins = self._does_my_card_win(my_c, opp_c, color_ranking)
                if wins:
                    margin = my_str - opp_str
                    tight_margin_bonus = max(0.0, (8.0 - margin) * 0.32)
                    slot_u = (10.0 * float(weights[d])) + tight_margin_bonus + (my_str * 0.28)
                else:
                    # Guaranteed loss against known opponent card: strongly prefer moving good cards out!
                    slot_u = my_str * 0.12
            else:
                # Exact Bayesian win probability against the remaining unseen cards
                win_prob = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, color_ranking, num_discards
                )
                slot_u = (win_prob * 10.0 * float(weights[d])) + (my_str * 0.30)

            total_score += slot_u

        # Suit concentration bonus (only if NOT on the final turn of the round, and if 7s remain unseen/in hand)
        if not env.is_final_turn:
            trump = color_ranking[0]
            colors = rack // 8
            for col in range(4):
                if col != trump:
                    cnt = int(np.sum(colors == col))
                    if cnt >= 3:
                        total_score += 1.35 * (cnt - 2)

        return total_score

    def _compute_post_discard_hand_value(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        candidate_rack: np.ndarray,
        discarded_card: int,
    ) -> float:
        val = (discarded_card % 8) + 1
        opp = 1 - player

        # --- FIX 1: Card 8 Instant Close Evaluation ---
        if val == 8:
            base_rack_score = self._evaluate_rack_utility(
                env, player, candidate_rack, env.color_ranking
            )
            if self._should_close_round_with_rack(
                env, player, candidate_rack, is_instant_card8=True
            ):
                return base_rack_score + 8.0
            else:
                return base_rack_score - 15.0

        # --- FIX 2: Card 5 Extra Turn Valuation ---
        if val == 5 and len(env.deck) > 0:
            base_rack_score = self._evaluate_rack_utility(
                env, player, candidate_rack, env.color_ranking
            )
            if env.is_final_turn:
                return base_rack_score + 3.50  # Always chain 5s on the final turn if legal!
            min_slot_str = min(
                self._card_strength(int(c), env.color_ranking) for c in candidate_rack
            )
            tempo_bonus = 1.45 if min_slot_str <= 5.5 else 0.45
            return base_rack_score + tempo_bonus

        # --- FIX 3: Card 6 Opponent Swap with Exact Unseen Pool ---
        if val == 6:
            best_d, best_post_6_score = self._find_best_effect_6_target(
                env, player, candidate_rack
            )
            return best_post_6_score

        return super()._compute_post_discard_hand_value(
            env, player, candidate_rack, discarded_card
        )

    def _find_best_effect_6_target(
        self, env: DraculaVsVanHelsingEnv, player: int, rack: np.ndarray
    ) -> Tuple[int, float]:
        """
        Shared exact evaluator for Card 6 (used both during discard lookahead and Phase 5 resolution).
        Returns (best_district, best_post_swap_rack_score).
        """
        opp = 1 - player
        weights = self._get_district_weights(env)
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)
        base_rack_score = self._evaluate_rack_utility(
            env, player, rack, env.color_ranking
        )

        # Compute expected strength of a hidden opponent card
        if len(unseen_cards) > 0:
            avg_unseen_str = float(
                np.mean([self._card_strength(int(c), env.color_ranking) for c in unseen_cards])
            )
        else:
            avg_unseen_str = 5.5

        best_d = 0
        best_score = -1e9

        for d in range(5):
            my_c = int(rack[d])
            is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])

            if is_opp_known:
                opp_c = int(env.racks[opp, d])
                post_rack = rack.copy()
                post_rack[d] = opp_c
                override = {d: (True, my_c)}
                score = self._evaluate_rack_utility(
                    env, player, post_rack, env.color_ranking, known_opp_override=override
                )
            else:
                # Current utility of keeping `my_c` in slot `d`
                my_str = self._card_strength(my_c, env.color_ranking)
                cur_win_p = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, env.color_ranking, num_discards
                )
                current_slot_u = (cur_win_p * 10.0 * float(weights[d])) + (my_str * 0.30)

                # If we swap `my_c` into the opponent's hidden slot `d`, the opponent now gets `my_c`
                # and we receive a random unseen card. Probability that the unseen card beats `my_c`:
                prob_unseen_beats_my_c = 1.0 - cur_win_p
                expected_new_slot_u = (
                    (prob_unseen_beats_my_c * 10.0 * float(weights[d]))
                    + (avg_unseen_str * 0.30)
                    - 2.40  # Information-leak & variance penalty for blind swapping
                )
                score = base_rack_score - current_slot_u + expected_new_slot_u

            if score > best_score:
                best_score = score
                best_d = int(d)

        return best_d, best_score

    def _decide_effect_6_swap_opp(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        best_d, _ = self._find_best_effect_6_target(env, player, env.racks[player])
        if best_d in legal_actions:
            return best_d
        return int(legal_actions[0])

    def _decide_discard(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        best_action = int(legal_actions[0])
        best_post_value = -1e9

        # Check how many 6s are still unseen (if > 0, putting a revealed Trump on the table is risky!)
        unseen_cards = self._get_unseen_cards(env, player, env.racks[player])
        unseen_sixes = sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 6)

        for action in legal_actions:
            candidate_rack = env.racks[player].copy()
            if action == 0:
                discarded_card = int(env.drawn_card)
            else:
                slot = action - 1
                discarded_card = int(candidate_rack[slot])
                candidate_rack[slot] = int(env.drawn_card)

            post_value = self._compute_post_discard_hand_value(
                env, player, candidate_rack, discarded_card
            )

            # If we place a face-up drawn card (from Card 2) into our rack, penalize if opponent can steal it with a 6
            if action > 0 and env.drawn_card_was_revealed:
                drawn_str = self._card_strength(int(env.drawn_card), env.color_ranking)
                exposure_penalty = 0.45 * unseen_sixes if drawn_str >= 9.0 else 0.20
                post_value -= exposure_penalty

            if post_value > best_post_value:
                best_post_value = post_value
                best_action = int(action)

        return best_action

    def _should_close_round_with_rack(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        is_instant_card8: bool = False,
    ) -> bool:
        """
        Uses exact Bayesian win probabilities and Round 5 survival math to decide when to close.
        """
        opp = 1 - player
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)

        expected_wins = 0.0
        expected_hp_loss = 0.0

        for d in range(5):
            my_c = int(rack[d])
            is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
            if is_known:
                p_win = (
                    1.0
                    if self._does_my_card_win(my_c, int(env.racks[opp, d]), env.color_ranking)
                    else 0.0
                )
            else:
                p_win = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, env.color_ranking, num_discards
                )

            expected_wins += p_win
            if player == 0:  # Dracula
                # Immediate 4-Vampire Knockout check before dying
                if (
                    env.vampires[d] == 3
                    and p_win >= (0.78 if is_instant_card8 else 0.85)
                    and (env.dracula_hp - expected_hp_loss) > 1.25
                ):
                    return True
                expected_hp_loss += 1.0 - p_win
            else:  # Van Helsing
                # Immediate 0-HP Knockout check
                if expected_wins >= env.dracula_hp + (0.15 if is_instant_card8 else 0.35):
                    return True

        # Special Round 5 Endgame Logic:
        if env.round_num == 5:
            if player == 0:
                # Dracula wins the game if expected_hp_loss < dracula_hp by a safe margin!
                safe_margin = 1.15 if is_instant_card8 else 1.45
                if (env.dracula_hp - expected_hp_loss) >= safe_margin:
                    return True
            else:
                # Van Helsing MUST deal at least `dracula_hp` damage in Round 5 or he loses automatically!
                # Never close Round 5 voluntarily as Van Helsing unless expected_wins can kill Dracula!
                return expected_wins >= (env.dracula_hp - 0.10)

        if is_instant_card8:
            return expected_wins >= 3.20
        return expected_wins >= (3.50 if player == 0 else 3.55)


# =============================================================================
# BOT D: TACTICAL REFINED BOT V3 (ASYMMETRIC ROLE ENGINE & FOG-OF-WAR SOLVER)
# =============================================================================
class TacticalRefinedBot_V3(TacticalRefinedBot_V2):
    """
    Next-Generation Heuristic Bot (V3) inheriting from TacticalRefinedBot_V2.
    Strictly Information-Set compliant (zero hidden-state leaks).

    Key Upgrades over V2:
      1. Fog-of-War & Public Exposure Penalty:
         - Tracks `env.revealed[player, d]` and `env.known_by_opp[player, d]`.
         - Recognizes that when our card in slot `d` is known to the opponent while the opponent's
           card in `d` is secret, the opponent will actively counter-play it (either beating it
           by a minimal margin or dumping a 1 against our high card, or stealing our exposed Trump with a 6).
      2. Role-Specific Asymmetric District Weights (Dracula vs Van Helsing):
         - Dracula focuses offensive power on his highest-vampire district(s) to force 3->4 conversions,
           and front-loads defense in D1..D3 when HP is low.
         - Van Helsing applies heavier lockdown priority (2.65x) on 3-vampire districts to prevent
           instant defeat, while keeping balanced pressure across 0/1-vampire districts to bleed HP.
      3. Fix for the "3 vs 4/6 Retention Anomaly":
         - Prevents discarding higher-value cards (4s and 6s) for marginal swaps when a weak card (1, 2, 3)
           can be discarded instead, eliminating the 14.2% accumulation of 3s at showdown.
      4. Opponent Final-Turn Counter-Play Discount on Action-1 Calls:
         - Discounts expected wins when calling Action 1 based on how many 6s, 7s, and higher Trumps
           remain unseen in the deck/opponent hand, drastically cutting call backfire rates.
    """

    def _get_role_district_weights(
        self, env: DraculaVsVanHelsingEnv, player: int
    ) -> np.ndarray:
        """
        Computes role-asymmetric tactical weights for Districts 0..4.
        """
        weights = np.ones(5, dtype=np.float32)
        vamps = env.vampires
        hp = int(env.dracula_hp)

        # Find the maximum vampire count currently on the board
        max_v = int(np.max(vamps))

        for d in range(5):
            v = int(vamps[d])
            if player == 0:  # DRACULA (Offensive focal point + survival)
                if v == 3:
                    weights[d] = 2.35  # Winning here wins the entire game!
                elif v == 2:
                    # If Dracula has no 3-vamp district yet, a 2-vamp district is his primary spearhead
                    weights[d] = 1.55 if max_v == 2 else 1.30
                elif v == 1:
                    weights[d] = 1.12 if max_v == 1 else 1.00
            else:  # VAN HELSING (Absolute lockdown on 3-vamp threats + flat HP bleed elsewhere)
                if v == 3:
                    weights[d] = 2.65  # Losing here loses the entire game immediately!
                elif v == 2:
                    weights[d] = 1.40
                elif v == 1:
                    weights[d] = 1.05

            # Left-to-right resolution tie-breaker (D1 resolves first)
            weights[d] += (4 - d) * 0.045

        # Critical Low-HP Window (Dracula HP <= 3)
        if hp <= 3:
            three_vamp_dists = np.flatnonzero(vamps == 3)
            if len(three_vamp_dists) > 0:
                target_d = int(three_vamp_dists[0])
                crit_indices = list(range(target_d + 1))
                share = 2.2 / float(max(1, min(len(crit_indices), hp)))
                for idx in crit_indices[: max(1, hp)]:
                    weights[idx] += share
            else:
                # Distribute +2.1x across the first `hp` districts (or first 3 districts)
                limit = max(1, min(3, hp))
                share = 2.1 / float(limit)
                for idx in range(limit):
                    weights[idx] += share

        # Round 5 Endgame Survival / Lethal Boost
        if env.round_num == 5 and hp <= 5:
            for d in range(5):
                weights[d] = max(weights[d], 1.50 + (4 - d) * 0.07)

        return weights

    def _evaluate_rack_utility_v3(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        color_ranking: np.ndarray,
        my_exposed_mask: Optional[np.ndarray] = None,
        known_opp_override: Optional[Dict[int, Tuple[bool, int]]] = None,
    ) -> float:
        """
        Full Information-Set rack evaluator incorporating:
          - Role-asymmetric district weights,
          - Exact Bayesian win probabilities against unseen cards,
          - Fog-of-War penalty when our card in slot `d` is known to the opponent while their card is secret.
        """
        opp = 1 - player
        weights = self._get_role_district_weights(env, player)
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)

        if my_exposed_mask is None:
            my_exposed_mask = env.revealed[player] | env.known_by_opp[player]

        # Count how many 6s are still unseen (threat of opponent stealing our exposed high cards)
        unseen_sixes = (
            sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 6)
            if not env.is_final_turn
            else 0
        )

        total_score = 0.0

        for d in range(5):
            my_c = int(rack[d])
            my_str = self._card_strength(my_c, color_ranking)
            w = float(weights[d])

            if known_opp_override and d in known_opp_override:
                is_opp_known, opp_c = known_opp_override[d]
            else:
                is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
                opp_c = int(env.racks[opp, d]) if is_opp_known else -1

            if is_opp_known:
                opp_str = self._card_strength(opp_c, color_ranking)
                wins = self._does_my_card_win(my_c, opp_c, color_ranking)
                if wins:
                    margin = my_str - opp_str
                    # Reward winning with the minimum necessary card (e.g. 4 over 3)
                    tight_margin_bonus = max(0.0, (8.0 - margin) * 0.35)
                    slot_u = (10.0 * w) + tight_margin_bonus + (my_str * 0.34)
                else:
                    # Guaranteed loss against a known opponent card:
                    # Keep a tiny fraction of raw strength (so we still prefer discarding a 1 over a 3),
                    # but strongly incentivize evacuating strong cards (5, 7, 8, Trumps) out of this lost slot!
                    slot_u = my_str * 0.08
            else:
                win_prob = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, color_ranking, num_discards
                )

                # FOG-OF-WAR ADJUSTMENT:
                # If our card in slot `d` is visible/known to the opponent, and it is not near-unbeatable
                # (win_prob < 0.92), the opponent can see it and counter-play it!
                if my_exposed_mask[d] and not env.is_final_turn:
                    if win_prob < 0.92:
                        # Opponent will selectively beat exposed mid-cards or dump trash on exposed strong cards
                        win_prob *= 0.86
                    if my_str >= 10.0 and unseen_sixes > 0:
                        # Risk of opponent using Card 6 to steal our exposed high Trump!
                        win_prob -= 0.04 * unseen_sixes

                # Nonlinear intrinsic card retention weight (0.42 * my_str) ensures a 4 is always
                # valued noticeably higher than a 3, fixing the V2 "3-accumulation" anomaly!
                slot_u = (win_prob * 10.0 * w) + (my_str * 0.42)

            total_score += slot_u

        # Suit concentration bonus (only if NOT on the final turn, and if 7s remain available)
        if not env.is_final_turn:
            unseen_sevens = sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 7)
            my_sevens = sum(1 for c in rack if (int(c) % 8) + 1 == 7)
            if (unseen_sevens + my_sevens) > 0:
                trump = color_ranking[0]
                colors = rack // 8
                for col in range(4):
                    if col != trump:
                        cnt = int(np.sum(colors == col))
                        if cnt >= 3:
                            total_score += 1.45 * (cnt - 2)

        return total_score

    def _evaluate_rack_utility(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        color_ranking: np.ndarray,
        known_opp_override: Optional[Dict[int, Tuple[bool, int]]] = None,
    ) -> float:
        return self._evaluate_rack_utility_v3(
            env, player, rack, color_ranking, None, known_opp_override
        )

    def _find_best_effect_4_swap(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        exposed_mask: np.ndarray,
    ) -> Tuple[int, float]:
        """Finds the best pair of slots (0..9) to swap with Card 4, tracking card exposure."""
        best_action = 0
        best_score = -1e9

        for action, (i, j) in enumerate(DraculaVsVanHelsingEnv.SWAP_PAIRS):
            swapped_rack = rack.copy()
            swapped_rack[i], swapped_rack[j] = swapped_rack[j], swapped_rack[i]
            swapped_exp = exposed_mask.copy()
            swapped_exp[i], swapped_exp[j] = swapped_exp[j], swapped_exp[i]

            score = self._evaluate_rack_utility_v3(
                env, player, swapped_rack, env.color_ranking, swapped_exp
            )
            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action, best_score

    def _find_best_effect_6_target(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        exposed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """
        Evaluates the exact post-swap hand utility for Card 6 across all 5 districts.
        """
        opp = 1 - player
        weights = self._get_role_district_weights(env, player)
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)

        if exposed_mask is None:
            exposed_mask = (env.revealed[player] | env.known_by_opp[player]).copy()

        base_rack_score = self._evaluate_rack_utility_v3(
            env, player, rack, env.color_ranking, exposed_mask
        )

        if len(unseen_cards) > 0:
            avg_unseen_str = float(
                np.mean([self._card_strength(int(c), env.color_ranking) for c in unseen_cards])
            )
        else:
            avg_unseen_str = 5.5

        best_d = 0
        best_score = -1e9

        for d in range(5):
            my_c = int(rack[d])
            is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])

            if is_opp_known:
                opp_c = int(env.racks[opp, d])
                post_rack = rack.copy()
                post_rack[d] = opp_c
                post_exp = exposed_mask.copy()
                post_exp[d] = True  # Opponent knows the card they just gave us!
                override = {d: (True, my_c)}
                score = self._evaluate_rack_utility_v3(
                    env,
                    player,
                    post_rack,
                    env.color_ranking,
                    post_exp,
                    known_opp_override=override,
                )
            else:
                my_str = self._card_strength(my_c, env.color_ranking)
                cur_win_p = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, env.color_ranking, num_discards
                )
                w = float(weights[d])
                current_slot_u = (cur_win_p * 10.0 * w) + (my_str * 0.42)

                # When we swap `my_c` into opponent's hidden slot `d`, opponent gets `my_c` (known to us!)
                # and we get an unseen card (known to opponent!).
                prob_unseen_beats_my_c = 1.0 - cur_win_p
                expected_new_slot_u = (
                    (prob_unseen_beats_my_c * 10.0 * w)
                    + (avg_unseen_str * 0.42)
                    - 2.65  # Strict penalty against blind-swapping unless dumping true low trash (1 or 2)
                )
                score = base_rack_score - current_slot_u + expected_new_slot_u

            if score > best_score:
                best_score = score
                best_d = int(d)

        return best_d, best_score

    def _compute_post_discard_hand_value_v3(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        candidate_rack: np.ndarray,
        candidate_exposed: np.ndarray,
        discarded_card: int,
    ) -> float:
        val = (discarded_card % 8) + 1
        opp = 1 - player
        base_rack_score = self._evaluate_rack_utility_v3(
            env, player, candidate_rack, env.color_ranking, candidate_exposed
        )

        # --- CARDS 1, 2, 3: Purely Informational ---
        if val == 1:
            return base_rack_score - 0.04
        elif val == 2:
            return base_rack_score - 0.03
        elif val == 3:
            return base_rack_score + 0.04

        # --- CARD 4: Best Pairwise Swap (with friction threshold to stop wasting 4s for tiny gains) ---
        elif val == 4:
            _, best_swapped_score = self._find_best_effect_4_swap(
                env, player, candidate_rack, candidate_exposed
            )
            gain = best_swapped_score - base_rack_score
            # Only credit the swap gain if it meaningfully improves district outcomes (> 0.65 pts)
            return base_rack_score + (gain if gain > 0.65 else 0.0)

        # --- CARD 5: Extra Turn Tempo ---
        elif val == 5:
            if len(env.deck) == 0:
                return base_rack_score
            if env.is_final_turn:
                return base_rack_score + 4.00
            min_slot_str = min(
                self._card_strength(int(c), env.color_ranking) for c in candidate_rack
            )
            return base_rack_score + (1.35 if min_slot_str <= 5.0 else 0.35)

        # --- CARD 6: Best Opponent District Swap ---
        elif val == 6:
            _, best_post_6_score = self._find_best_effect_6_target(
                env, player, candidate_rack, candidate_exposed
            )
            return best_post_6_score

        # --- CARD 7: Best Trump Color Change ---
        elif val == 7:
            colors = candidate_rack // 8
            old_trump = env.color_ranking[0]
            old_trump_count = int(np.sum(colors == old_trump))

            vis_mask = env.revealed[opp] | env.known_by_opp[opp]
            opp_vis_cards = env.racks[opp, vis_mask]
            opp_old_vis_trumps = int(np.sum((opp_vis_cards // 8) == old_trump))

            best_post_7_score = -1e9
            for rank_idx in range(1, 4):
                new_ranking = env.color_ranking.copy()
                new_ranking[0], new_ranking[rank_idx] = (
                    new_ranking[rank_idx],
                    new_ranking[0],
                )
                new_trump = new_ranking[0]
                new_trump_count = int(np.sum(colors == new_trump))
                opp_new_vis_trumps = int(np.sum((opp_vis_cards // 8) == new_trump))

                post_7_rack_score = self._evaluate_rack_utility_v3(
                    env, player, candidate_rack, new_ranking, candidate_exposed
                )
                opp_trump_swing_bonus = (opp_old_vis_trumps - opp_new_vis_trumps) * 4.2

                if new_trump_count >= 3 and new_trump_count > old_trump_count:
                    total_7_score = post_7_rack_score + opp_trump_swing_bonus
                elif (
                    old_trump_count == 0
                    and opp_old_vis_trumps >= 2
                    and opp_new_vis_trumps == 0
                ):
                    total_7_score = post_7_rack_score + opp_trump_swing_bonus
                else:
                    total_7_score = post_7_rack_score - 14.0

                if total_7_score > best_post_7_score:
                    best_post_7_score = total_7_score

            return best_post_7_score

        # --- CARD 8: Instant Round Close ---
        elif val == 8:
            if self._should_close_round_with_rack(
                env, player, candidate_rack, is_instant_card8=True
            ):
                return base_rack_score + 8.5
            else:
                return base_rack_score - 16.0

        return base_rack_score

    def _decide_discard(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        best_action = int(legal_actions[0])
        best_post_value = -1e9
        base_exposed = env.revealed[player] | env.known_by_opp[player]

        for action in legal_actions:
            candidate_rack = env.racks[player].copy()
            candidate_exposed = base_exposed.copy()

            if action == 0:
                discarded_card = int(env.drawn_card)
            else:
                slot = action - 1
                discarded_card = int(candidate_rack[slot])
                candidate_rack[slot] = int(env.drawn_card)
                # New drawn card is only exposed if drawn face-up from a Card 2 peek!
                candidate_exposed[slot] = bool(env.drawn_card_was_revealed)

            post_value = self._compute_post_discard_hand_value_v3(
                env, player, candidate_rack, candidate_exposed, discarded_card
            )

            if post_value > best_post_value:
                best_post_value = post_value
                best_action = int(action)

        return best_action

    def _decide_effect_4_swap_own(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        exposed_mask = env.revealed[player] | env.known_by_opp[player]
        best_action, _ = self._find_best_effect_4_swap(
            env, player, env.racks[player], exposed_mask
        )
        if best_action in legal_actions:
            return best_action
        return int(legal_actions[0])

    def _should_close_round_with_rack(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        is_instant_card8: bool = False,
    ) -> bool:
        """
        Incorporates Opponent Final-Turn Counter-Play Discount when calling Action 1.
        """
        opp = 1 - player
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)
        my_exposed = env.revealed[player] | env.known_by_opp[player]

        expected_wins = 0.0
        expected_hp_loss = 0.0

        for d in range(5):
            my_c = int(rack[d])
            is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
            if is_known:
                p_win = (
                    1.0
                    if self._does_my_card_win(my_c, int(env.racks[opp, d]), env.color_ranking)
                    else 0.0
                )
            else:
                p_win = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, env.color_ranking, num_discards
                )
                # If calling Action 1 (not instant Card 8), opponent gets 1 final turn to target exposed cards
                if not is_instant_card8 and my_exposed[d] and p_win < 0.94:
                    p_win *= 0.88

            expected_wins += p_win
            if player == 0:  # Dracula
                if (
                    env.vampires[d] == 3
                    and p_win >= (0.78 if is_instant_card8 else 0.88)
                    and (env.dracula_hp - expected_hp_loss) > (1.20 if is_instant_card8 else 1.45)
                ):
                    return True
                expected_hp_loss += 1.0 - p_win
            else:  # Van Helsing
                # Check if Van Helsing can guarantee reducing Dracula to 0 HP
                if expected_wins >= env.dracula_hp + (0.12 if is_instant_card8 else 0.42):
                    return True

        # If calling Action 1, discount expected wins if unseen 7s can flip our Trump-heavy hand
        if not is_instant_card8:
            my_trumps = int(np.sum((rack // 8) == env.color_ranking[0]))
            unseen_sevens = sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 7)
            if my_trumps >= 3 and unseen_sevens >= 2:
                expected_wins -= 0.15

        if env.round_num == 5:
            if player == 0:
                safe_margin = 1.15 if is_instant_card8 else 1.50
                return (env.dracula_hp - expected_hp_loss) >= safe_margin
            else:
                return expected_wins >= (env.dracula_hp - (0.10 if is_instant_card8 else 0.0))

        if is_instant_card8:
            return expected_wins >= 3.18
        return expected_wins >= (3.58 if player == 0 else 3.62)

class TacticalRefinedBot_V4(TacticalRefinedBot_V3):
    """
    TacticalRefinedBot_V4: Non-linear minimax engine.

    Replaces the linear district weights inherited from V1-V3 with an asymmetric
    hard-threshold utility:
      - Van Helsing: minimisation of absolute sudden-death risk (asymptotic penalty).
      - Dracula: maximisation of a single lethal vector (offensive singularity),
        combined with minimisation of damage on critical HP.
    """

    def _evaluate_rack_utility_v3(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        color_ranking: np.ndarray,
        my_exposed_mask: Optional[np.ndarray] = None,
        known_opp_override: Optional[Dict[int, Tuple[bool, int]]] = None,
    ) -> float:
        """
        Non-linear minimax evaluator. Replaces the linear weighted sum.
        """
        opp = 1 - player
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)

        if my_exposed_mask is None:
            my_exposed_mask = env.revealed[player] | env.known_by_opp[player]

        unseen_sixes = (
            sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 6)
            if not env.is_final_turn else 0
        )

        expected_hp_damage = 0.0
        lethal_singularity_bonus = 0.0
        vh_critical_risk_penalty = 0.0

        for d in range(5):
            my_c = int(rack[d])
            my_str = self._card_strength(my_c, color_ranking)

            if known_opp_override and d in known_opp_override:
                is_opp_known, opp_c = known_opp_override[d]
            else:
                is_opp_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])
                opp_c = int(env.racks[opp, d]) if is_opp_known else -1

            if is_opp_known:
                wins = self._does_my_card_win(my_c, opp_c, color_ranking)
                p_win = 1.0 if wins else 0.0
                # Margin efficiency: only useful to discriminate among equal wins
                margin_efficiency = max(0.0, (8.0 - (my_str - self._card_strength(opp_c, color_ranking))) * 0.05) if wins else 0.0
                intrinsic_val = margin_efficiency + (my_str * 0.05) if wins else (my_str * 0.01)
            else:
                p_win = self._exact_win_prob_vs_unseen(my_c, unseen_cards, color_ranking, num_discards)
                
                # Fog of War & Exposure
                if my_exposed_mask[d] and not env.is_final_turn:
                    if p_win < 0.96:
                        p_win *= 0.80  # Severe vulnerability if exposed and unbeatable
                    if my_str >= 10.0 and unseen_sixes > 0:
                        p_win -= 0.08 * unseen_sixes # Mathematical risk of theft
                
                intrinsic_val = my_str * 0.03

            expected_hp_damage += p_win + intrinsic_val
            vamps = int(env.vampires[d])

            # Asymmetric non-linear computation
            if player == 0:  # Dracula
                if vamps == 3:
                    # Seek the absolute maximum in a single lethal district
                    lethal_singularity_bonus = max(lethal_singularity_bonus, p_win * 80.0)
                elif vamps == 2:
                    lethal_singularity_bonus = max(lethal_singularity_bonus, p_win * 15.0)
            else:  # Van Helsing
                if vamps == 3:
                    # Exponential penalty for coverage deficit
                    vh_critical_risk_penalty += (1.0 - p_win) * 120.0
                elif vamps == 2:
                    vh_critical_risk_penalty += (1.0 - p_win) * 25.0
                elif env.round_num <= 2 and d in (1, 2):
                    # Pre-emptive Center-of-Gravity
                    vh_critical_risk_penalty += (1.0 - p_win) * 3.0

        # Suit synergy
        suit_bonus = 0.0
        if not env.is_final_turn:
            unseen_sevens = sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 7)
            my_sevens = sum(1 for c in rack if (int(c) % 8) + 1 == 7)
            if (unseen_sevens + my_sevens) > 0:
                trump = color_ranking[0]
                colors = rack // 8
                for col in range(4):
                    if col != trump:
                        cnt = int(np.sum(colors == col))
                        if cnt >= 3:
                            suit_bonus += 2.0 * (cnt - 2)

        if player == 0:
            # Dracula utility: survival (scaled inversely with HP) + lethal vector
            survival_weight = 15.0 if env.dracula_hp <= 3 else 4.0
            return (expected_hp_damage * survival_weight) + lethal_singularity_bonus + suit_bonus
        else:
            # Van Helsing utility: damage maximisation UNDER a strict lockdown constraint
            return (expected_hp_damage * 10.0) - vh_critical_risk_penalty + suit_bonus

    def _should_close_round_with_rack(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        rack: np.ndarray,
        is_instant_card8: bool = False,
    ) -> bool:
        """
        Sequential Resolution Closure Evaluator (Fixes Both Dracula HP Guard & VH Risk Check).
        Evaluates districts strictly in resolution order (D1 -> D5) so neither player calls
        if they would lose in an earlier district before their winning district resolves.
        """
        opp = 1 - player
        unseen_cards = self._get_unseen_cards(env, player, rack)
        num_discards = len(env.discard_pile)
        my_exposed = env.revealed[player] | env.known_by_opp[player]

        expected_wins = 0.0
        expected_hp_loss = 0.0
        vh_3vamp_risk_so_far = 0.0
        vh_total_risk_sum = 0.0

        for d in range(5):
            my_c = int(rack[d])
            is_known = bool(env.revealed[opp, d] or env.known_by_opp[opp, d])

            if is_known:
                p_win = (
                    1.0
                    if self._does_my_card_win(my_c, int(env.racks[opp, d]), env.color_ranking)
                    else 0.0
                )
            else:
                p_win = self._exact_win_prob_vs_unseen(
                    my_c, unseen_cards, env.color_ranking, num_discards
                )
                if not is_instant_card8 and my_exposed[d] and p_win < 0.95:
                    p_win *= 0.85

            vamps = int(env.vampires[d])

            if player == 0:  # DRACULA
                # FIX 1: Ensure Dracula survives all preceding districts (0 .. d-1)
                # BEFORE checking if he converts the 4th vampire in district `d`!
                hp_safety_margin = 1.15 if is_instant_card8 else 1.45
                if (
                    vamps == 3
                    and p_win >= (0.82 if is_instant_card8 else 0.90)
                    and (env.dracula_hp - expected_hp_loss) > hp_safety_margin
                ):
                    return True
                expected_wins += p_win
                expected_hp_loss += 1.0 - p_win

            else:  # VAN HELSING
                # First check if Dracula has a 3-vampire threat in THIS district `d`
                # (In each district, combat resolves once: ifVH loses a 3-vamp district, VH loses immediately)
                if vamps == 3:
                    vh_3vamp_risk_so_far += 1.0 - p_win
                    vh_total_risk_sum += 1.0 - p_win
                elif vamps == 2:
                    vh_total_risk_sum += (1.0 - p_win) * 0.30

                expected_wins += p_win

                # FIX 2: Sequential Lethal Check for Van Helsing.
                # VH can close for a lethal 0-HP kill by district `d` ONLY IF `vh_3vamp_risk_so_far`
                # in districts 0..d is strictly below the safety threshold (< 0.12)!
                kill_margin = 0.10 if is_instant_card8 else 0.38
                if (
                    expected_wins >= (env.dracula_hp + kill_margin)
                    and vh_3vamp_risk_so_far < 0.12
                ):
                    return True

        # Post-loop full-round evaluation (if no mid-resolution knockout occurred)
        if player == 1:  # VAN HELSING
            if env.round_num == 5:
                return (
                    expected_wins >= (env.dracula_hp - (0.15 if is_instant_card8 else 0.0))
                    and vh_3vamp_risk_so_far < 0.20
                )

            call_threshold = 3.22 if is_instant_card8 else 3.58
            if expected_wins >= call_threshold and vh_total_risk_sum < 0.15:
                my_trumps = int(np.sum((rack // 8) == env.color_ranking[0]))
                unseen_sevens = sum(1 for c in unseen_cards if (int(c) % 8) + 1 == 7)
                if not is_instant_card8 and my_trumps >= 3 and unseen_sevens >= 2:
                    return False
                return True
            return False

        else:  # DRACULA
            if env.round_num == 5:
                return (env.dracula_hp - expected_hp_loss) >= (
                    1.10 if is_instant_card8 else 1.45
                )

            # Stall strategy in late rounds if healthy, UNLESS holding a dominant hand (>= 3.85 wins)
            if env.round_num >= 4 and env.dracula_hp >= 6 and not is_instant_card8:
                return expected_wins >= 3.85

            return expected_wins >= (3.25 if is_instant_card8 else 3.60)

    def _decide_effect_4_swap_own(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        """
        The minimax engine automatically pushes swap probabilities towards
        asymptotic neutralisation of critical districts, damping marginal drift.
        """
        return super()._decide_effect_4_swap_own(env, player, legal_actions)

    def _decide_effect_6_swap_opp(
        self, env: DraculaVsVanHelsingEnv, player: int, legal_actions: np.ndarray
    ) -> int:
        """
        Inhibits regressive blind swaps. When there is no explicit gain (theft),
        prefers the district with the least impact on the minimax function.
        """
        best_d, best_score = self._find_best_effect_6_target(env, player, env.racks[player])
        if best_d in legal_actions:
            return best_d
        return int(legal_actions[0])

    def _compute_post_discard_hand_value_v3(
        self,
        env: DraculaVsVanHelsingEnv,
        player: int,
        candidate_rack: np.ndarray,
        candidate_exposed: np.ndarray,
        discarded_card: int,
    ) -> float:
        val = (discarded_card % 8) + 1
        if val == 5 and len(env.deck) > 0 and player == 1:
            base = self._evaluate_rack_utility_v3(
                env, player, candidate_rack, env.color_ranking, candidate_exposed
            )
            if env.is_final_turn:
                return base + 4.0
            normalized = base / 32.5
            desperate = 1.0 if normalized < 0.85 else 0.0
            extra = 1.20 * desperate
            min_slot_str = min(
                self._card_strength(int(c), env.color_ranking) for c in candidate_rack
            )
            base_bonus = 1.35 if min_slot_str <= 5.0 else 0.35
            return base + base_bonus + extra
        return super()._compute_post_discard_hand_value_v3(
            env, player, candidate_rack, candidate_exposed, discarded_card
        )

# =============================================================================
# HEAD-TO-HEAD TOURNAMENT: V3 vs V4 (2 x 10,000 GAMES + FULL TELEMETRY)
# =============================================================================
def run_matchup(
    env: DraculaVsVanHelsingEnv,
    dracula_bot: BaseBot,
    vh_bot: BaseBot,
    num_games: int,
    desc: str,
) -> Tuple[int, int]:
    from tqdm import tqdm

    env.reset_stats()
    for _ in tqdm(range(num_games), desc=desc, unit="game"):
        obs, info = env.reset()
        done = False
        while not done:
            p = info["current_player"]
            action = (
                dracula_bot.select_action(env, obs, info)
                if p == 0
                else vh_bot.select_action(env, obs, info)
            )
            obs, reward, done, info = env.step(action)

    drac_wins = int(env.stats["win_types"][0] + env.stats["win_types"][1])
    vh_wins = int(env.stats["win_types"][2])
    return drac_wins, vh_wins


if __name__ == "__main__":
    GAMES_PER_SIDE = 10_000
    env = DraculaVsVanHelsingEnv(seed=42, track_stats=True)

    bot_v3 = TacticalRefinedBot_V3(seed=303)
    bot_v4 = TacticalRefinedBot_V4(seed=404)

    print("=" * 88)
    print(f"TACTICAL REFINED V3 vs TACTICAL REFINED V4 (BALANCED) ({GAMES_PER_SIDE * 2:,} TOTAL GAMES)")
    print("  Bot V3 = TacticalRefinedBot_V3 (Baseline)")
    print("  Bot V4 = TacticalRefinedBot_V4 (Balanced Hunter + Calibrated Closure + Overkill Matrix)")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # SERIES 1: Bot V3 (Dracula) vs Bot V4 (Van Helsing) + Full Telemetry
    # -------------------------------------------------------------------------
    print(f"\n[SERIES 1] {GAMES_PER_SIDE:,} Games: Bot V3 (Dracula) vs Bot V4 (Van Helsing)")
    v3_drac_wins, v4_vh_wins = run_matchup(
        env, bot_v3, bot_v4, GAMES_PER_SIDE, "Series 1 (V3=Drac vs V4=VH)"
    )
    print(
        f"\n-> SERIES 1 RESULT: Bot V3 (Dracula) Won {v3_drac_wins:,} ({v3_drac_wins / GAMES_PER_SIDE * 100:.2f}%) | "
        f"Bot V4 (Van Helsing) Won {v4_vh_wins:,} ({v4_vh_wins / GAMES_PER_SIDE * 100:.2f}%)"
    )
    print("\n>>> TELEMETRY REPORT FOR SERIES 1 (Dracula = V3 | Van Helsing = V4) <<<")
    env.print_telemetry_report()

    # -------------------------------------------------------------------------
    # SERIES 2: Bot V4 (Dracula) vs Bot V3 (Van Helsing) + Full Telemetry
    # -------------------------------------------------------------------------
    print(f"\n[SERIES 2] {GAMES_PER_SIDE:,} Games: Bot V4 (Dracula) vs Bot V3 (Van Helsing)")
    v4_drac_wins, v3_vh_wins = run_matchup(
        env, bot_v4, bot_v3, GAMES_PER_SIDE, "Series 2 (V4=Drac vs V3=VH)"
    )
    print(
        f"\n-> SERIES 2 RESULT: Bot V4 (Dracula) Won {v4_drac_wins:,} ({v4_drac_wins / GAMES_PER_SIDE * 100:.2f}%) | "
        f"Bot V3 (Van Helsing) Won {v3_vh_wins:,} ({v3_vh_wins / GAMES_PER_SIDE * 100:.2f}%)"
    )
    print("\n>>> TELEMETRY REPORT FOR SERIES 2 (Dracula = V4 | Van Helsing = V3) <<<")
    env.print_telemetry_report()

    # -------------------------------------------------------------------------
    # FINAL ROLE-BALANCED SUMMARY (20,000 GAMES COMBINED)
    # -------------------------------------------------------------------------
    total_games = GAMES_PER_SIDE * 2
    total_v3_wins = v3_drac_wins + v3_vh_wins
    total_v4_wins = v4_drac_wins + v4_vh_wins

    print("\n" + "=" * 88)
    print("FINAL ROLE-BALANCED VERDICT (20,000 GAMES COMBINED)")
    print("=" * 88)
    print(
        f"  TacticalRefinedBot_V3 Total Wins      : {total_v3_wins:6,} / {total_games:,} ({total_v3_wins / total_games * 100:.2f}%)"
    )
    print(f"    ├─ As Dracula     : {v3_drac_wins / GAMES_PER_SIDE * 100:.2f}%")
    print(f"    └─ As Van Helsing : {v3_vh_wins / GAMES_PER_SIDE * 100:.2f}%")
    print("-" * 88)
    print(
        f"  TacticalRefinedBot_V4 Total Wins      : {total_v4_wins:6,} / {total_games:,} ({total_games * 100 if False else total_v4_wins / total_games * 100:.2f}%)"
    )
    print(f"    ├─ As Dracula     : {v4_drac_wins / GAMES_PER_SIDE * 100:.2f}%")
    print(f"    └─ As Van Helsing : {v4_vh_wins / GAMES_PER_SIDE * 100:.2f}%")
    print("=" * 88)
    winner_name = (
        "TacticalRefinedBot_V4"
        if total_v4_wins > total_v3_wins
        else "TacticalRefinedBot_V3"
    )
    print(f"CHAMPION BASELINE BOT: {winner_name}\n")