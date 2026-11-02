import time
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from obs_features import build_features

class DraculaVsVanHelsingEnv:
    """
    Production-ready, audited Reinforcement Learning environment for 'Dracula vs Van Helsing'.
    Features:
      - Finite State Machine (FSM) with 10-action discrete space and exact legal Action Masking.
      - Leak-free 655-feature Information Set observation vector for Imperfect-Information RL.
      - Built-in multi-agent step deltas (HP lost, Vampires added, Dual Rewards) in `info` for Step 2/3.
      - Complete 12-section Game-Theoretic & Tactical Telemetry Engine.

    Players:
        0 = Dracula
        1 = Van Helsing

    Card Encoding (integers 0..31):
        color = card_id // 8        (0=Red, 1=Yellow, 2=Purple, 3=Green)
        value = (card_id % 8) + 1   (values 1 to 8)
    """

    # Finite State Machine (FSM) Phases
    PHASE_TURN_START = 0         # Actions: 0=Draw card, 1=Call End of Round
    PHASE_CHOOSE_DISCARD = 1     # Actions: 0=Discard drawn card, 1..5=Swap with rack slot 0..4
    PHASE_EFFECT_1_REV_OWN = 2   # Actions: 0..4=Reveal your own hidden card in slot 0..4
    PHASE_EFFECT_3_REV_OPP = 3   # Actions: 0..4=Reveal opponent's hidden card in slot 0..4
    PHASE_EFFECT_4_SWAP_OWN = 4  # Actions: 0..9=Swap one of the 10 pairs of your own slots
    PHASE_EFFECT_6_SWAP_OPP = 5  # Actions: 0..4=Swap cards with opponent in district 0..4
    PHASE_EFFECT_7_TRUMP = 6     # Actions: 1..3=Swap Trump (rank 0) with color at rank 1, 2, or 3
    PHASE_GAME_OVER = 7

    # Observation & Action Space Constants (for Gymnasium / PettingZoo / PyTorch wrappers)
    OBS_DIM = 655
    ACTION_DIM = 10

    # All 10 possible slot-pair combinations for Card 4's swap effect
    SWAP_PAIRS = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
    ]

    def __init__(
        self,
        seed: Optional[int] = None,
        always_dracula_first: bool = False,
        track_stats: bool = True,
    ):
        """
        Args:
            seed: Optional random seed for reproducibility.
            always_dracula_first: If True, Dracula starts every round (1..5).
                                  If False, turn order continues naturally across rounds.
            track_stats: If True, records comprehensive gameplay telemetry across games.
        """
        self.rng = np.random.default_rng(seed)
        self._version = 0
        self._obs_cache = {}
        self.return_obs_from_step = True
        self.always_dracula_first = always_dracula_first
        self.track_stats = track_stats
        self.reset_stats()
        self.reset()

    # =========================================================================
    # 0. COMPREHENSIVE TELEMETRY AND STATISTICS TRACKER
    # =========================================================================
    def reset_stats(self):
        """Resets all cumulative statistical counters to zero."""
        self.stats = {
            # --- [1] Macro Game, Pacing, Margin of Victory & Comebacks ---
            "games_played": 0,
            "total_rounds_played": 0,
            "total_turns_played": 0,
            "total_discards": 0,
            "win_types": np.zeros(3, dtype=np.int64),          # [0=Drac_4_Vamps, 1=Drac_R5_Surv, 2=VH_0_HP]
            "game_end_round": np.zeros(5, dtype=np.int64),     # Rounds 1..5
            "decisive_district": np.zeros((2, 5), dtype=np.int64),
            "drac_hp_on_drac_win": 0,
            "max_vamps_on_vh_win": 0,
            "total_vamps_on_vh_win": 0,
            "drac_comeback_wins_low_hp": 0,    # Dracula wins after reaching <= 3 HP
            "vh_comeback_wins_3vamps": 0,      # Van Helsing wins after at least one district reached 3 Vampires

            # --- [2 & 3] Round Triggers & Full-Round Score Distribution ---
            "round_end_reasons": np.zeros(3, dtype=np.int64),  # [0=Call, 1=Card_8, 2=Deck_Empty]
            "full_round_drac_wins_dist": np.zeros(6, dtype=np.int64),

            # --- [4] Initiative, Calling Timing, Efficiency & Backfire Rates ---
            "round_starter_count": np.zeros(2, dtype=np.int64),
            "round_starter_district_wins": 0,
            "round_second_district_wins": 0,
            "calls_by_player": np.zeros(2, dtype=np.int64),
            "call_discard_pile_sum": np.zeros(2, dtype=np.int64),   # Sum of len(discard_pile) when Action 1 is called
            "card8_end_by_player": np.zeros(2, dtype=np.int64),
            "card8_discard_pile_sum": np.zeros(2, dtype=np.int64),  # Sum of len(discard_pile) before Card 8 is discarded
            "closer_initiations": np.zeros(2, dtype=np.int64),
            "closer_districts_won": np.zeros(2, dtype=np.int64),
            "closer_backfires": np.zeros(2, dtype=np.int64),
            "final_turn_card5_extensions": np.zeros(2, dtype=np.int64),

            # --- [5] Hand Upgrade Efficiency (Initial Deal vs Showdown) ---
            "initial_val_sum": np.zeros(2, dtype=np.int64),
            "initial_trumps_sum": np.zeros(2, dtype=np.int64),
            "untouched_slots_at_showdown": np.zeros(2, dtype=np.int64),

            # --- [6] Tactical Card Effect Mastery (Cards 1, 2, 3, 4, 6, 7) ---
            "trump_changes_by_player": np.zeros(2, dtype=np.int64),
            "trump_promoted_from_rank": np.zeros(4, dtype=np.int64),
            "effect7_own_trump_gain_sum": np.zeros(2, dtype=np.int64),         # New trumps in own rack - Old trumps in own rack
            "effect7_vis_opp_trump_delta_sum": np.zeros(2, dtype=np.int64),    # Change in known/visible opponent trumps
            "effect6_count": np.zeros(2, dtype=np.int64),
            "effect6_net_power_delta_sum": np.zeros(2, dtype=np.float64),
            "effect6_stole_trump_count": np.zeros(2, dtype=np.int64),
            "effect4_count": np.zeros(2, dtype=np.int64),
            "effect4_high_card_to_threat": np.zeros(2, dtype=np.int64),
            "effect1_revealed_val_sum": np.zeros(2, dtype=np.int64),           # Sum of card values self-revealed via Card 1
            "effect1_count": np.zeros(2, dtype=np.int64),
            "revealed_top_draws": np.zeros(2, dtype=np.int64),                 # Times player drew a face-up card (after Card 2)
            "revealed_top_kept": np.zeros(2, dtype=np.int64),                  # Times player kept that face-up card in rack
            "wasted_reveals_card1": np.zeros(2, dtype=np.int64),
            "wasted_reveals_card3": np.zeros(2, dtype=np.int64),

            # --- [7] Card Economy & Discards ---
            "discarded_values": np.zeros((2, 8), dtype=np.int64),
            "discarded_trumps": np.zeros(2, dtype=np.int64),
            "discard_source": np.zeros((2, 6), dtype=np.int64),

            # --- [8 & 9] Showdown Composition & Combat Physics ---
            "showdown_values": np.zeros((2, 5, 8), dtype=np.int64),
            "showdown_trumps": np.zeros((2, 5), dtype=np.int64),
            "trumps_per_hand_dist": np.zeros((2, 6), dtype=np.int64),
            "district_wins": np.zeros((2, 5), dtype=np.int64),
            "combat_win_types": np.zeros(4, dtype=np.int64),
            "non_trump_win_margin_sum": np.zeros(2, dtype=np.int64),
            "non_trump_win_count": np.zeros(2, dtype=np.int64),

            # --- [10] Information Asymmetry & Fog-of-War ---
            "showdown_visibility": np.zeros((2, 3), dtype=np.int64),
            "wins_by_visibility": np.zeros((2, 2), dtype=np.int64),

            # --- [11 & 12] Spatial & Asymmetric Urgency State Telemetry ---
            "effect_district_targets": {
                "effect_1_rev_own": np.zeros((2, 5), dtype=np.int64),
                "effect_3_rev_opp": np.zeros((2, 5), dtype=np.int64),
                "effect_4_swap_own": np.zeros((2, 5), dtype=np.int64),
                "effect_6_swap_opp": np.zeros((2, 5), dtype=np.int64),
            },
            "vamp_state_combats": np.zeros(4, dtype=np.int64),
            "vamp_state_drac_wins": np.zeros(4, dtype=np.int64),
            "vamp_state_val_sum": np.zeros((2, 4), dtype=np.int64),
            "vamp_state_trumps": np.zeros((2, 4), dtype=np.int64),
            "crit_hp_rounds": 0,
            "crit_hp_early_val_sum": np.zeros(2, dtype=np.int64),
            "crit_hp_late_val_sum": np.zeros(2, dtype=np.int64),
            "crit_hp_early_trumps": np.zeros(2, dtype=np.int64),
            "crit_hp_late_trumps": np.zeros(2, dtype=np.int64),
        }

    # =========================================================================
    # 1. GAME SETUP AND INITIALIZATION
    # =========================================================================
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to the beginning of Round 1."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._version += 1
        self.round_num = 1
        self.dracula_hp = 12
        self.vampires = np.zeros(5, dtype=np.int8)
        self.color_ranking = self.rng.permutation(4).astype(np.int8)

        self.current_player = 0
        self.done = False
        self.winner = None
        self.win_reason = ""

        # Step-level delta trackers for RL reward shaping / logging
        self.step_hp_lost = 0
        self.step_vampires_added = 0
        self.round_just_ended = False

        self._start_new_round()
        return self.get_observation(self.current_player), self._get_info()

    def _start_new_round(self):
        """Shuffles all 32 cards, deals 5 cards to each player in order, and resets round state."""
        deck_array = self.rng.permutation(32).astype(np.int8)

        self.racks = np.zeros((2, 5), dtype=np.int8)
        self.racks[0] = deck_array[0:5]
        self.racks[1] = deck_array[5:10]

        self.slot_untouched = np.ones((2, 5), dtype=bool)

        self.deck = list(deck_array[10:])
        self.discard_pile = []
        # PUBLIC history (leak-free): who discarded what, how often each slot was replaced
        self.discard_owner = np.full(32, -1, dtype=np.int8)
        self.slot_change_count = np.zeros((2, 5), dtype=np.int8)
        self.turns_taken = np.zeros(2, dtype=np.int8)

        self.revealed = np.zeros((2, 5), dtype=bool)
        self.known_by_opp = np.zeros((2, 5), dtype=bool)

        self.top_deck_revealed = False
        self.drawn_card = -1
        self.drawn_card_was_revealed = False

        self.end_of_round_called = False
        self.caller_player = -1
        self.round_closer_player = -1
        self.is_final_turn = False

        # Public action context for leak-free Information Set observations
        self.last_public_event = np.zeros(7, dtype=np.float32)

        if self.always_dracula_first:
            self.current_player = 0

        self.round_starter = self.current_player

        if self.track_stats:
            self.stats["round_starter_count"][self.round_starter] += 1
            trump_col = self.color_ranking[0]
            for p in range(2):
                for d in range(5):
                    c = self.racks[p, d]
                    self.stats["initial_val_sum"][p] += (c % 8) + 1
                    if (c // 8) == trump_col:
                        self.stats["initial_trumps_sum"][p] += 1

        self.phase = self.PHASE_TURN_START

    # =========================================================================
    # 2. LEGAL ACTION MASKING
    # =========================================================================
    def get_legal_actions(self) -> np.ndarray:
        """
        Returns a binary mask of length 10 (dtype=int8).
        1 = Legal action in the current phase, 0 = Illegal action.
        """
        mask = np.zeros(10, dtype=np.int8)
        if self.done or self.phase == self.PHASE_GAME_OVER:
            return mask

        p = self.current_player
        opp = 1 - p

        if self.phase == self.PHASE_TURN_START:
            mask[0] = 1
            if len(self.discard_pile) >= 6 and not self.end_of_round_called:
                mask[1] = 1

        elif self.phase == self.PHASE_CHOOSE_DISCARD:
            can_discard_8 = len(self.discard_pile) >= 6
            drawn_val = (self.drawn_card % 8) + 1
            if drawn_val != 8 or can_discard_8:
                mask[0] = 1

            for slot in range(5):
                slot_card_val = (self.racks[p, slot] % 8) + 1
                if slot_card_val != 8 or can_discard_8:
                    mask[slot + 1] = 1

        elif self.phase == self.PHASE_EFFECT_1_REV_OWN:
            for slot in range(5):
                if not self.revealed[p, slot]:
                    mask[slot] = 1

        elif self.phase == self.PHASE_EFFECT_3_REV_OPP:
            for slot in range(5):
                if not self.revealed[opp, slot]:
                    mask[slot] = 1

        elif self.phase == self.PHASE_EFFECT_4_SWAP_OWN:
            mask[0:10] = 1

        elif self.phase == self.PHASE_EFFECT_6_SWAP_OPP:
            mask[0:5] = 1

        elif self.phase == self.PHASE_EFFECT_7_TRUMP:
            mask[1:4] = 1

        return mask

    # =========================================================================
    # 3. STEP FUNCTION AND FINITE STATE MACHINE TRANSITIONS
    # =========================================================================
    def _card_combat_score(self, card_id: int) -> float:
        """Quantifies a card's current combat strength (0.7 to 16.0) for tactical telemetry."""
        col, val = card_id // 8, (card_id % 8) + 1
        if col == self.color_ranking[0]:
            return 8.0 + val
        rank = int(np.where(self.color_ranking == col)[0][0])
        return float(val) - (rank * 0.1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes a single decision step for `self.current_player`.
        Returns: (next_player_observation, dracula_terminal_reward, done, info_dict)
        """
        self._version += 1
        legal_mask = self.get_legal_actions()
        assert 0 <= action < 10 and legal_mask[action] == 1, (
            f"Illegal action {action} in phase {self.phase}! Valid mask: {legal_mask}"
        )

        # Reset per-step delta indicators
        self.step_hp_lost = 0
        self.step_vampires_added = 0
        self.round_just_ended = False

        p = self.current_player
        opp = 1 - p

        # --- PHASE 0: TURN START ---
        if self.phase == self.PHASE_TURN_START:
            if self.track_stats:
                self.stats["total_turns_played"] += 1

            if action == 0:  # Draw top card from deck
                self.drawn_card = self.deck.pop(0)
                self.turns_taken[p] += 1
                self.drawn_card_was_revealed = self.top_deck_revealed
                if self.track_stats and self.drawn_card_was_revealed:
                    self.stats["revealed_top_draws"][p] += 1
                self.top_deck_revealed = False
                self.phase = self.PHASE_CHOOSE_DISCARD

            elif action == 1:  # Call End of Round
                self.end_of_round_called = True
                self.caller_player = p
                self.round_closer_player = p
                if self.track_stats:
                    self.stats["calls_by_player"][p] += 1
                    self.stats["call_discard_pile_sum"][p] += len(self.discard_pile)
                self.current_player = opp
                self.is_final_turn = True
                self.phase = self.PHASE_TURN_START

        # --- PHASE 1: CHOOSE WHICH CARD TO DISCARD ---
        elif self.phase == self.PHASE_CHOOSE_DISCARD:
            self.last_public_event.fill(0.0)
            self.last_public_event[0] = float(p)

            if action == 0:
                discarded_card = self.drawn_card
            else:
                slot = action - 1
                discarded_card = self.racks[p, slot]
                self.racks[p, slot] = self.drawn_card
                self.slot_untouched[p, slot] = False
                self.slot_change_count[p, slot] += 1
                self.revealed[p, slot] = self.drawn_card_was_revealed
                self.known_by_opp[p, slot] = self.drawn_card_was_revealed
                self.last_public_event[2 + slot] = 1.0
                if self.track_stats and self.drawn_card_was_revealed:
                    self.stats["revealed_top_kept"][p] += 1

            disc_val = (discarded_card % 8) + 1
            self.discard_owner[int(discarded_card)] = p
            self.last_public_event[1] = disc_val / 8.0

            if self.track_stats:
                self.stats["total_discards"] += 1
                self.stats["discarded_values"][p, disc_val - 1] += 1
                self.stats["discard_source"][p, action] += 1
                if (discarded_card // 8) == self.color_ranking[0]:
                    self.stats["discarded_trumps"][p] += 1
                if disc_val == 8:
                    # Record how many cards were already in the discard pile BEFORE this 8 was added
                    self.stats["card8_discard_pile_sum"][p] += len(self.discard_pile)

            self.drawn_card = -1
            self.drawn_card_was_revealed = False
            self.discard_pile.append(discarded_card)

            self._trigger_card_effect(discarded_card)

        # --- PHASE 2: CARD 1 EFFECT (REVEAL 1 OWN CARD) ---
        elif self.phase == self.PHASE_EFFECT_1_REV_OWN:
            self.revealed[p, action] = True
            self.known_by_opp[p, action] = True
            self.last_public_event[2 + action] = 1.0
            if self.track_stats:
                self.stats["effect_district_targets"]["effect_1_rev_own"][p, action] += 1
                self.stats["effect1_count"][p] += 1
                self.stats["effect1_revealed_val_sum"][p] += (self.racks[p, action] % 8) + 1
            self._end_turn(discarded_val=1)

        # --- PHASE 3: CARD 3 EFFECT (REVEAL 1 OPPONENT CARD) ---
        elif self.phase == self.PHASE_EFFECT_3_REV_OPP:
            self.revealed[opp, action] = True
            self.known_by_opp[opp, action] = True
            self.last_public_event[2 + action] = 1.0
            if self.track_stats:
                self.stats["effect_district_targets"]["effect_3_rev_opp"][p, action] += 1
            self._end_turn(discarded_val=3)

        # --- PHASE 4: CARD 4 EFFECT (SWAP 2 OWN CARDS) ---
        elif self.phase == self.PHASE_EFFECT_4_SWAP_OWN:
            i, j = self.SWAP_PAIRS[action]
            if self.track_stats:
                self.stats["effect4_count"][p] += 1
                self.stats["effect_district_targets"]["effect_4_swap_own"][p, i] += 1
                self.stats["effect_district_targets"]["effect_4_swap_own"][p, j] += 1
                score_i = self._card_combat_score(self.racks[p, i])
                score_j = self._card_combat_score(self.racks[p, j])
                if (score_i > score_j and self.vampires[j] > self.vampires[i]) or (
                    score_j > score_i and self.vampires[i] > self.vampires[j]
                ):
                    self.stats["effect4_high_card_to_threat"][p] += 1

            self.racks[p, i], self.racks[p, j] = self.racks[p, j], self.racks[p, i]
            self.slot_untouched[p, i] = False
            self.slot_untouched[p, j] = False
            self.revealed[p, i], self.revealed[p, j] = self.revealed[p, j], self.revealed[p, i]
            self.known_by_opp[p, i], self.known_by_opp[p, j] = (
                self.known_by_opp[p, j],
                self.known_by_opp[p, i],
            )
            self.last_public_event[2 + i] = 1.0
            self.last_public_event[2 + j] = 1.0
            self._end_turn(discarded_val=4)

        # --- PHASE 5: CARD 6 EFFECT (SWAP 1 CARD WITH OPPONENT IN SAME DISTRICT) ---
        elif self.phase == self.PHASE_EFFECT_6_SWAP_OPP:
            d = action
            if self.track_stats:
                self.stats["effect6_count"][p] += 1
                self.stats["effect_district_targets"]["effect_6_swap_opp"][p, d] += 1
                given_score = self._card_combat_score(self.racks[p, d])
                received_score = self._card_combat_score(self.racks[opp, d])
                self.stats["effect6_net_power_delta_sum"][p] += (received_score - given_score)
                if (self.racks[opp, d] // 8) == self.color_ranking[0] and (
                    self.racks[p, d] // 8
                ) != self.color_ranking[0]:
                    self.stats["effect6_stole_trump_count"][p] += 1

            self.racks[p, d], self.racks[opp, d] = self.racks[opp, d], self.racks[p, d]
            self.slot_untouched[p, d] = False
            self.slot_untouched[opp, d] = False
            self.revealed[p, d], self.revealed[opp, d] = self.revealed[opp, d], self.revealed[p, d]
            self.known_by_opp[p, d] = True
            self.known_by_opp[opp, d] = True
            self.last_public_event[2 + d] = 1.0
            self._end_turn(discarded_val=6)

        # --- PHASE 6: CARD 7 EFFECT (SWAP TRUMP TOKEN WITH ANOTHER COLOR TOKEN) ---
        elif self.phase == self.PHASE_EFFECT_7_TRUMP:
            old_trump = self.color_ranking[0]
            new_trump = self.color_ranking[action]

            if self.track_stats:
                self.stats["trump_changes_by_player"][p] += 1
                self.stats["trump_promoted_from_rank"][action] += 1
                # 1. Own hand Trump gain (fully known to active player)
                my_old_t = int(np.sum((self.racks[p] // 8) == old_trump))
                my_new_t = int(np.sum((self.racks[p] // 8) == new_trump))
                self.stats["effect7_own_trump_gain_sum"][p] += (my_new_t - my_old_t)

                # 2. Known/visible opponent Trump delta (only cards known to active player)
                vis_mask = self.revealed[opp] | self.known_by_opp[opp]
                opp_vis_cards = self.racks[opp, vis_mask]
                opp_old_vis_t = int(np.sum((opp_vis_cards // 8) == old_trump))
                opp_new_vis_t = int(np.sum((opp_vis_cards // 8) == new_trump))
                self.stats["effect7_vis_opp_trump_delta_sum"][p] += (opp_new_vis_t - opp_old_vis_t)

            self.color_ranking[0], self.color_ranking[action] = (
                self.color_ranking[action],
                self.color_ranking[0],
            )
            self._end_turn(discarded_val=7)

        dracula_reward = 0.0
        if self.done:
            dracula_reward = 1.0 if self.winner == 0 else -1.0

        obs_out = self.get_observation(self.current_player) if self.return_obs_from_step else None
        return obs_out, dracula_reward, self.done, self._get_info()

    # =========================================================================
    # 4. CARD EFFECTS AND END-OF-TURN / END-OF-ROUND LOGIC
    # =========================================================================
    def _trigger_card_effect(self, card_id: int):
        val = (card_id % 8) + 1
        p = self.current_player
        opp = 1 - p

        if val == 1:
            if np.all(self.revealed[p]):
                if self.track_stats:
                    self.stats["wasted_reveals_card1"][p] += 1
                self._end_turn(discarded_val=1)
            else:
                self.phase = self.PHASE_EFFECT_1_REV_OWN
        elif val == 2:
            if len(self.deck) > 0:
                self.top_deck_revealed = True
            self._end_turn(discarded_val=2)
        elif val == 3:
            if np.all(self.revealed[opp]):
                if self.track_stats:
                    self.stats["wasted_reveals_card3"][p] += 1
                self._end_turn(discarded_val=3)
            else:
                self.phase = self.PHASE_EFFECT_3_REV_OPP
        elif val == 4:
            self.phase = self.PHASE_EFFECT_4_SWAP_OWN
        elif val == 5:
            self._end_turn(discarded_val=5)
        elif val == 6:
            self.phase = self.PHASE_EFFECT_6_SWAP_OPP
        elif val == 7:
            self.phase = self.PHASE_EFFECT_7_TRUMP
        elif val == 8:
            self._end_turn(discarded_val=8)

    def _end_turn(self, discarded_val: int):
        p = self.current_player
        opp = 1 - p

        # Priority 1: Card 8 discarded -> Round ends immediately
        if discarded_val == 8:
            self.round_closer_player = p
            if self.track_stats:
                self.stats["round_end_reasons"][1] += 1
                self.stats["card8_end_by_player"][p] += 1
            self.current_player = opp
            self._resolve_round()
            return

        # Priority 2: Draw deck is empty -> Round ends immediately
        if len(self.deck) == 0:
            self.round_closer_player = -1
            if self.track_stats:
                self.stats["round_end_reasons"][2] += 1
            self.current_player = opp
            self._resolve_round()
            return

        # Priority 3: Card 5 discarded (and deck is not empty) -> Extra turn
        if discarded_val == 5:
            if self.is_final_turn and self.track_stats:
                self.stats["final_turn_card5_extensions"][p] += 1
            self.phase = self.PHASE_TURN_START
            return

        # Priority 4: Final turn after opponent called End of Round finishes
        if self.is_final_turn:
            if self.track_stats:
                self.stats["round_end_reasons"][0] += 1
            self.current_player = opp
            self._resolve_round()
            return

        # Priority 5: Normal turn transition
        self.current_player = opp
        self.phase = self.PHASE_TURN_START

    # =========================================================================
    # 5. DISTRICT RESOLUTION AND VICTORY CHECKS
    # =========================================================================
    def _resolve_round(self):
        """
        Records showdown card distributions, resolves Districts 1..5 in order,
        and checks for immediate victory after each district.
        """
        self.round_just_ended = True
        trump_color = self.color_ranking[0]
        is_crit_hp = self.dracula_hp <= 3

        if self.track_stats:
            self.stats["total_rounds_played"] += 1
            if is_crit_hp:
                self.stats["crit_hp_rounds"] += 1

            for p in range(2):
                trumps_in_hand = 0
                self.stats["untouched_slots_at_showdown"][p] += int(np.sum(self.slot_untouched[p]))
                for d in range(5):
                    c = self.racks[p, d]
                    val = (c % 8) + 1
                    col = c // 8
                    self.stats["showdown_values"][p, d, val - 1] += 1
                    is_t = (col == trump_color)
                    if is_t:
                        self.stats["showdown_trumps"][p, d] += 1
                        trumps_in_hand += 1

                    if is_crit_hp:
                        if d in (0, 1):
                            self.stats["crit_hp_early_val_sum"][p] += val
                            if is_t:
                                self.stats["crit_hp_early_trumps"][p] += 1
                        elif d in (3, 4):
                            self.stats["crit_hp_late_val_sum"][p] += val
                            if is_t:
                                self.stats["crit_hp_late_trumps"][p] += 1

                    if self.revealed[p, d]:
                        self.stats["showdown_visibility"][p, 0] += 1
                    elif self.known_by_opp[p, d]:
                        self.stats["showdown_visibility"][p, 1] += 1
                    else:
                        self.stats["showdown_visibility"][p, 2] += 1

                self.stats["trumps_per_hand_dist"][p, trumps_in_hand] += 1

        districts_won_this_round = [0, 0]

        for d in range(5):
            card_drac = self.racks[0, d]
            card_vh = self.racks[1, d]
            vamp_count_before = int(self.vampires[d])

            dracula_wins_district, combat_type = self._compare_cards(card_drac, card_vh, trump_color)
            winner_p = 0 if dracula_wins_district else 1
            districts_won_this_round[winner_p] += 1

            if self.track_stats:
                self.stats["combat_win_types"][combat_type] += 1
                self.stats["vamp_state_combats"][vamp_count_before] += 1
                if dracula_wins_district:
                    self.stats["vamp_state_drac_wins"][vamp_count_before] += 1

                val_d, col_d = (card_drac % 8) + 1, card_drac // 8
                val_vh, col_vh = (card_vh % 8) + 1, card_vh // 8
                self.stats["vamp_state_val_sum"][0, vamp_count_before] += val_d
                self.stats["vamp_state_val_sum"][1, vamp_count_before] += val_vh
                if col_d == trump_color:
                    self.stats["vamp_state_trumps"][0, vamp_count_before] += 1
                if col_vh == trump_color:
                    self.stats["vamp_state_trumps"][1, vamp_count_before] += 1

                if combat_type == 2:
                    margin = abs(val_d - val_vh)
                    self.stats["non_trump_win_margin_sum"][winner_p] += margin
                    self.stats["non_trump_win_count"][winner_p] += 1

                is_winner_card_public = 0 if self.revealed[winner_p, d] else 1
                self.stats["wins_by_visibility"][winner_p, is_winner_card_public] += 1

                if winner_p == self.round_starter:
                    self.stats["round_starter_district_wins"] += 1
                else:
                    self.stats["round_second_district_wins"] += 1

            if dracula_wins_district:
                self.vampires[d] += 1
                self.step_vampires_added += 1
                if self.track_stats:
                    self.stats["district_wins"][0, d] += 1

                if self.vampires[d] >= 4:
                    self.done = True
                    self.winner = 0
                    self.win_reason = f"Dracula converted all 4 inhabitants in District {d + 1}"
                    self.phase = self.PHASE_GAME_OVER
                    if self.track_stats:
                        self._record_closer_stats(districts_won_this_round, game_winner=0)
                        self.stats["games_played"] += 1
                        self.stats["win_types"][0] += 1
                        self.stats["game_end_round"][self.round_num - 1] += 1
                        self.stats["decisive_district"][0, d] += 1
                        self.stats["drac_hp_on_drac_win"] += int(self.dracula_hp)
                        if self.dracula_hp <= 3:
                            self.stats["drac_comeback_wins_low_hp"] += 1
                    return
            else:
                self.dracula_hp -= 1
                self.step_hp_lost += 1
                if self.track_stats:
                    self.stats["district_wins"][1, d] += 1

                if self.dracula_hp <= 0:
                    self.done = True
                    self.winner = 1
                    self.win_reason = f"Van Helsing reduced Dracula to 0 HP in District {d + 1}"
                    self.phase = self.PHASE_GAME_OVER
                    if self.track_stats:
                        self._record_closer_stats(districts_won_this_round, game_winner=1)
                        self.stats["games_played"] += 1
                        self.stats["win_types"][2] += 1
                        self.stats["game_end_round"][self.round_num - 1] += 1
                        self.stats["decisive_district"][1, d] += 1
                        self.stats["max_vamps_on_vh_win"] += int(np.max(self.vampires))
                        self.stats["total_vamps_on_vh_win"] += int(np.sum(self.vampires))
                        if np.max(self.vampires) >= 3:
                            self.stats["vh_comeback_wins_3vamps"] += 1
                    return

        if self.track_stats:
            self._record_closer_stats(districts_won_this_round, game_winner=None)
            self.stats["full_round_drac_wins_dist"][districts_won_this_round[0]] += 1

        if self.round_num >= 5:
            self.done = True
            self.winner = 0
            self.win_reason = "Dracula survived until the end of Round 5"
            self.phase = self.PHASE_GAME_OVER
            if self.track_stats:
                self.stats["games_played"] += 1
                self.stats["win_types"][1] += 1
                self.stats["game_end_round"][4] += 1
                self.stats["drac_hp_on_drac_win"] += int(self.dracula_hp)
                if self.dracula_hp <= 3:
                    self.stats["drac_comeback_wins_low_hp"] += 1
        else:
            self.round_num += 1
            self._start_new_round()

    def _record_closer_stats(self, districts_won: list, game_winner: Optional[int]):
        closer = self.round_closer_player
        if closer in (0, 1):
            opp = 1 - closer
            self.stats["closer_initiations"][closer] += 1
            self.stats["closer_districts_won"][closer] += districts_won[closer]
            if districts_won[opp] >= 3 or (game_winner is not None and game_winner == opp):
                self.stats["closer_backfires"][closer] += 1

    def _compare_cards(self, card_drac: int, card_vh: int, trump_color: int) -> Tuple[bool, int]:
        col_d, val_d = card_drac // 8, (card_drac % 8) + 1
        col_vh, val_vh = card_vh // 8, (card_vh % 8) + 1

        is_trump_d = col_d == trump_color
        is_trump_vh = col_vh == trump_color

        if is_trump_d and not is_trump_vh:
            return True, 0
        if is_trump_vh and not is_trump_d:
            return False, 0

        if is_trump_d and is_trump_vh:
            return val_d > val_vh, 1

        if val_d != val_vh:
            return val_d > val_vh, 2

        rank_d = int(np.where(self.color_ranking == col_d)[0][0])
        rank_vh = int(np.where(self.color_ranking == col_vh)[0][0])
        return rank_d < rank_vh, 3

    # =========================================================================
    # 6. LEAK-FREE INFORMATION-SET OBSERVATION ENCODING FOR RL AGENTS
    # =========================================================================
    def get_observation(self, player: int) -> np.ndarray:
        """Base 655 features + rich leak-free features (cached per state & player)."""
        hit = self._obs_cache.get(player)
        if hit is not None and hit[0] == self._version:
            return hit[1].copy()
        obs = np.concatenate([self._base_observation(player), build_features(self, player)])
        self._obs_cache[player] = (self._version, obs)
        return obs.copy()

    def _base_observation(self, player: int) -> np.ndarray:
        """
        Builds a 655-dimensional float32 observation vector representing the exact
        Information Set of `player` (zero hidden-state leaks, complete public action context).
        """
        opp = 1 - player
        obs = []

        # 1. Player identity (2), Round one-hot (5), Phase one-hot (8) -> 15 features
        obs.extend([1.0, 0.0] if player == 0 else [0.0, 1.0])
        round_oh = [0.0] * 5
        round_oh[self.round_num - 1] = 1.0
        obs.extend(round_oh)
        phase_oh = [0.0] * 8
        phase_oh[self.phase] = 1.0
        obs.extend(phase_oh)

        # 2. Normalized public board state -> 10 features
        obs.append(self.dracula_hp / 12.0)
        obs.extend([v / 4.0 for v in self.vampires])
        obs.append(len(self.deck) / 22.0)
        obs.append(len(self.discard_pile) / 22.0)
        obs.append(1.0 if self.end_of_round_called else 0.0)
        obs.append(1.0 if self.is_final_turn else 0.0)

        # 3. Color Ranking hierarchy (4 ranks x 4 color one-hot) -> 16 features
        for pos in range(4):
            col_oh = [0.0] * 4
            col_oh[self.color_ranking[pos]] = 1.0
            obs.extend(col_oh)

        # 4. Own Rack (5 slots x 7 features = 35 features)
        for s in range(5):
            c = self.racks[player, s]
            col_oh = [0.0] * 4
            col_oh[c // 8] = 1.0
            val_norm = ((c % 8) + 1) / 8.0
            obs.extend(col_oh)
            obs.append(val_norm)
            obs.append(1.0 if self.revealed[player, s] else 0.0)
            obs.append(1.0 if (self.revealed[player, s] or self.known_by_opp[player, s]) else 0.0)

        # 5. Opponent Rack (5 slots x 6 features = 30 features)
        for s in range(5):
            is_known = self.revealed[opp, s] or self.known_by_opp[opp, s]
            if is_known:
                c = self.racks[opp, s]
                col_oh = [0.0] * 4
                col_oh[c // 8] = 1.0
                val_norm = ((c % 8) + 1) / 8.0
                obs.extend(col_oh)
                obs.append(val_norm)
                obs.append(1.0 if self.revealed[opp, s] else 0.5)
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 6. Currently Drawn Card (only visible during PHASE_CHOOSE_DISCARD) -> 5 features
        if self.phase == self.PHASE_CHOOSE_DISCARD and self.drawn_card != -1:
            col_oh = [0.0] * 4
            col_oh[self.drawn_card // 8] = 1.0
            val_norm = ((self.drawn_card % 8) + 1) / 8.0
            obs.extend(col_oh)
            obs.append(val_norm)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 7. Revealed Top Card of Deck (when Effect 2 is active) -> 5 features
        if self.top_deck_revealed and len(self.deck) > 0:
            top_c = self.deck[0]
            col_oh = [0.0] * 4
            col_oh[top_c // 8] = 1.0
            val_norm = ((top_c % 8) + 1) / 8.0
            obs.extend(col_oh)
            obs.append(val_norm)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 8. Top Card of Discard Pile (Most recently discarded card) -> 5 features
        if len(self.discard_pile) > 0:
            last_disc = self.discard_pile[-1]
            col_oh = [0.0] * 4
            col_oh[last_disc // 8] = 1.0
            val_norm = ((last_disc % 8) + 1) / 8.0
            obs.extend(col_oh)
            obs.append(val_norm)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 9. Last Public Event Context (who acted, value discarded, districts affected) -> 7 features
        obs.extend(self.last_public_event.tolist())

        # 10. Full Discard Pile Memory (32 binary flags for card counting) -> 32 features
        discard_mask = [0.0] * 32
        for c in self.discard_pile:
            discard_mask[c] = 1.0
        obs.extend(discard_mask)

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        r_drac = 0.0
        if self.done:
            r_drac = 1.0 if self.winner == 0 else -1.0

        return {
            "round": self.round_num,
            "current_player": self.current_player,
            "phase": self.phase,
            "dracula_hp": self.dracula_hp,
            "vampires": self.vampires.copy(),
            "action_mask": self.get_legal_actions(),
            "round_just_ended": self.round_just_ended,
            "step_hp_lost": self.step_hp_lost,
            "step_vampires_added": self.step_vampires_added,
            "rewards_both": (r_drac, -r_drac),
            "winner": self.winner,
            "win_reason": self.win_reason,
        }

    # =========================================================================
    # 7. COMPREHENSIVE 12-SECTION TELEMETRY REPORT GENERATOR
    # =========================================================================
    def print_telemetry_report(self):
        """Prints a complete 12-section statistical breakdown of all gameplay metrics."""
        s = self.stats
        g = max(1, s["games_played"])
        r = max(1, s["total_rounds_played"])
        tot_combats = max(1, int(np.sum(s["combat_win_types"])))

        drac_wins = int(s["win_types"][0] + s["win_types"][1])
        vh_wins = int(s["win_types"][2])

        print("\n" + "=" * 88)
        print("DRACULA VS VAN HELSING — COMPLETE 12-SECTION RESEARCH & RL TELEMETRY REPORT")
        print("=" * 88)

        # SECTION 1: MACRO GAME, PACING, MARGIN OF VICTORY & COMEBACKS
        print("[1] MACRO GAME OUTCOMES, PACING, MARGIN OF VICTORY & CLUTCH COMEBACKS")
        print(f"  Total Games Simulated           : {g:,}")
        print(f"  Total Rounds Played             : {r:,} (Avg: {r / g:.2f} rounds/game)")
        print(f"  Average Turns per Round         : {s['total_turns_played'] / r:.2f} turns ({s['total_discards'] / r:.2f} discards/round)")
        print(f"  Dracula Total Wins              : {drac_wins:,} ({drac_wins / g * 100:.2f}%)")
        print(f"    ├─ By 4 Vampires in District  : {s['win_types'][0]:,} ({s['win_types'][0] / g * 100:.2f}%)")
        print(f"    └─ By Round 5 Survival        : {s['win_types'][1]:,} ({s['win_types'][1] / g * 100:.2f}%)")
        print(f"  Van Helsing Total Wins (0 HP)   : {vh_wins:,} ({vh_wins / g * 100:.2f}%)")
        print(f"  Avg Dracula HP Left on Drac Win : {s['drac_hp_on_drac_win'] / max(1, drac_wins):.2f} HP (Clutch Wins at <=3 HP: {s['drac_comeback_wins_low_hp'] / max(1, drac_wins) * 100:.1f}%)")
        print(f"  Avg Max Vamps in a Dist (VH Win): {s['max_vamps_on_vh_win'] / max(1, vh_wins):.2f} vamps (Clutch Wins vs 3-Vamp Threat: {s['vh_comeback_wins_3vamps'] / max(1, vh_wins) * 100:.1f}%)")
        print("-" * 88)

        # SECTION 2: GAME LENGTH & ROUND-END TRIGGERS
        print("[2] GAME DURATION BY ROUND & ROUND-ENDING TRIGGERS")
        for idx in range(5):
            cnt = s["game_end_round"][idx]
            print(f"  Games Ending in Round {idx + 1}         : {cnt:8,} ({cnt / g * 100:5.2f}%)")
        print("  " + "-" * 45)
        re_names = ["Voluntary Call (Action 1)", "Card 8 Instant Close     ", "Deck Exhaustion (22 crds)"]
        for idx, name in enumerate(re_names):
            cnt = s["round_end_reasons"][idx]
            print(f"  Round Ended via {name}: {cnt:8,} ({cnt / r * 100:5.2f}%)")
        print("-" * 88)

        # SECTION 3: FULL-ROUND SCORE DISTRIBUTION (SWEEPS VS CLOSE ROUNDS)
        tot_full_rounds = max(1, int(np.sum(s["full_round_drac_wins_dist"])))
        print("[3] FULL-ROUND SCORE DISTRIBUTION (DRACULA - VAN HELSING DISTRICT SPLIT)")
        score_strs = [
            f"{k}-{5-k}: {s['full_round_drac_wins_dist'][k] / tot_full_rounds * 100:4.1f}%"
            for k in range(6)
        ]
        print("  " + " | ".join(score_strs))
        print("-" * 88)

        # SECTION 4: INITIATIVE, CALLING TIMING, EFFICIENCY & BACKFIRE RATES
        print("[4] INITIATIVE, CALL TIMING, ROUND-CLOSING EFFICIENCY & BACKFIRE RATES")
        tot_starter_wins = s["round_starter_district_wins"]
        tot_second_wins = s["round_second_district_wins"]
        print(f"  Round Starter District Win Rate : {tot_starter_wins / tot_combats * 100:.2f}% (vs Second Player: {tot_second_wins / tot_combats * 100:.2f}%)")
        for p_idx, p_name in enumerate(["Dracula    ", "Van Helsing"]):
            calls = s["calls_by_player"][p_idx]
            avg_call_disc = s["call_discard_pile_sum"][p_idx] / max(1, calls)
            c8 = s["card8_end_by_player"][p_idx]
            avg_c8_disc = s["card8_discard_pile_sum"][p_idx] / max(1, c8)
            inits = max(1, s["closer_initiations"][p_idx])
            avg_d_won = s["closer_districts_won"][p_idx] / inits
            backfire_pct = s["closer_backfires"][p_idx] / inits * 100
            ext5 = s["final_turn_card5_extensions"][p_idx]
            print(
                f"  {p_name} -> Calls: {calls:,} (at {avg_call_disc:.1f} discs) | Card-8s: {c8:,} (at {avg_c8_disc:.1f} discs) | "
                f"Won: {avg_d_won:.2f}/5 | Backfire: {backfire_pct:.1f}% | '5'-Chains: {ext5:,}"
            )
        print("-" * 88)

        # SECTION 5: HAND UPGRADE EFFICIENCY (INITIAL DEAL -> SHOWDOWN DELTA)
        print("[5] HAND UPGRADE EFFICIENCY (INITIAL DEAL -> SHOWDOWN DELTA)")
        tot_show_cards = r * 5
        vals_array = np.arange(1, 9)
        for p_idx, p_name in enumerate(["Dracula    ", "Van Helsing"]):
            init_val = s["initial_val_sum"][p_idx] / tot_show_cards
            final_val = np.sum(s["showdown_values"][p_idx] * vals_array) / tot_show_cards
            init_tr = s["initial_trumps_sum"][p_idx] / r
            final_tr = np.sum(s["showdown_trumps"][p_idx]) / r
            untouched_pct = s["untouched_slots_at_showdown"][p_idx] / tot_show_cards * 100
            print(
                f"  {p_name} -> Card Value: {init_val:.2f} -> {final_val:.2f} (Delta: {final_val - init_val:+.2f}) | "
                f"Trumps/Hand: {init_tr:.2f} -> {final_tr:.2f} (Delta: {final_tr - init_tr:+.2f}) | Untouched Slots: {untouched_pct:.1f}%"
            )
        print("-" * 88)

        # SECTION 6: TACTICAL CARD EFFECT MASTERY (CARDS 1, 2, 3, 4, 6, 7)
        print("[6] TACTICAL CARD EFFECT MASTERY (CARDS 1, 2, 4, 6, 7 & WASTED REVEALS)")
        for p_idx, p_name in enumerate(["Dracula    ", "Van Helsing"]):
            c7_cnt = max(1, s["trump_changes_by_player"][p_idx])
            c7_own = s["effect7_own_trump_gain_sum"][p_idx] / c7_cnt
            c7_opp = s["effect7_vis_opp_trump_delta_sum"][p_idx] / c7_cnt
            c6_cnt = max(1, s["effect6_count"][p_idx])
            c6_delta = s["effect6_net_power_delta_sum"][p_idx] / c6_cnt
            c6_steal = s["effect6_stole_trump_count"][p_idx] / c6_cnt * 100
            c4_cnt = max(1, s["effect4_count"][p_idx])
            c4_threat = s["effect4_high_card_to_threat"][p_idx] / c4_cnt * 100
            c1_avg_val = s["effect1_revealed_val_sum"][p_idx] / max(1, s["effect1_count"][p_idx])
            top_keep_pct = s["revealed_top_kept"][p_idx] / max(1, s["revealed_top_draws"][p_idx]) * 100
            print(
                f"  {p_name} -> Card 7 Own Trump Gain: {c7_own:+.2f} (Vis Opp: {c7_opp:+.2f}) | "
                f"Card 6 Power Delta: {c6_delta:+.2f} (Stole Trump: {c6_steal:.1f}%) | "
                f"Card 4 To-Threat: {c4_threat:.1f}%\n"
                f"                 Card 1 Self-Revealed Avg Val: {c1_avg_val:.2f} | "
                f"Peeked Deck-Top (Card 2) Kept Rate: {top_keep_pct:.1f}%"
            )
        print("-" * 88)

        # SECTION 7: DISCARD FREQUENCY & CARD ECONOMY
        tot_disc_d = max(1, int(np.sum(s["discarded_values"][0])))
        tot_disc_vh = max(1, int(np.sum(s["discarded_values"][1])))
        print("[7] DISCARDED CARD NUMBERS (1 to 8) & TRUMP DISCARD DISCIPLINE")
        print("  Card # |   Dracula Discards   | Van Helsing Discards |    Combined Share")
        for val_idx in range(8):
            d_cnt = s["discarded_values"][0, val_idx]
            vh_cnt = s["discarded_values"][1, val_idx]
            comb_pct = (d_cnt + vh_cnt) / (tot_disc_d + tot_disc_vh) * 100
            print(
                f"    {val_idx + 1}    | {d_cnt:9,} ({d_cnt / tot_disc_d * 100:5.2f}%) | "
                f"{vh_cnt:9,} ({vh_cnt / tot_disc_vh * 100:5.2f}%)   | {comb_pct:6.2f}%"
            )
        print(f"  Trump Cards Discarded           : Dracula: {s['discarded_trumps'][0] / tot_disc_d * 100:.2f}% | VH: {s['discarded_trumps'][1] / tot_disc_vh * 100:.2f}%")
        print(f"  Immediate Draw-and-Trash Rate   : Dracula: {s['discard_source'][0, 0] / tot_disc_d * 100:.2f}% | VH: {s['discard_source'][1, 0] / tot_disc_vh * 100:.2f}%")
        print(f"  Wasted Reveals (Auto-Skipped)   : Card 1: D={s['wasted_reveals_card1'][0]:,}, VH={s['wasted_reveals_card1'][1]:,} | Card 3: D={s['wasted_reveals_card3'][0]:,}, VH={s['wasted_reveals_card3'][1]:,}")
        print("-" * 88)

        # SECTION 8: SHOWDOWN CARD NUMBERS & DOMINANT (TRUMP) HAND DENSITY
        show_d_vals = np.sum(s["showdown_values"][0], axis=0)
        show_vh_vals = np.sum(s["showdown_values"][1], axis=0)
        print("[8] SHOWDOWN CARD NUMBERS (1 to 8) & DOMINANT (TRUMP) HAND CONCENTRATION")
        print("  Card # |    Dracula Racks     |  Van Helsing Racks   |    Combined Share")
        for val_idx in range(8):
            d_cnt = show_d_vals[val_idx]
            vh_cnt = show_vh_vals[val_idx]
            comb_pct = (d_cnt + vh_cnt) / (2 * tot_show_cards) * 100
            print(
                f"    {val_idx + 1}    | {d_cnt:9,} ({d_cnt / tot_show_cards * 100:5.2f}%) | "
                f"{vh_cnt:9,} ({vh_cnt / tot_show_cards * 100:5.2f}%)   | {comb_pct:6.2f}%"
            )
        print("  Trump Count per 5-Card Hand Dist: " + " | ".join(
            [f"{k}T: D={s['trumps_per_hand_dist'][0, k] / r * 100:.1f}%, VH={s['trumps_per_hand_dist'][1, k] / r * 100:.1f}%" for k in range(5)]
        ))
        print("-" * 88)

        # SECTION 9: COMBAT RESOLUTION HIERARCHY & OVERKILL
        print("[9] HOW DISTRICT COMBATS ARE WON (RESOLUTION HIERARCHY & OVERKILL)")
        c_names = [
            "Single Trump beats Non-Trump ",
            "Trump beats Lower Trump      ",
            "Higher Number (Neither Trump)",
            "Color Ranking Tie-Breaker    ",
        ]
        for idx, name in enumerate(c_names):
            cnt = s["combat_win_types"][idx]
            print(f"  {name}  : {cnt:9,} ({cnt / tot_combats * 100:5.2f}%)")
        ovk_d = s["non_trump_win_margin_sum"][0] / max(1, s["non_trump_win_count"][0])
        ovk_vh = s["non_trump_win_margin_sum"][1] / max(1, s["non_trump_win_count"][1])
        print(f"  Avg Non-Trump Win Margin (Overkill): Dracula: +{ovk_d:.2f} pts | Van Helsing: +{ovk_vh:.2f} pts")
        print("-" * 88)

        # SECTION 10: INFORMATION ASYMMETRY & FOG-OF-WAR AT SHOWDOWN
        print("[10] INFORMATION VISIBILITY AT SHOWDOWN (FOG OF WAR)")
        for p_idx, p_name in enumerate(["Dracula    ", "Van Helsing"]):
            pub = s["showdown_visibility"][p_idx, 0] / tot_show_cards * 100
            priv = s["showdown_visibility"][p_idx, 1] / tot_show_cards * 100
            sec = s["showdown_visibility"][p_idx, 2] / tot_show_cards * 100
            pub_wins = s["wins_by_visibility"][p_idx, 0]
            hid_wins = s["wins_by_visibility"][p_idx, 1]
            tot_p_wins = max(1, pub_wins + hid_wins)
            print(
                f"  {p_name} Racks -> Public Face-Up: {pub:4.1f}% | Known via Swap-6: {priv:4.1f}% | Secret: {sec:4.1f}% "
                f"(Wins from Face-Up: {pub_wins / tot_p_wins * 100:4.1f}%)"
            )
        print("-" * 88)

        # SECTION 11: DISTRICT-BY-DISTRICT SPATIAL ANALYTICS & TARGETING
        print("[11] DISTRICT-BY-DISTRICT PERFORMANCE (DISTRICTS 1 TO 5) & EFFECT TARGETING")
        print("  Dist | Resolved | Drac Win% | Drac 4-Vamp KO | VH 0-HP KO | Avg Showdown Val (D / VH) | Trump% (D / VH)")
        for d in range(5):
            d_w = s["district_wins"][0, d]
            vh_w = s["district_wins"][1, d]
            res = max(1, d_w + vh_w)
            d_win_pct = d_w / res * 100
            d_ko = s["decisive_district"][0, d]
            vh_ko = s["decisive_district"][1, d]
            avg_val_d = np.sum(s["showdown_values"][0, d] * vals_array) / r
            avg_val_vh = np.sum(s["showdown_values"][1, d] * vals_array) / r
            trump_d_pct = s["showdown_trumps"][0, d] / r * 100
            trump_vh_pct = s["showdown_trumps"][1, d] / r * 100
            print(
                f"   D{d + 1}  | {res:8,} |  {d_win_pct:6.2f}%  | {d_ko:8,} ({d_ko / max(1, s['win_types'][0]) * 100:4.1f}%) | "
                f"{vh_ko:6,} ({vh_ko / max(1, vh_wins) * 100:4.1f}%) |     {avg_val_d:.2f}  /  {avg_val_vh:.2f}      | "
                f"{trump_d_pct:4.1f}% / {trump_vh_pct:4.1f}%"
            )
        eff_labels = [
            ("effect_1_rev_own", "  Target Share Card 1 (Rev Own) "),
            ("effect_3_rev_opp", "  Target Share Card 3 (Rev Opp) "),
            ("effect_4_swap_own", "  Target Share Card 4 (Swap Own)"),
            ("effect_6_swap_opp", "  Target Share Card 6 (Swap Opp)"),
        ]
        for key, label in eff_labels:
            arr = np.sum(s["effect_district_targets"][key], axis=0)
            tot_e = max(1, int(np.sum(arr)))
            dist_str = " | ".join([f"D{d+1}: {arr[d] / tot_e * 100:4.1f}%" for d in range(5)])
            print(f"{label} -> {dist_str}")
        print("-" * 88)

        # SECTION 12: ASYMMETRIC CRITICAL STATE RESPONSE (VAMPIRES 0..3 & DRACULA LOW HP <= 3)
        print("[12] ASYMMETRIC CRITICAL STATE RESPONSE (VAMPIRE THREAT 0..3 & DRACULA LOW HP <= 3)")
        print("  Vamps in Dist | Combats  | Drac Win% | Avg Card Value (D / VH) | Dominant Trump% Used (D / VH)")
        for v in range(4):
            c_cnt = max(1, s["vamp_state_combats"][v])
            d_win_pct = s["vamp_state_drac_wins"][v] / c_cnt * 100
            avg_v_d = s["vamp_state_val_sum"][0, v] / c_cnt
            avg_v_vh = s["vamp_state_val_sum"][1, v] / c_cnt
            tr_d = s["vamp_state_trumps"][0, v] / c_cnt * 100
            tr_vh = s["vamp_state_trumps"][1, v] / c_cnt * 100
            label = f"{v} Vampires (CRITICAL)" if v == 3 else f"{v} Vampires           "
            print(
                f"  {label} | {s['vamp_state_combats'][v]:8,} |  {d_win_pct:6.2f}%  |      {avg_v_d:.2f}  /  {avg_v_vh:.2f}       |       {tr_d:5.1f}%  /  {tr_vh:5.1f}%"
            )
        crit_r = max(1, s["crit_hp_rounds"] * 2)
        print("  " + "-" * 78)
        print(f"  When Dracula HP <= 3 ({s['crit_hp_rounds']:,} showdowns) -> Front-Loading Defense in D1-D2 vs D4-D5:")
        print(
            f"    Dracula     -> D1-D2 Avg Val: {s['crit_hp_early_val_sum'][0] / crit_r:.2f} (Trump: {s['crit_hp_early_trumps'][0] / crit_r * 100:.1f}%) "
            f"vs D4-D5 Avg Val: {s['crit_hp_late_val_sum'][0] / crit_r:.2f} (Trump: {s['crit_hp_late_trumps'][0] / crit_r * 100:.1f}%)"
        )
        print(
            f"    Van Helsing -> D1-D2 Avg Val: {s['crit_hp_early_val_sum'][1] / crit_r:.2f} (Trump: {s['crit_hp_early_trumps'][1] / crit_r * 100:.1f}%) "
            f"vs D4-D5 Avg Val: {s['crit_hp_late_val_sum'][1] / crit_r:.2f} (Trump: {s['crit_hp_late_trumps'][1] / crit_r * 100:.1f}%)"
        )
        print("=" * 88 + "\n")


# =============================================================================
# AUTOMATED SELF-DIAGNOSTIC INTEGRITY CHECK & 100,000-GAME BENCHMARK
# =============================================================================
def verify_environment_integrity(env: DraculaVsVanHelsingEnv):
    """Mathematically verifies that all telemetry counters sum to exact invariants."""
    s = env.stats
    g = s["games_played"]
    r = s["total_rounds_played"]

    assert g == np.sum(s["win_types"]), "Win types sum does not match games_played!"
    assert g == np.sum(s["game_end_round"]), "Game end rounds sum does not match games_played!"
    assert r == np.sum(s["round_end_reasons"]), "Round end reasons sum does not match total_rounds_played!"
    assert r * 5 == np.sum(s["showdown_values"][0]), "Dracula showdown card count mismatch!"
    assert r * 5 == np.sum(s["showdown_values"][1]), "Van Helsing showdown card count mismatch!"
    assert r * 5 == np.sum(s["showdown_visibility"][0]), "Visibility tracking sum mismatch!"
    assert np.sum(s["combat_win_types"]) == np.sum(s["district_wins"]), "Combat win types vs district wins mismatch!"
    assert np.sum(s["vamp_state_combats"]) == np.sum(s["district_wins"]), "Vampire state combat sum mismatch!"
    print("[INTEGRITY CHECK PASSED]: All 12 telemetry sections and invariants verified 100% exact.")


# =============================================================================
# STRESS TEST & FULL 12-SECTION TELEMETRY RUN (WITH PROGRESS BAR & ETA)
# =============================================================================
if __name__ == "__main__":
    NUM_GAMES = 10_000
    env = DraculaVsVanHelsingEnv(seed=42, always_dracula_first=False, track_stats=True)
    rng = np.random.default_rng(42)

    print(f"Simulating {NUM_GAMES:,} games with telemetry tracking...")
    
    # Wrap the main game loop with tqdm for a real-time progress bar and ETA
    with tqdm(total=NUM_GAMES, desc="Running Simulations", unit="game") as pbar:
        for _ in range(NUM_GAMES):
            obs, info = env.reset()
            done = False
            while not done:
                mask = info["action_mask"]
                legal_actions = np.flatnonzero(mask)
                action = int(rng.choice(legal_actions))
                obs, reward, done, info = env.step(action)
            
            pbar.update(1)

    verify_environment_integrity(env)
    env.print_telemetry_report()