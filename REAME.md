# Dracula vs Van Helsing: Imitation Learning, Reinforcement Learning and League Self-Play for an Imperfect-Information Card Game

> A complete, reproducible pipeline that takes hand-engineered heuristic bots (the *V4 minimax bot* being the strongest) and turns them into two neural agents (one per role) that beat them: **behavioural cloning → DAgger → PPO fine-tuning with an adaptive curriculum → league self-play**.

---

## Abstract

*Dracula vs Van Helsing* is a two-player, asymmetric, zero-sum card game with hidden information, a stochastic deck, five rounds and a sudden-death victory condition. We built a leak-free simulator, a ladder of five increasingly sophisticated heuristic bots (Greedy → V1 → V2 → V3 → V4) and a four-stage learning pipeline for both roles. A first attempt with a 160-feature observation plateaued at **20.8 %** win rate against the strongest bot (V4) as Dracula despite 2 M PPO steps. After diagnosing the bottleneck as *representational* (a deterministic, information-set-compliant teacher was imitated with only 72 % accuracy), we redesigned the observation (160 → 655 features) and rebuilt the pipeline. The final agents win **55.8 %** (Dracula) and **55.0 %** (Van Helsing) of their games against V4 over 20 000 evaluation games each, versus a V4-vs-V4 mirror of **50.9 %** and **49.1 %**, i.e. roughly **+5 to +6 percentage points above the teacher**. The improvement holds against opponents that were never part of the training league, is transitive across the two roles, and is not accompanied by an increase of exploitability under a (limited) exploiter probe.

---

## Table of contents

1. [Summary of results](#1-summary-of-results)
2. [Repository layout](#2-repository-layout)
3. [The game](#3-the-game)
4. [The heuristic bot ladder](#4-the-heuristic-bot-ladder)
5. [Why not train from scratch? Difficulties and design rationale](#5-why-not-train-from-scratch-difficulties-and-design-rationale)
6. [Method](#6-method)
7. [Results](#7-results)
8. [Reproducing the experiments (commands)](#8-reproducing-the-experiments-commands)
9. [Conclusions, limitations and future work](#9-conclusions-limitations-and-future-work)
10. [References](#10-references)
11. [Appendix](#11-appendix)

---

## 1. Summary of results

Win rate of our agents against the teacher bot **V4** (deterministic agent play, fresh seeds; 95 % CI half-widths from the normal approximation):

| Role | V4 mirror (V4 vs V4) | BC | DAgger | PPO (2 M steps) | **League self-play (gen. 10)** |
|---|---|---|---|---|---|
| Dracula | 50.9 % ± 1.0 (10 000 g.) | 43.60 % (5 000 g.) | 45.84 % (5 000 g.) | 53.12 % ± 1.0 (10 000 g.) | **55.76 % ± 0.69** (20 000 g.) |
| Van Helsing | 49.1 % ± 1.0 (same games) | 42.82 % (5 000 g.) | 42.72 % (5 000 g.) | 52.34 % ± 1.0 (10 000 g.) | **55.04 % ± 0.69** (20 000 g.) |

Key findings:

* **The agents outperform their teacher.** Relative to the V4 mirror the final agents are about **+4.9 ± 1.2 pp** (Dracula) and **+5.9 ± 1.2 pp** (Van Helsing) better.
* **Features were the decisive change.** The legacy 160-feature pipeline reached only 20.8 % against V4. With the 655-feature observation, plain behavioural cloning already reaches 43–44 %.
* **League self-play adds a further +2.6 to +2.9 pp** over the PPO agents against V4 and +3.9 / +2.6 pp head-to-head against the frozen PPO agents of the other role.
* **Robustness.** On five held-out opponents the final agents are better than or equal to the PPO agents most comparisons and never significantly worse. Strength composes transitively across roles.
* **No evidence of higher exploitability.** Exploiter agents trained for 400 k steps win less often against the final agents than against the PPO agents.

---

## 2. Repository layout

| File | Purpose |
|---|---|
| `dvvh_env.py` | Game simulator (finite-state machine, 10-action space with exact legal masks, leak-free information-set observations, telemetry engine). `get_observation` returns the 655-d vector. |
| `obs_features.py` | Builder of the 495 extra features (rank-canonical frame, Bayesian unseen-card statistics, sequential-resolution DP, action-aligned look-ahead). |
| `bots.py` | `RandomBot`, `GreedyHeuristicBot`, `TacticalRefinedBot` (V1), `_V2`, `_V3`, `_V4`. `python bots.py` runs head-to-head series with telemetry. |
| `rl_wrappers.py` | Gymnasium wrapper (`DraculaVsVanHelsingMaskEnv`): action masks, opponent pool, shaping, `SB3PolicyBotAdapter` (use a trained model as a bot). |
| `rl_callbacks.py` | `AdaptiveCurriculumAndEloCallback`, `AdaptiveLR`: evaluation, PFSP-style curriculum, Elo, LR control, state persistence. |
| `warmup_bc.py` | Stage 1: behavioural-cloning warm-up from a sequence of teachers. |
| `dagger_bc.py` | Stage 2: DAgger fine-tuning against V4. |
| `train_curriculum.py` | Stage 3: PPO (MaskablePPO) with BC auxiliary loss, critic warm-up, adaptive curriculum. |
| `train_selfplay.py`, `sp_common.py` | Stage 4: league self-play (alternating roles, PFSP, snapshot gating, metrics). |
| `exploit_probe.py` | Exploitability proxy (trains an exploiter against a frozen target). |
| `final_validation.py`, `eval_checkpoint.py`, `robustness_eval.py`, `head_to_head.py`, `diagnose_agent.py` | Evaluation and diagnostics tools. |
| `results/gen10/` | Raw metrics of the final generation (validation, held-out, self-play `metrics.jsonl`, `state.json`, exploitability probes). |
| `requirements.txt`, `.gitignore` | Dependencies and ignore rules (checkpoints, caches and logs are not versioned). |

Directories created at run time: `checkpoints/{Dracula,VanHelsing}/`, `bc_cache/`, `bc_history/`, `logs/`, `selfplay/`, `probes_long/`.

---

## 3. The game

### 3.1 Rules (as implemented in `dvvh_env.py`)

* **Players:** player 0 = **Dracula**, player 1 = **Van Helsing**.
* **Cards:** 32 cards = 4 colours × values 1–8. A random **colour ranking** (a permutation of the 4 colours) is drawn at the start of the game; the colour of rank 0 is the **trump**. The ranking can be changed during the game only by a card-7 effect.
* **Board:** 5 *districts* (D1–D5). Every district keeps a counter of **vampires** (0–3). Dracula starts with **12 HP**.
* **Round setup:** all 32 cards are reshuffled; each player receives 5 face-down cards (their *rack*; slot *d* faces the opponent's slot *d* in district *d*); the remaining 22 cards form the draw deck; the discard pile is empty. The game lasts **5 rounds**; turn order continues across rounds.
* **A turn:** the active player either **draws** the top card of the deck and then *discards the drawn card or swaps it with one of their rack slots* (the card that ends up on the discard pile triggers its effect), or **calls the end of the round** (the opponent then plays one final turn).
* **Card effects by the value of the discarded card:**

| Value | Effect |
|---|---|
| 1 | Reveal one of your own face-down cards |
| 2 | Reveal the top card of the deck |
| 3 | Reveal one of the opponent's face-down cards |
| 4 | Swap two of your own cards (10 possible pairs) |
| 5 | Extra turn |
| 6 | Swap your card with the opponent's card in a chosen district |
| 7 | Change the trump (swap the rank-0 colour with the colour of rank 1, 2 or 3) |
| 8 | End the round immediately |

* **Round end:** triggered by a voluntary call (followed by the opponent's final turn), by an 8 being discarded or by the exhaustion of the deck.
* **Showdown:** districts are resolved in order D1 → D5 by comparing the two cards of each district: a trump beats a non-trump; two trumps → higher value wins; two non-trumps → higher value wins; equal values → the colour with the better rank wins.
  * If **Dracula wins** the district, its vampire counter increases; at **4 vampires** Dracula wins the game immediately ("converted all 4 inhabitants").
  * If **Van Helsing wins** the district, Dracula loses 1 HP; at **0 HP** Van Helsing wins immediately.
  * If Dracula survives the showdown of **round 5** he wins.

The game is therefore asymmetric: Dracula wants to concentrate wins in one district (a *lethal vector*), Van Helsing wants to bleed HP while locking down 3-vampire districts.

### 3.2 Formal view

* **Imperfect information:** each player sees its own rack, the public discard pile, public events and the cards that were revealed to them; the opponent's face-down cards and the deck order are hidden.
* **Action space:** 10 discrete actions whose meaning depends on the FSM phase (`TURN_START`, `CHOOSE_DISCARD`, `EFFECT_1_REV_OWN`, `EFFECT_3_REV_OPP`, `EFFECT_4_SWAP_OWN`, `EFFECT_6_SWAP_OPP`, `EFFECT_7_TRUMP`); an **exact legal-action mask** is available at every step.
* **Horizon:** a typical game lasts about **155–158 decisions in total (≈ 78 per player)**. Between agents of the final generation: the game ends in round 3 in 2–3 % of games, round 4 in ≈ 49 %, round 5 in ≈ 48–50 %; ≈ 47–50 % of the games end by conversion (4 vampires), the others by HP depletion or round-5 survival.
* **Reward:** +1 for a win, −1 for a loss (terminal, zero-sum).

### 3.3 Random-play baseline: a structural advantage for Dracula

Simulating **100 000 games with both players choosing uniformly among their legal actions** gives

| | Wins | Win rate |
|---|---|---|
| Dracula | 52 831 | **52.83 %** |
| Van Helsing | 47 169 | 47.17 % |

The 95 % CI of the Dracula win rate is ±0.31 pp, so the advantage over 50 % is highly significant (z ≈ 17.9). The role asymmetry therefore *exists even without strategy*: every role-specific result in this report must be compared with a **role-specific baseline**. The advantage shrinks, but does not necessarily vanish, with skill: in the V4 mirror Dracula wins 50.9 % ± 1.0 (10 000 games).

---

## 4. The heuristic bot ladder

All bots are deterministic functions of the **information set** (no hidden state is read; we audited the accesses to the opponent's rack of V4, which are always guarded by public-knowledge masks). Each version inherits from the previous one (`Greedy → V1 → V2 → V3 → V4`) and fixes its weaknesses.

| Bot | Class | Main ideas / improvements over the previous version |
|---|---|---|
| **Random** | `RandomBot` | Uniform over legal actions. Baseline (Dracula wins 52.83 %). |
| **Greedy** | `GreedyHeuristicBot` | Post-resolution heuristic: evaluates a rack by the expected outcome of each district, with district multipliers 2.5× (3 vampires), 1.5× (2 vampires), and +3.0× distributed over the first districts when Dracula's HP ≤ 3; chooses discards/swaps/effects greedily and decides when to call the round. |
| **V1** | `TacticalRefinedBot` | Contained multipliers (2.0× / 1.3× / +2.0×), a small positional bonus favouring early districts (resolution order), and several tactical fixes in the discard, call and effect decisions. |
| **V2** | `TacticalRefinedBot_V2` | **Exact Bayesian card counting** (the set of unseen cards is known exactly; win probabilities are computed against it with a model of the opponent's *selection bias*), detection of *nut cards* (unbeatable because all higher trumps are gone), a **round-5 survival / lethal endgame solver**, a bug fix for the instant-close card 8, a *final-turn mode* (no speculative building on the last turn) and **counter-theft awareness** for the card-6 effect. |
| **V3** | `TacticalRefinedBot_V3` | **Fog-of-war / public-exposure penalty** (a card known to the opponent will be counter-played), **role-specific asymmetric district weights** (Dracula concentrates on his highest-vampire district and front-loads defence when HP is low; Van Helsing locks down 3-vampire districts with a 2.65× weight), fix of the "3 vs 4/6 retention anomaly" (weak cards are discarded before 4s and 6s), and a discount of the value of calling the round according to the unseen 6s/7s/trumps the opponent could use on the last turn. |
| **V4** | `TacticalRefinedBot_V4` | **Non-linear minimax engine.** Replaces the linear district weights of V1–V3 by an asymmetric, threshold-based utility: Van Helsing minimises the absolute risk of sudden death (asymptotic penalty), Dracula maximises the single lethal vector ("offensive singularity") together with damage minimisation at critical HP. Calibrated closing logic and an overkill matrix for card-6/card-4 decisions. |

### 4.1 Head-to-head win rates between consecutive versions

Each version was played against its predecessor in **both seats** (the "as Dracula / as Van Helsing" figures are the win rates of that bot in that role against the other bot in the opposite role).

| Match-up | Newer bot: win rates | as Dracula | as Van Helsing 
|---|---|---|---|
| Greedy vs Random | – | **99.32 %** | **99.05 %** |
| V1 vs Greedy | **V1: (61.88 %)** | 66.59 % | 57.17 %  |
| V2 vs V1 | **V2: (56.49 %)** | 62.13 % | 50.86 % |
| V3 vs V2 | **V3: (54.88 %)** | 62.36 % | 47.41 % 
| V4 vs V3 | **V4: (53.25 %)** | 56.42 % | 50.08 % |

Observations:

* Every bot beats its predecessor, but the **margin shrinks with each iteration** (61.9 % → 56.5 % → 54.9 % → 53.3 %): classic diminishing returns of hand-engineering.
* Because of the Dracula advantage, results must be read per seat. V4's gain over V3 lies mostly in the Dracula seat, as Van Helsing V4 is essentially at parity with V3.
* V4 vs V4 gives Dracula 50.9 % and V3 as Dracula against V4 as Van Helsing wins 49.9 %, so **V3 and V4 are very close in overall strength**.
* V4 is the **teacher** of the whole pipeline and the main yardstick.

---

## 5. Why not train from scratch? Difficulties and design rationale

We did **not** run a from-random-initialisation PPO ablation. The reasons below are a-priori arguments supported by the behaviour of the first (warm-started) pipeline.

1. **Sparse, delayed, noisy reward.** The only unbiased signal is the win/loss at the end of a game of ≈ 78 own decisions. A random policy loses against even the Greedy bot 99 % of the time, so a from-scratch learner would receive an almost constant −1. The signal only becomes informative once the policy is already competent.
2. **Partial observability and a stochastic deck.** The value of a state depends on hidden cards, critics reach an explained variance of only ≈ 0.5–0.6 even for strong agents (see §7.4). Advantage estimates are intrinsically noisy.
3. **A strong, structured opponent family.** V4 is a minimax-style bot with exact card counting. Beating it requires reproducing non-trivial computations (Bayesian win probabilities, sequential district resolution) that a generic MLP must discover from raw bits.
4. **Reward shaping is hazardous.** Intermediate rewards (+0.06 per vampire, −0.04 per HP) are *not* potential-based, so they can change the optimal policy (Ng et al., 1999 [3]). Cumulatively they can reach ≈ 1.2, comparable with the terminal reward. We use them only in the early legacy runs and set the shaping weight to **0** in all final stages.
5. **Role asymmetry.** The two roles have different objectives, baselines and difficulty, so two separate learners with separate critics are required.
6. **Compounding error of pure imitation.** Behavioural cloning is trained on the *teacher's* state distribution but evaluated on the *learner's*. Errors accumulate quadratically with the horizon (Ross et al., 2011 [1]). In our legacy pipeline BC reached 72.43 % validation accuracy but only 13.87 % win rate against V4.
7. **Imitation, then refinement.** We therefore (i) clone the teacher family with **behavioural cloning**, (ii) correct its distribution shift with **DAgger**, (iii) improve beyond the teacher with **PPO** and finally (iv) **league self-play**.

### 5.1 What went wrong in the first (160-feature) pipeline

The legacy pipeline taught the lessons that shaped the final design:

* **Curriculum starvation:** unlock thresholds were above what the agent could achieve (it stalled at 46.8 % against Greedy with an unlock threshold of 55 %), so 96 % of the transitions were spent against one opponent.
* **Threshold flip-flop:** V4 spawned only when V3's win-rate EMA was above 0.26, which oscillated around the threshold, V4 received only ≈ 100 k transitions in total.
* **Learning-rate schedule exhausted** halfway through the training (the SB3 progress variable depended on the *total* timesteps argument).
* **Policy simplification:** the agent reproduced the marginal action frequencies of V4 but not its conditional behaviour.
* **Observation bottleneck:** a deterministic, information-set-compliant teacher was imitated at only 72 %. V4 computes quantities that the 160-d vector does not expose (exact win probabilities against unseen cards, sequential district resolution). This diagnosis motivated the redesign of §6.2.

---

## 6. Method

### 6.1 Environment, wrapper and training interface

* **Simulator:** `DraculaVsVanHelsingEnv`. A finite-state machine with exact action masking, observations are *information-set compliant*.
* **Gymnasium wrapper:** `DraculaVsVanHelsingMaskEnv` (`rl_wrappers.py`).
  * The learner (*ego*) is fixed to one role; the wrapper plays the opponent's turns internally using a pool of opponents (bots or trained models) sampled at every episode reset with probabilities that can be changed live (`set_sampling_probs`).
  * `action_masks()` is exposed for `MaskablePPO` (invalid-action masking, Huang & Ontañón [4]).
  * The episode `info` dictionary exposes `ego_won` and `opponent_name`, used for per-opponent statistics.
  * `SB3PolicyBotAdapter` wraps a trained model as a bot, so that models can be used as opponents, evaluation subjects or league members.
* **Vectorisation:** `SubprocVecEnv` with 8 environments, `VecMonitor`.
* **Algorithm:** `MaskablePPO` from `sb3-contrib` (Stable-Baselines3 [8]).

### 6.2 Observation design

#### 6.2.1 The legacy 160-d vector

| Block | Dim |
|---|---|
| Player identity (2), round one-hot (5), phase one-hot (8) | 15 |
| Normalised public board (vampires per district, HP, deck/discard counters, …) | 10 |
| Colour-ranking hierarchy (4 ranks × 4 colours one-hot) | 16 |
| Own rack (5 slots × 7 features) | 35 |
| Opponent rack (5 slots × 6 features; only what the player may know) | 30 |
| Currently drawn card | 5 |
| Revealed top card of the deck | 5 |
| Top of the discard pile | 5 |
| Last public event context | 7 |
| Discard-pile memory (32 binary flags, card counting) | 32 |

#### 6.2.2 The extended 655-d vector (160 legacy + 495 new)

Design principles: (i) **leak-free**, (ii) **rank-canonical** (cards are described by *colour rank* and value, not by absolute colour), (iii) **exact tactical quantities** that V2–V4 compute internally are given as inputs. All 495 features are deterministic functions of the player's information set and are in [−1, 1].

| Block | Dim | Content |
|---|---|---|
| Own rack, rank frame | 65 | per slot: rank one-hot (4), value one-hot (8), normalised combat strength |
| Opponent known cards, rank frame | 75 | per slot: revealed / privately-known flags + card features |
| Drawn card, revealed top card, last discard | 39 | same card encoding |
| Rank-frame memory | 96 | discarded cards, *seen* cards, cards discarded *by the opponent* (public history) |
| Unseen-card statistics | 16 | counts per value/rank, expected strength, probability of a trump / high card for a hidden opponent card (with selection-bias weights) |
| Per-district block | 80 | vampire one-hot, position, known/win/lose flags, strength margin, win probability vs the unseen pool (raw and exposure-adjusted), exposure flag, "nut" flag, number of unseen cards that beat the card, slot-replacement counts of both players |
| Sequential round resolution (DP) | 16 | exact dynamic programme over D1→D5 under district independence: probability that Dracula converts, that Van Helsing kills, survival probability, expected HP lost / vampires added, reachability of each district, a squashed utility |
| Global context | 29 | HP bucket, max-vampire one-hot, who called, legal-action mask (10), HP per remaining district, turns taken, … |
| Action-aligned one-step look-ahead | 79 | utility change of every candidate action (discard/swap, swap-pair, new trump), arg-max flags, discard-effect one-hot, expected strength gain of a card-6 swap |

The **rank-canonical encoding** removes a symmetry: with the legacy absolute-colour encoding the network must learn the interaction `colour × ranking` separately for each of the 24 colour permutations, in the rank frame "is it a trump?" is a single coordinate. Public history features (who discarded which card, how often each slot was replaced) give information on the opponent's retention behaviour.

### 6.3 Network architecture

* **Policy and value networks:** two **independent** MLPs (actor and critic do not share layers in modern SB3 when `net_arch` is a list), each `655 → 512 → 256 → output` with ReLU, orthogonal initialisation (SB3 default). The actor has ≈ 0.47 M parameters (335 872 + 131 328 + 2 570), the critic ≈ 0.47 M, i.e. ≈ 0.94 M in total.
* **Why two hidden layers?** A 3-layer `[512, 512, 256]` network was judged oversized for ≈ 1 M BC samples (≈ 2.1 samples per parameter versus the commonly recommended 5–10 with 160 inputs). With the 655-d input the first layer is larger, so the ratio of the 2-layer network is again ≈ 2 samples/parameter. The observed gap between training and validation accuracy (96.6 % vs 89.6 % after DAgger) shows mild overfitting, which we mitigate with data aggregation (DAgger), weight decay (1e-5), cosine learning-rate decay and early stopping, rather than with a larger network. Widening the network was *not* needed: the bottleneck is data, not capacity.
* **Invalid-action masking:** logits of illegal actions are set to −∞ before the softmax; it is applied identically in BC, DAgger, PPO and league training, and in evaluation through `SB3PolicyBotAdapter`.
* The **same architecture** is used for BC, DAgger and PPO so that weights transfer directly.

### 6.4 Stage 1 — Behavioural-cloning warm-up (`warmup_bc.py`)

* **Teachers (sequential curriculum):** Greedy (10 k steps) → V1 (20 k) → V2 (30 k) → V3 (100 k) → V4 (800 k) = **960 000 demonstration samples**. Phases use 30/30/30/40/60 epochs and learning rates 3e-4/3e-4/3e-4/2.5e-4/2e-4. Weaker teachers preserve their priors while the strongest teacher installs the target policy.
* **Demonstration collection:** the teacher plays the learner's role; the opponent is sampled from all other bots (Random, Greedy, V1–V4 except the teacher), so the data cover a variety of opponent styles. Demonstrations are cached (`bc_cache/`).
* **Loss and optimiser:** masked cross-entropy on `(observation, legal mask, teacher action)`, AdamW-style weight decay 1e-5, batch 256, cosine annealing, **90/10 train/validation split**, early stopping with patience 10.
* **Evaluation:** 5 000 games against each of the six benchmark bots.

### 6.5 Stage 2 — DAgger (`dagger_bc.py`)

* **Idea:** BC is trained on teacher states; DAgger rolls out the *learner*, queries V4 on every learner-visited state and aggregates the labelled pairs (Ross et al. [1]).
* **Configuration:** 4 iterations max, **200 000 learner-visited states per iteration** (rollout opponents: Random, Greedy, V1, V2, V3), new data **oversampled ×2** (≈ 29.4 % of the aggregate), 40 epochs per iteration with learning rates `[1e-4, 5e-5, 3e-5]`, warm-start from the previous model.
* **Early stopping:** stop if the win-rate improvement against V4 is below +1 pp (5 000-game evaluation).
* **Selection note:** differences below ≈ 1.4 pp between iterations are not statistically distinguishable at 5 000 games (SE ≈ 0.7 pp each); the iteration with the highest validation accuracy is used.

### 6.6 Stage 3 — PPO fine-tuning with an adaptive curriculum (`train_curriculum.py`, `rl_callbacks.py`)

**Initialisation and anchoring**

* The PPO actor is initialised from the DAgger checkpoint. The **critic is not trained by DAgger** (random initialisation), so for the first **150 000 steps the actor is frozen** and only the critic is trained (with a 5× learning rate). This prevents noisy advantages from damaging the imitation prior. During the warm-up the win rate against V4 stays at the DAgger baseline, which doubles as an integrity check.
* **Auxiliary BC loss** (`MaskablePPOWithBC`): after each PPO update, a few supervised steps on V4 demonstrations keep the policy close to the prior. Only V4-labelled data are used (the cache also contains labels of weaker teachers that conflict with V4), the coefficient decays linearly from 0.20 to 0 over 800 k steps.
* **Shaping weight: 0.** The terminal ±1 reward is sufficient and avoids the policy bias of non-potential-based shaping.

**PPO hyper-parameters (final)**

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Parallel envs | 8 | Rollout size | 4096 × 8 = 32 768 |
| Batch size / epochs | 1024 / 4 | γ / GAE λ | 1.0 / 0.95 |
| Clip range | 0.10 | `target_kl` | 0.010 |
| Entropy coef. | 0.003 | Max grad norm / vf coef. | 0.5 / 0.5 |
| LR | adaptive, high 5e-5 | Critic warm-up | 150 k steps |
| Total steps | 2 M (2 031 616 effective) | Seed | 42 |

Rationale: small, low-variance steps (large rollouts, few epochs, tight clip/KL) are appropriate when fine-tuning from a strong prior; γ = 1 matches the undiscounted win/loss objective; the entropy coefficient is kept small because the policy is already stochastic where the imitation is uncertain.

**`AdaptiveCurriculumAndEloCallback`**

1. **Evaluation every 100 k steps** against Random, Greedy, V1, V2, V3 (250 games each) and **V4 (1 000 games)**; EMA (α = 0.6) of every win rate.
2. **Opponent sampling (PFSP-style):** weight `p(1 − p)²` over *unlocked* opponents (hard-but-beatable opponents get more mass) plus a probability floor. Unlock rule: opponent *i* unlocks when the EMA against *i − 1* exceeds a threshold; once unlocked an opponent **stays unlocked** (latch, fixing the flip-flop of the legacy run). EMAs are **seeded with the DAgger win rates** so that all opponents are available from step 0.
3. **Random is excluded from training** (it is beaten > 99 % of the time: zero gradient) but stays in the evaluation as a regression check.
4. **Pooled V4 estimate:** the V4 win rate is pooled over the last three evaluations (3 000 games, SE ≈ 0.9 pp instead of 1.6–2.2 pp). It is used for the streak/graduation counter and best-checkpoint saving (promotion threshold set relative to the role's mirror).
5. **Adaptive learning rate** (`AdaptiveLR`): constant until the pooled V4 win rate reaches `ramp_start` (0.50 Dracula / 0.48 Van Helsing), then geometric decay to 0.30× at `target` (0.56 / 0.54), then exponential decay with floor 5e-6.
6. **Elo tracking** against fixed anchors (Random 1000, Greedy 1300, V1 1400, V2 1450, V3 1525, V4 1560, K = 32), JSON **state persistence** for resuming, snapshots at every evaluation and diagnostic logging (explained variance, value loss, entropy, KL, episode reward and length).

### 6.7 Stage 4 — League self-play (`train_selfplay.py`)

The final stage removes the teacher: both agents train against a *league* of opponents of the opposite role.

* **Two independent PPO learners** (one per role, initialised from the frozen PPO agents `FROZEN_v1`). Because each role has its own critic, the advantage baselines are *role-conditioned by construction* (the goal of role-conditioned advantage estimation in asymmetric self-play, cf. SPIRAL [9]).
* **Alternating phases:** `phase_steps = 100 000` per role per cycle (each call rounds up to 4 rollouts), 10 cycles → **≈ 1.31 M self-play steps per role**.
* **League for role R = opponents of the opposite role:**

| Member | Type | Probability floor |
|---|---|---|
| `FROZEN_v1` of the opposite role | neural, stochastic play | 10 % |
| V4 | bot (permanent anchor) | 12 % |
| V3 | bot (permanent anchor) | 8 % |
| V4 with 10 % random moves | bot (style diversity) | 4 % |
| Mixture of V2/V3/V4 per decision | bot (style diversity) | 4 % |
| Snapshots of the opposite role's own training (≤ 5, FIFO) | neural | 5 % each |

* **PFSP:** the sampling weight is proportional to `max(1 − win_rate, 0.05)` (win rates estimated from training episodes with 100 pseudo-games of prior, refreshed every 25 k steps), then the floors above are enforced. The league has 5 members at the start and at most 10 (maximum considered: 12).
* **Training is stochastic** (sampled actions), **evaluation is deterministic**.
* **Constant hyper-parameters:** LR 3e-5, entropy 0.003, clip 0.10, `target_kl` 0.01, rollout 4096 × 8, batch 1024, 4 epochs, γ = 1, λ = 0.95. No BC loss, no shaping.
* **Snapshot admission (gate):** after each phase the snapshot plays 2 000 deterministic games against the *frozen* `FROZEN_v1` of the other role; it enters the league used by the other role if it is not collapsed (win rate ≥ baseline − 2 pp) and at least 2 cycles passed since the last admission (a strict "significantly better" rule is available but underpowered: with 2 000 games SE ≈ 1.1 pp).
* **Per-cycle metrics:** (a) head-to-head against `FROZEN_v1` (difference against the frozen baseline, z-test); (b) win rate against the neural league members (health indicator: the interval must overlap [49 %, 51 %]); (c) exploitability from `exploit_probe.py` (when available); V4 regression check every 2 cycles.
* **Exploitability probe** (`exploit_probe.py`): an *exploiter* of the opposite role, initialised from its frozen weights, is trained for 400 k (or 1 M) PPO steps against **one frozen, deterministic target**; its final win rate is a (lower-bound) proxy of the target's exploitability.

### 6.8 Evaluation protocol and statistics

* **Deterministic play** of the agents, fresh seeds, games split over processes; 95 % CI half-width `1.96·√(p(1 − p)/N)`; differences of independent proportions are tested with a z-test.
* **Standard errors:** N = 1 000 → 1.6 pp; N = 5 000 → 0.7 pp; N = 10 000 → 0.5 pp.
* **Paired baselines** (the same opponent is played by the new agent and by the reference agent) and **role-specific mirrors** are used to control for opponent difficulty and the role asymmetry.
* **Avoiding the winner's curse:** in-training evaluations are noisy (and ≈ 1.8–2.4 pp *lower* than the final 10 000-game figures); `last.zip` (not the "best" checkpoint) is validated; decision rules are fixed *before* looking at the data.
* **Leak-free validation** of every observation change.

---

## 7. Results

### 7.1 Teacher baselines and the mirror

| Match-up | Result |
|---|---|
| V4 (Dracula) vs V4 (Van Helsing), pooled 10 000 games | **Dracula 50.91 %**, Van Helsing 49.09 % (±0.98) |
| V4 vs V4, 1 000-game diagnostic: outcomes | conversion 49.3 %, HP depletion 47.3 %, round-5 survival 3.4 % |
| V3 (D) vs V4 (VH), 1 500 games | 49.47 % ± 2.5 |

### 7.2 BC and DAgger (655-d pipeline)

**Dracula**

| Opponent | BC (5 000 g.) | DAgger iter 1 (5 000 g.) |
|---|---|---|
| Random | 99.46 % | – |
| Greedy | 67.66 % | 69.20 % |
| V1 | 60.08 % | 60.96 % |
| V2 | 52.44 % | 54.54 % |
| V3 | 49.82 % | 52.18 % |
| **V4** | **43.60 %** | **45.84 %** (+2.76 pp; iteration 2: 45.04 %, −0.80 pp → early stop) |

BC validation accuracy: Greedy 73.7 %, V1 76.6 %, V2 76.8 %, V3 80.1 %, **V4 84.4 %**; after DAgger (1 360 000 samples): **89.6 %** (train 96.6 %).

**Van Helsing**

| Opponent | BC (5 000 g.) | DAgger iter 1 (5 000 g.) |
|---|---|---|
| Random | 99.50 % | – |
| Greedy | 63.84 % | 64.46 % |
| V1 | 55.14 % | 55.36 % |
| V2 | 47.26 % | 47.12 % |
| V3 | 42.02 % | 42.78 % |
| **V4** | **42.82 %** | **42.72 %** (+0.54 pp vs baseline 42.18 % → early stop) |

BC validation accuracy: 69.8 / 76.2 / 75.8 / 79.6 / **84.7 %** (teachers Greedy→V4); after DAgger: **89.5 %**.

Comparison with the legacy pipeline (Dracula, BC): validation accuracy 72.43 % → **84.4 %** and win rate against V4 13.87 % → **43.60 %**.

### 7.3 PPO fine-tuning

Win rate against V4 during PPO (1 000 games per evaluation, pooled = last three evaluations):

| Step | Dracula eval | Dracula pooled | Van Helsing eval | Van Helsing pooled |
|---|---|---|---|---|
| (DAgger baseline) | 45.84 | 45.84 | 42.72 | 42.72 |
| 100 k | 45.1 | 45.5 | 43.2 | 43.0 |
| 200 k | 44.6 | 45.2 | 45.1 | 43.7 |
| 300 k | 47.4 | 45.7 | 41.1 | 43.1 |
| 400 k | 46.5 | 46.2 | 44.5 | 43.6 |
| 500 k | 46.5 | 46.8 | 42.1 | 42.6 |
| 600 k | 46.6 | 46.5 | 44.3 | 43.6 |
| 700 k | 49.4 | 47.5 | 44.2 | 43.5 |
| 800 k | 46.6 | 47.5 | 44.7 | 44.4 |
| 900 k | 49.0 | 48.3 | 44.5 | 44.5 |
| 1.0 M | 50.2 | 48.6 | 47.7 | 45.6 |
| 1.1 M | 51.9 | 50.4 | 47.5 | 46.6 |
| 1.2 M | 51.6 | 51.2 | 49.4 | 48.2 |
| 1.3 M | 50.0 | 51.2 | 47.6 | 48.2 |
| 1.4 M | 52.3 | 51.3 | 49.3 | 48.8 |
| 1.5 M | 51.4 | 51.2 | 48.6 | 48.5 |
| 1.6 M | 50.5 | 51.4 | 51.7 | 49.9 |
| 1.7 M | 50.5 | 50.8 | 48.2 | 49.5 |
| 1.8 M | 51.4 | 50.8 | 51.4 | 50.4 |
| 1.9 M | 52.0 | 51.3 | 49.1 | 49.6 |
| 2.0 M | 51.3 | 51.6 | 49.6 | 50.0 |

Reading:

* **Dracula** improves between 0.6 M and 1.1 M steps (≈ 46 % → 51 %), then plateaus (mean of the last 10 evaluations 51.3 %). This coincides with the decay of the BC anchor to zero at 800 k steps (consistent with, but not proof of, the anchor acting as a brake).
* **Van Helsing** improves later and more slowly (≈ 43.7 % for 100–900 k, 48.4 % for 1.0–1.5 M, 50.0 % for 1.6–2.0 M) and had not fully plateaued.
* The in-training evaluations are *lower* than the final 10 000-game measurement, which is the reference.

Training diagnostics: explained variance 0.47–0.59 (Dracula) / 0.50–0.62 (Van Helsing); value loss ≈ 0.12; KL 0.003–0.013; entropy ≈ 0.19–0.20 nats (Dracula) and 0.284 → 0.244 (Van Helsing); ≈ 77 own decisions per episode; internal Elo 1257 → 1550 (Dracula), 1249 → 1509 (Van Helsing).

**Final PPO agents (`FROZEN_v1`, deterministic, 10 000 games):**

| | vs V4 | vs V3 |
|---|---|---|
| Dracula | **53.12 % ± 0.98** (independent re-measurement: 52.48 % on 5 000 g.) | 58.06 % ± 0.97 |
| Van Helsing | **52.34 % ± 0.98** (independent: 52.77 % on 3 000 g.) | 52.56 % ± 0.98 |

Against the role-specific mirrors this is **+2.0 ± 1.3 pp** (Dracula, z ≈ 3.1) and **+3.3 ± 1.3 pp** (Van Helsing, z ≈ 5.0). The agents' wins by district closely follow V4's.

### 7.4 Diagnostics of the PPO agents

* **Agreement with V4 on agent-visited states:** overall **81.6 %** for both roles. Lowest in `CHOOSE_DISCARD` (65 %) and `EFFECT_4_SWAP_OWN` (63–64 %); highest in `TURN_START` (≈ 98 %). The agreement is the same in games won and lost (81.7 % vs 81.4 %), i.e. deviations from V4 are neither systematically good nor bad: V4's action is not the unique optimum in many states.
* **Calling the round:** the agents call less often than V4 (Van Helsing 1.6 % vs 3.1 % of `TURN_START` decisions; Dracula 3.4 % vs 4.3 %) and play slightly longer games (round 5 reached in 53 % of the agents' games vs 45 % for V4).
* **Policy regularity:** a small MLP imitates the agents' actions better than V4's (BC accuracy 77.2 % vs 73.8 % Van Helsing; 74.2 % vs 73.0 % Dracula), consistent with smoother decision boundaries.
* **Warning on `OBS SUFFICIENCY`:** the ≈ 97 % winner-prediction accuracy printed by `diagnose_agent.py` is inflated because train/test are split by *sample* and states of the same game appear in both sets; it should not be interpreted as an information measurement.
* **Head-to-head (`FROZEN_v1` pair):** Dracula 49.44 % ± 1.39 (deterministic), 51.24 % (sampled actions), against a transitivity prediction `a + (1 − b) − m = 0.5291 + 0.4756 − 0.5091 = 49.56 %` (z = −0.11).

### 7.5 League self-play (10 cycles)

Per-cycle metric (a): win rate of each role's snapshot (2 000 deterministic games) against the *frozen PPO agent* of the other role (baseline: Dracula 49.44 %, Van Helsing 50.56 %; SE ≈ 1.1 pp):

| Cycle | Dracula vs FROZEN VH | z | Van Helsing vs FROZEN D | z |
|---|---|---|---|---|
| 1 | 49.85 | +0.31 | 52.05 | +1.13 |
| 2 | 52.80 | +2.54 | 47.25 | −2.50 |
| 3 | 51.65 | +1.67 | 50.15 | −0.31 |
| 4 | 50.80 | +1.03 | 52.10 | +1.16 |
| 5 | 50.95 | +1.14 | 50.00 | −0.42 |
| 6 | 52.55 | +2.35 | 49.15 | −1.07 |
| 7 | 51.70 | +1.71 | 51.85 | +0.98 |
| 8 | 52.75 | +2.50 | 51.20 | +0.48 |
| 9 | 55.25 | +4.41 | 52.70 | +1.62 |
| 10 | 52.70 | +2.47 | 52.35 | +1.35 |

Win rate against V4 (every second cycle, 2 000 games): Dracula 53.70 / 53.20 / 55.05 / 54.10 / 54.85 %, Van Helsing 54.20 / 53.05 / 51.90 / 54.00 / 54.25 %. Linear trend of metric (a): ≈ +0.3 pp per cycle for both roles.

Notes:

* Snapshots were admitted in cycles 1, 3, 5, 7, 9 for both roles; the league grew from 5 to 10 members. Only one Dracula snapshot (cycle 9) passed the strict "significantly better" gate; none did for Van Helsing, because the per-cycle test (SE ≈ 1.1 pp) is underpowered: the 10 000-game validation shows that Van Helsing *did* improve (§7.6). Therefore the validation uses `last.zip` of both roles.
* The win rate against the neural league members stayed close to 50 % (47.8–55.0 %), as expected from PFSP; the metric is a health indicator, not evidence of improvement, because it is a moving target.

### 7.6 Final validation of generation 10 (10 000 games, fresh seeds)

| | vs V4 (Δ vs `FROZEN_v1`) | vs `FROZEN_v1` of the other role (Δ vs baseline) |
|---|---|---|
| **Dracula** | **56.15 % ± 0.97** (+3.24 ± 1.26 pp, z = +5.05); independent protocol: 55.37 % ± 0.97 | **53.32 % ± 0.98** (+3.88 ± 1.70 pp, z = +4.48) |
| **Van Helsing** | **54.80 % ± 0.98** (+2.36 ± 1.30 pp, z = +3.56); independent protocol: 55.27 % ± 0.97 | **53.12 % ± 0.98** (+2.56 ± 1.70 pp, z = +2.96) |

Additional comparisons:

| | `FROZEN_v1` | Generation 10 |
|---|---|---|
| Dracula vs V3 | 58.06 % | **61.23 %** |
| Van Helsing vs V3 | 52.56 % | **53.76 %** |
| New Dracula vs new Van Helsing (deterministic, 5 000 g.) | 49.44 % | **50.96 % ± 1.39** (51.08 % with sampled actions) |
| Transitivity prediction | 49.56 % | 50.44 % (z = +0.49) |

All pre-registered decision rules (no V4 regression, beats `FROZEN_v1` with z ≥ 1.96, no robustness degradation ≥ 3 pp) are **passed for both roles**. The `champion.zip` (best snapshot on the noisy per-cycle gate: Dracula cycle 9) is statistically indistinguishable from `last.zip` (56.03 % vs 56.15 % against V4) and is not used.

### 7.7 Robustness: opponents that were not in the training league

The league anchors included V4, V3, V4 with 10 % random moves and a V2/V3/V4 mixture, so those conditions are *not* independent of the training. The following **held-out** opponents were never used for training (3 000 games each; paired comparison with the `FROZEN_v1` agent of the same role):

| Opponent | Dracula: `FROZEN_v1` → gen. 10 | Δ (z) | Van Helsing: `FROZEN_v1` → gen. 10 | Δ (z) |
|---|---|---|---|---|
| Greedy | 76.13 → 78.23 | +2.10 (+1.94) | 71.60 → 73.73 | +2.13 (+1.85) |
| V1 | 68.90 → 71.43 | +2.53 (+2.15) | 63.33 → 65.27 | +1.93 (+1.56) |
| V2 | 62.47 → 66.30 | +3.83 (+3.10) | 55.80 → 58.40 | +2.60 (+2.04) |
| V4 with 25 % random moves | 86.00 → 86.40 | +0.40 (+0.45) | 84.43 → 85.27 | +0.83 (+0.90) |
| V3 with 25 % random moves | 85.10 → 87.93 | +2.83 (+3.22) | 85.57 → 85.33 | −0.23 (−0.26) |

Dracula improves on 5/5 conditions (mean +2.3 pp), Van Helsing on 4/5 (mean +1.5 pp), and **no condition is significantly worse** (rule: ≥ 3 pp worse with z ≤ −2). The gain is therefore not specific to the league anchors; for Van Helsing it is more modest than for Dracula.

In-league perturbed conditions (for completeness): vs V4 with 10 % random moves Dracula +1.17 (z = +0.99), Van Helsing +3.27 (z = +2.76); vs a V3/V4 mixture Dracula +3.10 (z = +2.43), Van Helsing +1.67 (z = +1.30).

### 7.8 Exploitability proxy

An exploiter of the opposite role is trained for 400 k steps against each frozen target (2 000 evaluation games for the final point, ±2.19 pp):

| Target | Exploiter win rate at step 0 | after 400 k steps | Δ over training |
|---|---|---|---|
| Dracula `FROZEN_v1` | 48.0 % | **52.95 %** | +4.95 |
| Dracula generation 10 | 47.4 % | **47.35 %** | −0.05 |
| Van Helsing `FROZEN_v1` | 52.0 % | **53.20 %** | +1.2 |
| Van Helsing generation 10 | 49.4 % | **49.40 %** | 0.0 |

The final exploiter win rate is lower against the new agents (−5.6 pp, z ≈ −3.5 for Dracula; −3.8 pp, z ≈ −2.4 for Van Helsing); part of that difference is simply the higher initial strength of the new agents. The *gain* of the exploiter during training is smaller against the new Dracula (−1.8 σ difference) and unchanged for Van Helsing. In all cases there is **no sign of increased exploitability**. The probe is a *lower bound* (one seed, 400 k steps, no search).

**Extended probes (1 M steps, seed 901):** the raw outputs are in `results/gen10/Dracula_v1_long.json`, `Dracula_v2_long.json`, `VanHelsing_v1_long.json`, `VanHelsing_v2_long.json` (`v1` = PPO agents, `v2` = generation 10). Print them with:

```powershell
python -c "import json,glob; [print(f, round(json.load(open(f))['exploiter_wr']*100,2), json.load(open(f))['games']) for f in sorted(glob.glob('results/gen10/*_long.json'))]"
```

### 7.9 Diagnostics of generation 10

| | Dracula (gen. 10) | Van Helsing (gen. 10) |
|---|---|---|
| Win rate vs V4 (1 000-game diagnostic) | 58.8 % (V4 mirror in the same seeds: 52.7 %) | 53.6 % (V4: 47.3 %) |
| Agreement with V4 (overall) | **80.0 %** | **80.2 %** |
| Agreement: TURN_START / DISCARD / REV_OWN / REV_OPP / SWAP_OWN / SWAP_OPP / TRUMP | 97.1 / 63.1 / 91.6 / 79.3 / 58.4 / 79.8 / 86.1 % | 96.9 / 62.9 / 89.1 / 79.7 / 63.9 / 86.2 / 85.4 % |
| Calls of the round at `TURN_START` (V4 value) | 2.8 % (4.3 %) | 1.0 % (3.1 %) |
| Games where the opposing side reaches 4 vampires (V4 value) | 53.7 % (49.3 %) | 44.7 % (49.3 %) |
| BC accuracy of a small MLP on the agent's actions (V4 value) | 75.1 % (73.0 %) | 76.8 % (73.8 %) |

After 1.3 M more steps the agents drift slowly away from V4 (agreement 81.6 % → 80.0–80.2 %), call the round even less often, and remain at least as strong against all opponents. The agreement does not differ between games won and lost (80.0 vs 79.9 %, 80.2 vs 80.3 %).

### 7.10 Learning curve against V4 across the whole project

```
Dracula  - win rate vs V4 (%)                         (1 '#' = 2 pp; V4 mirror = 50.9)
Legacy BC (160 features)               13.9  #######
Legacy DAgger                          18.3  #########
Legacy PPO (2 M steps)                 20.8  ##########
BC (655 features)                      43.6  ######################
DAgger                                 45.8  #######################
PPO (2 M steps)                        53.1  ###########################
League self-play (gen. 10)             55.8  ############################
V4 mirror (reference)                  50.9  #########################

Van Helsing - win rate vs V4 (%)                      (V4 mirror = 49.1)
BC (655 features)                      42.8  #####################
DAgger                                 42.7  #####################
PPO (2 M steps)                        52.3  ##########################
League self-play (gen. 10)             55.0  ###########################
V4 mirror (reference)                  49.1  ########################
```

Win rates against every benchmark bot at each stage (deterministic play; the Greedy/V1/V2 entries of the PPO and gen.-10 columns come from the 3 000-game held-out comparison, the V3/V4 entries from 10 000-game evaluations):

| Opponent | Dracula BC | DAgger | PPO (`v1`) | Gen. 10 | Van Helsing BC | DAgger | PPO (`v1`) | Gen. 10 |
|---|---|---|---|---|---|---|---|---|
| Random | 99.46 | – | – | – | 99.50 | – | – | – |
| Greedy | 67.66 | 69.20 | 76.13 | 78.23 | 63.84 | 64.46 | 71.60 | 73.73 |
| V1 | 60.08 | 60.96 | 68.90 | 71.43 | 55.14 | 55.36 | 63.33 | 65.27 |
| V2 | 52.44 | 54.54 | 62.47 | 66.30 | 47.26 | 47.12 | 55.80 | 58.40 |
| V3 | 49.82 | 52.18 | 58.06 | 61.23 | 42.02 | 42.78 | 52.56 | 53.76 |
| **V4** | **43.60** | **45.84** | **53.12** | **55.76** | **42.82** | **42.72** | **52.34** | **55.04** |

---

## 8. Reproducing the experiments (commands)

All commands were run from the repository root in **Windows PowerShell** (the same commands work in a POSIX shell if paths are adapted). Output directories (`checkpoints/`, `bc_cache/`, `bc_history/`, `logs/`, `selfplay/`) are created automatically; `logs/` must exist before using `Tee-Object`.

### 8.0 Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir logs
```

### 8.1 Environment sanity checks and baselines

```powershell
# Random-play baseline + telemetry report (edit NUM_GAMES at the bottom of the file; 100 000 games were used for the 52.83 % figure)
python dvvh_env.py

# Bot-vs-bot series with telemetry (edit the pairs / GAMES_PER_SIDE at the bottom of the file)
python bots.py

# Leak, dynamics-equivalence and speed tests of the 655-d observation
python test_obs.py
```

### 8.2 Stage 1 — Behavioural-cloning warm-up (both roles)

```powershell
python warmup_bc.py --role 0 --eval-games 5000 | Tee-Object logs\bc_Dracula.log
python warmup_bc.py --role 1 --eval-games 5000 | Tee-Object logs\bc_VH.log
```

### 8.3 Stage 2 — DAgger (both roles)

```powershell
python dagger_bc.py --role 0 --iterations 4 --steps-per-iter 200000 --eval-games 5000 | Tee-Object logs\dagger_Dracula.log
python dagger_bc.py --role 1 --iterations 4 --steps-per-iter 200000 --eval-games 5000 | Tee-Object logs\dagger_VH.log

# Dracula only: the best iteration (iteration 1) replaced the final DAgger checkpoint (manual step)
Copy-Item .\checkpoints\Dracula\dagger_Dracula_iter1.zip .\checkpoints\Dracula\dagger_Dracula_final.zip -Force
```

(For Van Helsing the early stop happened after iteration 1, so `dagger_VanHelsing_final.zip` already is iteration 1.)

### 8.4 Stage 3 — PPO fine-tuning

```powershell
# Dracula (defaults seed the curriculum with the Dracula DAgger win rates)
python train_curriculum.py --role 0 --steps 2000000 --dagger-init --use-bc-aux

# Van Helsing (role-specific LR ramp, promotion threshold and EMA seeds)
python train_curriculum.py --role 1 --steps 2000000 --dagger-init --use-bc-aux `
    --lr-ramp-start 0.48 --lr-target 0.54 --promotion 0.50 `
    --init-ema "Random=0.995,Greedy=0.6446,V1=0.5536,V2=0.4712,V3=0.4278,V4=0.4272" `
    | Tee-Object logs\ppo_VH.log
```

Note: `--steps` is the number of steps of *this invocation*; with `--resume` SB3 adds it to the current step counter.

### 8.5 Evaluation of the PPO agents and freezing `v1`

```powershell
python eval_checkpoint.py --role 0 --games 10000 --opponents V4 V3 --models ./checkpoints/Dracula/ppo_Dracula_final.zip
python eval_checkpoint.py --role 1 --games 10000 --opponents V4 V3 --models ./checkpoints/VanHelsing/ppo_VanHelsing_final.zip

python diagnose_agent.py --model ./checkpoints/Dracula/ppo_Dracula_final.zip --opp V4 --episodes 1000
python diagnose_agent.py --model ./checkpoints/VanHelsing/ppo_VanHelsing_final.zip --ego 1 --opp V4 --episodes 1000

# Robustness / mirror (paired with V4 playing the same seat)
python robustness_eval.py --role 0 --model ./checkpoints/Dracula/ppo_Dracula_final.zip --games 5000 --conditions V4
python robustness_eval.py --role 0 --model ./checkpoints/Dracula/ppo_Dracula_final.zip --games 3000 --conditions V4_eps25 Mix_V3V4 Mix_V2V3V4
python robustness_eval.py --role 1 --model ./checkpoints/VanHelsing/ppo_VanHelsing_final.zip --games 3000 --conditions V4 V4_eps10 V4_eps25

# Freeze the PPO agents (used as league anchors and as baselines)
Copy-Item .\checkpoints\Dracula\ppo_Dracula_final.zip .\checkpoints\Dracula\FROZEN_Dracula_v1.zip
Copy-Item .\checkpoints\VanHelsing\ppo_VanHelsing_final.zip .\checkpoints\VanHelsing\FROZEN_VH_v1.zip

# Head-to-head of the frozen pair, with the transitivity prediction a + (1 - b) - m
python head_to_head.py --dracula ./checkpoints/Dracula/FROZEN_Dracula_v1.zip --vh ./checkpoints/VanHelsing/FROZEN_VH_v1.zip --games 5000 --mode both --a 0.5291 --b 0.5244 --m 0.5091 --pred-sd 0.008 | Tee-Object logs\h2h.log
```

### 8.6 Stage 4 — League self-play

```powershell
# Optional smoke test (a few minutes)
python train_selfplay.py --cycles 1 --phase-steps 8192 --envs 2 --n-steps 512 --batch-size 256 --eval-games 40 --v4-eval-games 40 --workers 2 --out-dir .\sp_smoke

# Baseline exploitability of the frozen PPO agents (400 k exploiter steps each)
python exploit_probe.py --target-role 0 --target ./checkpoints/Dracula/FROZEN_Dracula_v1.zip --tag baseline
python exploit_probe.py --target-role 1 --target ./checkpoints/VanHelsing/FROZEN_VH_v1.zip --tag baseline

# League self-play: 10 cycles x 100 k steps per role
python train_selfplay.py --cycles 10 --phase-steps 100000 | Tee-Object logs\selfplay.log

# Exploitability of the generation-10 agents
python exploit_probe.py --target-role 0 --target ./selfplay/Dracula/last.zip --tag gen10
python exploit_probe.py --target-role 1 --target ./selfplay/VanHelsing/last.zip --tag gen10
```

To extend a finished run (backup first, `last.zip` is overwritten): `python train_selfplay.py --cycles 20 --phase-steps 100000 --resume --eval-games 4000`.

### 8.7 Final validation

```powershell
# Full validation against V4, FROZEN_v1, robustness and balance (10 000 games each; ~28 minutes with 8 workers)
python final_validation.py --new-dracula ./selfplay/Dracula/last.zip --new-vh ./selfplay/VanHelsing/last.zip --games 10000 | Tee-Object logs\final_last.log

# Held-out opponents (never in the league; ~19 minutes)
python final_validation.py --new-dracula ./selfplay/Dracula/last.zip --new-vh ./selfplay/VanHelsing/last.zip --only-robust --robust-games 3000 --robust-conditions Greedy V1 V2 V4_eps25 V3_eps25 | Tee-Object logs\final_heldout.log

# Cross-check with the independent protocols used for the PPO agents
python eval_checkpoint.py --role 0 --games 10000 --opponents V4 V3 --models ./selfplay/Dracula/last.zip
python eval_checkpoint.py --role 1 --games 10000 --opponents V4 V3 --models ./selfplay/VanHelsing/last.zip
python head_to_head.py --dracula ./selfplay/Dracula/last.zip --vh ./selfplay/VanHelsing/last.zip --games 5000 --mode both --a 0.5615 --b 0.5480 --m 0.5091 --pred-sd 0.008
python diagnose_agent.py --model ./selfplay/Dracula/last.zip --opp V4 --episodes 1000
python diagnose_agent.py --model ./selfplay/VanHelsing/last.zip --ego 1 --opp V4 --episodes 1000
```

### 8.8 Release packaging and extended exploitability probes

```powershell
# Back up generation 10 (frozen "v2") and the raw self-play outputs
Copy-Item .\selfplay\Dracula\last.zip .\checkpoints\Dracula\FROZEN_Dracula_v2.zip -Force
Copy-Item .\selfplay\VanHelsing\last.zip .\checkpoints\VanHelsing\FROZEN_VH_v2.zip -Force
Copy-Item -Recurse .\selfplay .\selfplay_gen10 -Force

# Extended (1 M steps) exploitability probes, seed 901, separate output directory
python exploit_probe.py --target-role 0 --target ./checkpoints/Dracula/FROZEN_Dracula_v2.zip --tag v2_long --steps 1000000 --seed 901 --out-dir ./probes_long
python exploit_probe.py --target-role 0 --target ./checkpoints/Dracula/FROZEN_Dracula_v1.zip --tag v1_long --steps 1000000 --seed 901 --out-dir ./probes_long
python exploit_probe.py --target-role 1 --target ./checkpoints/VanHelsing/FROZEN_VH_v2.zip --tag v2_long --steps 1000000 --seed 901 --out-dir ./probes_long
python exploit_probe.py --target-role 1 --target ./checkpoints/VanHelsing/FROZEN_VH_v1.zip --tag v1_long --steps 1000000 --seed 901 --out-dir ./probes_long
```

---

## 9. Conclusions, limitations and future work

### 9.1 Conclusions

1. **Observation design was the key.** The same teacher (V4) that could be cloned to only 13.9 % win rate with 160 features is cloned to 43.6 % with 655 features; the representational change (rank-canonical encoding, exact unseen-card statistics, sequential-resolution DP, look-ahead) matters more than any optimiser setting we tuned.
2. **Imitation → refinement works.** The trajectory of the Dracula agent against V4 is **13.9 → 18.3 → 20.8 % (legacy)** and, with the redesigned pipeline, **43.6 (BC) → 45.8 (DAgger) → 53.1 (PPO) → 55.8 % (league self-play)**; for Van Helsing **42.8 → 42.7 → 52.3 → 55.0 %**. PPO contributes the largest step (+7 to +10 pp), DAgger a small one, and self-play a further +2.6 to +2.9 pp.
3. **The agents exceed the teacher.** Against the V4 mirrors (50.9 % / 49.1 %) the final agents are about +5 to +6 pp better, with a 95 % CI of ±1.2 pp.
4. **The result is robust and consistent.** The improvement holds on opponents outside the league (9/10 comparisons favourable, none significantly worse), strength composes transitively across the two roles, and exploiter probes do not find higher exploitability.
5. **Sound statistics matter as much as the algorithm.** Pooled estimates, paired baselines, pre-registered decision rules and large final evaluations (10 000 games) prevented several false conclusions (e.g. a spurious 54 % for V3 as Dracula from 500 games, or a flat in-training V4 curve that hid a +2 pp gap).

### 9.2 Limitations

* **All opponents are heuristics or our own agents.** No human play, no independent search-based opponent (e.g. information-set MCTS); absolute strength is therefore not established.
* **Exploitability is only bounded from below** (one seed per probe, 400 k–1 M steps, no search-based best response).
* **Per-cycle gates were underpowered** (2 000 games, SE ≈ 1.1 pp); the champion selection mechanism did not pick the best Van Helsing snapshot; we validated `last.zip` instead.
* **Not a controlled ablation.** The legacy and the new pipelines differ in several aspects (features, data volume, PPO configuration, curriculum fixes). Only the feature-gating experiment isolates the effect of the observation.
* **Single training seed** per role and per stage; run-to-run variance is unknown.
* **Evaluation uses deterministic play**; stochastic play is ≈ 1–2 pp weaker for some roles (not significant in our tests).
* **Capacity/data ratio** of the 655-input network is ≈ 2 samples/parameter (mild overfitting at the BC stage).
* **Role asymmetry** (Dracula +2.8 pp under random play, +0.9 ± 1.0 pp in the V4 mirror) requires role-specific baselines; cross-role comparisons must be made carefully.

### 9.3 Future work

* An independent, search-based benchmark (information-set Monte-Carlo tree search with determinisation) and human evaluation.
* Stronger exploiters (several seeds, longer budgets, population-based exploiters, league exploiters / main exploiters) and a full best-response computation on abstractions of the game.
* Asymmetric actor-critic (critic with privileged access to the hidden cards and deck order) to reduce advantage variance (Pinto et al. [7]); GAE λ closer to 1 for the ≈ 78-step horizon.
* Longer self-play (with larger gate evaluations, e.g. 4 000 games) and a more diverse league; several seeds to quantify variance.
* Distillation or search-augmented training (policy improvement via lookahead on the exact simulator with sampled hidden states).

---

## 10. References

1. S. Ross, G. Gordon, J. A. Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger). AISTATS 2011.
2. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
3. A. Y. Ng, D. Harada, S. Russell. *Policy invariance under reward transformations: theory and application to reward shaping.* ICML 1999.
4. S. Huang, S. Ontañón. *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* FLAIRS 2022.
5. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
6. O. Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature 575, 2019 (league training, prioritised fictitious self-play).
7. L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, P. Abbeel. *Asymmetric Actor Critic for Image-Based Robot Learning.* RSS 2018.
8. A. Raffin et al. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR 22, 2021.
9. B. Liu et al. *SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning* (role-conditioned advantage estimation), 2025.
10. J. Heinrich, M. Lanctot, D. Silver. *Fictitious Self-Play in Extensive-Form Games.* ICML 2015.
11. M. Lanctot et al. *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning* (PSRO). NeurIPS 2017.

---

## 11. Appendix

### A. Statistical reference

| N games | SE at p = 0.5 | 95 % CI half-width |
|---|---|---|
| 500 | 2.24 pp | 4.4 pp |
| 1 000 | 1.58 pp | 3.1 pp |
| 2 000 | 1.12 pp | 2.2 pp |
| 5 000 | 0.71 pp | 1.4 pp |
| 10 000 | 0.50 pp | 0.98 pp |
| 20 000 | 0.35 pp | 0.69 pp |

Difference of two independent proportions: `z = (p₁ − p₂) / √(p₁(1 − p₁)/N₁ + p₂(1 − p₂)/N₂)`. Transitivity prediction for the head-to-head of two agents A (Dracula) and B (Van Helsing): `P(A beats B) ≈ a + (1 − b) − m` with `a` = win rate of A against V4, `b` = win rate of B against V4 and `m` = V4-vs-V4 Dracula win rate.

### B. Lessons learned (engineering)

* Validate every observation change with a **hidden-information permutation test**; derived features are an easy way to leak.
* After DAgger the **critic is random**: freeze the actor and warm up the value function before PPO.
* SB3's `learn(total_timesteps)` is **additive on resume**; schedules based on progress remaining must be designed accordingly (we use a win-rate-driven LR instead).
* Never trust a 250–1 000-game evaluation to choose between checkpoints; use pooled windows and a large independent final evaluation.
* Make unlock rules **monotone (latches)** to avoid curriculum oscillations; seed EMAs with measured baselines.
* Loading custom objects (LR schedules) from pickled checkpoints in worker processes is fragile: load with `custom_objects` overrides.
* On Windows, `SubprocVecEnv` and process pools use `spawn`: every entry point needs a `if __name__ == "__main__":` guard.

### C. Checkpoints produced

| File | Meaning |
|---|---|
| `checkpoints/<Role>/bc_warmup_<Role>_model.zip` | BC model |
| `checkpoints/<Role>/dagger_<Role>_iter{k}.zip`, `dagger_<Role>_final.zip` | DAgger iterations / selected |
| `checkpoints/<Role>/ppo_<Role>_final.zip`, `snapshot_*`, `best_*` | PPO outputs |
| `checkpoints/Dracula/FROZEN_Dracula_v1.zip`, `checkpoints/VanHelsing/FROZEN_VH_v1.zip` | frozen PPO agents (league anchors, baselines) |
| `selfplay/<Role>/last.zip`, `snapshots/snap_cXXX.zip`, `champion.zip` (if any) | league self-play outputs |
| `checkpoints/Dracula/FROZEN_Dracula_v2.zip`, `checkpoints/VanHelsing/FROZEN_VH_v2.zip` | **final agents (generation 10)** |

### D. Citation

If you use this project, please cite the repository and the references above.