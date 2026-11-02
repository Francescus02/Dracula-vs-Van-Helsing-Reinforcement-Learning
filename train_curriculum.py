"""
PPO fine-tuning for Dracula vs Van Helsing.

Features:
  - Init from DAgger checkpoint (--dagger-init).
  - Resume from PPO checkpoint (--resume) with continuous LR, annealing, callback state.
  - BC auxiliary loss (--use-bc-aux) to prevent catastrophic forgetting of the DAgger prior.
  - Adaptive curriculum callback with per-eval state persistence.

Usage:
    # First run: 500k steps from DAgger init + BC aux loss
    python train_curriculum.py --role 0 --steps 500000 --dagger-init --use-bc-aux

    # Resume to 1.5M steps
    python train_curriculum.py --role 0 --steps 1500000 --resume --use-bc-aux

    # Baseline (from scratch)
    python train_curriculum.py --role 0 --steps 3000000
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import MaskablePPO

from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer

from rl_wrappers import make_masked_dvvh_env
from rl_callbacks import AdaptiveCurriculumAndEloCallback, AdaptiveLR


# =============================================================================
# MASKABLE PPO WITH BC AUXILIARY LOSS
# =============================================================================
class MaskablePPOWithBC(MaskablePPO):
    """
    MaskablePPO with an auxiliary behavioral-cloning loss on a fixed dataset.

    After each rollout (via super().train()), performs `bc_steps_per_rollout`
    minibatch SGD steps on a separately-maintained dataset with its own Adam
    optimizer. The BC optimizer's LR is `main_lr * bc_coef(t)` where `bc_coef`
    decays linearly from `bc_coef_start` to `bc_coef_end` over `bc_decay_steps`.

    This preserves the DAgger prior during the vulnerable early PPO phase
    without preventing exploration. The BC dataset is NOT saved in the .zip
    (it would inflate the file by ~600 MB); it is re-injected on load.
    """

    def __init__(
        self,
        *args,
        bc_dataset: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        bc_coef_start: float = 0.30,
        bc_coef_end: float = 0.03,
        bc_decay_steps: int = 1_000_000,
        bc_steps_per_rollout: int = 50,
        critic_warmup_steps: int = 0,
        critic_warmup_lr_boost: float = 5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._critic_warmup_steps = critic_warmup_steps
        self._critic_warmup_lr_boost = critic_warmup_lr_boost
        self._bc_obs = None
        self._bc_masks = None
        self._bc_acts = None
        if bc_dataset is not None:
            obs, masks, acts = bc_dataset
            self._bc_obs = th.as_tensor(obs, dtype=th.float32)
            self._bc_masks = th.as_tensor(masks, dtype=th.bool)
            self._bc_acts = th.as_tensor(acts, dtype=th.long)
            print(f"[BC AUX] dataset: {len(self._bc_obs):,} samples")
        self._bc_coef_start = bc_coef_start
        self._bc_coef_end = bc_coef_end
        self._bc_decay_steps = bc_decay_steps
        self._bc_steps = bc_steps_per_rollout
        self._bc_optimizer: Optional[th.optim.Adam] = None
        self._bc_step_count = 0
        self._bc_first_call = True
        # Set by the callback: scales the BC anchor by the ratcheted win rate vs V4 (None = time decay)
        self.bc_scale: Optional[float] = None

    def _bc_coef(self) -> float:
        if self._bc_obs is None:
            return 0.0
        if getattr(self, "bc_scale", None) is not None:
            return float(self._bc_coef_start * self.bc_scale)
        t = min(1.0, self.num_timesteps / max(1, self._bc_decay_steps))
        return float(
            self._bc_coef_start + t * (self._bc_coef_end - self._bc_coef_start)
        )

    def _bc_train(self) -> None:
        coef = self._bc_coef()
        if coef < 1e-5 or self._bc_obs is None:
            return

        policy = self.policy
        policy.set_training_mode(True)

        if self._bc_optimizer is None:
            self._bc_optimizer = th.optim.Adam(policy.parameters(), lr=1e-4)
            if self._bc_first_call and self.verbose >= 1:
                print(
                    f"[BC AUX] optimizer created | dataset={len(self._bc_obs):,} "
                    f"| coef={coef:.3f} | steps/rollout={self._bc_steps}"
                )
                self._bc_first_call = False

        main_lr = float(self.lr_schedule(self._current_progress_remaining))
        for pg in self._bc_optimizer.param_groups:
            pg["lr"] = main_lr * coef

        n = len(self._bc_obs)
        for _ in range(self._bc_steps):
            idx = th.randint(0, n, (self.batch_size,))
            obs = self._bc_obs[idx]
            masks = self._bc_masks[idx]
            acts = self._bc_acts[idx]
            dist = policy.get_distribution(obs, action_masks=masks)
            log_prob = dist.log_prob(acts)
            bc_loss = -log_prob.mean()
            self._bc_optimizer.zero_grad()
            bc_loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            self._bc_optimizer.step()
            self._bc_step_count += 1

    def _set_actor_frozen(self, frozen: bool) -> None:
        params = list(self.policy.mlp_extractor.policy_net.parameters()) + list(
            self.policy.action_net.parameters()
        )
        for p in params:
            p.requires_grad_(not frozen)

    def train(self) -> None:
        """
        Critic warm-up: a DAgger/BC checkpoint has a trained actor but a RANDOM critic.
        With a random critic the GAE advantages are noise and the first PPO updates would
        damage the imitation prior. For the first `critic_warmup_steps` timesteps the actor is
        frozen and only the value network is trained (with a boosted LR).
        """
        warm = self.num_timesteps < getattr(self, "_critic_warmup_steps", 0)
        ctrl = self.lr_schedule
        self._set_actor_frozen(warm)
        if hasattr(ctrl, "boost"):
            ctrl.boost = self._critic_warmup_lr_boost if warm else 1.0
        super().train()
        if not warm:
            self._bc_train()

    def save(self, path, exclude=None, include=None) -> None:
        """Strip BC tensors before save (saves ~600 MB) and persist BC optimizer state."""
        saved_attrs = {}
        for attr in ("_bc_obs", "_bc_masks", "_bc_acts", "_bc_optimizer"):
            if hasattr(self, attr):
                saved_attrs[attr] = getattr(self, attr)
                setattr(self, attr, None)

        # Persist BC optimizer moments to a sidecar file (parameter-reference safe)
        if saved_attrs.get("_bc_optimizer") is not None:
            bc_opt_path = os.path.splitext(str(path))[0] + ".bc_optimizer.pt"
            th.save(saved_attrs["_bc_optimizer"].state_dict(), bc_opt_path)

        try:
            super().save(path, exclude=exclude, include=include)
        finally:
            for attr, val in saved_attrs.items():
                setattr(self, attr, val)


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class TrainConfig:
    ego_player: int = 0
    total_timesteps: int = 1_500_000
    n_envs: int = 8
    seed: int = 42
    save_root: str = "./checkpoints"
    log_root: str = "./tensorboard_logs"
    bc_cache_root: str = "./bc_cache"

    # PPO (fine-tuning from a strong imitation prior: small, low-variance steps)
    learning_rate_start: float = 5e-5     # lr_high of AdaptiveLR
    lr_min: float = 5e-6
    lr_ramp_start: float = 0.50           # V4 pooled WR where the LR starts to decrease
    lr_target: float = 0.56               # V4 pooled WR where LR = 0.30 x lr_high
    n_steps: int = 4096                   # x n_envs = 32,768 samples / rollout
    batch_size: int = 1024
    n_epochs: int = 4
    gamma: float = 1.0                    # win/loss is the objective: undiscounted
    gae_lambda: float = 0.95
    clip_range: float = 0.10
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.010
    net_arch: List[int] = field(default_factory=lambda: [512, 256])
    critic_warmup_steps: int = 150_000
    critic_warmup_lr_boost: float = 5.0

    # Callback
    eval_freq: int = 100_000
    eval_games_per_bot: int = 250
    v4_eval_games: int = 1000
    initial_shaping_weight: float = 0.0   # shaping is NOT potential-based -> off by default
    annealing_fraction: float = 0.55
    promotion_threshold: float = 0.52     # pooled V4 WR (last 3 evals) to count as a pass
    required_consecutive_passes: int = 2
    train_exclude: str = "Random"         # evaluated but never sampled for training
    bc_wr_gating: bool = False
    init_ema: str = ""                    # e.g. "Random=0.99,Greedy=0.7,V1=0.6,V2=0.55,V3=0.5,V4=0.45"

    # BC auxiliary anchor (V4 demonstrations only, time-decayed)
    use_bc_aux: bool = False
    bc_teachers: str = "V4"
    bc_coef_start: float = 0.20
    bc_coef_end: float = 0.0
    bc_decay_steps: int = 800_000
    bc_steps_per_rollout: int = 50

    # Init / resume
    init_from: Optional[str] = None
    resume: bool = False
    dagger_init: bool = False


# Win rates of the DAgger iter-1 checkpoint (5000 games each). They seed the EMA/pooled
# estimates so that every opponent is unlocked from step 0 and the LR starts consistent.
DAGGER_INIT_EMA = {
    "Random": 0.990, "Greedy": 0.692, "V1_Tactical": 0.610,
    "V2_Bayesian": 0.545, "V3_Asymmetric": 0.522, "V4_Minimax": 0.458,
}


_EMA_ALIASES = {"Random": "Random", "Greedy": "Greedy", "V1": "V1_Tactical", "V2": "V2_Bayesian",
                "V3": "V3_Asymmetric", "V4": "V4_Minimax"}


def _parse_init_ema(text: str) -> dict:
    """Parse '--init-ema Random=0.99,Greedy=0.7,V1=0.6,V2=0.55,V3=0.5,V4=0.45' (win rates in [0,1]).
    Falls back to the Dracula DAgger baselines when empty -> ALWAYS pass it for another role."""
    out = dict(DAGGER_INIT_EMA)
    if not text:
        return out
    for part in text.split(","):
        k, v = part.split("=")
        out[_EMA_ALIASES.get(k.strip(), k.strip())] = float(v)
    return out


def linear_schedule(initial_value: float, floor: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return max(floor, progress_remaining * initial_value)
    return func


# =============================================================================
# HELPERS
# =============================================================================
def _role_name(ego_player: int) -> str:
    return "Dracula" if ego_player == 0 else "VanHelsing"


def _checkpoint_paths(cfg: TrainConfig) -> Tuple[str, str, str]:
    role = _role_name(cfg.ego_player)
    out_dir = os.path.join(cfg.save_root, role)
    os.makedirs(out_dir, exist_ok=True)
    final_ckpt = os.path.join(out_dir, f"ppo_{role}_final.zip")
    dagger_ckpt = os.path.join(out_dir, f"dagger_{role}_final.zip")
    state_path = os.path.join(out_dir, f"ppo_{role}_callback_state.json")
    return final_ckpt, dagger_ckpt, state_path


def _load_bc_dataset(cfg: "TrainConfig") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    role = _role_name(cfg.ego_player)
    cache_dir = os.path.join(cfg.bc_cache_root, role)
    if not os.path.isdir(cache_dir):
        print(f"[BC AUX] cache not found at {cache_dir}; BC aux disabled")
        return None
    teachers = [t.strip() for t in cfg.bc_teachers.split(",") if t.strip()]
    files = sorted(
        f for f in os.listdir(cache_dir)
        if f.endswith(".npz") and any(f.startswith(f"teacher_{t}_") for t in teachers)
    )
    if not files:
        print(f"[BC AUX] no .npz files in {cache_dir}; BC aux disabled")
        return None
    obs_l, mask_l, act_l = [], [], []
    for fname in files:
        data = np.load(os.path.join(cache_dir, fname))
        obs_l.append(data["obs"])
        mask_l.append(data["masks"])
        act_l.append(data["acts"])
    obs = np.concatenate(obs_l, axis=0)
    masks = np.concatenate(mask_l, axis=0)
    acts = np.concatenate(act_l, axis=0)
    print(f"[BC AUX] loaded {len(obs):,} samples from {len(files)} files")
    return obs, masks, acts


# =============================================================================
# ENV & MODEL
# =============================================================================
def build_vec_env(cfg: TrainConfig) -> VecMonitor:
    env_fns = [
        make_masked_dvvh_env(
            rank=i,
            ego_player=cfg.ego_player,
            shaping_weight=cfg.initial_shaping_weight,
            base_seed=cfg.seed,
        )
        for i in range(cfg.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)
    return VecMonitor(vec_env, info_keywords=("ego_won", "ep_shaping_reward"))


def build_model(env: VecMonitor, cfg: TrainConfig, log_path: str, lr_ctrl: AdaptiveLR) -> MaskablePPO:
    policy_kwargs = dict(net_arch=cfg.net_arch, activation_fn=nn.ReLU)
    custom_logger = configure(log_path, ["csv", "tensorboard"])
    bc_dataset = _load_bc_dataset(cfg) if cfg.use_bc_aux else None

    if cfg.init_from and os.path.exists(cfg.init_from):
        print(f"[INIT] Loading model from {cfg.init_from}")
        model = MaskablePPOWithBC.load(cfg.init_from, env=env)

        # Override training-time hyperparameters (schema may differ from saved)
        model.learning_rate = lr_ctrl
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        model.clip_range = get_schedule_fn(cfg.clip_range)
        model.ent_coef = cfg.ent_coef
        model.n_epochs = cfg.n_epochs
        model.batch_size = cfg.batch_size
        model.n_steps = cfg.n_steps
        model.gamma = cfg.gamma
        model.gae_lambda = cfg.gae_lambda
        model.vf_coef = cfg.vf_coef
        model.max_grad_norm = cfg.max_grad_norm
        model.target_kl = cfg.target_kl
        model.rollout_buffer = MaskableRolloutBuffer(
            model.n_steps, model.observation_space, model.action_space, model.device,
            gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs,
        )
        model._critic_warmup_steps = cfg.critic_warmup_steps
        model._critic_warmup_lr_boost = cfg.critic_warmup_lr_boost

        # Inject BC dataset fresh (was stripped at save time)
        if bc_dataset is not None:
            obs, masks, acts = bc_dataset
            model._bc_obs = th.as_tensor(obs, dtype=th.float32)
            model._bc_masks = th.as_tensor(masks, dtype=th.bool)
            model._bc_acts = th.as_tensor(acts, dtype=th.long)
            print(f"[BC AUX] injected {len(model._bc_obs):,} samples into loaded model")
        else:
            model._bc_obs = None
            model._bc_masks = None
            model._bc_acts = None
        model._bc_coef_start = cfg.bc_coef_start
        model.bc_scale = None
        model._bc_coef_end = cfg.bc_coef_end
        model._bc_decay_steps = cfg.bc_decay_steps
        model._bc_steps = cfg.bc_steps_per_rollout
        model.ent_coef = cfg.ent_coef
        model._bc_step_count = 0
        model._bc_first_call = True

        # Restore BC optimizer moments from sidecar if available
        bc_opt_path = os.path.splitext(str(cfg.init_from))[0] + ".bc_optimizer.pt"
        if os.path.exists(bc_opt_path) and model._bc_obs is not None:
            model._bc_optimizer = th.optim.Adam(model.policy.parameters(), lr=1e-4)
            try:
                model._bc_optimizer.load_state_dict(
                    th.load(bc_opt_path, map_location="cpu")
                )
                print(f"[BC AUX] restored BC optimizer moments from {bc_opt_path}")
            except Exception as e:
                print(f"[BC AUX] could not restore BC optimizer ({e}); starting fresh")
                model._bc_optimizer = None
        else:
            model._bc_optimizer = None

        model.set_logger(custom_logger)
        return model

    model = MaskablePPOWithBC(
        "MlpPolicy",
        env,
        learning_rate=lr_ctrl,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
        seed=cfg.seed,
        bc_dataset=bc_dataset,
        bc_coef_start=cfg.bc_coef_start,
        bc_coef_end=cfg.bc_coef_end,
        bc_decay_steps=cfg.bc_decay_steps,
        bc_steps_per_rollout=cfg.bc_steps_per_rollout,
        critic_warmup_steps=cfg.critic_warmup_steps,
        critic_warmup_lr_boost=cfg.critic_warmup_lr_boost,
    )
    model.set_logger(custom_logger)
    return model


# =============================================================================
# TRAIN
# =============================================================================
def train(cfg: TrainConfig) -> None:
    role = _role_name(cfg.ego_player)
    run_name = f"{role}_PPO_Seed{cfg.seed}"
    log_path = os.path.join(cfg.log_root, run_name)
    final_ckpt, dagger_ckpt, state_path = _checkpoint_paths(cfg)

    if cfg.resume:
        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(
                f"--resume requested but {final_ckpt} does not exist."
            )
        cfg.init_from = final_ckpt
        cfg.initial_shaping_weight = 0.0
        print(f"[MODE] RESUME from {final_ckpt}")
        print(f"[MODE] Callback state: {state_path}")
    elif cfg.dagger_init:
        if not os.path.exists(dagger_ckpt):
            raise FileNotFoundError(
                f"--dagger-init requested but {dagger_ckpt} does not exist."
            )
        cfg.init_from = dagger_ckpt
        # Fresh run: remove stale callback state from previous tests
        if os.path.exists(state_path):
            os.remove(state_path)
            print(f"[FRESH] Removed stale callback state at {state_path}")
        print(f"[MODE] INIT from DAgger checkpoint {dagger_ckpt}")
    else:
        cfg.init_from = None
        print("[MODE] FROM SCRATCH (no init)")

    print("=" * 80)
    print(f"STARTING PPO TRAINING FOR: {role}")
    print(f"Target total steps: {cfg.total_timesteps:,}")
    print(f"Envs: {cfg.n_envs} | Seed: {cfg.seed}")
    print(f"BC aux loss: {cfg.use_bc_aux}")
    print(f"LR start: {cfg.learning_rate_start:.2e} | Ent coef: {cfg.ent_coef}")
    print("=" * 80)

    env = build_vec_env(cfg)
    lr_ctrl = AdaptiveLR(lr_high=cfg.learning_rate_start, lr_min=cfg.lr_min,
                         ramp_start=cfg.lr_ramp_start, target=cfg.lr_target)
    model = build_model(env, cfg, log_path, lr_ctrl)
    if cfg.use_bc_aux and cfg.bc_wr_gating:
        model.bc_scale = 1.0
    print(f"[CFG] critic warm-up: {cfg.critic_warmup_steps:,} steps | shaping w0={cfg.initial_shaping_weight} | "
          f"excluded from training: {cfg.train_exclude}")

    if cfg.resume:
        print(f"[RESUME] model.num_timesteps = {model.num_timesteps:,}")

    callback = AdaptiveCurriculumAndEloCallback(
        ego_player=cfg.ego_player,
        total_timesteps=cfg.total_timesteps,
        eval_freq=cfg.eval_freq,
        eval_games_per_bot=cfg.eval_games_per_bot,
        initial_shaping_weight=cfg.initial_shaping_weight,
        annealing_fraction=cfg.annealing_fraction,
        promotion_winrate_threshold=cfg.promotion_threshold,
        required_consecutive_passes=cfg.required_consecutive_passes,
        save_dir=cfg.save_root,
        deterministic_eval=True,
        verbose=1,
        state_path=state_path,
        lr_controller=lr_ctrl,
        v4_eval_games=cfg.v4_eval_games,
        bc_wr_gating=cfg.bc_wr_gating,
        initial_ema=_parse_init_ema(cfg.init_ema) if cfg.dagger_init else None,
        train_exclude=[x.strip() for x in cfg.train_exclude.split(",") if x.strip()],
    )

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=not cfg.resume,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving current model...")
    finally:
        model.save(final_ckpt)
        print(f"\n[SAVE] Final model -> {final_ckpt}")
        env.close()


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=int, choices=[0, 1], required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument(
        "--steps", type=int, default=2_000_000,
        help="Timesteps to run in THIS invocation. On --resume SB3 ADDS them to the current "
             "num_timesteps (resume at 500k with --steps 1500000 ends at 2.0M).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dagger-init", action="store_true", help="Init from dagger_{role}_final.zip")
    parser.add_argument("--resume", action="store_true", help="Resume from ppo_{role}_final.zip")
    parser.add_argument("--use-bc-aux", action="store_true")
    parser.add_argument("--no-bc-aux", action="store_true")
    for name, typ in [
        ("lr-high", float), ("lr-min", float), ("lr-ramp-start", float), ("lr-target", float),
        ("ent-coef", float), ("n-steps", int), ("batch-size", int), ("n-epochs", int),
        ("gamma", float), ("clip-range", float), ("target-kl", float),
        ("eval-freq", int), ("eval-games", int), ("v4-eval-games", int),
        ("shaping", float), ("critic-warmup-steps", int), ("bc-teachers", str),
        ("bc-coef-start", float), ("bc-coef-end", float), ("bc-decay-steps", int),
        ("bc-steps-per-rollout", int), ("promotion", float), ("passes", int),
        ("train-exclude", str), ("init-ema", str),
    ]:
        parser.add_argument(f"--{name}", type=typ, default=None)

    args = parser.parse_args()
    if args.resume and args.dagger_init:
        parser.error("--resume and --dagger-init are mutually exclusive")

    m = {  # CLI name -> TrainConfig field
        "lr_high": "learning_rate_start", "lr_min": "lr_min", "lr_ramp_start": "lr_ramp_start",
        "lr_target": "lr_target", "ent_coef": "ent_coef", "n_steps": "n_steps",
        "batch_size": "batch_size", "n_epochs": "n_epochs", "gamma": "gamma",
        "clip_range": "clip_range", "target_kl": "target_kl", "eval_freq": "eval_freq",
        "eval_games": "eval_games_per_bot", "v4_eval_games": "v4_eval_games",
        "shaping": "initial_shaping_weight", "critic_warmup_steps": "critic_warmup_steps",
        "bc_teachers": "bc_teachers", "bc_coef_start": "bc_coef_start", "bc_coef_end": "bc_coef_end",
        "bc_decay_steps": "bc_decay_steps", "bc_steps_per_rollout": "bc_steps_per_rollout",
        "promotion": "promotion_threshold", "passes": "required_consecutive_passes",
        "train_exclude": "train_exclude", "init_ema": "init_ema",
    }
    overrides = {m[k]: v for k, v in vars(args).items() if k in m and v is not None}
    if not (args.resume or args.dagger_init):  # from scratch: classic PPO settings
        overrides.setdefault("learning_rate_start", 3e-4)
        overrides.setdefault("ent_coef", 0.01)
        overrides.setdefault("initial_shaping_weight", 1.0)
        overrides.setdefault("critic_warmup_steps", 0)
        overrides.setdefault("gamma", 0.995)

    cfg = TrainConfig(
        ego_player=args.role, n_envs=args.envs, total_timesteps=args.steps, seed=args.seed,
        resume=args.resume, dagger_init=args.dagger_init,
        use_bc_aux=(args.dagger_init or args.resume) and not args.no_bc_aux,
        **overrides,
    )
    train(cfg)