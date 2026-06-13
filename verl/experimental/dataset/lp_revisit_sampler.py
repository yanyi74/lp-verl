# Copyright 2025
# LP-GRPO v2: Learning-Progress driven dense-revisit curriculum sampler.
#
# Maintains a small "active pool" so that prompts are revisited densely enough
# to estimate a reliable, smoothed learning-progress signal (dema) via double-EMA.
# Drives: (1) which prompts to (re)visit each step, (2) drop mastered/stuck prompts,
# (3) expose dema_map for the advantage estimator (lp_grpo_v2) to read.
#
# Implements verl's AbstractCurriculumSampler interface:
#   __iter__ : yields prompt indices (dataset positions) for the next epoch
#   update(batch): called after each train step, updates per-prompt state
"""LPRevisitSampler — dense-revisit curriculum sampler for LP-GRPO v2."""
import math
import numpy as np
from collections.abc import Sized
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class LPRevisitSampler(AbstractCurriculumSampler):
    """Learning-progress driven dense-revisit sampler.

    Pool mechanics (per prompt, keyed by persistent index `pid`):
      base[pid]  : slow EMA baseline of p_t          (beta)
      dema[pid]  : EMA of (p_t - base)  = progress    (alpha)
      stuck[pid] : consecutive (p_t<=eps & |dema|<theta) count
      visit[pid] : how many times this prompt was trained

    Selection score for a prompt currently in the active pool:
      score = 0                              if stuck>=K (cooling)
            = 0                              if visit>=M and dema<=theta (graduate)
            = (1 + lam*max(0,dema)) * sqrt(4*p_t*(1-p_t))   otherwise
      (difficulty is intentionally NOT in the score; it lives in the advantage
       weight w=(1-p0)^g. Sampling only cares "is there gradient signal now".)

    Each step a batch is filled with `revisit_ratio` weighted-sampled pool prompts
    + the rest fresh from reserve. mastered/stuck prompts leave the pool; reserve
    and cooldown-expired prompts refill it.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.N = len(data_source)
        c = data_config.get("lp_sampler", {}) or {}
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        self.pool_size = int(c.get("pool_size", 2000))
        self.revisit_ratio = float(c.get("revisit_ratio", 0.7))
        self.alpha = float(c.get("alpha", 0.4))      # dema EMA
        self.beta = float(c.get("beta", 0.2))        # base EMA
        self.theta = float(c.get("theta", 0.05))     # progress deadzone
        self.K = int(c.get("stuck_K", 3))            # stuck -> cooldown
        self.M = int(c.get("max_visit", 6))          # soft visit cap
        self.cooldown_T = int(c.get("cooldown_T", 100))
        self.lam = float(c.get("lam", 3.0))          # progress boost in score
        self.eps = float(c.get("eps", 0.05))         # ~0/8 and ~8/8 thresholds
        self.total_steps = int(c.get("total_steps", 10_000_000))
        self.seed = int(data_config.get("seed", 1) or 1)
        self.rng = np.random.default_rng(self.seed)

        # per-prompt persistent state (index pid in [0, N))
        self.base = np.full(self.N, np.nan, dtype=np.float64)  # nan = uninitialized
        self.dema = np.zeros(self.N, dtype=np.float64)
        self.stuck = np.zeros(self.N, dtype=np.int32)
        self.visit = np.zeros(self.N, dtype=np.int32)
        self.p0 = np.full(self.N, 0.5, dtype=np.float64)       # filled from p_zero if available

        # try to load offline p_zero as base/p0 initial value
        self._load_p_zero()

        # pool partition
        all_idx = list(range(self.N))
        self.rng.shuffle(all_idx)
        init_pool = min(self.pool_size, self.N)
        self.active = set(all_idx[:init_pool])
        self.reserve = list(all_idx[init_pool:])     # FIFO-ish queue
        self.cooldown = {}                           # pid -> remaining steps
        self._step = 0

        # SHARED with advantage estimator: trainer will point lp_state["dema_map"] here
        self.dema_map = {}

    # ---- offline p_zero ----
    def _load_p_zero(self):
        ds = self.data_source
        df = getattr(ds, "dataframe", None)
        if df is None or "p_zero" not in getattr(df, "column_names", []):
            return
        try:
            pz = list(df["p_zero"])
            extra = list(df["extra_info"]) if "extra_info" in df.column_names else [None] * len(pz)
            for p, ex in zip(pz, extra):
                if p is None:
                    continue
                idx = (ex or {}).get("index", None) if isinstance(ex, dict) else None
                if idx is None or not (0 <= idx < self.N):
                    continue
                self.p0[idx] = float(p)
                self.base[idx] = float(p)  # warm-start baseline with prior
        except Exception as e:
            print(f"[LPRevisitSampler] p_zero load skipped: {e}")

    # ---- scoring ----
    def _score(self, pid):
        if self.stuck[pid] >= self.K:
            return 0.0
        if self.visit[pid] >= self.M and self.dema[pid] <= self.theta:
            return 0.0
        # use base as current p_t proxy when prompt not yet seen this round
        p_t = self.base[pid] if not math.isnan(self.base[pid]) else self.p0[pid]
        p_t = min(max(p_t, 0.0), 1.0)
        sigma_gate = math.sqrt(max(4.0 * p_t * (1.0 - p_t), 0.0))
        prog_boost = 1.0 + self.lam * max(0.0, self.dema[pid])
        return prog_boost * sigma_gate + 1e-6  # +eps so fresh-ish pool prompts still selectable

    def _refill_pool(self):
        # bring cooldown-expired back, then reserve
        revived = [pid for pid, t in self.cooldown.items() if t <= 0]
        for pid in revived:
            del self.cooldown[pid]
            self.stuck[pid] = 0
            self.active.add(pid)
        while len(self.active) < self.pool_size and self.reserve:
            self.active.add(self.reserve.pop())

    def __len__(self):
        # number of prompts yielded per epoch == dataset size (keeps dataloader len sane)
        return self.N

    def __iter__(self):
        # Generator: yield indices one at a time; DataLoader slices into batches.
        # We yield in chunks of batch_size, re-selecting from the pool each chunk so
        # state updated by update() between batches takes effect.
        n_yield = 0
        while n_yield < self.N:
            self._refill_pool()
            pool = list(self.active)
            if not pool:
                # fallback: pure reserve / random
                pool = list(self.rng.integers(0, self.N, size=self.batch_size))
            scores = np.array([self._score(pid) for pid in pool], dtype=np.float64)
            n_re = min(int(self.batch_size * self.revisit_ratio), len(pool))
            chosen = []
            if scores.sum() > 0 and n_re > 0:
                p = scores / scores.sum()
                k = min(n_re, int((scores > 0).sum()))
                if k > 0:
                    chosen = list(self.rng.choice(pool, size=k, replace=False, p=p))
            # fill the rest with fresh reserve prompts
            n_fresh = self.batch_size - len(chosen)
            for _ in range(n_fresh):
                if self.reserve:
                    pid = self.reserve.pop()
                    self.active.add(pid)
                    chosen.append(pid)
                elif pool:
                    chosen.append(int(self.rng.choice(pool)))
            for pid in chosen:
                if n_yield >= self.N:
                    break
                yield int(pid)
                n_yield += 1

    # ---- state update (called by trainer after each step) ----
    def update(self, batch: DataProto) -> None:
        self._step += 1
        # decay cooldowns
        for pid in list(self.cooldown.keys()):
            self.cooldown[pid] -= 1

        if "index" not in batch.non_tensor_batch:
            return
        idx_arr = batch.non_tensor_batch["index"]
        # group reward by uid to get per-prompt p_t. token_level_scores summed.
        scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
        uid_arr = batch.non_tensor_batch.get("uid", idx_arr)

        # aggregate p_t per persistent index
        from collections import defaultdict
        acc = defaultdict(list)
        for i in range(len(idx_arr)):
            acc[int(idx_arr[i])].append(float(scores[i]))
        for pid, vals in acc.items():
            if not (0 <= pid < self.N):
                continue
            p_t = float(np.mean([1.0 if v > 0.5 else 0.0 for v in vals]))  # pass-rate
            if math.isnan(self.base[pid]):
                self.base[pid] = p_t
            d = p_t - self.base[pid]
            self.dema[pid] = (1 - self.alpha) * self.dema[pid] + self.alpha * d
            self.base[pid] = (1 - self.beta) * self.base[pid] + self.beta * p_t
            self.visit[pid] += 1
            # stuck: ~0/8 and no progress
            if p_t <= self.eps and abs(self.dema[pid]) < self.theta:
                self.stuck[pid] += 1
            else:
                self.stuck[pid] = 0
            # expose smoothed progress to advantage estimator
            self.dema_map[pid] = float(self.dema[pid])

            # pool transitions
            if pid in self.active:
                if p_t >= 1.0 - self.eps:                      # mastered -> graduate
                    self.active.discard(pid)
                elif self.stuck[pid] >= self.K:                # stuck -> cooldown
                    self.active.discard(pid)
                    self.cooldown[pid] = self.cooldown_T
                elif self.visit[pid] >= self.M and self.dema[pid] <= self.theta:
                    self.active.discard(pid)                   # seen enough, plateaued
                    self.reserve.insert(0, pid)
