# Copyright 2025
# LP-GRPO v2: Learning-Progress dense-revisit curriculum sampler (SLIM version).
#
# Design principle: NO fixed pool / NO fixed revisit-ratio / NO cooldown counters.
# Every prompt in the whole dataset competes by a single value score; revisit,
# eviction, and coverage all EMERGE from the score + two interpretable knobs:
#   min_interval : a just-trained prompt must wait this many steps (controls
#                  revisit density AND guarantees the policy changed between
#                  visits so the progress signal carries new info)
#   max_visit    : a prompt trained this many times is retired (anti-overfit)
#
# score(prompt) = (1 + dema+) * sqrt(4 p_t (1-p_t))
#   - sqrt(4 p_t(1-p_t)) = signal gate: ~0 for solved(p_t->1)/unlearnable(p_t->0)
#     -> mastered & stuck prompts auto-fade (no stuck_K / cooldown needed)
#   - (1 + dema+) = progress boost: prompts that are improving get sampled more
#   - never-seen prompts: last_visit=-inf, p_t=p0 -> always eligible + base score
#     -> coverage emerges automatically
#
# dema (smoothed learning progress, double-EMA) is exposed via self.dema_map for
# the advantage estimator (lp_grpo_v2) to read.
"""LPRevisitSampler (slim) — value-scored dense-revisit sampler for LP-GRPO v2."""
import math
import numpy as np
from collections.abc import Sized
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class LPRevisitSampler(AbstractCurriculumSampler):
    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.N = len(data_source)
        c = data_config.get("lp_sampler", {}) or {}
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        # --- the only two mechanism knobs ---
        self.min_interval = int(c.get("min_interval", 30))  # steps a prompt must wait before re-visit
        self.M = int(c.get("max_visit", 5))                 # hard cap: retire after M visits
        # --- signal-processing knobs (have theory) ---
        self.alpha = float(c.get("alpha", 0.4))             # dema EMA (progress smoothing)
        self.beta = float(c.get("beta", 0.2))               # base EMA (slow baseline)
        self.theta = float(c.get("theta", 0.05))            # progress deadzone (for metrics)
        self.eps = float(c.get("eps", 0.05))                # ~0/8 and ~8/8 thresholds
        self.new_floor = float(c.get("new_floor", 1.0))     # min score for unseen prompts (coverage)
        self.seed = int(data_config.get("seed", 1) or 1)
        self.rng = np.random.default_rng(self.seed)

        # per-prompt persistent state (index pid in [0, N))
        self.base = np.full(self.N, np.nan, dtype=np.float64)       # slow EMA baseline; nan=unseen
        self.dema = np.zeros(self.N, dtype=np.float64)              # smoothed progress
        self.visit = np.zeros(self.N, dtype=np.int32)              # total times trained
        self.last_visit = np.full(self.N, -10**9, dtype=np.int64)  # step of last visit; -inf=never
        self.p0 = np.full(self.N, 0.5, dtype=np.float64)           # offline prior (filled if available)

        self._load_p_zero()
        self._step = 0
        # SHARED with advantage estimator
        self.dema_map = {}
        self.metrics = {}

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

    # ---- value score for a prompt (scalar; used by get_metrics) ----
    def _score(self, pid):
        if self.visit[pid] >= self.M:
            return 0.0
        if self._step - self.last_visit[pid] < self.min_interval:
            return 0.0
        p_t = self.base[pid] if not math.isnan(self.base[pid]) else self.p0[pid]
        p_t = min(max(p_t, 0.0), 1.0)
        signal_gate = math.sqrt(max(4.0 * p_t * (1.0 - p_t), 0.0))
        prog_boost = 1.0 + max(0.0, self.dema[pid])
        return prog_boost * signal_gate + 1e-3

    # ---- vectorized score over all prompts (used every batch in __iter__) ----
    def _scores_all(self):
        p = np.where(np.isnan(self.base), self.p0, self.base)
        p = np.clip(p, 0.0, 1.0)
        gate = np.sqrt(np.maximum(4.0 * p * (1.0 - p), 0.0))
        sc = (1.0 + np.maximum(0.0, self.dema)) * gate + 1e-3
        # NEW: unseen prompts get a guaranteed base score so coverage is not
        # starved by high-scoring revisited prompts (fixes "new prompts never
        # get in"). new-prompt floor >= typical revisit score keeps exploration.
        unseen = self.visit == 0
        sc[unseen] = np.maximum(sc[unseen], self.new_floor)
        # retired: trained M times -> never sampled again (hard, permanent)
        sc[self.visit >= self.M] = 0.0
        # cooling: just-trained prompt waits min_interval steps
        sc[self._step - self.last_visit < self.min_interval] = 0.0
        return sc

    def __len__(self):
        return self.N

    def __iter__(self):
        # Yield N indices per epoch, re-selected from the WHOLE dataset by value
        # score. The DataLoader may pull several batches before update() runs, so
        # self.visit can lag. We keep a provisional pick-count within this pass and
        # fold it in, so a prompt is never picked more than (M - already_visited)
        # times here -> hard cap M is respected even under prefetch.
        n_yield = 0
        prov = np.zeros(self.N, dtype=np.int32)  # provisional picks this pass
        while n_yield < self.N:
            scores = self._scores_all()
            # respect hard cap including provisional picks not yet in self.visit
            scores[self.visit + prov >= self.M] = 0.0
            elig = scores > 0
            n_elig = int(elig.sum())
            take = min(self.batch_size, self.N - n_yield)
            if n_elig >= take:
                p = scores / scores.sum()
                chosen = self.rng.choice(self.N, size=take, replace=False, p=p)
            else:
                head = list(np.where(elig)[0])
                rest = list(np.where(~elig)[0])
                self.rng.shuffle(rest)
                chosen = np.array((head + rest)[:take])
            for pid in chosen:
                pid = int(pid)
                prov[pid] += 1
                yield pid
                n_yield += 1
                if n_yield >= self.N:
                    break

    # ---- state update (called by trainer after each step) ----
    def update(self, batch: DataProto) -> None:
        self._step += 1
        if "index" not in batch.non_tensor_batch:
            return
        idx_arr = batch.non_tensor_batch["index"]
        scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()

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
            self.last_visit[pid] = self._step
            self.dema_map[pid] = float(self.dema[pid])

    def get_metrics(self):
        """Bias-free global stats over all visited prompts."""
        seen = self.visit > 0
        nseen = int(seen.sum())
        if nseen == 0:
            return {}
        dema_seen = self.dema[seen]
        return {
            "sampler/n_seen": nseen,
            "sampler/coverage_frac": nseen / self.N,
            "sampler/n_retired": int((self.visit >= self.M).sum()),
            "sampler/dema_global_mean": float(dema_seen.mean()),
            "sampler/dema_global_pos_rate": float((dema_seen > self.theta).mean()),
            "sampler/dema_global_neg_rate": float((dema_seen < -self.theta).mean()),
            "sampler/visit_mean_seen": float(self.visit[seen].mean()),
            "sampler/visit_max": int(self.visit.max()),
            "sampler/frac_at_cap": float((self.visit[seen] >= self.M).mean()),
            "sampler/n_eligible_now": int((self._scores_all() > 0).sum()),
        }
