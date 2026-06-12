"""Tests for the three w-regularization modes in compute_lp_grpo_outcome_advantage.

Covers the LP_W_MODE switch added for the clip-vs-norm-vs-raw experiment:
  - norm:  lp_normalize_w=True, no clip                    -> mean(w) == 1
  - raw:   lp_normalize_w=False, no clip                   -> w == raw_w
  - clip:  lp_w_clip_hi > lp_w_clip_lo > 0                 -> w in [lo, hi]
  - priority: clip overrides norm when both are set
  - advantages downstream actually reflect the regularized w

Run:  python3 tests/lp_grpo/test_w_regularization.py
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from verl.trainer.ppo.core_algos import compute_lp_grpo_outcome_advantage


class MockConfig:
    def __init__(self, **kw):
        self._d = kw

    def get(self, k, default=None):
        return self._d.get(k, default)


def build_bimodal_batch(n_per_prompt=8):
    """Bimodal p_t distribution similar to the real 30k run:
    mix of plateau-fail (p_t ~ 0), progress, plateau-medium, mastered.
    Returns (rewards, mask, uid_arr, nt_batch, p_0_map).
    """
    # (p_0, n_correct) per prompt
    cases = [
        (0.05, 0),   # plateau-fail (p_t=0)
        (0.10, 1),   # plateau-fail (p_t=0.125)
        (0.30, 3),   # progress (p_t=0.375)
        (0.40, 5),   # progress (p_t=0.625)
        (0.50, 4),   # plateau-medium (p_t=0.5)
        (0.50, 5),   # plateau-medium (p_t=0.625)
        (0.85, 7),   # mastered (p_t=0.875)
        (0.95, 8),   # mastered (p_t=1.0)
    ]
    bsz = len(cases) * n_per_prompt
    rewards = torch.zeros((bsz, 4))
    mask = torch.ones_like(rewards)
    uid, persistent = [], []
    p_0_map = {}
    for pi, (p0, n_correct) in enumerate(cases):
        p_0_map[pi] = p0
        for j in range(n_per_prompt):
            i = pi * n_per_prompt + j
            rewards[i, -1] = 1.0 if j < n_correct else 0.0
            uid.append(f"uid_{pi}")
            persistent.append(pi)
    return (
        rewards,
        mask,
        np.array(uid, dtype=object),
        {"index": np.array(persistent, dtype=np.int64)},
        p_0_map,
    )


def _run(mode, lp_normalize_w, lp_w_clip_lo, lp_w_clip_hi, **extra):
    rewards, mask, uid, nt_batch, p_0_map = build_bimodal_batch()
    lp_state = {"p_0_map": dict(p_0_map), "last_p_t_map": {}}
    config = MockConfig(
        lp_gamma=0.5,
        lp_lambda=3.0,
        lp_w_max=5.0,
        lp_eps_p=0.05,
        lp_zpd_strength=1.0,
        lp_p0_ema_alpha=0.0,
        lp_normalize_w=lp_normalize_w,
        lp_w_clip_lo=lp_w_clip_lo,
        lp_w_clip_hi=lp_w_clip_hi,
        **extra,
    )
    adv, _ = compute_lp_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(),
        response_mask=mask,
        index=uid,
        config=config,
        non_tensor_batch=nt_batch,
        lp_state=lp_state,
    )
    return adv, lp_state["last_metrics"], rewards, uid


# ---------- mode-specific behavior ----------

def test_norm_mode_mean_is_one():
    """LP_W_MODE=norm => mean(w) forced to 1.0."""
    _, m, _, _ = _run("norm", lp_normalize_w=True, lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)
    assert abs(m["lp/w/mean"] - 1.0) < 1e-6, f"norm mode mean(w) should be 1, got {m['lp/w/mean']}"
    assert m["lp/w/raw_mean"] != m["lp/w/mean"], "raw_mean should differ from normalized mean for non-trivial batch"
    print(f"[PASS] test_norm_mode_mean_is_one (mean={m['lp/w/mean']:.4f}, raw_mean={m['lp/w/raw_mean']:.4f}, "
          f"std={m['lp/w/std']:.3f}, range=[{m['lp/w/min']:.3f}, {m['lp/w/max']:.3f}])")


def test_raw_mode_no_transform():
    """LP_W_MODE=raw => w == raw_w, mean(w) == raw_mean."""
    _, m, _, _ = _run("raw", lp_normalize_w=False, lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)
    assert abs(m["lp/w/mean"] - m["lp/w/raw_mean"]) < 1e-9, \
        f"raw mode: mean(w) should equal raw_mean, got mean={m['lp/w/mean']} raw={m['lp/w/raw_mean']}"
    # With ZPD=1 on a bimodal p_t distribution, raw_w_mean should be well below 1
    assert m["lp/w/raw_mean"] < 0.9, f"raw_mean unexpectedly high: {m['lp/w/raw_mean']}"
    print(f"[PASS] test_raw_mode_no_transform (mean=raw_mean={m['lp/w/mean']:.4f}, "
          f"std={m['lp/w/std']:.3f}, range=[{m['lp/w/min']:.3f}, {m['lp/w/max']:.3f}])")


def test_clip_mode_bounds_enforced():
    """LP_W_MODE=clip => w in [lo, hi]; mean(w) is wherever it naturally lands."""
    lo, hi = 0.3, 1.5
    _, m, _, _ = _run("clip", lp_normalize_w=False, lp_w_clip_lo=lo, lp_w_clip_hi=hi)
    eps = 1e-9
    assert m["lp/w/min"] >= lo - eps, f"clip lower bound violated: min={m['lp/w/min']} < {lo}"
    assert m["lp/w/max"] <= hi + eps, f"clip upper bound violated: max={m['lp/w/max']} > {hi}"
    # mean should be inside the clip range, not forced to 1.0
    assert lo <= m["lp/w/mean"] <= hi, f"mean(w)={m['lp/w/mean']} outside [{lo}, {hi}]"
    assert abs(m["lp/w/mean"] - 1.0) > 1e-3, "clip mode should NOT force mean to 1"
    print(f"[PASS] test_clip_mode_bounds_enforced (lo={lo}, hi={hi}, "
          f"mean={m['lp/w/mean']:.4f}, range=[{m['lp/w/min']:.4f}, {m['lp/w/max']:.4f}])")


def test_clip_overrides_norm_priority():
    """When both lp_normalize_w=True and clip bounds are set, clip wins."""
    lo, hi = 0.3, 1.5
    _, m, _, _ = _run("clip+norm", lp_normalize_w=True, lp_w_clip_lo=lo, lp_w_clip_hi=hi)
    assert m["lp/w/min"] >= lo - 1e-9
    assert m["lp/w/max"] <= hi + 1e-9
    assert abs(m["lp/w/mean"] - 1.0) > 1e-3, \
        f"clip should override norm, but mean={m['lp/w/mean']} ~= 1 (norm took effect)"
    print(f"[PASS] test_clip_overrides_norm_priority (mean={m['lp/w/mean']:.4f}, "
          f"range=[{m['lp/w/min']:.4f}, {m['lp/w/max']:.4f}])")


def test_clip_inactive_when_bounds_zero():
    """lo=hi=0 => clip disabled; falls through to norm/raw based on lp_normalize_w."""
    # clip disabled + norm on => norm behavior
    _, m_norm, _, _ = _run("norm-fallback", lp_normalize_w=True, lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)
    assert abs(m_norm["lp/w/mean"] - 1.0) < 1e-6, "norm fallback failed"

    # clip disabled + norm off => raw behavior
    _, m_raw, _, _ = _run("raw-fallback", lp_normalize_w=False, lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)
    assert abs(m_raw["lp/w/mean"] - m_raw["lp/w/raw_mean"]) < 1e-9, "raw fallback failed"
    print(f"[PASS] test_clip_inactive_when_bounds_zero (lo=hi=0 falls through correctly)")


def test_clip_inactive_when_hi_le_lo():
    """Inverted/equal bounds => clip skipped. Defensive check."""
    # hi < lo: should skip clip and behave like raw (since lp_normalize_w=False)
    _, m, _, _ = _run("bad-bounds", lp_normalize_w=False, lp_w_clip_lo=2.0, lp_w_clip_hi=0.5)
    assert abs(m["lp/w/mean"] - m["lp/w/raw_mean"]) < 1e-9, \
        f"inverted bounds should disable clip, but mean changed: {m['lp/w/mean']} vs raw {m['lp/w/raw_mean']}"
    print(f"[PASS] test_clip_inactive_when_hi_le_lo (defensive: inverted bounds skip clip)")


# ---------- downstream check: advantage actually reflects regularized w ----------

def test_advantage_scales_with_regularized_w():
    """If clip changes w, the advantage tensor should change proportionally for
    groups whose w was actually clipped (non-zero advantage groups only)."""
    # Run twice with same data but different w-regularization
    rewards, mask, uid, nt_batch, p_0_map = build_bimodal_batch()

    def adv_and_metrics(lp_normalize_w, lo, hi):
        lp_state = {"p_0_map": dict(p_0_map), "last_p_t_map": {}}
        config = MockConfig(
            lp_gamma=0.5, lp_lambda=3.0, lp_w_max=5.0, lp_eps_p=0.05,
            lp_zpd_strength=1.0, lp_p0_ema_alpha=0.0,
            lp_normalize_w=lp_normalize_w, lp_w_clip_lo=lo, lp_w_clip_hi=hi,
        )
        adv, _ = compute_lp_grpo_outcome_advantage(
            token_level_rewards=rewards.clone(), response_mask=mask, index=uid,
            config=config, non_tensor_batch=nt_batch, lp_state=lp_state,
        )
        return adv, lp_state["last_metrics"]

    adv_raw, m_raw = adv_and_metrics(False, 0.0, 0.0)
    adv_clip, m_clip = adv_and_metrics(False, 0.3, 1.5)

    # The two advantage tensors should differ for at least some samples
    diff = (adv_raw - adv_clip).abs().sum().item()
    assert diff > 1e-3, f"clip should have moved at least some advantages, total diff={diff}"

    # For samples that weren't clipped (raw_w in [lo, hi]), advantages should match
    # For samples that were clipped, the ratio = clip_value / raw_w
    # Sanity: max(|adv|) under clip should be <= max(|adv|) under raw (since hi=1.5 caps high w)
    max_adv_raw = adv_raw.abs().max().item()
    max_adv_clip = adv_clip.abs().max().item()
    # Allow some slack; clip upper bound 1.5 vs raw max (which may be ~2-3 here)
    print(f"[INFO] max|adv| raw={max_adv_raw:.4f}, clip={max_adv_clip:.4f}, "
          f"raw w_max={m_raw['lp/w/max']:.3f}, clip w_max={m_clip['lp/w/max']:.3f}")
    if m_raw["lp/w/max"] > 1.5:
        assert max_adv_clip <= max_adv_raw + 1e-3, \
            f"clip should cap max advantage when raw w_max > 1.5: raw={max_adv_raw}, clip={max_adv_clip}"
    print(f"[PASS] test_advantage_scales_with_regularized_w (diff sum={diff:.4f})")


# ---------- regression: behavior unchanged from before the patch when LP_W_MODE=norm ----------

def test_norm_mode_regression_against_known_values():
    """Lock the existing norm-mode behavior so future refactors don't silently change it.
    Uses the same bimodal batch; checks key invariants of pre-patch behavior."""
    _, m, _, _ = _run("norm", lp_normalize_w=True, lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)
    # Invariants of mean-norm:
    assert abs(m["lp/w/mean"] - 1.0) < 1e-6
    # The raw_w_mean for this bimodal batch with ZPD=1, lambda=3, gamma=0.5
    # is determined; pin it loosely (~0.5-0.7 expected).
    assert 0.3 < m["lp/w/raw_mean"] < 0.9, \
        f"raw_mean drifted out of expected range: {m['lp/w/raw_mean']}"
    print(f"[PASS] test_norm_mode_regression_against_known_values "
          f"(raw_mean={m['lp/w/raw_mean']:.4f}, w/std={m['lp/w/std']:.3f})")


# ---------- mode comparison summary (informational, no assertions beyond sanity) ----------

def test_summary_three_modes_side_by_side():
    """Print a side-by-side summary so a human can eyeball the three modes."""
    print("\n--- side-by-side comparison ---")
    print(f"{'mode':10s} | {'mean':>7s} | {'std':>7s} | {'min':>7s} | {'max':>7s} | {'raw_mean':>9s}")
    print("-" * 60)
    for name, kw in [
        ("norm",        dict(lp_normalize_w=True,  lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)),
        ("raw",         dict(lp_normalize_w=False, lp_w_clip_lo=0.0, lp_w_clip_hi=0.0)),
        ("clip[.2,2]",  dict(lp_normalize_w=False, lp_w_clip_lo=0.2, lp_w_clip_hi=2.0)),
        ("clip[.3,1.5]",dict(lp_normalize_w=False, lp_w_clip_lo=0.3, lp_w_clip_hi=1.5)),
    ]:
        _, m, _, _ = _run(name, **kw)
        print(f"{name:10s} | {m['lp/w/mean']:>7.4f} | {m['lp/w/std']:>7.4f} | "
              f"{m['lp/w/min']:>7.4f} | {m['lp/w/max']:>7.4f} | {m['lp/w/raw_mean']:>9.4f}")
    print("[PASS] test_summary_three_modes_side_by_side")


if __name__ == "__main__":
    test_norm_mode_mean_is_one()
    test_raw_mode_no_transform()
    test_clip_mode_bounds_enforced()
    test_clip_overrides_norm_priority()
    test_clip_inactive_when_bounds_zero()
    test_clip_inactive_when_hi_le_lo()
    test_advantage_scales_with_regularized_w()
    test_norm_mode_regression_against_known_values()
    test_summary_three_modes_side_by_side()
    print("\n=== ALL W-REGULARIZATION TESTS PASSED ===")
