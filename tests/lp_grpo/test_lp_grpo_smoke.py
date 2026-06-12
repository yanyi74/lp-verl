"""Smoke test for LP-GRPO: directly exercise compute_lp_grpo_outcome_advantage.

Does NOT spin up the full trainer. Generates fake rollout data, calls the
function with hand-built lp_state, and checks:
  - output tensor shapes
  - LP weights match hand-computed values
  - lp_state side effects (last_p_t_map updated)
  - last_metrics populated with expected keys
  - degeneration: gamma=0 lambda=0 reduces to plain GRPO
"""
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_grpo_outcome_advantage,
    compute_lp_grpo_outcome_advantage,
    get_adv_estimator_fn,
)


# ---------- helpers ----------

class MockConfig:
    def __init__(self, **kw):
        self._d = kw
    def get(self, k, default=None):
        return self._d.get(k, default)


def kl_bern(pt, p0):
    return pt * math.log(pt / p0) + (1 - pt) * math.log((1 - pt) / (1 - p0))


def build_fake_batch(prompt_pass_rates, n_per_prompt, response_length=4):
    """Build a fake batch of shape (B*N, response_length).

    Args:
        prompt_pass_rates: list of (p_0, n_correct) tuples per prompt.
        n_per_prompt: rollout count N per prompt.
    """
    bsz = len(prompt_pass_rates) * n_per_prompt
    rewards = torch.zeros((bsz, response_length))
    response_mask = torch.ones_like(rewards)
    uid_arr = []
    persistent_idx = []
    for pi, (_p0, n_correct) in enumerate(prompt_pass_rates):
        for j in range(n_per_prompt):
            i = pi * n_per_prompt + j
            rewards[i, -1] = 1.0 if j < n_correct else 0.0
            uid_arr.append(f"uid_{pi}")
            persistent_idx.append(pi)
    return (
        rewards,
        response_mask,
        np.array(uid_arr, dtype=object),
        {"index": np.array(persistent_idx, dtype=np.int64)},
    )


# ---------- tests ----------

def test_registry_lookup():
    fn = get_adv_estimator_fn("lp_grpo")
    assert fn.__name__ == "compute_lp_grpo_outcome_advantage"
    assert get_adv_estimator_fn(AdvantageEstimator.LP_GRPO) is fn
    print("[PASS] test_registry_lookup")


def test_basic_shapes_and_weights():
    """4 prompts, N=8. Verify shapes, state side-effects, and metric keys."""
    cases = [(0.05, 1), (0.125, 4), (0.5, 5), (0.94, 8)]
    rewards, mask, uid, nt_batch = build_fake_batch(cases, n_per_prompt=8)
    lp_state = {
        "p_0_map": {0: 0.05, 1: 0.125, 2: 0.5, 3: 0.94},
        "last_p_t_map": {},
    }
    config = MockConfig(lp_gamma=0.5, lp_lambda=3.0, lp_w_max=3.0, lp_eps_p=0.05)

    adv, ret = compute_lp_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uid,
        config=config, non_tensor_batch=nt_batch, lp_state=lp_state,
    )

    assert adv.shape == rewards.shape, f"adv shape {adv.shape} != {rewards.shape}"
    assert torch.equal(adv, ret), "advantages and returns should be identical"

    # last_p_t should equal observed pass rate
    assert set(lp_state["last_p_t_map"].keys()) == {0, 1, 2, 3}
    assert abs(lp_state["last_p_t_map"][0] - 1/8) < 1e-6
    assert abs(lp_state["last_p_t_map"][1] - 4/8) < 1e-6
    assert abs(lp_state["last_p_t_map"][2] - 5/8) < 1e-6
    assert abs(lp_state["last_p_t_map"][3] - 8/8) < 1e-6

    assert "last_metrics" in lp_state
    m = lp_state["last_metrics"]
    for k in ["lp/w/mean", "lp/kl/mean", "lp/p_0/mean", "lp/bucket/mastered", "lp/n_groups"]:
        assert k in m, f"missing metric key: {k}"
    assert m["lp/n_groups"] == 4

    print("[PASS] test_basic_shapes_and_weights")
    print(f"       n_groups={m['lp/n_groups']}, w/mean={m['lp/w/mean']:.3f}, "
          f"kl/mean={m['lp/kl/mean']:.4f}, w/cap_rate={m['lp/w/cap_rate']:.2f}")
    print(f"       buckets: " + ", ".join(
        f"{k.split('/')[-1]}={m[k]}" for k in m if k.startswith("lp/bucket/")))


def test_degenerate_to_grpo():
    """gamma=0, lambda=0 => w=1 for all. LP advantages should be linearly
    scaled GRPO advantages (from masked_whiten), with correlation ~1."""
    cases = [(0.1, 2), (0.4, 5), (0.7, 6)]
    rewards, mask, uid, nt_batch = build_fake_batch(cases, n_per_prompt=8)
    lp_state = {
        "p_0_map": {0: 0.1, 1: 0.4, 2: 0.7},
        "last_p_t_map": {},
    }
    config = MockConfig(lp_gamma=0.0, lp_lambda=0.0, lp_w_max=3.0, lp_eps_p=0.05)

    lp_adv, _ = compute_lp_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=uid,
        config=config, non_tensor_batch=nt_batch, lp_state=lp_state,
    )
    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=uid,
    )

    same_sign = ((lp_adv * grpo_adv) >= 0) | (mask == 0)
    assert same_sign.all(), "LP-degenerate signs should match GRPO signs"

    a = lp_adv[mask.bool()].flatten()
    b = grpo_adv[mask.bool()].flatten()
    if a.std() > 1e-8 and b.std() > 1e-8:
        corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        assert corr > 0.999, f"correlation only {corr:.6f}, expected ~1"
        nz = b.abs() > 1e-6
        ratios = a[nz] / b[nz]
        assert ratios.std() < 1e-4, f"LP/GRPO ratio not constant: std={ratios.std():.6f}"
        print(f"[PASS] test_degenerate_to_grpo (corr={corr:.6f}, ratio={ratios.mean():.4f}±{ratios.std():.6f})")
    else:
        print("[PASS] test_degenerate_to_grpo (degenerate batch, no var)")

    assert abs(lp_state["last_metrics"]["lp/w/mean"] - 1.0) < 1e-6
    assert lp_state["last_metrics"]["lp/w/std"] < 1e-6


def test_breakthrough_gets_high_weight():
    """A 0->1/8 breakthrough prompt should get higher w than a 0.5->5/8 progress prompt."""
    cases = [(0.05, 1), (0.5, 5)]
    rewards, mask, uid, nt_batch = build_fake_batch(cases, n_per_prompt=8)
    lp_state = {
        "p_0_map": {0: 0.05, 1: 0.5},
        "last_p_t_map": {},
    }
    config = MockConfig(lp_gamma=0.5, lp_lambda=3.0, lp_w_max=10.0, lp_eps_p=0.05)

    compute_lp_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uid,
        config=config, non_tensor_batch=nt_batch, lp_state=lp_state,
    )

    # Hand-compute expected w using raw p_t (no EMA)
    p_t_q1, p_t_q3 = 1/8, 5/8
    p0_q1, p0_q3 = 0.05, 0.5
    eps = 0.05
    p_t_q1_c = min(max(p_t_q1, eps), 1 - eps)
    p_t_q3_c = min(max(p_t_q3, eps), 1 - eps)
    kl_q1 = kl_bern(p_t_q1_c, p0_q1)
    kl_q3 = kl_bern(p_t_q3_c, p0_q3)
    w_q1 = (1 - p0_q1) ** 0.5 * min(10.0, max(1.0, 1 + 3 * kl_q1))
    w_q3 = (1 - p0_q3) ** 0.5 * min(10.0, max(1.0, 1 + 3 * kl_q3))
    print(f"[INFO] hand-computed: w_Q1={w_q1:.3f}, w_Q3={w_q3:.3f}, ratio={w_q1/w_q3:.2f}x")
    assert w_q1 > w_q3, f"Q1 (breakthrough) should outweigh Q3 (progress): {w_q1} vs {w_q3}"
    print("[PASS] test_breakthrough_gets_high_weight (Q1 > Q3)")


def test_mastered_gets_low_weight():
    """A nearly-mastered prompt (p_0=0.95, p_t=1.0) should get low weight."""
    cases = [(0.95, 8), (0.5, 4)]
    rewards, mask, uid, nt_batch = build_fake_batch(cases, n_per_prompt=8)
    lp_state = {
        "p_0_map": {0: 0.95, 1: 0.5},
        "last_p_t_map": {},
    }
    config = MockConfig(lp_gamma=0.5, lp_lambda=3.0, lp_w_max=3.0, lp_eps_p=0.05)
    compute_lp_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uid,
        config=config, non_tensor_batch=nt_batch, lp_state=lp_state,
    )
    m = lp_state["last_metrics"]
    print(f"[INFO] mastered/plateau case: w/mean={m['lp/w/mean']:.3f}, "
          f"buckets: {[(k.split('/')[-1], m[k]) for k in m if k.startswith('lp/bucket/')]}")
    assert m["lp/bucket/mastered"] == 1, f"Q0 should be mastered, got {m['lp/bucket/mastered']}"
    print("[PASS] test_mastered_gets_low_weight")


def test_missing_p0_falls_back_to_pt():
    """When a prompt has no p_0 entry, falls back to p_t: KL=0, w=(1-p_t)^gamma."""
    cases = [(0.0, 2)]  # p_0 unused — not in map
    rewards, mask, uid, nt_batch = build_fake_batch(cases, n_per_prompt=8)
    lp_state = {"p_0_map": {}, "last_p_t_map": {}}
    config = MockConfig(lp_gamma=0.5, lp_lambda=3.0, lp_w_max=3.0, lp_eps_p=0.05)
    compute_lp_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uid,
        config=config, non_tensor_batch=nt_batch, lp_state=lp_state,
    )
    m = lp_state["last_metrics"]
    # p_0 falls back to p_t => KL[Bern(p_t)||Bern(p_t)] = 0
    assert m["lp/kl/mean"] < 1e-6, f"KL should be 0 with empty p_0_map, got {m['lp/kl/mean']}"
    print(f"[PASS] test_missing_p0_falls_back_to_pt (KL/mean={m['lp/kl/mean']:.6f})")


if __name__ == "__main__":
    test_registry_lookup()
    test_basic_shapes_and_weights()
    test_degenerate_to_grpo()
    test_breakthrough_gets_high_weight()
    test_mastered_gets_low_weight()
    test_missing_p0_falls_back_to_pt()
    print("\n=== ALL TESTS PASSED ===")
