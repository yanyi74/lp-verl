# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"
    LP_GRPO = "lp_grpo"
    LP_GRPO_ASYNC = "lp_grpo_async"
    LP_GRPO_V11 = "lp_grpo_v11"
    LP_GRPO_V2 = "lp_grpo_v2"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    lp_state: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object
        non_tensor_batch: `(Optional[dict])`
            non-tensor batch data; if it contains "index" (persistent prompt id)
            and lp_state has p_0_map, LP bucket metrics are logged (gradients unaffected).
        lp_state: `(Optional[dict])`
            mutable LP state dict; if provided alongside non_tensor_batch["index"]
            and p_0_map, LP monitoring metrics are written to lp_state["last_metrics"].

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    # LP monitoring: compute bucket stats for analysis without touching gradients.
    # Only runs when the dataset provides p_zero (persistent p_0 per prompt).
    _grpo_lp_monitoring(
        index=index,
        id2mean=id2mean,
        id2std=id2std,
        non_tensor_batch=non_tensor_batch,
        lp_state=lp_state,
        config=config,
        epsilon=epsilon,
    )

    return scores, scores


def _grpo_lp_monitoring(
    index: np.ndarray,
    id2mean: dict,
    id2std: Optional[dict],
    non_tensor_batch: Optional[dict],
    lp_state: Optional[dict],
    config: Optional[Any],
    epsilon: float = 1e-6,
) -> None:
    """Compute LP bucket metrics for monitoring without modifying advantages.

    Writes results to lp_state["last_metrics"] so the trainer can log them
    alongside LP-GRPO metrics for direct comparison. Mirrors the same
    post-processing pipeline (clip / useful-mean-norm) that LP-GRPO uses, so
    baseline runs produce directly comparable lp/w/* metrics.
    """
    if (
        lp_state is None
        or not lp_state.get("p_0_map")
        or non_tensor_batch is None
        or "index" not in non_tensor_batch
    ):
        return

    lp_eps_p = config.get("lp_eps_p", 0.05) if config is not None else 0.05
    lp_gamma = config.get("lp_gamma", 0.5) if config is not None else 0.5
    lp_lambda = config.get("lp_lambda", 3.0) if config is not None else 3.0
    lp_zpd_strength = config.get("lp_zpd_strength", 1.0) if config is not None else 1.0
    lp_w_clip_lo = config.get("lp_w_clip_lo", 0.0) if config is not None else 0.0
    lp_w_clip_hi = config.get("lp_w_clip_hi", 0.0) if config is not None else 0.0
    lp_normalize_w = config.get("lp_normalize_w", False) if config is not None else False

    p_0_map = lp_state["p_0_map"]
    last_p_t_map = lp_state.setdefault("last_p_t_map", {})
    persistent_idx_arr = non_tensor_batch["index"]

    uid_to_persistent = {}
    for i, uid in enumerate(index):
        if uid not in uid_to_persistent:
            uid_to_persistent[uid] = persistent_idx_arr[i]

    bucket_counts = {k: 0 for k in ("breakthrough", "progress", "plateau", "mastered", "regressing")}
    log_w, log_kl, log_fdiff, log_fprog, log_p0, log_pt = [], [], [], [], [], []
    plateau_pt = []
    # Track per-uid raw w + whether the group is "useful" (σ_g > eps) so we can
    # apply the same post-processing the LP-GRPO branch does.
    uid_to_w: dict = {}
    uid_useful: dict = {}

    for uid, group_mean in id2mean.items():
        persistent_idx = uid_to_persistent.get(uid)
        if persistent_idx is None or persistent_idx not in p_0_map:
            continue
        p_t = float(group_mean.item())
        p_0 = float(p_0_map[persistent_idx])
        last_p_t_map[persistent_idx] = p_t

        p_0_c = min(max(p_0, lp_eps_p), 1.0 - lp_eps_p)
        p_t_c = min(max(p_t, lp_eps_p), 1.0 - lp_eps_p)
        delta_abs = abs(p_t - p_0)
        kl = float(p_t_c * np.log(p_t_c / p_0_c) + (1.0 - p_t_c) * np.log((1.0 - p_t_c) / (1.0 - p_0_c)))

        if p_t > p_0:
            f_diff = (1.0 - p_0_c) ** lp_gamma
            f_prog = 1.0 + lp_lambda * delta_abs
        else:
            f_diff = (1.0 - p_t_c) ** lp_gamma
            f_prog = 1.0
        zpd = (4.0 * p_t_c * (1.0 - p_t_c)) ** lp_zpd_strength if lp_zpd_strength > 0 else 1.0
        w = f_diff * f_prog * zpd
        uid_to_w[uid] = w
        if id2std is not None and uid in id2std:
            uid_useful[uid] = float(id2std[uid].item()) > epsilon
        else:
            uid_useful[uid] = True  # if std unknown, assume useful

        bucket = _classify_lp_bucket(p_0, p_t)
        bucket_counts[bucket] += 1
        if bucket == "plateau":
            plateau_pt.append(p_t)
        log_w.append(w); log_kl.append(kl); log_fdiff.append(f_diff)
        log_fprog.append(f_prog); log_p0.append(p_0); log_pt.append(p_t)

    if not log_w:
        return

    w_arr_raw = np.asarray(log_w, dtype=np.float64)
    kl_arr = np.asarray(log_kl, dtype=np.float64)
    raw_w_mean = float(w_arr_raw.mean()) if len(w_arr_raw) > 0 else 1.0
    useful_w_raw = np.asarray([w for uid, w in uid_to_w.items() if uid_useful.get(uid, True)],
                              dtype=np.float64)
    useful_w_mean = float(useful_w_raw.mean()) if len(useful_w_raw) > 0 else raw_w_mean
    n_useful = len(useful_w_raw)

    # Mirror LP-GRPO post-processing so baseline monitoring stats are comparable.
    if lp_w_clip_hi > lp_w_clip_lo > 0:
        w_arr = np.clip(w_arr_raw, lp_w_clip_lo, lp_w_clip_hi)
    elif lp_normalize_w and useful_w_mean > 0:
        w_arr = w_arr_raw / useful_w_mean
    else:
        w_arr = w_arr_raw

    lp_state["last_metrics"] = {
        "lp/w/mean": float(w_arr.mean()), "lp/w/std": float(w_arr.std()),
        "lp/w/min": float(w_arr.min()), "lp/w/max": float(w_arr.max()),
        "lp/w/raw_mean": raw_w_mean,
        "lp/w/useful_mean": useful_w_mean,
        "lp/w/n_useful": n_useful,
        "lp/w/cap_rate": 0.0,
        "lp/kl/mean": float(kl_arr.mean()), "lp/kl/p50": float(np.median(kl_arr)),
        "lp/kl/p95": float(np.quantile(kl_arr, 0.95)),
        "lp/f_diff/mean": float(np.mean(log_fdiff)),
        "lp/f_prog/mean": float(np.mean(log_fprog)),
        "lp/p_0/mean": float(np.mean(log_p0)),
        "lp/p_t/mean": float(np.mean(log_pt)),
        **{f"lp/bucket/{k}": v for k, v in bucket_counts.items()},
        "lp/n_groups": len(log_w),
        **({"lp/plateau/p_t_mean": float(np.mean(plateau_pt)),
            "lp/plateau/p_t_low_rate": float(np.mean(np.array(plateau_pt) < 0.3))} if plateau_pt else {}),
    }


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.LP_GRPO)  # or simply: @register_adv_est("lp_grpo")
def compute_lp_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    lp_state: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LP-GRPO: GRPO advantage multiplied by per-prompt weight w(p_0, p_t).

    Improving  (p_t > p_0): w = (1-p_0)^gamma * clip(1 + lambda*KL[Bern(p_t)||Bern(p_0)], 1, w_max)
    Regressing (p_t <= p_0): w = (1-p_t)^gamma

    p_0 is the per-prompt pass rate at epoch start; p_t is this step's pass rate
    from N rollouts. Using p_t for f_diff when regressing ensures forgotten prompts
    recover appropriate gradient weight despite a high historical p_0.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: per-step group id (uuid), used to group rollouts of the same prompt.
        epsilon: numerical stability for std normalization.
        norm_adv_by_std_in_grpo: if True, GRPO base; if False, Dr.GRPO base.
        config: AlgoConfig with lp_gamma, lp_lambda, lp_w_max, lp_eps_p.
        non_tensor_batch: must contain "index" (persistent prompt id from dataset).
        lp_state: mutable dict with "p_0_map" (required) and "last_p_t_map"
            (created on demand). Mutated in place.

    Returns:
        (advantages, returns), both shape (bs, response_length).
    """
    if non_tensor_batch is None or "index" not in non_tensor_batch:
        raise ValueError(
            "lp_grpo requires non_tensor_batch['index'] (persistent prompt id). "
            "Make sure compute_advantage forwards non_tensor_batch."
        )
    if lp_state is None or "p_0_map" not in lp_state:
        raise ValueError(
            "lp_grpo requires lp_state with 'p_0_map'. The trainer must initialize "
            "lp_state via a baseline rollout before training starts."
        )

    if config is not None:
        lp_gamma = config.get("lp_gamma", 0.5)
        lp_lambda = config.get("lp_lambda", 3.0)
        lp_w_max = config.get("lp_w_max", 3.0)
        lp_eps_p = config.get("lp_eps_p", 0.05)
        lp_normalize_w = config.get("lp_normalize_w", False)
        lp_w_clip_lo = config.get("lp_w_clip_lo", 0.0)
        lp_w_clip_hi = config.get("lp_w_clip_hi", 0.0)
        lp_zpd_strength = config.get("lp_zpd_strength", 0.0)
        lp_p0_ema_alpha = config.get("lp_p0_ema_alpha", 0.0)
        lp_breakthrough_boost = config.get("lp_breakthrough_boost", 1.0)
        lp_progress_boost = config.get("lp_progress_boost", 1.0)
        lp_regressing_penalty = config.get("lp_regressing_penalty", 1.0)
    else:
        lp_gamma, lp_lambda, lp_w_max, lp_eps_p = 0.5, 3.0, 3.0, 0.05
        lp_normalize_w = False
        lp_w_clip_lo = 0.0
        lp_w_clip_hi = 0.0
        lp_zpd_strength = 0.0
        lp_p0_ema_alpha = 0.0
        lp_breakthrough_boost = 1.0
        lp_progress_boost = 1.0
        lp_regressing_penalty = 1.0

    p_0_map = lp_state["p_0_map"]
    last_p_t_map = lp_state.setdefault("last_p_t_map", {})
    persistent_idx_arr = non_tensor_batch["index"]

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # uid -> persistent prompt index (same across all N rollouts of the group)
        uid_to_persistent = {}
        for i in range(bsz):
            uid = index[i]
            if uid not in uid_to_persistent:
                uid_to_persistent[uid] = persistent_idx_arr[i]

        # Per-group LP weight + bookkeeping for logging
        uid_to_w: dict = {}
        log_w, log_kl, log_fdiff, log_fprog, log_p0, log_pt = [], [], [], [], [], []
        plateau_pt = []
        bucket_counts = {
            "breakthrough": 0,  # p_0<0.2, delta>0.15
            "progress": 0,      # delta>0.05
            "plateau": 0,       # |delta|<=0.05, p_0 in [0.2, 0.8]
            "mastered": 0,      # p_0>0.8 and delta>=0
            "regressing": 0,    # delta<-0.05
        }
        for uid, group_mean in id2mean.items():
            persistent_idx = uid_to_persistent[uid]
            p_t = float(group_mean.item())  # binary reward => group mean = success rate

            # Fallback: first epoch without p_zero column uses p_t as p_0 (KL=0, w=f_diff only).
            p_0 = float(p_0_map.get(persistent_idx, p_t))
            last_p_t_map[persistent_idx] = p_t

            p_0_c = min(max(p_0, lp_eps_p), 1.0 - lp_eps_p)
            p_t_c = min(max(p_t, lp_eps_p), 1.0 - lp_eps_p)

            # KL kept for logging/metrics compatibility only — f_prog uses |Δp|.
            kl = float(
                p_t_c * np.log(p_t_c / p_0_c)
                + (1.0 - p_t_c) * np.log((1.0 - p_t_c) / (1.0 - p_0_c))
            )
            delta_abs = abs(p_t - p_0)

            # Progress signal: linear in |Δp|, naturally bounded in [1, 1+λ].
            # Replaces clip(1+λ·KL, 1, w_max) — |Δp| is numerically stable, doesn't
            # need eps clipping or w_max cap, and is robust to N=8 binomial noise.
            if p_t > p_0:
                # Improving: weight by baseline difficulty, amplify by |Δp|.
                f_diff = (1.0 - p_0_c) ** lp_gamma
                f_prog = 1.0 + lp_lambda * delta_abs
            else:
                # Regressing/plateau: weight by current difficulty only; no amplification.
                f_diff = (1.0 - p_t_c) ** lp_gamma
                f_prog = 1.0

            # ZPD (two-sided): w *= (4·p_t·(1-p_t))^β. Acts as GRPO-signal detector
            # and signal-aware resource allocator — mastered (p_t→1), plateau-easy
            # and plateau-hard (p_t in {≈0, ≈1}) get compressed; plateau-medium
            # (p_t≈0.5) retains full weight. lp_zpd_strength=0 disables.
            if lp_zpd_strength > 0:
                zpd_factor = (4.0 * p_t_c * (1.0 - p_t_c)) ** lp_zpd_strength
            else:
                zpd_factor = 1.0

            # Bucket-aware boost/penalty: directly steer each learning state.
            # Classify with shared helper so adv-reweight and adaptive-N agree.
            bucket = _classify_lp_bucket(p_0, p_t)
            if bucket == "breakthrough":
                bucket_factor = lp_breakthrough_boost
            elif bucket == "progress":
                bucket_factor = lp_progress_boost
            elif bucket == "regressing":
                bucket_factor = lp_regressing_penalty
            else:  # plateau / mastered: no extra factor (mastered already crushed by ZPD)
                bucket_factor = 1.0

            w = f_diff * f_prog * zpd_factor * bucket_factor
            uid_to_w[uid] = w

            # Per-step EMA update on p_0 (replaces epoch-end整批覆盖). alpha=0 freezes
            # p_0 entirely; alpha=1 reduces to the old "overwrite each occurrence" behavior.
            # We update AFTER computing w so the current step uses the old p_0.
            if lp_p0_ema_alpha > 0:
                new_p_0 = (1.0 - lp_p0_ema_alpha) * p_0 + lp_p0_ema_alpha * p_t
                p_0_map[persistent_idx] = min(max(new_p_0, lp_eps_p), 1.0 - lp_eps_p)

            log_w.append(w); log_kl.append(kl); log_fdiff.append(f_diff)
            log_fprog.append(f_prog); log_p0.append(p_0); log_pt.append(p_t)
            bucket_counts[bucket] += 1
            if bucket == "plateau":
                plateau_pt.append(p_t)

        # Stash per-step metrics on lp_state for the trainer to consume.
        w_arr_raw = np.asarray(log_w, dtype=np.float64)
        kl_arr = np.asarray(log_kl, dtype=np.float64)

        # w regularization (priority: hard-clip > useful-mean-norm > raw):
        # - hard-clip: bounded w, no drifting denominator.
        # - useful-mean-norm: forces mean(w)=1 over USEFUL groups (σ_g > eps).
        #   Stuck groups (σ_g≈0, i.e. all-correct or all-wrong, A=0 regardless of w)
        #   are excluded from the denominator — they consume zero gradient budget
        #   so they shouldn't dilute the scaling factor. This makes effective LR on
        #   useful samples exactly match nominal LR (= baseline GRPO at same nominal).
        # - raw: use w directly; effective LR scales with raw_w_mean.
        raw_w_mean = float(w_arr_raw.mean()) if len(w_arr_raw) > 0 else 1.0
        # Useful mean: average over groups with non-trivial reward variance.
        useful_uids = [uid for uid, std in id2std.items() if float(std.item()) > epsilon]
        if useful_uids:
            useful_w_raw = np.asarray([uid_to_w[uid] for uid in useful_uids], dtype=np.float64)
            useful_w_mean = float(useful_w_raw.mean()) if len(useful_w_raw) > 0 else raw_w_mean
        else:
            useful_w_mean = raw_w_mean
        if lp_w_clip_hi > lp_w_clip_lo > 0:
            for uid in uid_to_w:
                uid_to_w[uid] = float(np.clip(uid_to_w[uid], lp_w_clip_lo, lp_w_clip_hi))
            w_arr = np.clip(w_arr_raw, lp_w_clip_lo, lp_w_clip_hi)
        elif lp_normalize_w and useful_w_mean > 0:
            scale = 1.0 / useful_w_mean
            for uid in uid_to_w:
                uid_to_w[uid] *= scale
            w_arr = w_arr_raw * scale
        else:
            w_arr = w_arr_raw

        lp_state["last_metrics"] = {
            "lp/w/mean": float(w_arr.mean()), "lp/w/std": float(w_arr.std()),
            "lp/w/min": float(w_arr.min()), "lp/w/max": float(w_arr.max()),
            "lp/w/raw_mean": raw_w_mean,
            "lp/w/useful_mean": useful_w_mean,
            "lp/w/n_useful": len(useful_uids),
            "lp/w/cap_rate": float((np.asarray(log_fprog) >= lp_w_max - 1e-6).mean()),
            "lp/kl/mean": float(kl_arr.mean()), "lp/kl/p50": float(np.median(kl_arr)),
            "lp/kl/p95": float(np.quantile(kl_arr, 0.95)),
            "lp/f_diff/mean": float(np.mean(log_fdiff)),
            "lp/f_prog/mean": float(np.mean(log_fprog)),
            "lp/p_0/mean": float(np.mean(log_p0)),
            "lp/p_t/mean": float(np.mean(log_pt)),
            **{f"lp/bucket/{k}": v for k, v in bucket_counts.items()},
            "lp/n_groups": len(uid_to_w),
            **({"lp/plateau/p_t_mean": float(np.mean(plateau_pt)),
                "lp/plateau/p_t_low_rate": float(np.mean(np.array(plateau_pt) < 0.3))} if plateau_pt else {}),
        }

        for i in range(bsz):
            w_i = uid_to_w[index[i]]
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon) * w_i
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) * w_i
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.LP_GRPO_ASYNC)
def compute_lp_grpo_async_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    lp_state: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LP-GRPO Async (v_async): asymmetric per-rollout weighting.

    Breaks the within-group symmetric invariant of GRPO by applying
    different weights to correct (r=1) vs wrong (r=0) rollouts.

    Formula (0 hyperparameters):
        base_w = 4 * p_t * (1 - p_t)                  # ZPD magnitude
        delta  = clip(p_t - p_0, -0.8, 0.8)           # asymmetric direction
        w+     = base_w * (1 + delta * (1 - p_t))     # for correct rollouts
        w-     = base_w * (1 - delta * p_t)           # for wrong rollouts
        A_i_final = w_i * A_i_GRPO

    Mathematical properties:
      - mean(w_i) within group == base_w (LR-preserving by construction)
      - w+ > 0, w- > 0 for delta in [-0.8, 0.8], p_t in [0,1]
      - group net signal = base_w * delta * N * p_t(1-p_t) / sigma
        This is NEW; both vanilla GRPO and symmetric LP have group net = 0.

    Falls back to vanilla GRPO weights (w=1) when sigma_g < eps_useful
    (degenerate group with no useful signal).
    """
    if non_tensor_batch is None or "index" not in non_tensor_batch:
        raise ValueError(
            "lp_grpo_async requires non_tensor_batch['index'] (persistent prompt id). "
            "Set up dataset same as lp_grpo (p_zero column + extra_info.index)."
        )
    if lp_state is None or "p_0_map" not in lp_state:
        raise ValueError(
            "lp_grpo_async requires lp_state with 'p_0_map'. "
            "Trainer must initialize lp_state same as lp_grpo path."
        )

    p_0_map = lp_state["p_0_map"]
    persistent_idx_arr = non_tensor_batch["index"]
    scores = token_level_rewards.sum(dim=-1)  # (B,)

    id2score = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}
    uid_to_persistent: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            uid = index[i]
            id2score[uid].append(scores[i])
            if uid not in uid_to_persistent:
                uid_to_persistent[uid] = persistent_idx_arr[i]

        for uid in id2score:
            vals = torch.stack(id2score[uid])
            if len(vals) == 1:
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = vals.mean()
                id2std[uid] = vals.std()

        # v_async: compute per-group (w+, w-)
        uid_to_w_plus: dict = {}
        uid_to_w_minus: dict = {}
        log_w_plus, log_w_minus, log_delta, log_net = [], [], [], []
        log_p0, log_pt = [], []
        log_active = 0

        for uid in id2score:
            sigma = float(id2std[uid].item())
            p_t = float(id2mean[uid].item())
            pid = uid_to_persistent[uid]
            # fallback for first-seen prompt: delta = 0 (degenerates to symmetric ZPD)
            p_0 = float(p_0_map.get(pid, p_t))

            base_w = 4.0 * p_t * (1.0 - p_t)
            delta = max(-0.8, min(0.8, p_t - p_0))

            # Degenerate group (all same reward): no useful signal; behave like vanilla GRPO
            if sigma < 1e-3:
                uid_to_w_plus[uid] = 1.0
                uid_to_w_minus[uid] = 1.0
                continue

            wp = base_w * (1.0 + delta * (1.0 - p_t))
            wm = base_w * (1.0 - delta * p_t)
            uid_to_w_plus[uid] = wp
            uid_to_w_minus[uid] = wm

            # Logging
            group_rewards = torch.stack(id2score[uid])
            n_plus = float((group_rewards > 0.5).sum().item())
            n_minus = float(len(group_rewards)) - n_plus
            net_signal = base_w * delta * (n_plus + n_minus) * p_t * (1.0 - p_t) / (sigma + epsilon)
            log_w_plus.append(wp)
            log_w_minus.append(wm)
            log_delta.append(delta)
            log_net.append(net_signal)
            log_p0.append(p_0)
            log_pt.append(p_t)
            if abs(delta) > 0.05:
                log_active += 1

        # Apply per-rollout: select w+ or w- by reward
        for i in range(bsz):
            uid = index[i]
            mu = id2mean[uid]
            sigma_g = id2std[uid]

            if norm_adv_by_std_in_grpo:
                base_adv = (scores[i] - mu) / (sigma_g + epsilon)
            else:
                base_adv = scores[i] - mu

            if scores[i].item() > 0.5:
                w = uid_to_w_plus[uid]
            else:
                w = uid_to_w_minus[uid]
            scores[i] = base_adv * w

        scores = scores.unsqueeze(-1) * response_mask

        # Stash metrics for trainer
        if log_w_plus:
            wp_arr = np.asarray(log_w_plus)
            wm_arr = np.asarray(log_w_minus)
            d_arr = np.asarray(log_delta)
            net_arr = np.asarray(log_net)
            lp_state["last_metrics"] = {
                "lp_async/w_plus/mean": float(wp_arr.mean()),
                "lp_async/w_plus/max": float(wp_arr.max()),
                "lp_async/w_plus/min": float(wp_arr.min()),
                "lp_async/w_minus/mean": float(wm_arr.mean()),
                "lp_async/w_minus/max": float(wm_arr.max()),
                "lp_async/w_minus/min": float(wm_arr.min()),
                "lp_async/delta/mean_abs": float(np.mean(np.abs(d_arr))),
                "lp_async/delta/max_abs": float(np.max(np.abs(d_arr))),
                "lp_async/delta/p50_abs": float(np.median(np.abs(d_arr))),
                "lp_async/net_signal/mean_abs": float(np.mean(np.abs(net_arr))),
                "lp_async/net_signal/max_abs": float(np.max(np.abs(net_arr))),
                "lp_async/active_groups": float(log_active),
                "lp_async/active_ratio": float(log_active) / max(1, len(log_w_plus)),
                "lp_async/n_useful_groups": float(len(log_w_plus)),
                "lp_async/p_0/mean": float(np.mean(log_p0)),
                "lp_async/p_t/mean": float(np.mean(log_pt)),
            }

    return scores, scores


@register_adv_est(AdvantageEstimator.LP_GRPO_V11)
def compute_lp_grpo_v11_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    lp_state: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LP-GRPO v1.1: fused ZPD+difficulty with tanh-bounded progress.

    Formula:
        w = 4 * p_t * (1 - p_t)^(gamma+1) * f_prog(delta)

        delta = p_t - p_0
        if delta >= 0:  f_prog = 1 + alpha * tanh(lam * delta)        (improving)
        if delta < 0:   f_prog = 1 + beta  * tanh(lam * |delta|)      (regressing)

    Config (read from AlgoConfig):
        lp_v11_gamma  (default 0.5)  - ZPD-difficulty shape
        lp_v11_alpha  (default 1.5)  - improving amplification strength
        lp_v11_beta   (default 0.5)  - regressing amplification (set = alpha for symmetric)
        lp_v11_lambda (default 5.0)  - tanh saturation speed

    Behavior:
        beta = alpha (e.g., 1.5): SYMMETRIC, improving and regressing equally amplified
        beta < alpha (e.g., 0.5): MILD ASYMMETRIC, improving stronger but regressing still > 1
        beta = 0:                  EXTREME ASYMMETRIC, regressing only gets f_prog = 1
    """
    if non_tensor_batch is None or "index" not in non_tensor_batch:
        raise ValueError(
            "lp_grpo_v11 requires non_tensor_batch['index'] (persistent prompt id). "
            "Set up dataset same as lp_grpo (p_zero column + extra_info.index)."
        )
    if lp_state is None or "p_0_map" not in lp_state:
        raise ValueError(
            "lp_grpo_v11 requires lp_state with 'p_0_map'. "
            "Trainer must initialize lp_state same as lp_grpo path."
        )

    # Read v1.1 hyperparameters from config
    if config is not None:
        v11_gamma = config.get("lp_v11_gamma", 0.5)
        v11_alpha = config.get("lp_v11_alpha", 1.5)
        v11_beta = config.get("lp_v11_beta", 0.5)
        v11_lambda = config.get("lp_v11_lambda", 5.0)
    else:
        v11_gamma, v11_alpha, v11_beta, v11_lambda = 0.5, 1.5, 0.5, 5.0

    p_0_map = lp_state["p_0_map"]
    persistent_idx_arr = non_tensor_batch["index"]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}
    uid_to_persistent: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            uid = index[i]
            id2score[uid].append(scores[i])
            if uid not in uid_to_persistent:
                uid_to_persistent[uid] = persistent_idx_arr[i]

        for uid in id2score:
            vals = torch.stack(id2score[uid])
            if len(vals) == 1:
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = vals.mean()
                id2std[uid] = vals.std()

        # Per-group v1.1 weight
        uid_to_w: dict = {}
        log_w, log_delta, log_fprog = [], [], []
        log_imp, log_reg, log_plat = 0, 0, 0
        log_p0, log_pt = [], []
        # Bucket counters (aligned with v1's lp/bucket/* for comparable logging)
        bucket_counts = {k: 0 for k in LP_BUCKET_NAMES}
        plateau_pt_vals = []

        for uid in id2score:
            p_t = float(id2mean[uid].item())
            pid = uid_to_persistent[uid]
            p_0 = float(p_0_map.get(pid, p_t))

            # Fused ZPD + difficulty
            fused = 4.0 * p_t * (1.0 - p_t) ** (v11_gamma + 1)

            # Asymmetric f_prog
            delta = p_t - p_0
            if delta >= 0:
                f_prog = 1.0 + v11_alpha * np.tanh(v11_lambda * delta)
                log_imp += 1
            else:
                f_prog = 1.0 + v11_beta * np.tanh(v11_lambda * abs(delta))
                log_reg += 1
            if abs(delta) < 1e-3:
                log_plat += 1

            w = fused * f_prog
            uid_to_w[uid] = w

            log_w.append(w)
            log_delta.append(delta)
            log_fprog.append(f_prog)
            log_p0.append(p_0)
            log_pt.append(p_t)

            # Bucket classification (same as v1)
            bucket = _classify_lp_bucket(p_0, p_t)
            bucket_counts[bucket] += 1
            if bucket == "plateau":
                plateau_pt_vals.append(p_t)

        # Apply per-group w to all rollouts (symmetric per-rollout)
        for i in range(bsz):
            uid = index[i]
            mu = id2mean[uid]
            sigma_g = id2std[uid]
            w = uid_to_w[uid]
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - mu) / (sigma_g + epsilon) * w
            else:
                scores[i] = (scores[i] - mu) * w

        scores = scores.unsqueeze(-1) * response_mask

        # Log metrics
        if log_w:
            w_arr = np.asarray(log_w)
            d_arr = np.asarray(log_delta)
            fp_arr = np.asarray(log_fprog)
            metrics = {
                # v1.1-specific (under lp_v11/ namespace)
                "lp_v11/w/mean": float(w_arr.mean()),
                "lp_v11/w/max": float(w_arr.max()),
                "lp_v11/w/min": float(w_arr.min()),
                "lp_v11/w/std": float(w_arr.std()),
                "lp_v11/delta/mean": float(d_arr.mean()),
                "lp_v11/delta/abs_mean": float(np.abs(d_arr).mean()),
                "lp_v11/f_prog/mean": float(fp_arr.mean()),
                "lp_v11/f_prog/max": float(fp_arr.max()),
                "lp_v11/improving_count": float(log_imp),
                "lp_v11/regressing_count": float(log_reg),
                "lp_v11/plateau_count": float(log_plat),
                "lp_v11/n_groups": float(len(log_w)),
                "lp_v11/config/alpha": float(v11_alpha),
                "lp_v11/config/beta": float(v11_beta),
                "lp_v11/config/asymmetry_ratio": float(v11_alpha / max(v11_beta, 1e-6)),
                # Aligned with v1 (lp_grpo) for cross-version comparability:
                "lp/w/mean": float(w_arr.mean()),
                "lp/w/max": float(w_arr.max()),
                "lp/w/min": float(w_arr.min()),
                "lp/w/std": float(w_arr.std()),
                "lp/w/raw_mean": float(w_arr.mean()),
                "lp/w/cap_rate": 0.0,
                "lp/f_prog/mean": float(fp_arr.mean()),
                "lp/p_0/mean": float(np.mean(log_p0)),
                "lp/p_t/mean": float(np.mean(log_pt)),
                "lp/n_groups": float(len(log_w)),
                **{f"lp/bucket/{k}": v for k, v in bucket_counts.items()},
            }
            if plateau_pt_vals:
                metrics["lp/plateau/p_t_mean"] = float(np.mean(plateau_pt_vals))
                metrics["lp/plateau/p_t_low_rate"] = float(np.mean(np.array(plateau_pt_vals) < 0.3))
            lp_state["last_metrics"] = metrics

    return scores, scores


@register_adv_est(AdvantageEstimator.LP_GRPO_V2)
def compute_lp_grpo_v2_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    lp_state: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LP-GRPO v2: difficulty anchor x continuous asymmetric progress factor.

    A_i = A_i^GRPO * w_x
    w_x = (1 - p0_x)^gamma * f_prog(dema_x)
      f_prog(d) = 1 + alpha * tanh(k*d)      if d >= 0  (improving)
                = 1 + beta  * tanh(k*|d|)     if d <  0  (regressing, beta<alpha)

    Key differences from v1/v11 (data-driven fixes):
      - difficulty anchor uses p0 (offline prior), NOT p_t  -> orthogonal to sigma
      - NO continuous ZPD 4p(1-p) (it is ~sigma^2, redundant, hurt high-p_t prompts)
      - progress signal `dema` is the SMOOTHED double-EMA learning progress
        maintained by the revisit sampler (lp_state["dema_map"]), reliable under
        dense revisit. Falls back to single-step (p_t - p0) if dema unavailable.

    The `dema_map` is updated by LPRevisitSampler.update(); here we only READ it.
    """
    if non_tensor_batch is None or "index" not in non_tensor_batch:
        raise ValueError("lp_grpo_v2 requires non_tensor_batch['index'].")
    if lp_state is None or "p_0_map" not in lp_state:
        raise ValueError("lp_grpo_v2 requires lp_state with 'p_0_map'.")

    if config is not None:
        gamma = config.get("lp_v2_gamma", 0.5)
        alpha = config.get("lp_v2_alpha", 1.0)
        beta = config.get("lp_v2_beta", 0.4)
        k = config.get("lp_v2_k", 8.0)
    else:
        gamma, alpha, beta, k = 0.5, 1.0, 0.4, 8.0

    p_0_map = lp_state["p_0_map"]
    dema_map = lp_state.get("dema_map", {})  # filled by sampler; may be empty early
    persistent_idx_arr = non_tensor_batch["index"]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}
    uid_to_persistent: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            uid = index[i]
            id2score[uid].append(scores[i])
            if uid not in uid_to_persistent:
                uid_to_persistent[uid] = persistent_idx_arr[i]
        for uid in id2score:
            vals = torch.stack(id2score[uid])
            if len(vals) == 1:
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = vals.mean()
                id2std[uid] = vals.std()

        uid_to_w: dict = {}
        log_w, log_dema, log_p0, log_fprog = [], [], [], []
        # bucket counts: two views for cross-version alignment
        #   bucket_pt  : classify by (p_0, p_t) — SAME as v1/v11 lp/bucket/* (comparable)
        #   bucket_dema: classify by (p_0, p_0+dema) — v2's actual smoothed signal
        bucket_pt = {kk: 0 for kk in LP_BUCKET_NAMES}
        bucket_dema = {kk: 0 for kk in LP_BUCKET_NAMES}
        for uid in id2score:
            pid = uid_to_persistent[uid]
            p_t = float(id2mean[uid].item())
            p_0 = float(p_0_map.get(pid, p_t))
            # progress: prefer smoothed dema from sampler; fallback to single-step
            if pid in dema_map:
                d = float(dema_map[pid])
            else:
                d = p_t - p_0
            # difficulty anchor (uses p_0, orthogonal to sigma); clip p0 to [eps,1-eps]
            p0c = min(max(p_0, 1e-3), 1.0 - 1e-3)
            f_diff = (1.0 - p0c) ** gamma
            # continuous asymmetric progress factor (bounded by tanh)
            if d >= 0:
                f_prog = 1.0 + alpha * np.tanh(k * d)
            else:
                f_prog = 1.0 + beta * np.tanh(k * abs(d))
            w = f_diff * f_prog
            uid_to_w[uid] = w
            log_w.append(w); log_dema.append(d); log_p0.append(p_0); log_fprog.append(f_prog)
            # bucket bookkeeping (aligned with v1/v11 + dema-based view)
            bucket_pt[_classify_lp_bucket(p_0, p_t)] += 1
            bucket_dema[_classify_lp_bucket(p_0, min(max(p_0 + d, 0.0), 1.0))] += 1

        for i in range(bsz):
            uid = index[i]
            mu, sig = id2mean[uid], id2std[uid]
            if norm_adv_by_std_in_grpo:
                base_adv = (scores[i] - mu) / (sig + epsilon)
            else:
                base_adv = scores[i] - mu
            scores[i] = base_adv * uid_to_w[uid]
        scores = scores.unsqueeze(-1) * response_mask

        if log_w:
            w_arr = np.asarray(log_w); d_arr = np.asarray(log_dema)
            lp_state["last_metrics"] = {
                "lp_v2/w/mean": float(w_arr.mean()), "lp_v2/w/max": float(w_arr.max()),
                "lp_v2/w/min": float(w_arr.min()), "lp_v2/w/std": float(w_arr.std()),
                "lp_v2/dema/mean": float(d_arr.mean()), "lp_v2/dema/abs_mean": float(np.abs(d_arr).mean()),
                "lp_v2/dema/pos_rate": float((d_arr > 0.05).mean()),
                "lp_v2/dema/neg_rate": float((d_arr < -0.05).mean()),
                "lp_v2/f_prog/mean": float(np.mean(log_fprog)),
                "lp_v2/p0/mean": float(np.mean(log_p0)),
                "lp_v2/n_groups": float(len(log_w)),
                "lp_v2/dema_from_sampler": float(sum(1 for uid in id2score if uid_to_persistent[uid] in dema_map)),
                # bucket counts aligned with v1/v11 lp/bucket/* (classify by p_0,p_t) for cross-version comparison
                **{f"lp/bucket/{kk}": v for kk, v in bucket_pt.items()},
                # v2's own dema-based buckets (classify by p_0, p_0+dema)
                **{f"lp_v2/bucket/{kk}": v for kk, v in bucket_dema.items()},
            }

    return scores, scores


LP_BUCKET_NAMES = ("breakthrough", "progress", "plateau", "mastered", "regressing")


def _classify_lp_bucket(p_0: float, p_t: float) -> str:
    """Shared 5-bucket classification used by advantage reweight and adaptive-N.

    Matches the inline logic in compute_lp_grpo_outcome_advantage so that both
    paths agree on which bucket each prompt belongs to.
    """
    delta = p_t - p_0
    if p_0 > 0.8 and delta >= 0:
        return "mastered"
    if delta < -0.05:
        return "regressing"
    if p_0 < 0.2 and delta > 0.15:
        return "breakthrough"
    if delta > 0.05:
        return "progress"
    return "plateau"


def compute_lp_n_allocation_bucket(
    persistent_indices,
    p_0_map: dict,
    last_p_t_map: dict,
    n_total: int,
    bucket_mult: dict,
    n_min: int = 2,
    n_max: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket adaptive N allocation with A+B hybrid for first-seen prompts.

    Motivation: weighted (w-proportional) allocation gives high-LP prompts more
    rollouts but does NOT specifically help plateau prompts — those are stuck
    because their reward signal is noisy (p_t~0.5 -> binom std ~0.18 at N=8).
    Bucket allocation lets us say "plateau gets 2x N" explicitly, decoupling N
    from the advantage weight.

    Two modes are blended per-prompt to handle the chicken-and-egg issue of
    needing p_t to classify the bucket:
      - A (seen before, has last_p_t): classify into 5 buckets, use bucket_mult.
      - B (first-seen, fallback): variance-driven, weight = plateau_mult * 4 *
        p_0*(1-p_0). This peaks at p_0=0.5 matching plateau, falls to 0 at
        p_0 in {0,1}. Rationale: with no p_t info, high binomial variance
        (~0.5 success rate) is the best proxy for "needs more rollouts".

    Budget is preserved exactly: sum(N_g) == n_total. Per-prompt N in [n_min, n_max].

    Args:
        persistent_indices: array-like length B, persistent prompt ids
        p_0_map / last_p_t_map: lagged baseline / last-seen p_t per pid
        n_total: total rollout budget (e.g., B * rollout.n)
        bucket_mult: dict mapping bucket name -> relative N multiplier
            (e.g., {"plateau": 2.0, "mastered": 0.5, ...})
        n_min: minimum N per prompt (>=2 keeps GRPO std defined)
        n_max: optional per-prompt cap

    Returns:
        n_g: int64 (B,), sum == n_total
        bucket_idx: int64 (B,), index into LP_BUCKET_NAMES; -1 for first-seen (B mode)
    """
    B = len(persistent_indices)
    if B == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    plateau_mult = float(bucket_mult.get("plateau", 2.0))

    bucket_idx = np.full(B, -1, dtype=np.int64)  # -1 = first-seen (B mode)
    mult = np.zeros(B, dtype=np.float64)
    for i, pid in enumerate(persistent_indices):
        p_0 = float(p_0_map.get(pid, 0.5))
        if pid in last_p_t_map:
            # A mode: bucket classification
            p_t = float(last_p_t_map[pid])
            name = _classify_lp_bucket(p_0, p_t)
            mult[i] = float(bucket_mult.get(name, 1.0))
            bucket_idx[i] = LP_BUCKET_NAMES.index(name)
        else:
            # B mode: variance-driven, scaled so peak (p_0=0.5) == plateau_mult
            mult[i] = plateau_mult * 4.0 * p_0 * (1.0 - p_0)
            # bucket_idx stays -1

    mult_sum = float(mult.sum())

    if B * n_min >= n_total:
        # Not enough budget for n_min everywhere: distribute as evenly as possible.
        base = n_total // B
        residual = n_total - base * B
        n_g = np.full(B, base, dtype=np.int64)
        if residual > 0:
            n_g[:residual] += 1
        return n_g, bucket_idx

    n_remaining = n_total - B * n_min
    if mult_sum <= 0:
        # Degenerate (e.g., all first-seen with p_0 in {0,1}): uniform fallback.
        base = n_remaining // B
        residual = n_remaining - base * B
        n_g = np.full(B, n_min + base, dtype=np.int64)
        if residual > 0:
            n_g[:residual] += 1
        return n_g, bucket_idx

    n_g_float = n_remaining * (mult / mult_sum)
    n_g_floor = np.floor(n_g_float).astype(np.int64)
    residual = int(n_remaining - n_g_floor.sum())
    if residual > 0:
        frac = n_g_float - n_g_floor
        order = np.argsort(-frac)
        n_g_floor[order[:residual]] += 1

    n_g = n_g_floor + n_min

    if n_max is not None:
        excess = int(np.maximum(n_g - n_max, 0).sum())
        n_g = np.minimum(n_g, n_max)
        if excess > 0:
            non_capped_idx = np.where(n_g < n_max)[0]
            if len(non_capped_idx) > 0:
                order = non_capped_idx[np.argsort(-mult[non_capped_idx])]
                k = 0
                while excess > 0 and k < 1000 * len(order):
                    j = order[k % len(order)]
                    if n_g[j] < n_max:
                        n_g[j] += 1
                        excess -= 1
                    k += 1

    return n_g, bucket_idx


def compute_lp_n_allocation(
    persistent_indices,
    p_0_map: dict,
    last_p_t_map: dict,
    n_total: int,
    lp_gamma: float = 0.5,
    lp_lambda: float = 3.0,
    lp_w_max: float = 5.0,
    lp_eps_p: float = 0.05,
    n_min: int = 2,
    n_max: Optional[int] = None,
) -> np.ndarray:
    """Compute per-prompt rollout count N_g using LP weights.

    Used by LP-GRPO adaptive rollout allocation: at each step, before generating
    rollouts, we estimate each prompt's learning value with the LP formula using
    lagged p_t (from last_p_t_map; falls back to p_0 for first-seen prompts).
    The N_g vector sums to n_total and each entry >= n_min.

    Allocation uses largest-remainder method on normalized w:
        share_g = (n_total - B*n_min) * w_g / sum(w_g)
        N_g = floor(share_g) + (rank-based +1 for top fractional remainders) + n_min

    Args:
        persistent_indices: array-like of length B, persistent prompt ids
        p_0_map: dict {persistent_idx -> baseline pass rate}
        last_p_t_map: dict {persistent_idx -> last observed pass rate}
        n_total: total rollout budget for this batch (e.g., batch_size * rollout.n)
        lp_gamma/lambda/w_max/eps_p: LP formula hyperparameters
        n_min: minimum rollouts per prompt (>=2 keeps GRPO std defined)
        n_max: optional per-prompt cap (None = no cap; LP w is bounded by w_max anyway)

    Returns:
        np.ndarray (B,) int64, with sum(N_g) == n_total (modulo edge cases),
        and (n_min <= N_g <= n_max if n_max set).
    """
    B = len(persistent_indices)
    if B == 0:
        return np.array([], dtype=np.int64)

    # Step 1: compute raw LP w per prompt using lagged p_t
    w = np.zeros(B, dtype=np.float64)
    for i, pid in enumerate(persistent_indices):
        p_0 = float(p_0_map.get(pid, 0.5))
        p_t = float(last_p_t_map.get(pid, p_0))  # fallback: KL=0 → f_prog=1

        p_0_c = min(max(p_0, lp_eps_p), 1.0 - lp_eps_p)
        p_t_c = min(max(p_t, lp_eps_p), 1.0 - lp_eps_p)

        kl = float(
            p_t_c * np.log(p_t_c / p_0_c)
            + (1.0 - p_t_c) * np.log((1.0 - p_t_c) / (1.0 - p_0_c))
        )
        if p_t_c > p_0_c:
            f_diff = (1.0 - p_0_c) ** lp_gamma
            f_prog = min(lp_w_max, max(1.0, 1.0 + lp_lambda * kl))
        else:
            f_diff = (1.0 - p_t_c) ** lp_gamma
            f_prog = 1.0
        w[i] = f_diff * f_prog

    # Step 2: reserve n_min for each prompt, allocate rest proportional to w
    if B * n_min >= n_total:
        # Not enough budget for n_min everywhere: distribute as evenly as possible
        base = n_total // B
        residual = n_total - base * B
        n_g = np.full(B, base, dtype=np.int64)
        if residual > 0:
            n_g[:residual] += 1
        return n_g

    n_remaining = n_total - B * n_min
    w_sum = float(w.sum())
    if w_sum <= 0:
        # Degenerate (shouldn't happen normally): uniform distribution of remainder
        base = n_remaining // B
        residual = n_remaining - base * B
        n_g = np.full(B, n_min + base, dtype=np.int64)
        if residual > 0:
            n_g[:residual] += 1
        return n_g

    # Largest-remainder allocation on n_remaining
    n_g_float = n_remaining * (w / w_sum)
    n_g_floor = np.floor(n_g_float).astype(np.int64)
    residual = int(n_remaining - n_g_floor.sum())
    if residual > 0:
        frac = n_g_float - n_g_floor
        order = np.argsort(-frac)
        n_g_floor[order[:residual]] += 1

    n_g = n_g_floor + n_min

    # Optional cap
    if n_max is not None:
        excess = int(np.maximum(n_g - n_max, 0).sum())
        n_g = np.minimum(n_g, n_max)
        if excess > 0:
            # Redistribute excess to non-capped prompts (round-robin by w order)
            non_capped_mask = n_g < n_max
            non_capped_idx = np.where(non_capped_mask)[0]
            if len(non_capped_idx) > 0:
                # Sort non-capped by w descending; cycle to add 1 each
                order = non_capped_idx[np.argsort(-w[non_capped_idx])]
                k = 0
                while excess > 0 and k < 1000 * len(order):
                    j = order[k % len(order)]
                    if n_g[j] < n_max:
                        n_g[j] += 1
                        excess -= 1
                    k += 1

    return n_g


@register_adv_est(AdvantageEstimator.GDPO)  # or simply: @register_adv_est("gdpo")
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO: Group reward-Decoupled Normalization Policy Optimization.

    Instead of summing all reward dimensions first (like GRPO), GDPO normalizes
    each reward dimension independently within each group before aggregation.
    This prevents a dominant reward signal from drowning out weaker ones.

    Mathematical formulation:
        Step 1 – Group-wise decoupled normalization (via GRPO per dimension):
            For each reward dimension k, within each group g:
            A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)

        Step 2 – Weighted aggregation:
            A_sum = Σ_k w_k · A_k

        Step 3 – Batch-level normalization (via masked_whiten):
            A_final = whiten(A_sum, response_mask)

    Args:
        token_level_rewards: (bs, response_length) – standard token-level rewards.
            Used as fallback when per-dimension rewards are not provided.
        response_mask: (bs, response_length)
        index: (bs,) – group id per sample (from ``uid``).
        epsilon: Numerical stability constant.
        norm_adv_by_std_in_grpo: Whether to normalize by std in GRPO.
        config: Algorithm configuration (optional).
        non_tensor_batch: Non-tensor batch data containing per-dimension reward scores.
        batch: Batch data containing prompts, attention_mask, etc.

    Note:
        Ref GDPO (https://arxiv.org/abs/2601.05242).

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) – same as advantages (outcome-only).
    """
    score_list = None
    reward_weights = None

    if config is not None and non_tensor_batch is not None and batch is not None:
        gdpo_reward_keys = config.get("gdpo_reward_keys", None)
        assert gdpo_reward_keys, (
            "GDPO requires 'algorithm.gdpo_reward_keys' listing the individual reward "
            "component keys returned by compute_score (e.g. ['format_reward', 'accuracy_reward'])."
        )
        device = token_level_rewards.device
        prompt_length = batch["prompts"].size(1)
        valid_response_length = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        score_list = []
        for key in gdpo_reward_keys:
            assert key in non_tensor_batch, (
                f"GDPO reward key '{key}' not found in non_tensor_batch. "
                f"Available keys: {list(non_tensor_batch.keys())}. "
                f"Make sure your compute_score returns a dict containing '{key}'."
            )
            comp = non_tensor_batch[key]
            rm_score = torch.tensor(np.asarray(comp, dtype=np.float32), device=device)
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0), device=device), valid_response_length] = rm_score
            score_list.append(rm_scores)

        gdpo_weights = config.get("gdpo_reward_weights", None)
        if gdpo_weights is not None:
            reward_weights = list(gdpo_weights)

    if score_list is None:
        score_list = [token_level_rewards]

    num_scores = len(score_list)

    if reward_weights is not None:
        weights = torch.tensor(reward_weights, dtype=torch.float32, device=token_level_rewards.device)
    else:
        weights = torch.ones(num_scores, dtype=torch.float32, device=token_level_rewards.device)

    new_advantage = None

    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

        if new_advantage is None:
            new_advantage = weights[i] * normalized_score
        else:
            new_advantage += weights[i] * normalized_score

    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: True
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-TV.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the TV divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-TV
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for DPPO-Binary-KL.

    See https://arxiv.org/pdf/2602.04879 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Note: the clip_ratio is different from the standard PPO, it is the KL divergence threshold for DPPO.
    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Instead of dual-clip PPO, we use truncated importance sampling (TIS) to clip the policy loss.
    # However, a large threshold is recommended to avoid performance degradation due to the truncation bias.
    # See Section 5.4 in https://arxiv.org/pdf/2602.04879 for more details.
    clip_ratio_c = config.get("clip_ratio_c", 20.0)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
    truncated_ratio = truncated_ratio.detach()

    # Compute valid mask for DPPO-Binary-KL
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    valid_mask = valid_mask.detach().float()

    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard DPPO)
    pg_clipfrac = verl_F.masked_mean((1.0 - valid_mask).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    # Strip the optional '+' suffix so e.g. "k3+" dispatches to "k3".
    base_kl_penalty = kl_penalty[:-1] if kl_penalty.endswith("+") else kl_penalty
    forward_score = kl_penalty_forward(logprob, ref_logprob, base_kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics
