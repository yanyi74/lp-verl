# LP-GRPO Async (v_async) Patch

asymmetric per-rollout weighting, 0 hyperparameters, drop-in compatible with existing lp_grpo infra.

## 三处改动

### 1. `verl/trainer/ppo/core_algos.py`

#### 1a. 给 `AdvantageEstimator` 加新成员（第 88-115 行附近）

在 `class AdvantageEstimator` 里加一行：

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    # ... existing ...
    LP_GRPO = "lp_grpo"
    LP_GRPO_ASYNC = "lp_grpo_async"   # ← 新增
    # ... existing ...
```

#### 1b. 加 advantage 函数（建议放在 `compute_lp_grpo_outcome_advantage` 之后）

```python
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
    """LP-GRPO Async: asymmetric per-rollout weighting.

    Breaks GRPO's symmetric within-group invariant by applying
    different weights to correct (r_i=1) vs wrong (r_i=0) rollouts.

    Formula (0 hyperparameters):
        base_w = 4 * p_t * (1 - p_t)                  # ZPD magnitude
        delta  = clip(p_t - p_0, -0.8, 0.8)           # asymmetric direction
        w+     = base_w * (1 + delta * (1 - p_t))     # for correct rollouts
        w-     = base_w * (1 - delta * p_t)           # for wrong rollouts
        A_i_final = w_i * A_i_GRPO                    # i picks w+ or w-

    Mathematical properties:
      - mean(w_i) within group == base_w (LR-preserving)
      - w+ > 0, w- > 0 for delta in [-0.8, 0.8], p_t in [0,1]
      - group net signal = base_w * delta * N * p_t(1-p_t) / sigma
        (this is NEW; symmetric LP / vanilla GRPO have group net = 0)

    Falls back to vanilla GRPO when sigma_g ~ 0 (no useful signal).
    """
    if non_tensor_batch is None or "index" not in non_tensor_batch:
        raise ValueError(
            "lp_grpo_async requires non_tensor_batch['index'] "
            "(persistent prompt id). Set up dataset same as lp_grpo."
        )
    if lp_state is None or "p_0_map" not in lp_state:
        raise ValueError(
            "lp_grpo_async requires lp_state with 'p_0_map'. "
            "Trainer must initialize same as lp_grpo."
        )

    p_0_map = lp_state["p_0_map"]
    persistent_idx_arr = non_tensor_batch["index"]
    scores = token_level_rewards.sum(dim=-1)  # (B,)

    # ---- Group statistics (same as standard GRPO) ----
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

        # ---- v_async: compute per-group (w+, w-) ----
        uid_to_w_plus: dict = {}
        uid_to_w_minus: dict = {}
        # Logging accumulators
        log_w_plus, log_w_minus, log_delta, log_net = [], [], [], []
        log_active = 0  # number of groups with non-trivial asymmetric

        for uid in id2score:
            sigma = float(id2std[uid].item())
            p_t = float(id2mean[uid].item())
            pid = uid_to_persistent[uid]

            # Fallback: first-seen prompt uses p_t as p_0 (delta=0, symmetric)
            p_0 = float(p_0_map.get(pid, p_t))

            base_w = 4.0 * p_t * (1.0 - p_t)
            delta = max(-0.8, min(0.8, p_t - p_0))

            # σ ≈ 0: degenerate group, no useful signal; behave like vanilla GRPO
            if sigma < 1e-3:
                uid_to_w_plus[uid] = 1.0
                uid_to_w_minus[uid] = 1.0
                continue

            wp = base_w * (1.0 + delta * (1.0 - p_t))
            wm = base_w * (1.0 - delta * p_t)
            uid_to_w_plus[uid] = wp
            uid_to_w_minus[uid] = wm

            # Logging
            n_plus = float((torch.stack(id2score[uid]) > 0.5).sum().item())
            n_minus = len(id2score[uid]) - n_plus
            net_signal = base_w * delta * (n_plus + n_minus) * p_t * (1.0 - p_t) / (sigma + epsilon)
            log_w_plus.append(wp)
            log_w_minus.append(wm)
            log_delta.append(delta)
            log_net.append(net_signal)
            if abs(delta) > 0.05:
                log_active += 1

        # ---- Apply per-rollout ----
        for i in range(bsz):
            uid = index[i]
            mu, sigma = id2mean[uid], id2std[uid]

            if norm_adv_by_std_in_grpo:
                base_adv = (scores[i] - mu) / (sigma + epsilon)
            else:
                base_adv = scores[i] - mu

            # Select w by reward (binary r_i ∈ {0,1})
            if scores[i].item() > 0.5:  # correct rollout
                w = uid_to_w_plus[uid]
            else:  # wrong rollout
                w = uid_to_w_minus[uid]

            scores[i] = base_adv * w

        scores = scores.unsqueeze(-1) * response_mask

        # ---- Logging into lp_state for trainer to pick up ----
        if log_w_plus:
            lp_state["last_metrics"] = {
                "lp_async/w_plus/mean": float(np.mean(log_w_plus)),
                "lp_async/w_plus/max": float(np.max(log_w_plus)),
                "lp_async/w_minus/mean": float(np.mean(log_w_minus)),
                "lp_async/w_minus/min": float(np.min(log_w_minus)),
                "lp_async/delta/mean_abs": float(np.mean(np.abs(log_delta))),
                "lp_async/delta/max_abs": float(np.max(np.abs(log_delta))),
                "lp_async/net_signal/mean_abs": float(np.mean(np.abs(log_net))),
                "lp_async/active_groups": float(log_active),
                "lp_async/active_ratio": float(log_active) / max(1, len(log_w_plus)),
            }

    return scores, scores
```

### 2. `verl/trainer/ppo/ray_trainer.py`

#### 2a. 在 `compute_advantage` 里加 dispatch（约第 220 行）

找到现有的 `lp_grpo` 分支：

```python
if adv_estimator in (AdvantageEstimator.LP_GRPO, "lp_grpo"):
    adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
    adv_kwargs["lp_state"] = lp_state
```

在它之后加：

```python
if adv_estimator in (AdvantageEstimator.LP_GRPO_ASYNC, "lp_grpo_async"):
    adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
    adv_kwargs["lp_state"] = lp_state
```

#### 2b. 让 p_0_map 加载逻辑也适用于 async（约第 1437 行）

找到这一行：

```python
self.config.algorithm.adv_estimator in (AdvantageEstimator.LP_GRPO, "lp_grpo")
```

改成：

```python
self.config.algorithm.adv_estimator in (
    AdvantageEstimator.LP_GRPO, "lp_grpo",
    AdvantageEstimator.LP_GRPO_ASYNC, "lp_grpo_async",
)
```

epoch-end log 那里同样改（约第 1820 行 if 条件）。

### 3. 不需要 config 改动

`lp_grpo_async` 无超参，复用 `lp_grpo` 的 `lp_state` 机制（`p_0_map` 从 dataset 的 `p_zero` 列加载）。

---

## 验证 patch 的脚本（基于你现有 lr=1e-5 setup）

复制 `examples/lp_grpo_trainer/30k_lr1e5/run_lp_grpo_zpd1_lr1e5_30k_multinode.sh` 改名为 `run_lp_grpo_async_lr1e5_30k_multinode.sh`：

```bash
# 改 1: 算法选择
algorithm.adv_estimator=lp_grpo_async

# 改 2: 删掉所有 lp_* 旋钮（v_async 不用）
# 删除: algorithm.lp_gamma, lp_lambda, lp_w_max, lp_eps_p, lp_zpd_strength,
#       lp_p0_ema_alpha, lp_normalize_w, lp_w_clip_lo, lp_w_clip_hi,
#       lp_breakthrough_boost, lp_progress_boost, lp_regressing_penalty

# 改 3: 总步数缩短，快速验证
trainer.total_training_steps=200    # 原 700+, 减到 200 看趋势
trainer.save_freq=200
trainer.test_freq=20                # 高频测 (原可能 50+)

# 其他保持不变 (lr=1e-5, batch=128, rollout.n=8)
```

---

## 跑出来该看什么

### 必看（前 200 步）
```
critic/score/mean              ← 跟 GRPO baseline 比哪条曲线高
actor/entropy                  ← v_async 应该 < 0.45 (不像 v1 飙到 0.5)
actor/grad_norm                ← 应该稳定 0.10-0.20
```

### v_async 特有指标
```
lp_async/active_groups         ← 通常 50-80 (out of 128, ~50% 有效)
lp_async/active_ratio          ← ~0.4-0.6
lp_async/delta/mean_abs        ← 通常 0.05-0.15
lp_async/net_signal/mean_abs   ← > 0 即有真实方向信号
lp_async/w_plus/mean           ← 接近 1.0
lp_async/w_minus/mean          ← 接近 1.0
lp_async/w_plus/max            ← < 1.8 (设计上界)
lp_async/w_minus/min           ← > 0 (无 bug 保证)
```

### 跟 v1 LP 在同 step 的对比
```
v1 LP lp/w/mean = 0.58     ← 漂移
v_async w mean ≈ 1.0       ← 守恒, 改善了 v1 的 silent LR 缩水

v1 LP lp/w/max = 3.5       ← 高 lr 时打崩源
v_async w_plus/max < 1.8   ← 安全
```

---

## 三种可能结果

```
A. v_async 前 100 步显著超 GRPO (+1% 以上):
   ⇒ asymmetric 机制 work, 跑完整 700 步确认
   ⇒ 后续考虑加 tanh / boost 改进

B. v_async ≈ GRPO (±0.5%):
   ⇒ asymmetric 信号太弱, 直接上 tanh 改进
   ⇒ score = base · (1 + tanh(2·delta) · ...)

C. v_async 显著输 GRPO (>1%):
   ⇒ 检查 patch 实现, 看 lp_async/* metrics 是否合理
   ⇒ 如果 metrics 正常但 score 输, asymmetric 方向可能错
```

---

## 一句话

```
   patch 总改动: ~80 行
   实施时间: 2-3 小时
   首次实验: 200 步 × 3 run (GRPO / v1 / v_async)
   总验证时间: 1 天内能拿到结论
```
