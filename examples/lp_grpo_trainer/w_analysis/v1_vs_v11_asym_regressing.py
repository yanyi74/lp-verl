"""v1 vs v1.1 asym: focused on regressing behavior depth"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

def v1_weight(p_0, p_t):
    if p_t > p_0:
        f_diff = (1 - p_0) ** 0.5
        f_prog = min(5.0, max(1.0, 1 + 5.0 * abs(p_t - p_0)))
    else:
        f_diff = (1 - p_t) ** 0.5
        f_prog = 1.0
    return f_diff * f_prog * (4 * p_t * (1 - p_t))

def v11_asym_weight(p_0, p_t, alpha=1.5, beta=0.5, gamma=0.5, lam=5.0):
    fused = 4 * p_t * (1 - p_t) ** (gamma + 1)
    delta = p_t - p_0
    if delta >= 0:
        f_prog = 1 + alpha * np.tanh(lam * delta)
    else:
        f_prog = 1 + beta * np.tanh(lam * abs(delta))
    return fused * f_prog

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (A) 1D cross-section: fix p_0, vary p_t through regressing region
ax = axes[0]
p_t_arr = np.linspace(0.02, 0.95, 100)
for p_0_fix, color in [(0.3, '#3498db'), (0.5, '#e67e22'), (0.7, '#27ae60'), (0.9, '#9b59b6')]:
    w_v1 = [v1_weight(p_0_fix, pt) for pt in p_t_arr]
    w_v11 = [v11_asym_weight(p_0_fix, pt) for pt in p_t_arr]
    ax.plot(p_t_arr, w_v1, '--', color=color, linewidth=1.5, alpha=0.6,
            label=f'v1, p_0={p_0_fix}')
    ax.plot(p_t_arr, w_v11, '-', color=color, linewidth=2,
            label=f'v1.1 asym, p_0={p_0_fix}')
    ax.axvline(p_0_fix, color=color, linestyle=':', alpha=0.3)
ax.set_xlabel('p_t', fontsize=12)
ax.set_ylabel('w', fontsize=12)
ax.set_title('(A) w vs p_t (fixed p_0)\n虚线=v1, 实线=v1.1 asym\n竖虚线 p_t=p_0 (左侧是 regressing)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# (B) Regressing depth vs w (delta_neg = p_0 - p_t)
ax = axes[1]
depths = np.linspace(0, 0.5, 50)
for p_0_fix, color in [(0.5, '#e67e22'), (0.7, '#27ae60'), (0.9, '#9b59b6')]:
    w_v1_arr = []
    w_v11_arr = []
    for d in depths:
        p_t = max(0.01, p_0_fix - d)
        w_v1_arr.append(v1_weight(p_0_fix, p_t))
        w_v11_arr.append(v11_asym_weight(p_0_fix, p_t))
    ax.plot(depths, w_v1_arr, '--', color=color, linewidth=1.5, alpha=0.6, label=f'v1, p_0={p_0_fix}')
    ax.plot(depths, w_v11_arr, '-', color=color, linewidth=2, label=f'v1.1 asym, p_0={p_0_fix}')
ax.set_xlabel('regressing depth (p_0 - p_t)', fontsize=12)
ax.set_ylabel('w', fontsize=12)
ax.set_title('(B) Regressing depth 越大, w 怎么变\n虚线=v1, 实线=v1.1 asym\n★ v1.1 asym 始终在 v1 之上',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (C) Ratio: v1.1 asym / v1 in regressing region
ax = axes[2]
for p_0_fix, color in [(0.3, '#3498db'), (0.5, '#e67e22'), (0.7, '#27ae60'), (0.9, '#9b59b6')]:
    ratio_arr = []
    p_t_reg = np.linspace(0.02, p_0_fix - 0.01, 50)
    for pt in p_t_reg:
        w_v1 = v1_weight(p_0_fix, pt)
        w_v11 = v11_asym_weight(p_0_fix, pt)
        ratio_arr.append(w_v11 / max(w_v1, 1e-6))
    depths = p_0_fix - p_t_reg
    ax.plot(depths, ratio_arr, '-', color=color, linewidth=2, label=f'p_0={p_0_fix}')
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('regressing depth (p_0 - p_t)', fontsize=12)
ax.set_ylabel('v1.1 asym / v1', fontsize=12)
ax.set_title('(C) ★ v1.1 asym 给 regressing 的 w 比 v1 高多少倍\n'
             '横线 = v1 baseline, 上方表示 v1.1 asym 给更多权重',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('v1 vs v1.1 asym: regressing 行为对比\n'
             '★ v1.1 asym 对 regressing 反而比 v1 更温和 (w 更高), 不是更深',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_v1_vs_v11asym_regressing.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved fig_v1_vs_v11asym_regressing.png")

# Print numerical comparison
print(f"\n=== p_0=0.5 fixed, regressing depth comparison ===")
print(f"{'p_t':>6} {'depth':>7} {'v1':>8} {'v1.1 asym':>10} {'ratio':>7}")
for pt in [0.45, 0.40, 0.30, 0.20, 0.10, 0.05]:
    p_0 = 0.5
    depth = p_0 - pt
    w_v1 = v1_weight(p_0, pt)
    w_v11 = v11_asym_weight(p_0, pt)
    print(f"{pt:>6.2f} {depth:>7.2f} {w_v1:>8.3f} {w_v11:>10.3f} {w_v11/w_v1:>6.2f}x")
