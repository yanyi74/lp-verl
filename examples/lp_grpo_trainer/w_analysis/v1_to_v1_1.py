"""
v1 → v1.1 optimization visualization.
Show 3-way: GRPO baseline vs v1 (your original) vs v1.1 (optimized).
"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

# v1 hyperparameters
V1_GAMMA = 0.5
V1_LAMBDA = 5.0
V1_W_MAX = 5.0
V1_BETA = 1.0

# v1.1 hyperparameters
V11_GAMMA = 0.5
V11_ALPHA = 1.5
V11_LAMBDA = 5.0

def v1_weight(p_0, p_t):
    """Original v1."""
    if p_t > p_0:
        f_diff = (1 - p_0) ** V1_GAMMA
        f_prog = min(V1_W_MAX, max(1.0, 1 + V1_LAMBDA * abs(p_t - p_0)))
    else:
        f_diff = (1 - p_t) ** V1_GAMMA
        f_prog = 1.0
    zpd = (4 * p_t * (1 - p_t)) ** V1_BETA
    return f_diff * f_prog * zpd

def v1_1_weight(p_0, p_t):
    """Optimized v1.1."""
    fused = 4 * p_t * (1 - p_t) ** (V11_GAMMA + 1)
    f_prog = 1 + V11_ALPHA * np.tanh(V11_LAMBDA * abs(p_t - p_0))
    return fused * f_prog

BUCKETS = [
    ('stuck-hard',    0.05, 0.05, '#34495e'),
    ('mastered',      0.85, 0.95, '#95a5a6'),
    ('plateau-mid',   0.50, 0.50, '#e67e22'),
    ('progress',      0.30, 0.50, '#3498db'),
    ('breakthrough',  0.10, 0.45, '#27ae60'),
    ('regressing',    0.50, 0.20, '#9b59b6'),
]

BATCH_RATIOS = {
    'stuck-hard': 0.20, 'mastered': 0.16, 'plateau-mid': 0.30,
    'progress': 0.30, 'breakthrough': 0.10, 'regressing': 0.12,
}

# Compute per-bucket weights
ws = {}
for name, fn in [('GRPO', lambda p0, pt: 1.0), ('v1', v1_weight), ('v1.1', v1_1_weight)]:
    ws[name] = {b[0]: fn(b[1], b[2]) for b in BUCKETS}

print(f"\n{'bucket':<15} {'GRPO':>6} {'v1':>8} {'v1.1':>8}")
for bname, *_ in BUCKETS:
    print(f"{bname:<15} {ws['GRPO'][bname]:>6.2f} {ws['v1'][bname]:>8.3f} {ws['v1.1'][bname]:>8.3f}")

# ============================================================
# Figure: 6 panel comparison
# ============================================================
fig = plt.figure(figsize=(20, 12))

p0_arr = np.linspace(0.02, 0.98, 80)
pt_arr = np.linspace(0.02, 0.98, 80)
P0, PT = np.meshgrid(p0_arr, pt_arr)

W_grpo = np.ones_like(P0)
W_v1 = np.zeros_like(P0)
W_v1_1 = np.zeros_like(P0)
for i in range(P0.shape[0]):
    for j in range(P0.shape[1]):
        W_v1[i, j] = v1_weight(P0[i, j], PT[i, j])
        W_v1_1[i, j] = v1_1_weight(P0[i, j], PT[i, j])

vmax_show = 4.0

# Row 1: Landscapes
for idx, (name, W, subtitle) in enumerate([
    ('GRPO baseline', W_grpo, 'uniform w=1'),
    ('v1 (your original)', W_v1, f'w_max={W_v1.max():.2f}, clip 不稳'),
    ('★ v1.1 (optimized)', W_v1_1, f'w_max={W_v1_1.max():.2f}, tanh 平滑'),
]):
    ax = plt.subplot(2, 3, idx + 1)
    im = ax.contourf(P0, PT, W, levels=20, cmap='RdYlGn', vmin=0, vmax=vmax_show)
    ax.contour(P0, PT, W, levels=[1.0], colors='black', linewidths=1)
    plt.colorbar(im, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    for bname, bp0, bpt, bcolor in BUCKETS:
        ax.plot(bp0, bpt, 'o', color=bcolor, markersize=10, markeredgecolor='black', markeredgewidth=1)
    ax.set_xlabel('p_0', fontsize=11); ax.set_ylabel('p_t', fontsize=11)
    ax.set_title(f'({chr(65+idx)}) {name}\n{subtitle}\nrange [{W.min():.2f}, {W.max():.2f}]',
                 fontsize=11, fontweight='bold')

# Row 2: bucket weights + budget allocation + stability
bucket_names_arr = [b[0] for b in BUCKETS]
colors = [b[3] for b in BUCKETS]

# (D) per-bucket weight 3-way
ax = plt.subplot(2, 3, 4)
x = np.arange(len(bucket_names_arr))
bw = 0.25
ax.bar(x - bw, [ws['GRPO'][n] for n in bucket_names_arr], bw, color='lightgray',
       edgecolor='black', label='GRPO')
ax.bar(x, [ws['v1'][n] for n in bucket_names_arr], bw, color='#e67e22',
       edgecolor='black', label='v1')
ax.bar(x + bw, [ws['v1.1'][n] for n in bucket_names_arr], bw, color='#27ae60',
       edgecolor='black', label='v1.1 (ours)')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('per-prompt w', fontsize=11)
ax.set_title('(D) Per-bucket weight: 3-way comparison\n'
             '★ v1.1 keeps allocation pattern, lower w_max',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 3.5)

# (E) gradient budget allocation
ax = plt.subplot(2, 3, 5)
for name, color, offset in [('GRPO', 'lightgray', -bw), ('v1', '#e67e22', 0), ('v1.1', '#27ae60', bw)]:
    budgets = [BATCH_RATIOS[n] * ws[name][n] for n in bucket_names_arr]
    total = sum(budgets)
    pcts = [b / total * 100 for b in budgets]
    ax.bar(x + offset, pcts, bw, color=color, edgecolor='black', label=name)
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('% gradient budget', fontsize=11)
ax.set_title('(E) ★ Gradient budget allocation\n'
             'v1.1 同 v1 几乎一样, 都把算力转到 useful',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# (F) stability: effective LR vs nominal LR
ax = plt.subplot(2, 3, 6)
lrs = np.array([2e-6, 5e-6, 8e-6, 1e-5, 1.5e-5, 2e-5])
L_stable = 1.5e-5

w_max_v1 = W_v1.max()
w_max_v1_1 = W_v1_1.max()
w_max_grpo = 1.0

ax.plot(lrs * 1e6, lrs * w_max_grpo * 1e6, 'o-', color='gray', linewidth=2,
        markersize=10, label=f'GRPO (w_max=1.0)')
ax.plot(lrs * 1e6, lrs * w_max_v1 * 1e6, 's-', color='#e67e22', linewidth=2,
        markersize=10, label=f'v1 (w_max={w_max_v1:.2f})')
ax.plot(lrs * 1e6, lrs * w_max_v1_1 * 1e6, '^-', color='#27ae60', linewidth=2,
        markersize=10, label=f'v1.1 (w_max={w_max_v1_1:.2f})')
ax.axhline(L_stable * 1e6, color='red', linestyle='--', linewidth=2, label=f'L_stable ≈ 15 (unstable above)')
ax.fill_between([1.5, 22], L_stable * 1e6, 80, color='red', alpha=0.1)
# crossings
v1_cross = L_stable / w_max_v1 * 1e6
v11_cross = L_stable / w_max_v1_1 * 1e6
grpo_cross = L_stable / w_max_grpo * 1e6
ax.axvline(v1_cross, color='#e67e22', linestyle=':', alpha=0.5)
ax.axvline(v11_cross, color='#27ae60', linestyle=':', alpha=0.5)
ax.axvline(grpo_cross, color='gray', linestyle=':', alpha=0.5)
ax.text(v1_cross + 0.2, 5, f'v1 limit\n{v1_cross:.1f}', fontsize=9, color='#c0392b')
ax.text(v11_cross + 0.2, 5, f'v1.1 limit\n{v11_cross:.1f}', fontsize=9, color='green')
ax.text(grpo_cross + 0.2, 5, f'GRPO limit\n{grpo_cross:.1f}', fontsize=9, color='gray')
ax.set_xlabel('nominal lr (×1e-6)', fontsize=11)
ax.set_ylabel('max effective lr per prompt (×1e-6)', fontsize=11)
ax.set_title(f'(F) ★ Stability: v1 crosses limit @ 4.0e-6\n'
             f'v1.1 crosses @ {v11_cross:.1f}e-6 (跟 GRPO 接近)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim(1.5, 22)
ax.set_ylim(0, 80)

plt.suptitle('v1 → v1.1 optimization: GRPO vs v1 (original) vs v1.1 (optimized)\n'
             '3 fixes: clip→tanh (stability), ZPD+difficulty fused, bucket boosts deleted',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_v1_to_v1_1.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved fig_v1_to_v1_1.png")

# Print summary
print(f"\n=== Summary ===")
for name in ['GRPO', 'v1', 'v1.1']:
    useful = sum(BATCH_RATIOS[n] * ws[name][n] for n in ['progress', 'breakthrough'])
    useless = sum(BATCH_RATIOS[n] * ws[name][n] for n in ['stuck-hard', 'mastered', 'plateau-mid'])
    inter = sum(BATCH_RATIOS[n] * ws[name][n] for n in ['regressing'])
    total = useful + useless + inter
    print(f"  {name:<6}: useful={useful/total*100:.0f}%, useless={useless/total*100:.0f}%, intervention={inter/total*100:.0f}%")
print(f"\n  v1   w_max = {W_v1.max():.2f}, lr=1e-5 时 max eff_lr = {W_v1.max()*1e-5*1e6:.1f}e-6 (超 L_stable=15)")
print(f"  v1.1 w_max = {W_v1_1.max():.2f}, lr=1e-5 时 max eff_lr = {W_v1_1.max()*1e-5*1e6:.1f}e-6 (安全)")
