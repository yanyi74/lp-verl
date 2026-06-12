"""v1.1 asymmetric version: left (improving) > right (regressing)"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

# v1 params
V1_GAMMA, V1_LAMBDA, V1_W_MAX, V1_BETA = 0.5, 5.0, 5.0, 1.0
# v1.1 params
V11_GAMMA, V11_ALPHA, V11_LAMBDA, V11_FLOOR = 0.5, 1.5, 5.0, 0.3

def v1_weight(p_0, p_t):
    if p_t > p_0:
        f_diff = (1 - p_0) ** V1_GAMMA
        f_prog = min(V1_W_MAX, max(1.0, 1 + V1_LAMBDA * abs(p_t - p_0)))
    else:
        f_diff = (1 - p_t) ** V1_GAMMA
        f_prog = 1.0
    return f_diff * f_prog * (4 * p_t * (1 - p_t)) ** V1_BETA

def v1_1_asymmetric_weight(p_0, p_t):
    """v1.1 asymmetric: max(floor, 1 + α·tanh(λ·Δp)), signed Δp."""
    fused = 4 * p_t * (1 - p_t) ** (V11_GAMMA + 1)
    delta = p_t - p_0  # signed
    f_prog = max(V11_FLOOR, 1 + V11_ALPHA * np.tanh(V11_LAMBDA * delta))
    return fused * f_prog

def v1_1_symmetric_weight(p_0, p_t):
    """v1.1 symmetric (for comparison): 1 + α·tanh(λ·|Δp|)"""
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
    'stuck-hard':0.20, 'mastered':0.16, 'plateau-mid':0.30,
    'progress':0.30, 'breakthrough':0.10, 'regressing':0.12,
}

# Compute weights
ws = {
    'GRPO': {b[0]: 1.0 for b in BUCKETS},
    'v1': {b[0]: v1_weight(b[1], b[2]) for b in BUCKETS},
    'v1.1_sym': {b[0]: v1_1_symmetric_weight(b[1], b[2]) for b in BUCKETS},
    'v1.1_asym': {b[0]: v1_1_asymmetric_weight(b[1], b[2]) for b in BUCKETS},
}

print(f"\n{'bucket':<15} {'GRPO':>6} {'v1':>8} {'v1.1 sym':>10} {'v1.1 ASYM':>10}")
for bname, *_ in BUCKETS:
    print(f"{bname:<15} {ws['GRPO'][bname]:>6.2f} {ws['v1'][bname]:>8.3f} "
          f"{ws['v1.1_sym'][bname]:>10.3f} {ws['v1.1_asym'][bname]:>10.3f}")

# ============================================================
# Figure: 6 panel comparison
# ============================================================
fig = plt.figure(figsize=(20, 12))

p0_arr = np.linspace(0.02, 0.98, 80)
pt_arr = np.linspace(0.02, 0.98, 80)
P0, PT = np.meshgrid(p0_arr, pt_arr)

W_grpo = np.ones_like(P0)
W_v1 = np.zeros_like(P0)
W_sym = np.zeros_like(P0)
W_asym = np.zeros_like(P0)
for i in range(P0.shape[0]):
    for j in range(P0.shape[1]):
        W_v1[i, j] = v1_weight(P0[i, j], PT[i, j])
        W_sym[i, j] = v1_1_symmetric_weight(P0[i, j], PT[i, j])
        W_asym[i, j] = v1_1_asymmetric_weight(P0[i, j], PT[i, j])

vmax = 4.0

# Top row: 4 landscapes
landscapes = [
    (W_grpo, 'GRPO baseline\nw=1 everywhere', f'range [1.0, 1.0]'),
    (W_v1, 'v1 (your original)\nregressing 区被压', f'range [{W_v1.min():.2f}, {W_v1.max():.2f}], w_max=3.74'),
    (W_sym, 'v1.1 symmetric\nleft = right (你之前不要的)', f'range [{W_sym.min():.2f}, {W_sym.max():.2f}], w_max={W_sym.max():.2f}'),
    (W_asym, '★ v1.1 asymmetric (你要的)\nleft >> right', f'range [{W_asym.min():.2f}, {W_asym.max():.2f}], w_max={W_asym.max():.2f}'),
]

for idx, (W, title, subtitle) in enumerate(landscapes):
    ax = plt.subplot(2, 4, idx + 1)
    im = ax.contourf(P0, PT, W, levels=20, cmap='RdYlGn', vmin=0, vmax=vmax)
    ax.contour(P0, PT, W, levels=[1.0], colors='black', linewidths=1)
    plt.colorbar(im, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    for bname, bp0, bpt, bcolor in BUCKETS:
        ax.plot(bp0, bpt, 'o', color=bcolor, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel('p_0', fontsize=10)
    ax.set_ylabel('p_t', fontsize=10)
    ax.set_title(f'({chr(65+idx)}) {title}\n{subtitle}', fontsize=10, fontweight='bold')
    # Annotate improving / regressing regions
    if idx == 3:  # asymmetric panel
        ax.annotate('improving\n(left, BRIGHT)', xy=(0.15, 0.6), fontsize=9, color='darkgreen',
                    fontweight='bold', ha='center')
        ax.annotate('regressing\n(right, DARK)', xy=(0.75, 0.3), fontsize=9, color='darkred',
                    fontweight='bold', ha='center')

# Bottom row: bucket weights bar chart + gradient budget + left/right
bucket_names_arr = [b[0] for b in BUCKETS]
colors = [b[3] for b in BUCKETS]
x = np.arange(len(bucket_names_arr))
bw = 0.18

ax = plt.subplot(2, 4, 5)
ax.bar(x - 1.5*bw, [ws['GRPO'][n] for n in bucket_names_arr], bw, color='lightgray', edgecolor='black', label='GRPO')
ax.bar(x - 0.5*bw, [ws['v1'][n] for n in bucket_names_arr], bw, color='#e67e22', edgecolor='black', label='v1')
ax.bar(x + 0.5*bw, [ws['v1.1_sym'][n] for n in bucket_names_arr], bw, color='#3498db', edgecolor='black', label='v1.1 sym')
ax.bar(x + 1.5*bw, [ws['v1.1_asym'][n] for n in bucket_names_arr], bw, color='#27ae60', edgecolor='black', label='v1.1 asym ★')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('per-prompt w', fontsize=11)
ax.set_title('(E) Per-bucket weights (4-way)\n'
             '★ v1.1 asym: breakthrough 1.77, regressing 0.17',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 3.5)

# Gradient budget allocation
ax = plt.subplot(2, 4, 6)
for name, color, offset in [
    ('GRPO', 'lightgray', -1.5*bw),
    ('v1', '#e67e22', -0.5*bw),
    ('v1.1_sym', '#3498db', 0.5*bw),
    ('v1.1_asym', '#27ae60', 1.5*bw),
]:
    budgets = [BATCH_RATIOS[n] * ws[name][n] for n in bucket_names_arr]
    total = sum(budgets)
    pcts = [b / total * 100 for b in budgets]
    ax.bar(x + offset, pcts, bw, color=color, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('% gradient budget', fontsize=11)
ax.set_title('(F) Gradient budget allocation', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Left vs Right comparison
ax = plt.subplot(2, 4, 7)
left_buckets = ['breakthrough', 'progress']
right_buckets = ['regressing', 'mastered']
left_label = 'left (BT + Prog)'
right_label = 'right (Reg + Mast)'

method_names = ['GRPO', 'v1', 'v1.1 sym', 'v1.1 asym']
left_ws = []
right_ws = []
for m in ['GRPO', 'v1', 'v1.1_sym', 'v1.1_asym']:
    lw = np.mean([ws[m][n] for n in left_buckets])
    rw = np.mean([ws[m][n] for n in right_buckets])
    left_ws.append(lw)
    right_ws.append(rw)

xmethod = np.arange(len(method_names))
ax.bar(xmethod - 0.2, left_ws, 0.4, color='#27ae60', edgecolor='black', label=left_label)
ax.bar(xmethod + 0.2, right_ws, 0.4, color='#c0392b', edgecolor='black', label=right_label)
for i, (l, r) in enumerate(zip(left_ws, right_ws)):
    ax.text(i - 0.2, l + 0.05, f'{l:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + 0.2, r + 0.05, f'{r:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(xmethod)
ax.set_xticklabels(method_names, fontsize=10)
ax.set_ylabel('mean w', fontsize=11)
ax.set_title('(G) ★ Left vs Right comparison\n'
             'asymmetric 把 left/right 比例拉到最大',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Left/right ratio
ax = plt.subplot(2, 4, 8)
ratios = [l/r if r > 0 else 0 for l, r in zip(left_ws, right_ws)]
bars = ax.bar(method_names, ratios, color=['lightgray', '#e67e22', '#3498db', '#27ae60'], edgecolor='black')
for bar, r in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{r:.1f}x', ha='center', fontsize=11, fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.4)
ax.set_ylabel('left / right ratio', fontsize=11)
ax.set_title('(H) left:right 权重比\n'
             '★ asymmetric 最显著 凸显 improving',
             fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('v1.1 asymmetric: left > right, 凸显进步因子\n'
             '(顶行: 4 个 landscape; 底行: bucket weights + budget + left/right 比)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_v1_1_asymmetric.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved fig_v1_1_asymmetric.png")
