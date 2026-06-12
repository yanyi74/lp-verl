"""
v1 vs GRPO baseline only - clean comparison
No new formulas, just show how v1 differs from GRPO.
"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

# v1 parameters (your actual setup)
LP_GAMMA = 0.5
LP_LAMBDA = 5.0
LP_W_MAX = 5.0
LP_BETA = 1.0

def v1_weight(p_0, p_t):
    """v1 LP-GRPO formula (your original)."""
    if p_t > p_0:
        f_diff = (1 - p_0) ** LP_GAMMA
        f_prog = min(LP_W_MAX, max(1.0, 1 + LP_LAMBDA * abs(p_t - p_0)))
    else:
        f_diff = (1 - p_t) ** LP_GAMMA
        f_prog = 1.0
    zpd = (4 * p_t * (1 - p_t)) ** LP_BETA
    return f_diff * f_prog * zpd

# 6 typical buckets
BUCKETS = [
    ('stuck-hard',    0.05, 0.05, '#34495e'),
    ('mastered',      0.85, 0.95, '#95a5a6'),
    ('plateau-mid',   0.50, 0.50, '#e67e22'),
    ('progress',      0.30, 0.50, '#3498db'),
    ('breakthrough',  0.10, 0.45, '#27ae60'),
    ('regressing',    0.50, 0.20, '#9b59b6'),
]

# Real batch distribution from your data (lr=1e-5, step 200-221)
BATCH_RATIOS = {
    'stuck-hard':     0.20,
    'mastered':       0.16,
    'plateau-mid':    0.30,
    'progress':       0.30,
    'breakthrough':   0.10,
    'regressing':     0.12,
}

# Compute v1 weights per bucket
bucket_ws_v1 = {b[0]: v1_weight(b[1], b[2]) for b in BUCKETS}
bucket_ws_grpo = {b[0]: 1.0 for b in BUCKETS}

# Print
print(f"\n{'bucket':<15} {'v1 raw w':>10} {'GRPO w':>8} {'v1/GRPO':>10}")
for bname, *_ in BUCKETS:
    v1_w = bucket_ws_v1[bname]
    grpo_w = bucket_ws_grpo[bname]
    print(f"{bname:<15} {v1_w:>10.3f} {grpo_w:>8.3f} {v1_w/grpo_w:>10.2f}x")

# ============================================================
# Figure: 4 panels showing v1 vs GRPO clean comparison
# ============================================================
fig = plt.figure(figsize=(20, 12))

# (A) v1 landscape on (p_0, p_t)
p0_arr = np.linspace(0.02, 0.98, 80)
pt_arr = np.linspace(0.02, 0.98, 80)
P0, PT = np.meshgrid(p0_arr, pt_arr)

W_v1 = np.zeros_like(P0)
for i in range(P0.shape[0]):
    for j in range(P0.shape[1]):
        W_v1[i, j] = v1_weight(P0[i, j], PT[i, j])
W_grpo = np.ones_like(P0)

ax = plt.subplot(2, 3, 1)
im = ax.contourf(P0, PT, W_grpo, levels=20, cmap='RdYlGn', vmin=0, vmax=4)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0', fontsize=11); ax.set_ylabel('p_t', fontsize=11)
ax.set_title(f'(A) GRPO baseline\nw = 1 everywhere (uniform)\n'
             f'range [1.00, 1.00]', fontsize=11, fontweight='bold')

ax = plt.subplot(2, 3, 2)
im = ax.contourf(P0, PT, W_v1, levels=20, cmap='RdYlGn', vmin=0, vmax=4)
ax.contour(P0, PT, W_v1, levels=[1.0], colors='black', linewidths=1)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0', fontsize=11); ax.set_ylabel('p_t', fontsize=11)
ax.set_title(f'(B) v1 LP-GRPO (your formula)\n'
             f'w = (1-p_0)^γ · clip(1+λ|Δp|,1,w_max) · (4p_t(1-p_t))^β\n'
             f'γ=0.5, λ=5, w_max=5, β=1, range [{W_v1.min():.2f}, {W_v1.max():.2f}]',
             fontsize=11, fontweight='bold')

# (C) annotated bucket positions explanation
ax = plt.subplot(2, 3, 3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='p_0=p_t (no change)')
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=20, markeredgecolor='black', markeredgewidth=2)
    ax.annotate(bname, xy=(bp0, bpt), xytext=(bp0+0.03, bpt+0.03),
                fontsize=10, color=bcolor, fontweight='bold')
ax.set_xlabel('p_0 (initial pass rate)', fontsize=11)
ax.set_ylabel('p_t (current pass rate)', fontsize=11)
ax.set_title('(C) 6 typical bucket locations\non (p_0, p_t) plane', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

# (D) per-bucket weight comparison
ax = plt.subplot(2, 3, 4)
bucket_names_arr = [b[0] for b in BUCKETS]
v1_ws = [bucket_ws_v1[n] for n in bucket_names_arr]
grpo_ws = [bucket_ws_grpo[n] for n in bucket_names_arr]
colors = [b[3] for b in BUCKETS]

x = np.arange(len(bucket_names_arr))
bar_w = 0.35
bars1 = ax.bar(x - bar_w/2, grpo_ws, bar_w, color='lightgray', edgecolor='black', label='GRPO (w=1)')
bars2 = ax.bar(x + bar_w/2, v1_ws, bar_w, color=colors, edgecolor='black', label='v1 LP-GRPO')

for bar, v in zip(bars1, grpo_ws):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.2f}', ha='center', fontsize=9)
for bar, v in zip(bars2, v1_ws):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

ax.axhline(1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('per-prompt weight w', fontsize=11)
ax.set_title('(D) Per-bucket weight: v1 vs GRPO\n'
             '★ v1 让 breakthrough=2.58, plateau=0.71, mastered=0.11\n'
             'GRPO 都是 1.0', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 3.5)

# (E) gradient budget allocation
ax = plt.subplot(2, 3, 5)
# Budget = batch_ratio * weight
budget_grpo = {n: BATCH_RATIOS[n] * 1.0 for n in bucket_names_arr}
budget_v1 = {n: BATCH_RATIOS[n] * bucket_ws_v1[n] for n in bucket_names_arr}
total_grpo = sum(budget_grpo.values())
total_v1 = sum(budget_v1.values())
budget_grpo_pct = {n: v/total_grpo*100 for n, v in budget_grpo.items()}
budget_v1_pct = {n: v/total_v1*100 for n, v in budget_v1.items()}

grpo_pcts = [budget_grpo_pct[n] for n in bucket_names_arr]
v1_pcts = [budget_v1_pct[n] for n in bucket_names_arr]

bars1 = ax.bar(x - bar_w/2, grpo_pcts, bar_w, color='lightgray', edgecolor='black', label='GRPO')
bars2 = ax.bar(x + bar_w/2, v1_pcts, bar_w, color=colors, edgecolor='black', label='v1 LP-GRPO')

for bar, v in zip(bars1, grpo_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{v:.0f}%', ha='center', fontsize=9)
for bar, v in zip(bars2, v1_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('% of total gradient budget', fontsize=11)
ax.set_title('(E) ★ Gradient budget allocation (batch_ratio × w)\n'
             'How v1 reallocates compute vs GRPO uniform',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# (F) summary: useful vs useless allocation
ax = plt.subplot(2, 3, 6)
useful_buckets = ['progress', 'breakthrough']
useless_buckets = ['stuck-hard', 'mastered', 'plateau-mid']
inter_buckets = ['regressing']

grpo_useful = sum(budget_grpo_pct[n] for n in useful_buckets)
grpo_useless = sum(budget_grpo_pct[n] for n in useless_buckets)
grpo_inter = sum(budget_grpo_pct[n] for n in inter_buckets)

v1_useful = sum(budget_v1_pct[n] for n in useful_buckets)
v1_useless = sum(budget_v1_pct[n] for n in useless_buckets)
v1_inter = sum(budget_v1_pct[n] for n in inter_buckets)

cats = ['useful\n(progress+breakthrough)', 'useless\n(stuck+mastered+plateau)', 'intervention\n(regressing)']
grpo_vals = [grpo_useful, grpo_useless, grpo_inter]
v1_vals = [v1_useful, v1_useless, v1_inter]
xcat = np.arange(len(cats))
bars1 = ax.bar(xcat - bar_w/2, grpo_vals, bar_w, color='lightgray', edgecolor='black', label='GRPO')
bars2 = ax.bar(xcat + bar_w/2, v1_vals, bar_w, color=['#27ae60', '#e74c3c', '#9b59b6'],
               edgecolor='black', label='v1 LP-GRPO')
for bar, v in zip(bars1, grpo_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v:.0f}%', ha='center', fontsize=11)
for bar, v in zip(bars2, v1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(xcat)
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel('% of total gradient budget', fontsize=11)
ax.set_title(f'(F) Useful vs Useless 算力分配\n'
             f'v1 reallocates {v1_useful-grpo_useful:+.0f}% to useful, '
             f'{v1_useless-grpo_useless:+.0f}% from useless',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 75)

plt.suptitle('v1 LP-GRPO vs GRPO baseline: weight + gradient budget reallocation\n'
             'Batch ratios from your real lr=1e-5 data',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_v1_vs_grpo_clean.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved fig_v1_vs_grpo_clean.png")
print(f"\n=== Summary ===")
print(f"  GRPO: useful={grpo_useful:.0f}%, useless={grpo_useless:.0f}%, intervention={grpo_inter:.0f}%")
print(f"  v1:   useful={v1_useful:.0f}%, useless={v1_useless:.0f}%, intervention={v1_inter:.0f}%")
print(f"  Δ:    useful{v1_useful-grpo_useful:+.0f}%, useless{v1_useless-grpo_useless:+.0f}%, intervention{v1_inter-grpo_inter:+.0f}%")
