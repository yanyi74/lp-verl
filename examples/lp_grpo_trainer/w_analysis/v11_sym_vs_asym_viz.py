"""
v1.1 sym (beta=alpha=1.5) vs v1.1 asym (beta=0.5, alpha=1.5)
Visualize w landscape and per-bucket weights for the two experimental variants.
"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

# Shared params
GAMMA, LAMBDA = 0.5, 5.0
# sym: beta = alpha
ALPHA_SYM, BETA_SYM = 1.5, 1.5
# asym: beta < alpha
ALPHA_ASYM, BETA_ASYM = 1.5, 0.5

def v11_weight(p_0, p_t, alpha, beta, gamma=GAMMA, lam=LAMBDA):
    fused = 4 * p_t * (1 - p_t) ** (gamma + 1)
    delta = p_t - p_0
    if delta >= 0:
        f_prog = 1 + alpha * np.tanh(lam * delta)
    else:
        f_prog = 1 + beta * np.tanh(lam * abs(delta))
    return fused * f_prog

def v1_weight(p_0, p_t):
    if p_t > p_0:
        f_diff = (1 - p_0) ** 0.5
        f_prog = min(5.0, max(1.0, 1 + 5.0 * abs(p_t - p_0)))
    else:
        f_diff = (1 - p_t) ** 0.5
        f_prog = 1.0
    return f_diff * f_prog * (4 * p_t * (1 - p_t))

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
    'v1.1 sym': {b[0]: v11_weight(b[1], b[2], ALPHA_SYM, BETA_SYM) for b in BUCKETS},
    'v1.1 asym': {b[0]: v11_weight(b[1], b[2], ALPHA_ASYM, BETA_ASYM) for b in BUCKETS},
}

print(f"\n{'bucket':<15} {'GRPO':>6} {'v1':>8} {'sym':>8} {'asym':>8}")
for bname, *_ in BUCKETS:
    print(f"{bname:<15} {ws['GRPO'][bname]:>6.2f} {ws['v1'][bname]:>8.3f} "
          f"{ws['v1.1 sym'][bname]:>8.3f} {ws['v1.1 asym'][bname]:>8.3f}")

# ============================================================
# Figure: 6 panel (2 landscapes + bucket bars + budget + L/R + ratio)
# ============================================================
fig = plt.figure(figsize=(20, 11))

p0_arr = np.linspace(0.02, 0.98, 80)
pt_arr = np.linspace(0.02, 0.98, 80)
P0, PT = np.meshgrid(p0_arr, pt_arr)

W_sym = np.zeros_like(P0)
W_asym = np.zeros_like(P0)
for i in range(P0.shape[0]):
    for j in range(P0.shape[1]):
        W_sym[i, j] = v11_weight(P0[i, j], PT[i, j], ALPHA_SYM, BETA_SYM)
        W_asym[i, j] = v11_weight(P0[i, j], PT[i, j], ALPHA_ASYM, BETA_ASYM)

vmax = 2.5

# (A) sym landscape
ax = plt.subplot(2, 3, 1)
im = ax.contourf(P0, PT, W_sym, levels=20, cmap='RdYlGn', vmin=0, vmax=vmax)
ax.contour(P0, PT, W_sym, levels=[1.0], colors='black', linewidths=1)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0', fontsize=11); ax.set_ylabel('p_t', fontsize=11)
ax.set_title(f'(A) v1.1 SYMMETRIC (alpha=beta=1.5)\n'
             f'range [{W_sym.min():.2f}, {W_sym.max():.2f}], w_max={W_sym.max():.2f}\n'
             f'left ≈ right (镜像对称)',
             fontsize=11, fontweight='bold')

# (B) asym landscape
ax = plt.subplot(2, 3, 2)
im = ax.contourf(P0, PT, W_asym, levels=20, cmap='RdYlGn', vmin=0, vmax=vmax)
ax.contour(P0, PT, W_asym, levels=[1.0], colors='black', linewidths=1)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0', fontsize=11); ax.set_ylabel('p_t', fontsize=11)
ax.set_title(f'(B) ★ v1.1 ASYMMETRIC (alpha=1.5, beta=0.5)\n'
             f'range [{W_asym.min():.2f}, {W_asym.max():.2f}], w_max={W_asym.max():.2f}\n'
             f'left > right (improving 更强)',
             fontsize=11, fontweight='bold')
ax.annotate('improving\n(BRIGHT)', xy=(0.15, 0.5), fontsize=9, color='darkgreen',
            fontweight='bold', ha='center')
ax.annotate('regressing\n(dim)', xy=(0.75, 0.25), fontsize=9, color='darkred',
            fontweight='bold', ha='center')

# (C) Difference: sym - asym
ax = plt.subplot(2, 3, 3)
DIFF = W_sym - W_asym
vdiff = max(abs(DIFF.min()), DIFF.max())
im = ax.contourf(P0, PT, DIFF, levels=20, cmap='RdBu_r', vmin=-vdiff, vmax=vdiff)
ax.contour(P0, PT, DIFF, levels=[0], colors='black', linewidths=1)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0', fontsize=11); ax.set_ylabel('p_t', fontsize=11)
ax.set_title(f'(C) Difference: sym - asym\n'
             f'red = sym 给 regressing 更多权重\n'
             f'range [{DIFF.min():.2f}, {DIFF.max():.2f}]',
             fontsize=11, fontweight='bold')

# (D) Per-bucket weight 4-way
ax = plt.subplot(2, 3, 4)
bucket_names_arr = [b[0] for b in BUCKETS]
colors = [b[3] for b in BUCKETS]
x = np.arange(len(bucket_names_arr))
bw = 0.18
ax.bar(x - 1.5*bw, [ws['GRPO'][n] for n in bucket_names_arr], bw, color='lightgray', edgecolor='black', label='GRPO')
ax.bar(x - 0.5*bw, [ws['v1'][n] for n in bucket_names_arr], bw, color='#e67e22', edgecolor='black', label='v1')
ax.bar(x + 0.5*bw, [ws['v1.1 sym'][n] for n in bucket_names_arr], bw, color='#3498db', edgecolor='black', label='v1.1 sym')
ax.bar(x + 1.5*bw, [ws['v1.1 asym'][n] for n in bucket_names_arr], bw, color='#27ae60', edgecolor='black', label='v1.1 asym ★')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('per-prompt w', fontsize=11)
ax.set_title('(D) Per-bucket weight comparison\n'
             '差别在 regressing: sym=1.35, asym=0.83',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 3.0)

# (E) Gradient budget
ax = plt.subplot(2, 3, 5)
for name, color, offset in [
    ('GRPO', 'lightgray', -1.5*bw),
    ('v1', '#e67e22', -0.5*bw),
    ('v1.1 sym', '#3498db', 0.5*bw),
    ('v1.1 asym', '#27ae60', 1.5*bw),
]:
    budgets = [BATCH_RATIOS[n] * ws[name][n] for n in bucket_names_arr]
    total = sum(budgets)
    pcts = [b / total * 100 for b in budgets]
    ax.bar(x + offset, pcts, bw, color=color, edgecolor='black', label=name if offset == -1.5*bw else None)
ax.set_xticks(x)
ax.set_xticklabels(bucket_names_arr, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('% gradient budget', fontsize=11)
ax.set_title('(E) Gradient budget allocation\n'
             'sym 给 regressing 更多 budget',
             fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (F) Improving vs Regressing summary
ax = plt.subplot(2, 3, 6)
improving_buckets = ['progress', 'breakthrough']
regressing_buckets = ['regressing']

method_names = ['GRPO', 'v1', 'sym', 'asym']
imp_ws, reg_ws, ratios = [], [], []
for m in ['GRPO', 'v1', 'v1.1 sym', 'v1.1 asym']:
    iw = np.mean([ws[m][n] for n in improving_buckets])
    rw = np.mean([ws[m][n] for n in regressing_buckets])
    imp_ws.append(iw)
    reg_ws.append(rw)
    ratios.append(iw / rw if rw > 0 else float('inf'))

xmethod = np.arange(len(method_names))
ax.bar(xmethod - 0.2, imp_ws, 0.4, color='#27ae60', edgecolor='black', label='improving (BT+Prog)')
ax.bar(xmethod + 0.2, reg_ws, 0.4, color='#c0392b', edgecolor='black', label='regressing')
for i, (l, r, rat) in enumerate(zip(imp_ws, reg_ws, ratios)):
    ax.text(i - 0.2, l + 0.05, f'{l:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + 0.2, r + 0.05, f'{r:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i, max(l, r) + 0.35, f'{rat:.1f}x', ha='center', fontsize=11,
            fontweight='bold', color='#8e44ad')
ax.set_xticks(xmethod)
ax.set_xticklabels(method_names, fontsize=11)
ax.set_ylabel('mean w', fontsize=11)
ax.set_title('(F) ★ improving vs regressing 比\n'
             'sym=1.3x (近对称), asym=2.1x (适度不对称)\n'
             'v1=2.9x (强不对称)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 3.0)

plt.suptitle('v1.1 实验对照: SYMMETRIC (beta=alpha=1.5) vs ASYMMETRIC (beta=0.5, alpha=1.5)\n'
             '两个 run, lr=1e-5, 唯一差别是 beta 值',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_v11_sym_vs_asym.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved fig_v11_sym_vs_asym.png")

# Summary
print(f"\n=== Per-bucket summary ===")
print(f"sym vs asym 差异仅在 regressing:")
print(f"  sym  regressing w  = {ws['v1.1 sym']['regressing']:.3f}")
print(f"  asym regressing w  = {ws['v1.1 asym']['regressing']:.3f}")
print(f"  sym/asym ratio     = {ws['v1.1 sym']['regressing']/ws['v1.1 asym']['regressing']:.2f}x")
