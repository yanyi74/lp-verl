"""
v_min_strong: emphasize Δp as the dominant signal so each bucket has
visibly distinct w. "Don't waste compute on useless prompts" intent.
"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

# 5 typical buckets the user mentioned
BUCKETS = [
    ('stuck-hard',     0.05, 0.05, '#34495e', 'doing 10 times still wrong'),
    ('mastered',       0.85, 0.95, '#95a5a6', 'doing once correct'),
    ('plateau-mid',    0.50, 0.50, '#e67e22', 'stuck in middle (sigma>0)'),
    ('progress',       0.30, 0.50, '#3498db', 'steady improving'),
    ('breakthrough',   0.10, 0.45, '#27ae60', 'just cracked hard prompt'),
]

# Three formula variants
def v_min_v1(p_0, p_t):
    """Current v_min: ZPD-dominated, weak differentiation"""
    base = 4 * p_t * (1 - p_t) ** 1.5
    amp = 1 + 0.5 * np.tanh(5 * np.abs(p_t - p_0))
    return base * amp

def v_min_v2_dp(p_0, p_t):
    """Delta-p dominated: sqrt(|Δp|) · (1-p_t)^γ"""
    return np.sqrt(np.abs(p_t - p_0) + 0.05) * (1 - p_t) ** 0.5

def v_min_v3_explicit(p_0, p_t):
    """Most aggressive: small baseline + sharp Δp boost, difficulty bias"""
    baseline = 0.1
    movement = np.tanh(5 * np.abs(p_t - p_0))
    difficulty = (1 - p_t) ** 0.5
    return (baseline + movement) * difficulty

# Sample realistic batch for normalization
np.random.seed(42)
B = 256
p_0s = np.random.beta(1.5, 2.5, B)
p_ts = np.clip(p_0s + np.random.normal(0.08, 0.15, B), 0.02, 0.98)
useful = (p_ts > 0.05) & (p_ts < 0.95)

def normalize_clip(formula_fn, lo=0.3, hi=1.7):
    raws = np.array([formula_fn(p_0s[i], p_ts[i]) for i in range(B)])
    Z = raws[useful].mean()
    return raws, Z

# ============================================================
# Fig 1: 3 formula variants, landscape + bucket bars
# ============================================================
fig = plt.figure(figsize=(20, 10))

# top row: heatmaps
p0_arr = np.linspace(0.02, 0.98, 60)
pt_arr = np.linspace(0.02, 0.98, 60)
P0, PT = np.meshgrid(p0_arr, pt_arr)

variants = [
    ('v_min current (ZPD-dominant)', v_min_v1, 'current: even spread'),
    ('v_min_v2 (Δp-dominant)', v_min_v2_dp, 'sharper: plateau-mid pushed down'),
    ('v_min_v3 (explicit threshold)', v_min_v3_explicit, 'sharpest: only movers get weight'),
]

for idx, (name, fn, desc) in enumerate(variants):
    raws, Z = normalize_clip(fn)

    W = np.zeros_like(P0)
    for i in range(P0.shape[0]):
        for j in range(P0.shape[1]):
            raw = fn(P0[i, j], PT[i, j])
            W[i, j] = np.clip(raw / Z, 0.3, 1.7)

    ax = plt.subplot(2, 3, idx + 1)
    im = ax.contourf(P0, PT, W, levels=20, cmap='RdYlGn', vmin=0.3, vmax=1.7)
    ax.contour(P0, PT, W, levels=[1.0], colors='black', linewidths=1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.colorbar(im, ax=ax)

    # Mark bucket positions
    for bname, bp0, bpt, bcolor, _ in BUCKETS:
        ax.plot(bp0, bpt, 'o', color=bcolor, markersize=14, markeredgecolor='black', markeredgewidth=2)
        offset_y = -0.07 if bpt > 0.5 else 0.07
        ax.annotate(bname, xy=(bp0, bpt), xytext=(bp0, bpt+offset_y),
                    fontsize=8, ha='center', color=bcolor, fontweight='bold')

    ax.set_xlabel('p_0', fontsize=11)
    ax.set_ylabel('p_t', fontsize=11)
    ax.set_title(f'{name}\n{desc}\nmax/min = {W.max()/W.min():.1f}x', fontsize=10, fontweight='bold')

# bottom row: bar chart per bucket for each variant
for idx, (name, fn, desc) in enumerate(variants):
    raws, Z = normalize_clip(fn)

    bucket_ws = []
    bucket_names = []
    bucket_colors = []
    for bname, bp0, bpt, bcolor, _ in BUCKETS:
        w = np.clip(fn(bp0, bpt) / Z, 0.3, 1.7)
        bucket_ws.append(w)
        bucket_names.append(bname)
        bucket_colors.append(bcolor)
    # GRPO baseline reference
    bucket_ws.append(1.0)
    bucket_names.append('GRPO\nbaseline')
    bucket_colors.append('#7f8c8d')

    ax = plt.subplot(2, 3, idx + 4)
    bars = ax.bar(range(len(bucket_names)), bucket_ws, color=bucket_colors, edgecolor='black')
    for bar, w in zip(bars, bucket_ws):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{w:.2f}', ha='center', fontsize=10, fontweight='bold')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(bucket_names)))
    ax.set_xticklabels(bucket_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('w (per-prompt weight)', fontsize=11)
    ax.set_title(f'Bucket weights | mean of [stuck,mast,plat] = {np.mean(bucket_ws[:3]):.2f}\n'
                 f'mean of [prog,break] = {np.mean(bucket_ws[3:5]):.2f}', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 2)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('"Compute follows learning": 3 formula variants for per-bucket differentiation\n'
             '(top: weight landscape with bucket markers; bottom: explicit per-bucket weights vs GRPO)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_vmin_bucket_differentiation.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved fig_vmin_bucket_differentiation.png")

# Also print the per-bucket numbers
print("\n=== Per-bucket weights (after normalize + clip) ===")
print(f"{'bucket':<15} {'v_min_v1':>10} {'v_min_v2':>10} {'v_min_v3':>10} {'GRPO':>6}")
for bname, bp0, bpt, _, _ in BUCKETS:
    ws = []
    for fn in [v_min_v1, v_min_v2_dp, v_min_v3_explicit]:
        raws, Z = normalize_clip(fn)
        ws.append(np.clip(fn(bp0, bpt) / Z, 0.3, 1.7))
    print(f"{bname:<15} {ws[0]:>10.3f} {ws[1]:>10.3f} {ws[2]:>10.3f} {1.0:>6.2f}")
