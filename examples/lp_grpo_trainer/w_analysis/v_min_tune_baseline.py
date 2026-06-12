"""Show how baseline parameter controls per-bucket weight ratios"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

BUCKETS = [
    ('stuck-hard',    0.05, 0.05, '#34495e'),
    ('mastered',      0.85, 0.95, '#95a5a6'),
    ('plateau-mid',   0.50, 0.50, '#e67e22'),
    ('progress',      0.30, 0.50, '#3498db'),
    ('breakthrough',  0.10, 0.45, '#27ae60'),
    ('regressing',    0.50, 0.20, '#9b59b6'),
]

# Real batch distribution from user's data (step 200-221)
BATCH_RATIOS = {
    'stuck-hard':     0.20,
    'mastered':       0.16,
    'plateau-mid':    0.30,  # mid-range plateau
    'progress':       0.30,
    'breakthrough':   0.10,
    'regressing':     0.12,
}

def v_min_with_baseline(p_0, p_t, baseline, amp_strength=1.0):
    movement = np.tanh(5 * np.abs(p_t - p_0))
    difficulty = (1 - p_t) ** 0.5
    return (baseline + amp_strength * movement) * difficulty

def compute_bucket_weights(baseline, amp_strength=1.0):
    # Simulate batch
    np.random.seed(42)
    B = 256
    p_0s = np.random.beta(1.5, 2.5, B)
    p_ts = np.clip(p_0s + np.random.normal(0.08, 0.15, B), 0.02, 0.98)
    useful = (p_ts > 0.05) & (p_ts < 0.95)
    raws = np.array([v_min_with_baseline(p_0s[i], p_ts[i], baseline, amp_strength) for i in range(B)])
    Z = raws[useful].mean()

    weights = {}
    for bname, bp0, bpt, _ in BUCKETS:
        raw = v_min_with_baseline(bp0, bpt, baseline, amp_strength)
        weights[bname] = np.clip(raw / Z, 0.3, 1.7)
    return weights

# Plot bucket weights for different baseline values
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

baselines = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(baselines)))

ax = axes[0]
x = np.arange(len(BUCKETS) + 1)  # +1 for GRPO ref
bucket_names = [b[0] for b in BUCKETS] + ['GRPO=1']

bar_width = 0.13
for idx, (baseline, color) in enumerate(zip(baselines, colors)):
    weights = compute_bucket_weights(baseline)
    ys = [weights[b[0]] for b in BUCKETS] + [1.0]
    offset = (idx - len(baselines)/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, ys, bar_width, color=color,
                  label=f'baseline={baseline}', edgecolor='black', linewidth=0.5)

ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='GRPO ref')
ax.set_xticks(x)
ax.set_xticklabels(bucket_names, rotation=20, ha='right')
ax.set_ylabel('per-bucket weight w')
ax.set_title(f'Baseline parameter controls per-bucket weight allocation\n'
             '(small baseline = sharp; large baseline = gentle)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 2)

# Second panel: gradient budget allocation per bucket (weight × batch ratio)
ax = axes[1]
for idx, (baseline, color) in enumerate(zip(baselines, colors)):
    weights = compute_bucket_weights(baseline)
    # Weight × ratio = "gradient share" for this bucket
    shares = [weights[b[0]] * BATCH_RATIOS[b[0]] for b in BUCKETS]
    total = sum(shares)
    shares_pct = [s / total * 100 for s in shares]

    offset = (idx - len(baselines)/2 + 0.5) * bar_width
    bars = ax.bar(x[:6] + offset, shares_pct, bar_width, color=color,
                  label=f'baseline={baseline}', edgecolor='black', linewidth=0.5)

ax.set_xticks(x[:6])
ax.set_xticklabels(bucket_names[:6], rotation=20, ha='right')
ax.set_ylabel('% of total gradient budget')
ax.set_title('Gradient budget allocation (weight × batch ratio)\n'
             'Shows where your compute is actually going',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 45)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_baseline_bucket_tuning.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved fig_baseline_bucket_tuning.png")

# Print recommended setting
print("\n=== Recommended baseline analysis ===")
print(f"{'baseline':>10} {'plat-mid':>10} {'breakthrough':>13} {'ratio break/plat':>17}")
for b in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    w = compute_bucket_weights(b)
    ratio = w['breakthrough'] / w['plateau-mid']
    print(f"{b:>10} {w['plateau-mid']:>10.3f} {w['breakthrough']:>13.3f} {ratio:>17.2f}")
