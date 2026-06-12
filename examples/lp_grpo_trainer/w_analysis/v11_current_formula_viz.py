"""Visualize current v1.1 formulas (sym and asym variants you're running)"""
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/w_analysis'

GAMMA = 0.5
LAMBDA = 5.0
ALPHA = 1.5
BETA_SYM = 1.5
BETA_ASYM = 0.5

def v11_w(p_0, p_t, alpha, beta, gamma=GAMMA, lam=LAMBDA):
    fused = 4 * p_t * (1 - p_t) ** (gamma + 1)
    delta = p_t - p_0
    if delta >= 0:
        f_prog = 1 + alpha * np.tanh(lam * delta)
    else:
        f_prog = 1 + beta * np.tanh(lam * abs(delta))
    return fused * f_prog

def base_w(p_t, gamma=GAMMA):
    return 4 * p_t * (1 - p_t) ** (gamma + 1)

def f_prog_imp(delta, alpha=ALPHA, lam=LAMBDA):
    return 1 + alpha * np.tanh(lam * delta)

def f_prog_reg(delta, beta, lam=LAMBDA):
    return 1 + beta * np.tanh(lam * abs(delta))

BUCKETS = [
    ('stuck-hard',    0.05, 0.05, '#34495e'),
    ('mastered',      0.85, 0.95, '#95a5a6'),
    ('plateau-mid',   0.50, 0.50, '#e67e22'),
    ('progress',      0.30, 0.50, '#3498db'),
    ('breakthrough',  0.10, 0.45, '#27ae60'),
    ('regressing',    0.50, 0.20, '#9b59b6'),
]

# ============================================================
# Figure: comprehensive current formula visualization
# ============================================================
fig = plt.figure(figsize=(20, 13))

# Top row: components decomposition
# (A) base_w shape
ax = plt.subplot(3, 4, 1)
p = np.linspace(0.01, 0.99, 200)
bw = base_w(p)
ax.plot(p, bw, 'b-', linewidth=2)
ax.fill_between(p, 0, bw, alpha=0.2)
peak_p = 1 / (GAMMA + 2)
ax.axvline(peak_p, color='red', linestyle='--', alpha=0.5, label=f'peak at p_t={peak_p:.2f}')
ax.set_xlabel('p_t', fontsize=11)
ax.set_ylabel('base_w', fontsize=11)
ax.set_title(f'(A) base_w = 4·p_t·(1-p_t)^(γ+1)\nγ={GAMMA}, range [0, {bw.max():.2f}]\n融合 ZPD + 难度',
             fontsize=10, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# (B) f_prog improving
ax = plt.subplot(3, 4, 2)
d = np.linspace(0, 1, 200)
fp_imp = f_prog_imp(d, alpha=ALPHA)
ax.plot(d, fp_imp, 'g-', linewidth=2, label=f'α={ALPHA}')
ax.axhline(1+ALPHA, color='red', linestyle='--', alpha=0.5, label=f'上限 1+α={1+ALPHA:.1f}')
ax.set_xlabel('Δp (improving 时 = p_t - p_0)', fontsize=11)
ax.set_ylabel('f_prog (improving)', fontsize=11)
ax.set_title(f'(B) improving: 1 + α·tanh(λ·Δp)\nα={ALPHA}, λ={LAMBDA}',
             fontsize=10, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.5, 3)

# (C) f_prog regressing (sym vs asym)
ax = plt.subplot(3, 4, 3)
d_abs = np.linspace(0, 1, 200)
fp_reg_sym = f_prog_reg(d_abs, beta=BETA_SYM)
fp_reg_asym = f_prog_reg(d_abs, beta=BETA_ASYM)
ax.plot(d_abs, fp_reg_sym, 'b-', linewidth=2, label=f'SYM β={BETA_SYM} (= α)')
ax.plot(d_abs, fp_reg_asym, 'r-', linewidth=2, label=f'ASYM β={BETA_ASYM}')
ax.axhline(1+BETA_SYM, color='blue', linestyle=':', alpha=0.5)
ax.axhline(1+BETA_ASYM, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel('|Δp| (regressing 时 = p_0 - p_t)', fontsize=11)
ax.set_ylabel('f_prog (regressing)', fontsize=11)
ax.set_title(f'(C) regressing: 1 + β·tanh(λ·|Δp|)\n★ sym/asym 差别在这',
             fontsize=10, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.5, 3)

# (D) Full f_prog signed (improving + regressing)
ax = plt.subplot(3, 4, 4)
d_signed = np.linspace(-1, 1, 400)
fp_full_sym = [f_prog_imp(d, ALPHA) if d >= 0 else f_prog_reg(abs(d), BETA_SYM) for d in d_signed]
fp_full_asym = [f_prog_imp(d, ALPHA) if d >= 0 else f_prog_reg(abs(d), BETA_ASYM) for d in d_signed]
ax.plot(d_signed, fp_full_sym, 'b-', linewidth=2, label='SYM (β=1.5)')
ax.plot(d_signed, fp_full_asym, 'r-', linewidth=2, label='ASYM (β=0.5)')
ax.axvline(0, color='black', linestyle='--', alpha=0.3)
ax.axhline(1, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('signed Δp', fontsize=11)
ax.set_ylabel('f_prog', fontsize=11)
ax.set_title('(D) Full f_prog (signed)\n左半 regressing, 右半 improving',
             fontsize=10, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Middle row: 2D landscapes
p0_arr = np.linspace(0.02, 0.98, 80)
pt_arr = np.linspace(0.02, 0.98, 80)
P0, PT = np.meshgrid(p0_arr, pt_arr)

W_sym = np.zeros_like(P0)
W_asym = np.zeros_like(P0)
for i in range(P0.shape[0]):
    for j in range(P0.shape[1]):
        W_sym[i, j] = v11_w(P0[i, j], PT[i, j], ALPHA, BETA_SYM)
        W_asym[i, j] = v11_w(P0[i, j], PT[i, j], ALPHA, BETA_ASYM)

vmax = 2.5

ax = plt.subplot(3, 4, 5)
im = ax.contourf(P0, PT, W_sym, levels=20, cmap='RdYlGn', vmin=0, vmax=vmax)
ax.contour(P0, PT, W_sym, levels=[1.0], colors='black', linewidths=1)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0'); ax.set_ylabel('p_t')
ax.set_title(f'(E) v1.1 SYM landscape\nα=β=1.5, w_max={W_sym.max():.2f}\nleft ≈ right',
             fontsize=10, fontweight='bold')

ax = plt.subplot(3, 4, 6)
im = ax.contourf(P0, PT, W_asym, levels=20, cmap='RdYlGn', vmin=0, vmax=vmax)
ax.contour(P0, PT, W_asym, levels=[1.0], colors='black', linewidths=1)
plt.colorbar(im, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
for bname, bp0, bpt, bcolor in BUCKETS:
    ax.plot(bp0, bpt, 'o', color=bcolor, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('p_0'); ax.set_ylabel('p_t')
ax.set_title(f'(F) v1.1 ASYM landscape\nα=1.5, β=0.5, w_max={W_asym.max():.2f}\nleft > right',
             fontsize=10, fontweight='bold')

# (G) Bucket weights bar chart
ax = plt.subplot(3, 4, 7)
ws_sym = {b[0]: v11_w(b[1], b[2], ALPHA, BETA_SYM) for b in BUCKETS}
ws_asym = {b[0]: v11_w(b[1], b[2], ALPHA, BETA_ASYM) for b in BUCKETS}
bucket_names = [b[0] for b in BUCKETS]
colors = [b[3] for b in BUCKETS]
x = np.arange(len(bucket_names))
ax.bar(x - 0.2, [ws_sym[n] for n in bucket_names], 0.4, color='#3498db', edgecolor='black', label='SYM')
ax.bar(x + 0.2, [ws_asym[n] for n in bucket_names], 0.4, color='#27ae60', edgecolor='black', label='ASYM')
for i, n in enumerate(bucket_names):
    ax.text(i - 0.2, ws_sym[n] + 0.03, f'{ws_sym[n]:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.text(i + 0.2, ws_asym[n] + 0.03, f'{ws_asym[n]:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.4, label='GRPO = 1')
ax.set_xticks(x)
ax.set_xticklabels(bucket_names, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('w', fontsize=11)
ax.set_title('(G) Per-bucket weight: SYM vs ASYM\n仅 regressing 不同', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 2.5)

# (H) w mean (problem area)
ax = plt.subplot(3, 4, 8)
# Simulate batch
np.random.seed(42)
B = 512
p_0s = np.random.beta(1.5, 2.5, B)
p_ts = np.clip(p_0s + np.random.normal(0.08, 0.15, B), 0.02, 0.98)
useful = (p_ts > 0.05) & (p_ts < 0.95)
ws_sym_b = np.array([v11_w(p_0s[i], p_ts[i], ALPHA, BETA_SYM) for i in range(B) if useful[i]])
ws_asym_b = np.array([v11_w(p_0s[i], p_ts[i], ALPHA, BETA_ASYM) for i in range(B) if useful[i]])
ax.hist(ws_sym_b, bins=30, color='#3498db', alpha=0.6, edgecolor='black', label=f'SYM mean={ws_sym_b.mean():.2f}')
ax.hist(ws_asym_b, bins=30, color='#27ae60', alpha=0.6, edgecolor='black', label=f'ASYM mean={ws_asym_b.mean():.2f}')
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='ideal mean=1')
ax.axvline(ws_sym_b.mean(), color='blue', linestyle=':', linewidth=2)
ax.axvline(ws_asym_b.mean(), color='green', linestyle=':', linewidth=2)
ax.set_xlabel('w', fontsize=11)
ax.set_ylabel('# prompts', fontsize=11)
ax.set_title(f'(H) ★ w 分布问题\nmean 远低于 1, effective LR 缩水!',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Bottom row: trajectory comparison + diagnosis
import re

GRPO_LOG = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/30k_lr1e5/logs/grpo_baseline_lr1e5_30k_2node_20260606_1504.log'
SYM_LOG = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/30k_lr1e5/logs/lp_grpo_v11_sym_a1.5_b1.5_lr1e5_30k_2node_20260611_1818.log'
ASYM_LOG = '/mnt/tidal-alsh01/dataset/perceptionVLM/yanyi/paper/code/verl/examples/lp_grpo_trainer/30k_lr1e5/logs/lp_grpo_v11_asym_a1.5_b0.5_lr1e5_30k_2node_20260611_1817.log'

def parse_log(f):
    rows = []
    with open(f) as fh:
        for line in fh:
            m = re.search(r'global_step:(\d+)', line)
            if not m: continue
            d = {'step': int(m.group(1))}
            for k in ['critic/score/mean', 'actor/entropy', 'lp_v11/w/mean']:
                mm = re.search(re.escape(k)+r':([0-9.eE+\-]+)', line)
                if mm:
                    try: d[k] = float(mm.group(1))
                    except: pass
            rows.append(d)
    return rows

try:
    g_log = parse_log(GRPO_LOG)
    s_log = parse_log(SYM_LOG)
    a_log = parse_log(ASYM_LOG)
    max_step = min(max(r['step'] for r in s_log), max(r['step'] for r in a_log))

    # (I) score trajectory
    ax = plt.subplot(3, 4, 9)
    def smooth(rows, key, window=5):
        steps = [r['step'] for r in rows if key in r]
        vals = [r[key] for r in rows if key in r]
        if len(vals) < window: return steps, vals
        sm = np.convolve(vals, np.ones(window)/window, mode='valid')
        return steps[window-1:], sm

    s_g, v_g = smooth(g_log, 'critic/score/mean')
    s_s, v_s = smooth(s_log, 'critic/score/mean')
    s_a, v_a = smooth(a_log, 'critic/score/mean')

    cutoff = max_step + 5
    ax.plot([x for x in s_g if x<=cutoff], [v for x,v in zip(s_g, v_g) if x<=cutoff], '-', color='gray', linewidth=2, label='GRPO')
    ax.plot(s_s, v_s, '-', color='#3498db', linewidth=2, label='SYM (β=1.5)')
    ax.plot(s_a, v_a, '-', color='#27ae60', linewidth=2, label='ASYM (β=0.5)')
    ax.axvline(47, color='red', linestyle=':', alpha=0.5, label='step 47 反转点')
    ax.set_xlabel('step', fontsize=11)
    ax.set_ylabel('critic/score/mean (smoothed)', fontsize=11)
    ax.set_title('(I) Score 实测 (前 ~70 步)\n早期 v1.1 赢, ~47 步后反转',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (J) entropy trajectory
    ax = plt.subplot(3, 4, 10)
    s_ge, v_ge = smooth(g_log, 'actor/entropy')
    s_se, v_se = smooth(s_log, 'actor/entropy')
    s_ae, v_ae = smooth(a_log, 'actor/entropy')
    ax.plot([x for x in s_ge if x<=cutoff], [v for x,v in zip(s_ge, v_ge) if x<=cutoff], '-', color='gray', linewidth=2, label='GRPO')
    ax.plot(s_se, v_se, '-', color='#3498db', linewidth=2, label='SYM')
    ax.plot(s_ae, v_ae, '-', color='#27ae60', linewidth=2, label='ASYM')
    ax.set_xlabel('step', fontsize=11)
    ax.set_ylabel('actor/entropy', fontsize=11)
    ax.set_title('(J) ★ Entropy 实测\nGRPO 稳定 ~0.35, v1.1 上升到 0.55!\n(policy 不收敛 = 反转根因)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (K) w_mean over training
    ax = plt.subplot(3, 4, 11)
    s_sw, v_sw = smooth(s_log, 'lp_v11/w/mean')
    s_aw, v_aw = smooth(a_log, 'lp_v11/w/mean')
    ax.plot(s_sw, v_sw, '-', color='#3498db', linewidth=2, label='SYM')
    ax.plot(s_aw, v_aw, '-', color='#27ae60', linewidth=2, label='ASYM')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='ideal=1.0')
    ax.set_xlabel('step', fontsize=11)
    ax.set_ylabel('w mean over useful', fontsize=11)
    ax.set_title('(K) w_mean 实测\nSYM ≈ 0.40, ASYM ≈ 0.35\neffective LR 缩水 60%!',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

except Exception as e:
    print(f"Log parsing failed: {e}")
    for idx in [9, 10, 11]:
        ax = plt.subplot(3, 4, idx)
        ax.text(0.5, 0.5, f'Log unavailable\n{str(e)[:60]}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

# (L) Summary table
ax = plt.subplot(3, 4, 12)
ax.axis('off')
summary_text = """
SUMMARY: v1.1 当前公式

formula:
  w = 4·p_t·(1-p_t)^(γ+1) · f_prog(Δp)

  improving (Δp≥0):
    f_prog = 1 + α·tanh(λ·Δp)
  regressing (Δp<0):
    f_prog = 1 + β·tanh(λ·|Δp|)

defaults:
  γ=0.5, α=1.5, λ=5

两个跑的 variant:
  SYM:  β=1.5 (regressing 等同放大)
  ASYM: β=0.5 (regressing mild 放大)

实测问题 (47 步反转):
  1. w_mean ≈ 0.4 → effective LR 缩水 60%
  2. entropy 升到 0.55 → policy 不收敛
  3. 早期 +0.020 → 晚期 -0.010

待解决:
  归一化 ✗ (会让 w_max 爆 4.6)
  推荐 v1.2: mean=1 by construction
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, family='monospace', verticalalignment='top')

plt.suptitle('v1.1 当前公式: 公式分解 + landscape + bucket + 实测诊断',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_v11_current_full.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved fig_v11_current_full.png")
