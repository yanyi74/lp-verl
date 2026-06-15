"""
真实测试 LPRevisitSampler:
- 抽取真实 sampler 类的代码(避开verl依赖), 实例化
- 造假 DataProto, 模拟 DataLoader 迭代 + 每step后 update
- 验证: visit_max<=M, coverage, visit_mean, dema, 新题能否进来
"""
import numpy as np, math, types, re

# ---- 1. 抽取真实 sampler 类源码, 替换掉 verl 依赖 ----
src = open("verl/experimental/dataset/lp_revisit_sampler.py").read()
# 去掉 verl import 行
src = re.sub(r'from verl.*\n', '', src)
src = re.sub(r'from omegaconf.*\n', '', src)
# 造桩: AbstractCurriculumSampler / DataProto / DictConfig
stub = """
class AbstractCurriculumSampler:
    pass
class DataProto:
    def __init__(s, batch, non_tensor_batch):
        s.batch=batch; s.non_tensor_batch=non_tensor_batch
DictConfig=dict
"""
ns = {}
exec(stub + src, ns)
LPRevisitSampler = ns["LPRevisitSampler"]

# ---- 2. 造假 dataset + config ----
N = 30000
class FakeDS:
    def __init__(s,n): s._n=n; s.dataframe=None
    def __len__(s): return s._n
class Cfg(dict):
    def get(s,k,d=None): return dict.get(s,k,d)
    def __getattr__(s,k):
        try: return s[k]
        except KeyError: raise AttributeError(k)
# p0分布(难50/中22/易28)
rng = np.random.default_rng(0)
p0 = np.concatenate([rng.uniform(0,0.3,15000),rng.uniform(0.3,0.7,7000),rng.uniform(0.7,1,8000)])

data_cfg = Cfg(train_batch_size=128, gen_batch_size=128, seed=1,
    lp_sampler=Cfg(min_interval=30, max_visit=5, alpha=0.4, beta=0.2, theta=0.05, new_floor=1.0))

s = LPRevisitSampler(FakeDS(N), data_cfg)
s.p0[:] = p0; s.base[:] = p0   # 手动塞p0(因为没真dataframe)

# ---- 3. 造假batch + 模拟训练动态 ----
class FakeTensor:
    def __init__(s,a): s.a=np.array(a)
    def sum(s,axis): return FakeTensor(s.a.sum(axis=axis))
    def cpu(s): return s
    def numpy(s): return s.a
def make_batch(idx_list, p_true_fn):
    # 每题8个rollout, reward由真实p决定
    scores=[]
    idx_rep=[]
    for pid in idx_list:
        pt=p_true_fn(pid)
        # token_level_scores: 这里直接给sum后的(每个rollout一个值0/1)...
        # 但sampler里 scores=batch.batch["token_level_scores"].sum(-1)
        # 我们直接让 sum(-1) 返回每个样本的对错. 简化: 每题1个样本(uid=index)
        scores.append(1.0 if rng.random()<pt else 0.0)
        idx_rep.append(pid)
    batch_t={"token_level_scores": FakeTensor(np.array(scores).reshape(-1,1))}
    ntb={"index": np.array(idx_rep)}
    return DataProto if False else ns["DataProto"](batch_t, ntb)

# 真实p: 中等题随访问进步, 难题卡, 易题已会
def p_true(pid):
    v=s.visit[pid]
    if p0[pid]<0.1: return 0.02            # 极难卡死
    if p0[pid]>0.85: return 0.95           # 易已会
    return min(p0[pid]+v*0.08, 0.9)        # 中等: 随访问进步

# ---- 4. 模拟3 epoch: __iter__产出indices, 按batch切, 每batch后update ----
print(f"开始测试: N={N}, min_interval={s.min_interval}, max_visit={s.M}, new_floor={s.new_floor}")
bs=128
for ep in range(3):
    it = iter(s)
    buf=[]
    cnt=0
    for idx in it:
        buf.append(idx)
        if len(buf)==bs:
            b=make_batch(buf, p_true)
            s.update(b)
            buf=[]
            cnt+=1
    # 每epoch报告
    seen=s.visit>0
    print(f"epoch{ep+1}: steps={cnt}, visit_max={s.visit.max()}, "
          f"coverage={seen.sum()}/{N}({seen.sum()/N*100:.0f}%), "
          f"visit_mean_seen={s.visit[seen].mean():.2f}, retired={int((s.visit>=s.M).sum())}")

print()
print("=== 最终验证 ===")
print(f"visit_max = {s.visit.max()}  {'✅ <=5' if s.visit.max()<=5 else '❌ 超过5!'}")
seen=s.visit>0
print(f"coverage = {seen.sum()/N*100:.0f}%  {'✅' if seen.sum()/N>0.8 else '⚠️偏低'}")
print(f"visit分布: 0次={np.sum(s.visit==0)}, 1-4次={np.sum((s.visit>=1)&(s.visit<5))}, 满5次={np.sum(s.visit>=5)}")
# 难题 vs 中等 vs 易 的平均访问
print(f"难题(p0<0.1)平均访问: {s.visit[p0<0.1].mean():.2f}")
print(f"中等(0.1-0.7)平均访问: {s.visit[(p0>=0.1)&(p0<0.7)].mean():.2f}")
print(f"易题(p0>0.85)平均访问: {s.visit[p0>0.85].mean():.2f}")
print(f"dema_global_mean: {s.dema[seen].mean():.3f}")
