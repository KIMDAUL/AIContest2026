"""(task − history) 잔여 토큰  ↔  candidate_elements 토큰 유사도 시각화

가설
-----
- task     : 목표를 이루기 위해 등장하는 feature 토큰 집합
- history  : 지금까지 수행한 단계 → task 토큰 중 일부가 이미 나옴
- 다음 행동의 정답은 "아직 처리되지 않은 task 토큰"과 연관될 가능성이 높음
  → remaining = set(task_tokens) − set(history_tokens)
- 따라서 remaining 의 핵심 토큰은 candidate_elements (특히 정답 candidate)의
  토큰과 높은 유사도를 가질 것으로 기대된다.

이 스크립트는 (1) remaining vs 모든 candidates 의 유사도 분포,
(2) 정답 candidate vs 정답 외 candidate 들에서 remaining 과의 유사도 차이를
시각화한다.

산출물 (notebooks/viz_out/):
  - remaining_vs_candidates.png        : Jaccard/Overlap/Coverage 분포
  - remaining_vs_target_vs_others.png  : 정답 vs 비정답 candidate 비교
  - remaining_size_distribution.png    : |remaining| 분포
  - remaining_vs_candidates_summary.txt
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'viz_out'
OUT.mkdir(exist_ok=True)

# ---------- 데이터 ----------
df = pd.read_csv(ROOT / 'data' / 'train.csv', encoding='utf-8', encoding_errors='replace')
mask = df['task'].notna() & df['history'].notna() & df['candidate_elements'].notna()
sub = df.loc[mask].reset_index(drop=True)
n = len(sub)

# ---------- 전처리 ----------
DOMAIN_STOP = {'task', 'step', 'click', 'type', 'select', 'enter'}
# UI 동작 동사(click/type/select/enter)는 history 에 항상 등장 → remaining 계산을
# 흐리지 않도록 도메인 불용어로 추가
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")

def tokenize(s: str) -> list[str]:
    return [t for t in (m.lower() for m in TOKEN_RE.findall(s))
            if t not in STOPWORDS and len(t) > 1]

def cand_text(cand_list) -> str:
    """candidate dict 리스트 → 토큰화 대상 문자열 (text + attrs 값만 모음)."""
    parts = []
    for c in cand_list:
        parts.append(c.get('text') or '')
        parts.append(c.get('attrs') or '')
    return ' '.join(parts)

# ---------- 행별 토큰 집합 만들기 ----------
task_tokens_list      = [tokenize(s) for s in sub['task'].astype(str)]
history_tokens_list   = [tokenize(s) for s in sub['history'].astype(str)]

cand_parsed = [json.loads(s) for s in sub['candidate_elements'].astype(str)]
cand_tokens_list      = [tokenize(cand_text(cs)) for cs in cand_parsed]

# 정답/비정답 candidate 토큰 — 공평한 비교를 위해 candidate 별로 따로 보관
target_ids   = sub['target_id'].astype(str).tolist()
target_tokens_list  = []          # 정답 1개의 토큰
per_other_tokens_list = []        # 비정답 candidate 각각의 토큰 리스트 (가변 길이 list-of-lists)
others_tokens_list  = []          # 비정답 합집합 (참고용)
for cs, tid in zip(cand_parsed, target_ids):
    tgt_text, oth_union, oth_each = [], [], []
    for c in cs:
        ctext = (c.get('text') or '') + ' ' + (c.get('attrs') or '')
        if c.get('candidate_id') == tid:
            tgt_text.append(ctext)
        else:
            oth_union.append(ctext)
            oth_each.append(tokenize(ctext))
    target_tokens_list.append(tokenize(' '.join(tgt_text)))
    others_tokens_list.append(tokenize(' '.join(oth_union)))
    per_other_tokens_list.append(oth_each)

# numpy(object) 보관 — 인덱스 i 가 같은 행
task_tokenized_array      = np.array(task_tokens_list,      dtype=object)
history_tokenized_array   = np.array(history_tokens_list,   dtype=object)
candidate_tokenized_array = np.array(cand_tokens_list,      dtype=object)
target_tokenized_array    = np.array(target_tokens_list,    dtype=object)
others_tokenized_array    = np.array(others_tokens_list,    dtype=object)

# 잔여 토큰: 집합 차집합 (task - history)
remaining_tokenized_array = np.array(
    [list(set(t) - set(h)) for t, h in zip(task_tokens_list, history_tokens_list)],
    dtype=object,
)

# ---------- 유사도 계산 ----------
def jaccard(A, B):
    if not A and not B: return 0.0
    return len(A & B) / len(A | B)
def overlap_min(A, B):
    if not A or not B: return 0.0
    return len(A & B) / min(len(A), len(B))
def dice(A, B):
    if not A and not B: return 0.0
    return 2 * len(A & B) / (len(A) + len(B))
def coverage(R, C):
    """remaining 이 candidates 안에 얼마나 들어있는가 — |R∩C|/|R|"""
    if not R: return np.nan
    return len(R & C) / len(R)

J_rc      = np.zeros(n)   # remaining vs all-candidates
O_rc      = np.zeros(n)
D_rc      = np.zeros(n)
COV_rc    = np.full(n, np.nan)
J_rt      = np.full(n, np.nan)  # remaining vs target (single candidate)
J_ro_mean = np.full(n, np.nan)  # remaining vs each non-target candidate, averaged
J_ro_max  = np.full(n, np.nan)  # remaining vs best non-target candidate
COV_rt    = np.full(n, np.nan)
COV_ro_mean = np.full(n, np.nan)
COV_ro_max  = np.full(n, np.nan)
target_rank = np.full(n, np.nan)  # 정답이 Coverage 기준 몇 등인지 (1=top)
remaining_sz = np.zeros(n, dtype=int)
task_sz      = np.zeros(n, dtype=int)
history_sz   = np.zeros(n, dtype=int)
cand_sz      = np.zeros(n, dtype=int)

for i in range(n):
    R = set(remaining_tokenized_array[i])
    C = set(candidate_tokenized_array[i])
    T = set(target_tokenized_array[i])
    remaining_sz[i] = len(R)
    task_sz[i]      = len(set(task_tokenized_array[i]))
    history_sz[i]   = len(set(history_tokenized_array[i]))
    cand_sz[i]      = len(C)
    J_rc[i] = jaccard(R, C)
    O_rc[i] = overlap_min(R, C)
    D_rc[i] = dice(R, C)
    COV_rc[i] = coverage(R, C)
    if R:
        J_rt[i] = jaccard(R, T)
        COV_rt[i] = coverage(R, T)
        # 비정답 candidate 들 — 각각 개별 점수
        per_other = per_other_tokens_list[i]
        if per_other:
            j_each   = [jaccard(R, set(o))  for o in per_other]
            cov_each = [coverage(R, set(o)) for o in per_other]
            J_ro_mean[i]   = float(np.mean(j_each))
            J_ro_max[i]    = float(np.max(j_each))
            COV_ro_mean[i] = float(np.mean(cov_each))
            COV_ro_max[i]  = float(np.max(cov_each))
            # 정답이 Coverage 기준 몇 등인지 (동률 시 평균 순위)
            scores = cov_each + [COV_rt[i]]
            order = np.argsort(-np.array(scores))  # 내림차순
            ranks = np.empty_like(order); ranks[order] = np.arange(1, len(scores) + 1)
            target_rank[i] = ranks[-1]

# ---------- 1) remaining vs all-candidates 분포 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, vals, name, color in zip(
    axes, [J_rc, O_rc, COV_rc],
    ['Jaccard(R, C)', 'Overlap-min(R, C)', 'Coverage |R∩C|/|R|'],
    ['#1f77b4', '#2ca02c', '#9467bd'],
):
    v = vals[~np.isnan(vals)]
    ax.hist(v, bins=40, color=color, edgecolor='white')
    ax.axvline(v.mean(),     color='red',   ls='--', lw=1, label=f'mean={v.mean():.3f}')
    ax.axvline(np.median(v), color='black', ls=':',  lw=1, label=f'median={np.median(v):.3f}')
    ax.set_xlim(0, 1); ax.set_xlabel(name); ax.set_title(name); ax.legend(fontsize=9)
axes[0].set_ylabel('rows')
fig.suptitle(f'remaining tokens (task−history)  ↔  candidate_elements (n={n})')
fig.tight_layout()
fig.savefig(OUT / 'remaining_vs_candidates.png', dpi=130)
plt.close(fig)

# ---------- 2) 정답 candidate vs 비정답 candidate (개별 비교, 공평) ----------
# others 는 비정답 candidate 들의 *평균 개별 점수*. 합집합과의 비교가 아니므로
# 크기 효과 없이 candidate 1개당 점수가 동등하게 비교됨.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, vt, vo, name in zip(
    axes,
    [J_rt,    COV_rt],
    [J_ro_mean, COV_ro_mean],
    ['Jaccard(R, candidate)', 'Coverage |R∩candidate|/|R|'],
):
    vt_ = vt[~np.isnan(vt)]
    vo_ = vo[~np.isnan(vo)]
    bins = np.linspace(0, 1, 41)
    ax.hist(vo_, bins=bins, alpha=0.55,
            label=f'non-target (mean over candidates)  μ={vo_.mean():.3f}', color='#888')
    ax.hist(vt_, bins=bins, alpha=0.65,
            label=f'target  μ={vt_.mean():.3f}', color='#d62728')
    ax.set_xlabel(name); ax.set_title(name); ax.legend(fontsize=9)
fig.suptitle('per-candidate comparison: target vs typical non-target')
fig.tight_layout()
fig.savefig(OUT / 'remaining_vs_target_vs_others.png', dpi=130)
plt.close(fig)

# ---------- 2b) 정답 순위 분포 ----------
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ranks_valid = target_rank[~np.isnan(target_rank)].astype(int)
max_rank = int(ranks_valid.max())
ax.hist(ranks_valid, bins=np.arange(1, max_rank + 2) - 0.5,
        color='#d62728', edgecolor='white')
top1 = (ranks_valid == 1).mean() * 100
top3 = (ranks_valid <= 3).mean() * 100
ax.set_xlabel('rank of target by Coverage(R, candidate)  — 1 = best')
ax.set_ylabel('rows')
ax.set_title(f'target rank among candidates  (top-1 = {top1:.1f}%, top-3 = {top3:.1f}%)')
fig.tight_layout()
fig.savefig(OUT / 'target_rank_distribution.png', dpi=130)
plt.close(fig)

# ---------- 3) remaining 크기 분포 ----------
fig, ax = plt.subplots(figsize=(8.5, 4.5))
hi = int(np.percentile(remaining_sz, 99)) + 1
ax.hist(remaining_sz, bins=np.arange(0, hi + 2) - 0.5, color='#4c72b0', edgecolor='white')
ax.set_xlabel('|remaining| = |task_tokens − history_tokens|')
ax.set_ylabel('rows')
ax.set_title(f'remaining-set size per row  (μ={remaining_sz.mean():.2f}, '
             f'empty={ (remaining_sz==0).sum()} rows)')
fig.tight_layout()
fig.savefig(OUT / 'remaining_size_distribution.png', dpi=130)
plt.close(fig)

# ---------- 4) 수치 요약 ----------
def _stats(v):
    v = v[~np.isnan(v)]
    return (f'  μ={v.mean():.4f}  median={np.median(v):.4f}  '
            f'p25={np.percentile(v,25):.4f}  p75={np.percentile(v,75):.4f}  '
            f'≥0.5={(v>=0.5).mean()*100:.1f}%  ≥0.3={(v>=0.3).mean()*100:.1f}%')

# 같은 행에서 target 의 개별 점수가 비정답들의 평균/최대보다 높은가?
valid = ~np.isnan(J_rt) & ~np.isnan(J_ro_mean)
J_target_gt_mean  = (J_rt[valid]   >  J_ro_mean[valid]).mean() * 100
J_target_gt_max   = (J_rt[valid]   >  J_ro_max[valid]).mean() * 100
J_target_ge_max   = (J_rt[valid]   >= J_ro_max[valid]).mean() * 100
COV_target_gt_mean = (COV_rt[valid]   >  COV_ro_mean[valid]).mean() * 100
COV_target_gt_max  = (COV_rt[valid]   >  COV_ro_max[valid]).mean() * 100
COV_target_ge_max  = (COV_rt[valid]   >= COV_ro_max[valid]).mean() * 100
ranks_valid = target_rank[~np.isnan(target_rank)].astype(int)
top1 = (ranks_valid == 1).mean() * 100
top3 = (ranks_valid <= 3).mean() * 100

lines = [
    f'rows compared: {n}  (rows with empty remaining: {(remaining_sz==0).sum()})',
    f'preprocessing: lowercase + [A-Za-z]start regex + stopword removal',
    f'  stopwords = sklearn ENGLISH_STOP_WORDS ({len(ENGLISH_STOP_WORDS)}) + domain {sorted(DOMAIN_STOP)}',
    '',
    '[set sizes per row]',
    f'  |task|       μ={task_sz.mean():.2f}  median={np.median(task_sz):.0f}',
    f'  |history|    μ={history_sz.mean():.2f}  median={np.median(history_sz):.0f}',
    f'  |remaining|  μ={remaining_sz.mean():.2f}  median={np.median(remaining_sz):.0f}  p95={np.percentile(remaining_sz,95):.0f}',
    f'  |candidates| μ={cand_sz.mean():.2f}  median={np.median(cand_sz):.0f}',
    '',
    '[remaining ↔ all candidates]',
    'Jaccard',     _stats(J_rc),
    'Overlap-min', _stats(O_rc),
    'Coverage',    _stats(COV_rc),
    '',
    '[remaining ↔ target candidate (single)]',
    'Jaccard',  _stats(J_rt),
    'Coverage', _stats(COV_rt),
    '',
    '[remaining ↔ non-target candidates — averaged per candidate]',
    'Jaccard  (mean over non-target candidates)',  _stats(J_ro_mean),
    'Jaccard  (max over non-target candidates)',   _stats(J_ro_max),
    'Coverage (mean over non-target candidates)',  _stats(COV_ro_mean),
    'Coverage (max over non-target candidates)',   _stats(COV_ro_max),
    '',
    '[fair per-candidate comparison: target vs typical non-target]',
    f'  Jaccard:  target > mean(others)  in {J_target_gt_mean:.1f}% of rows',
    f'  Jaccard:  target > max(others)   in {J_target_gt_max:.1f}% of rows',
    f'  Jaccard:  target ≥ max(others)   in {J_target_ge_max:.1f}% of rows',
    f'  Coverage: target > mean(others)  in {COV_target_gt_mean:.1f}% of rows',
    f'  Coverage: target > max(others)   in {COV_target_gt_max:.1f}% of rows',
    f'  Coverage: target ≥ max(others)   in {COV_target_ge_max:.1f}% of rows',
    '',
    '[target rank among candidates by Coverage(R, candidate)]',
    f'  top-1 = {top1:.1f}%   top-3 = {top3:.1f}%   '
    f'mean rank = {ranks_valid.mean():.2f}  median = {int(np.median(ranks_valid))}',
]
summary = '\n'.join(lines)
(OUT / 'remaining_vs_candidates_summary.txt').write_text(summary, encoding='utf-8')
print(summary.encode('utf-8', errors='replace').decode('utf-8'))

# ---------- 샘플 ----------
print('\n--- sample row 0 ---')
print('task     :', sorted(set(task_tokenized_array[0])))
print('history  :', sorted(set(history_tokenized_array[0])))
print('remaining:', sorted(set(remaining_tokenized_array[0])))
print('target   :', sorted(set(target_tokenized_array[0])))
print(f'Jaccard(R,target)={J_rt[0]:.3f}  Coverage(R,target)={COV_rt[0]:.3f}')
print(f'\n[saved] {OUT}')
