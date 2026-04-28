"""베이스라인: Coverage(remaining, candidate) argmax 로 target_id 예측

remaining = set(task_tokens) - set(history_tokens) (전처리 후)
점수 = |remaining ∩ candidate_tokens| / |remaining|
       (remaining 이 비면 fallback 으로 Jaccard(task, candidate) 사용)
가장 점수 높은 candidate 의 candidate_id 를 예측. 동률은 candidate 등장 순서로
첫 번째 (보통 DOM 순) 채택.

train.csv 의 정답(target_id, op)으로 정확도 측정.

비교 baseline:
  - random           : candidate 중 무작위 1개 (이론상 1/|candidates|)
  - first            : 첫 candidate 고정
  - coverage         : 본 휴리스틱 (위)
  - coverage+tag-op  : op 까지 함께 예측 (tag → op 규칙) — combined 정확도

산출물 (notebooks/viz_out/):
  - baseline_coverage_summary.txt
  - baseline_coverage_topk.png        : top-1/3/5 accuracy
  - baseline_coverage_by_remaining.png: |remaining| 별 정확도
"""
from pathlib import Path
import json, re
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
# target_id, op 가 정답 — 둘 다 있어야 평가 가능
mask = df[['task','history','candidate_elements','target_id']].notna().all(axis=1)
sub = df.loc[mask].reset_index(drop=True)
n = len(sub)

# ---------- 전처리 ----------
DOMAIN_STOP = {'task', 'step', 'click', 'type', 'select', 'enter'}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
def tokenize(s: str):
    return [t for t in (m.lower() for m in TOKEN_RE.findall(s or ''))
            if t not in STOPWORDS and len(t) > 1]
def cand_text(c):
    return (c.get('text') or '') + ' ' + (c.get('attrs') or '')

# ---------- 후보 / 정답 ----------
cand_parsed = [json.loads(s) for s in sub['candidate_elements'].astype(str)]
task_tok    = [set(tokenize(s)) for s in sub['task'].astype(str)]
hist_tok    = [set(tokenize(s)) for s in sub['history'].fillna('').astype(str)]
remaining   = [t - h for t, h in zip(task_tok, hist_tok)]

target_ids = sub['target_id'].astype(str).tolist()
ops_true   = sub['op'].astype(str).tolist() if 'op' in sub.columns else None

# ---------- 점수 함수 ----------
def coverage(R: set, C: set) -> float:
    if not R: return 0.0
    return len(R & C) / len(R)
def jaccard(A: set, B: set) -> float:
    if not A and not B: return 0.0
    return len(A & B) / len(A | B)

# ---------- 행별 예측 ----------
pred_target_cov   = []
pred_target_first = []
pred_target_rand  = []
target_in_topk    = {1: [], 3: [], 5: []}     # cov 기준
score_per_row     = []                         # (점수 배열, target_idx) 디버그용
remaining_sz      = np.array([len(r) for r in remaining])
cand_count        = np.array([len(c) for c in cand_parsed])

rng = np.random.default_rng(0)

for i in range(n):
    cs   = cand_parsed[i]
    cids = [c['candidate_id'] for c in cs]
    R    = remaining[i]
    C_each = [set(tokenize(cand_text(c))) for c in cs]

    # 본 휴리스틱
    if R:
        scores = np.array([coverage(R, C) for C in C_each])
    else:
        # remaining 이 비면 task 전체로 fallback
        T = task_tok[i]
        scores = np.array([jaccard(T, C) for C in C_each])
    # argmax (동률 시 첫 번째 = 등장 순서)
    best = int(np.argmax(scores))
    pred_target_cov.append(cids[best])
    pred_target_first.append(cids[0])
    pred_target_rand.append(cids[rng.integers(len(cids))])

    # top-k (점수 내림차순; 동률은 등장 순)
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    target_idx = cids.index(target_ids[i]) if target_ids[i] in cids else -1
    rank = order.index(target_idx) + 1 if target_idx >= 0 else len(scores) + 1
    for k in target_in_topk:
        target_in_topk[k].append(int(rank <= k))
    score_per_row.append((scores, target_idx))

pred_target_cov   = np.array(pred_target_cov)
pred_target_first = np.array(pred_target_first)
pred_target_rand  = np.array(pred_target_rand)
target_arr        = np.array(target_ids)

acc_cov   = (pred_target_cov   == target_arr).mean() * 100
acc_first = (pred_target_first == target_arr).mean() * 100
acc_rand  = (pred_target_rand  == target_arr).mean() * 100
top1 = np.mean(target_in_topk[1]) * 100
top3 = np.mean(target_in_topk[3]) * 100
top5 = np.mean(target_in_topk[5]) * 100

# ---------- op 예측 (tag → op 단순 규칙) ----------
TAG2OP = {
    'input':    'TYPE',
    'textarea': 'TYPE',
    'select':   'SELECT',
    'button':   'CLICK',
    'a':        'CLICK',
    'link':     'CLICK',
    'checkbox': 'CHECK',
    'radio':    'CHECK',
}
def tag_to_op(tag: str, default='CLICK'):
    return TAG2OP.get((tag or '').lower(), default)

acc_op = None
acc_combined = None
op_dist = None
if ops_true is not None:
    pred_op = []
    for i, tid in enumerate(pred_target_cov):
        cs = cand_parsed[i]
        tag = next((c.get('tag') for c in cs if c['candidate_id'] == tid), '')
        pred_op.append(tag_to_op(tag))
    pred_op = np.array(pred_op)
    op_true_arr = np.array(ops_true)
    acc_op = (pred_op == op_true_arr).mean() * 100
    acc_combined = ((pred_target_cov == target_arr) & (pred_op == op_true_arr)).mean() * 100
    # op 분포
    op_dist = pd.Series(op_true_arr).value_counts().to_dict()

# ---------- |remaining| 구간별 정확도 ----------
def acc_by_bin(values, bins):
    out = []
    for lo, hi in bins:
        m = (remaining_sz >= lo) & (remaining_sz <= hi)
        if m.sum() == 0:
            out.append((f'{lo}-{hi}', 0, 0.0)); continue
        out.append((f'{lo}-{hi}', int(m.sum()),
                    float((pred_target_cov[m] == target_arr[m]).mean() * 100)))
    return out
bins = [(0,0),(1,3),(4,6),(7,9),(10,14),(15,99)]
bin_acc = acc_by_bin(pred_target_cov, bins)

# ---------- 시각화 ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ['random', 'first', 'top-1\n(coverage)', 'top-3', 'top-5']
vals   = [acc_rand, acc_first, top1, top3, top5]
colors = ['#bbb','#888','#1f77b4','#1f77b4','#1f77b4']
bars = ax.bar(labels, vals, color=colors, edgecolor='black')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('accuracy (%)'); ax.set_ylim(0, max(vals)+10)
ax.set_title(f'target_id baseline accuracy  (n={n})')
fig.tight_layout(); fig.savefig(OUT / 'baseline_coverage_topk.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
xs = [b[0] for b in bin_acc]
ys = [b[2] for b in bin_acc]
ns = [b[1] for b in bin_acc]
bars = ax.bar(xs, ys, color='#4c72b0', edgecolor='black')
for b, v, c in zip(bars, ys, ns):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.1f}%\n(n={c})',
            ha='center', fontsize=9)
ax.set_xlabel('|remaining| bin'); ax.set_ylabel('top-1 accuracy (%)')
ax.set_ylim(0, max(ys)+12)
ax.set_title('coverage top-1 accuracy by |remaining| size')
fig.tight_layout(); fig.savefig(OUT / 'baseline_coverage_by_remaining.png', dpi=130); plt.close(fig)

# ---------- 요약 ----------
lines = [
    f'evaluated rows: {n}   (rows with empty remaining: {(remaining_sz==0).sum()})',
    f'mean |candidates| per row: {cand_count.mean():.2f}',
    f'random baseline (1/|candidates|): {(1/cand_count).mean()*100:.2f}%',
    '',
    '[target_id accuracy]',
    f'  random           : {acc_rand:.2f}%',
    f'  first-candidate  : {acc_first:.2f}%',
    f'  coverage top-1   : {top1:.2f}%   (== argmax(coverage))',
    f'  coverage top-3   : {top3:.2f}%',
    f'  coverage top-5   : {top5:.2f}%',
    '',
    '[op accuracy via tag→op rule]'
        if acc_op is not None else '[op] (skipped: no op column)',
]
if acc_op is not None:
    lines += [
        f'  op rule = {TAG2OP}',
        f'  op accuracy (given predicted target_id): {acc_op:.2f}%',
        f'  combined (target_id AND op correct)    : {acc_combined:.2f}%',
        f'  op label distribution (true) : {op_dist}',
    ]
lines += ['', '[top-1 accuracy by |remaining|]']
for label, c, a in bin_acc:
    lines.append(f'  |R|={label:>5s}  n={c:5d}  acc={a:5.2f}%')

summary = '\n'.join(lines)
(OUT / 'baseline_coverage_summary.txt').write_text(summary, encoding='utf-8')
print(summary.encode('utf-8', errors='replace').decode('utf-8'))
print(f'\n[saved] {OUT}')
