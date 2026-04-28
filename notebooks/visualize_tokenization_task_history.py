"""행별 토큰화 + 불용어 제거 → 행별 단어 유사도 시각화

- task_tokenized_array[i]    : i번째 행 task   컬럼의 (토큰화 + 불용어 제거) 단어 리스트
- history_tokenized_array[i] : i번째 행 history 컬럼의 (토큰화 + 불용어 제거) 단어 리스트
같은 인덱스 i 는 같은 행을 가리키므로 두 컬럼 단어 유사도를 행별로 비교할 수 있다.

산출물 (notebooks/viz_out/):
  - row_word_similarity.png         : Jaccard / Overlap / Dice 분포 히스토그램
  - row_word_similarity_scatter.png : (|task|, |history|) 산점도, 색=Jaccard
  - row_word_similarity_summary.txt : 수치 요약
"""
from pathlib import Path
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
mask = df['task'].notna() & df['history'].notna()
sub = df.loc[mask].reset_index(drop=True)
n = len(sub)

# ---------- 전처리: 토큰화 + 불용어 제거 ----------
DOMAIN_STOP = {'task', 'step'}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")

def tokenize(s: str) -> list[str]:
    return [t for t in (m.lower() for m in TOKEN_RE.findall(s))
            if t not in STOPWORDS and len(t) > 1]

# 행별 토큰 리스트를 np.array(dtype=object)로 보관
task_tokenized_array = np.array(
    [tokenize(s) for s in sub['task'].astype(str).tolist()],
    dtype=object,
)
history_tokenized_array = np.array(
    [tokenize(s) for s in sub['history'].astype(str).tolist()],
    dtype=object,
)
assert len(task_tokenized_array) == len(history_tokenized_array) == n
# task_tokenized_array[i] ↔ history_tokenized_array[i] 가 같은 행

# ---------- 행별 단어 유사도 ----------
# Jaccard  = |A∩B| / |A∪B|        : 집합 동일성, 0..1
# Overlap  = |A∩B| / min(|A|,|B|)  : 작은 쪽 기준 포함률
# Dice     = 2|A∩B| / (|A|+|B|)    : 평균 기준 겹침
jaccard = np.zeros(n)
overlap = np.zeros(n)
dice    = np.zeros(n)
inter_sz = np.zeros(n, dtype=int)
union_sz = np.zeros(n, dtype=int)

for i in range(n):
    A = set(task_tokenized_array[i])
    B = set(history_tokenized_array[i])
    inter = A & B
    union = A | B
    inter_sz[i] = len(inter)
    union_sz[i] = len(union)
    jaccard[i] = len(inter) / len(union) if union else 0.0
    overlap[i] = len(inter) / min(len(A), len(B)) if A and B else 0.0
    dice[i]    = 2 * len(inter) / (len(A) + len(B)) if (A or B) else 0.0

task_lens = np.array([len(set(t)) for t in task_tokenized_array])
hist_lens = np.array([len(set(t)) for t in history_tokenized_array])

# ---------- 1) 유사도 분포 히스토그램 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, vals, name, color in zip(
    axes,
    [jaccard, overlap, dice],
    ['Jaccard', 'Overlap (min)', 'Dice'],
    ['#1f77b4', '#2ca02c', '#d62728'],
):
    ax.hist(vals, bins=40, color=color, edgecolor='white')
    ax.axvline(vals.mean(),    color='red',   ls='--', lw=1, label=f'mean={vals.mean():.3f}')
    ax.axvline(np.median(vals), color='black', ls=':',  lw=1, label=f'median={np.median(vals):.3f}')
    ax.set_xlim(0, 1)
    ax.set_xlabel(name); ax.set_title(name)
    ax.legend(fontsize=9)
axes[0].set_ylabel('rows')
fig.suptitle(f'per-row word similarity (task ↔ history, n={n})')
fig.tight_layout()
fig.savefig(OUT / 'row_word_similarity.png', dpi=130)
plt.close(fig)

# ---------- 2) 토큰 수 vs 유사도 산점도 ----------
fig, ax = plt.subplots(figsize=(8.5, 6))
sc = ax.scatter(
    task_lens, hist_lens,
    c=jaccard, cmap='viridis', s=8, alpha=0.55, vmin=0, vmax=1,
)
ax.set_xlabel('|unique tokens in task|')
ax.set_ylabel('|unique tokens in history|')
ax.set_title('row-level token-set sizes (color = Jaccard)')
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Jaccard similarity')
fig.tight_layout()
fig.savefig(OUT / 'row_word_similarity_scatter.png', dpi=130)
plt.close(fig)

# ---------- 3) 수치 요약 ----------
def stats(v):
    return (f'  μ={v.mean():.4f}  median={np.median(v):.4f}  '
            f'p25={np.percentile(v,25):.4f}  p75={np.percentile(v,75):.4f}  '
            f'≥0.5={(v>=0.5).mean()*100:.1f}%  ≥0.3={(v>=0.3).mean()*100:.1f}%')

lines = [
    f'rows compared (task & history both non-null): {n}',
    f'rows with missing history                    : {(~mask).sum()}',
    f'preprocessing: lowercase + [A-Za-z]start regex + stopword removal',
    f'  stopwords = sklearn ENGLISH_STOP_WORDS ({len(ENGLISH_STOP_WORDS)}) + domain {sorted(DOMAIN_STOP)}',
    '',
    '[unique-token count per row]',
    f'  task    : μ={task_lens.mean():.2f}  median={np.median(task_lens):.0f}  p95={np.percentile(task_lens,95):.0f}',
    f'  history : μ={hist_lens.mean():.2f}  median={np.median(hist_lens):.0f}  p95={np.percentile(hist_lens,95):.0f}',
    '',
    '[per-row word similarity]',
    f'Jaccard      |A∩B|/|A∪B|',
    stats(jaccard),
    f'Overlap(min) |A∩B|/min(|A|,|B|)',
    stats(overlap),
    f'Dice         2|A∩B|/(|A|+|B|)',
    stats(dice),
    '',
    f'[intersection size]  μ={inter_sz.mean():.2f}  median={np.median(inter_sz):.0f}  max={inter_sz.max()}',
    f'[union size]         μ={union_sz.mean():.2f}  median={np.median(union_sz):.0f}  max={union_sz.max()}',
    f'[rows with empty intersection] {(inter_sz==0).sum()}  ({(inter_sz==0).mean()*100:.2f}%)',
]
summary = '\n'.join(lines)
(OUT / 'row_word_similarity_summary.txt').write_text(summary, encoding='utf-8')
print(summary)
print(f'\n[saved] {OUT}')

# ---------- 샘플 확인 ----------
print('\n--- sample row 0 ---')
print(f'task tokens    ({len(task_tokenized_array[0])}): {task_tokenized_array[0][:20]}')
print(f'history tokens ({len(history_tokenized_array[0])}): {history_tokenized_array[0][:20]}')
print(f'Jaccard={jaccard[0]:.3f}  Overlap={overlap[0]:.3f}  Dice={dice[0]:.3f}')
