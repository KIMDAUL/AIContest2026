"""task vs history 텍스트 분포 비교 시각화

생성물 (notebooks/viz_out/):
  - task_history_length.png    : 글자/단어 길이 분포 비교
  - task_history_topwords.png  : 상위 단어 빈도 비교
  - task_history_similarity.png: 행별 TF-IDF 코사인 유사도 분포
  - task_history_summary.txt   : 어휘 겹침/JS divergence 등 수치 요약
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'viz_out'
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / 'data' / 'train.csv', encoding='utf-8', encoding_errors='replace')
mask = df['history'].notna() & df['task'].notna()
task = df.loc[mask, 'task'].astype(str).tolist()
hist = df.loc[mask, 'history'].astype(str).tolist()
n = len(task)

# 전처리: 소문자화 + 알파벳 시작 토큰만 추출 + 영어 불용어 제거
# (sklearn ENGLISH_STOP_WORDS 기준 — 'task' 자체가 두 컬럼에서 거의 동일 빈도이고
#  의미 비교에 도움이 안 되므로 도메인 불용어로 추가)
DOMAIN_STOP = {'task', 'step'}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
def tokenize(s: str, *, drop_stop: bool = True):
    toks = (t.lower() for t in TOKEN_RE.findall(s))
    if drop_stop:
        return [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return [t for t in toks if len(t) > 1]

# ---------- 1) 길이 분포 ----------
task_chars = np.array([len(s) for s in task])
hist_chars = np.array([len(s) for s in hist])
# 길이는 원시 토큰 수 (불용어 포함) — "텍스트가 얼마나 긴가"를 보는 지표
task_words = np.array([len(tokenize(s, drop_stop=False)) for s in task])
hist_words = np.array([len(tokenize(s, drop_stop=False)) for s in hist])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (a, b, label) in zip(axes, [
    (task_chars, hist_chars, 'characters'),
    (task_words, hist_words, 'words'),
]):
    hi = int(np.percentile(np.concatenate([a, b]), 99))
    bins = np.linspace(0, hi, 60)
    ax.hist(a, bins=bins, alpha=0.55, label=f'task (μ={a.mean():.0f})', color='#1f77b4')
    ax.hist(b, bins=bins, alpha=0.55, label=f'history (μ={b.mean():.0f})', color='#ff7f0e')
    ax.set_xlabel(label); ax.set_ylabel('count'); ax.legend()
    ax.set_title(f'length in {label}')
fig.suptitle(f'task vs history — length distribution (n={n})')
fig.tight_layout()
fig.savefig(OUT / 'task_history_length.png', dpi=130)
plt.close(fig)

# ---------- 2) 어휘 통계 + 상위 단어 ----------
task_tokens = [tok for s in task for tok in tokenize(s)]
hist_tokens = [tok for s in hist for tok in tokenize(s)]
task_cnt = Counter(task_tokens)
hist_cnt = Counter(hist_tokens)
vocab_task = set(task_cnt)
vocab_hist = set(hist_cnt)
inter = vocab_task & vocab_hist
union = vocab_task | vocab_hist
jaccard = len(inter) / max(1, len(union))

# JS divergence (전체 토큰 분포)
all_vocab = sorted(union)
idx = {w: i for i, w in enumerate(all_vocab)}
p = np.zeros(len(all_vocab)); q = np.zeros(len(all_vocab))
for w, c in task_cnt.items(): p[idx[w]] = c
for w, c in hist_cnt.items(): q[idx[w]] = c
p = p / p.sum(); q = q / q.sum()
m = 0.5 * (p + q)
def kl(a, b):
    mask = a > 0
    return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
js = 0.5 * kl(p, m) + 0.5 * kl(q, m)  # 0..1 (log2)

# 상위 단어 비교 (양쪽 합산 빈도 기준 상위 25개)
combined = Counter()
for w in union:
    combined[w] = task_cnt.get(w, 0) + hist_cnt.get(w, 0)
top = [w for w, _ in combined.most_common(25)]
t_freq = np.array([task_cnt.get(w, 0) for w in top]) / max(1, sum(task_cnt.values()))
h_freq = np.array([hist_cnt.get(w, 0) for w in top]) / max(1, sum(hist_cnt.values()))

fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(top))
ax.barh(y - 0.2, t_freq, height=0.4, color='#1f77b4', label='task')
ax.barh(y + 0.2, h_freq, height=0.4, color='#ff7f0e', label='history')
ax.set_yticks(y); ax.set_yticklabels(top)
ax.invert_yaxis()
ax.set_xlabel('relative frequency')
ax.set_title('top 25 tokens — relative frequency in each column')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'task_history_topwords.png', dpi=130)
plt.close(fig)

# ---------- 3) 행별 TF-IDF 코사인 유사도 ----------
vec = TfidfVectorizer(
    token_pattern=r"[A-Za-z][A-Za-z0-9_-]+",
    lowercase=True,
    min_df=2,
    stop_words=list(STOPWORDS),
)
vec.fit(task + hist)
T = vec.transform(task); H = vec.transform(hist)
# 행별 dot product (정규화된 TF-IDF → 코사인)
T_norm = T.multiply(1 / (np.sqrt(T.multiply(T).sum(axis=1)) + 1e-9))
H_norm = H.multiply(1 / (np.sqrt(H.multiply(H).sum(axis=1)) + 1e-9))
sims = np.asarray(T_norm.multiply(H_norm).sum(axis=1)).ravel()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(sims, bins=50, color='#4c72b0', edgecolor='white')
ax.axvline(sims.mean(), color='red', linestyle='--', label=f'mean={sims.mean():.3f}')
ax.axvline(np.median(sims), color='black', linestyle=':', label=f'median={np.median(sims):.3f}')
ax.set_xlabel('TF-IDF cosine similarity (per row)')
ax.set_ylabel('count')
ax.set_title(f'per-row task↔history similarity (n={n})')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'task_history_similarity.png', dpi=130)
plt.close(fig)

# ---------- 4) 수치 요약 ----------
lines = [
    f'rows compared (both non-null): {n}',
    f'rows with missing history    : {(~mask).sum()}',
    f'preprocessing: lowercase + alpha-start token regex + stopword removal',
    f'  stopwords = sklearn ENGLISH_STOP_WORDS ({len(ENGLISH_STOP_WORDS)}) + domain {sorted(DOMAIN_STOP)}',
    f'  (length metrics below use RAW tokens; vocab/freq/cosine use FILTERED tokens)',
    '',
    '[length]',
    f'  task    chars  μ={task_chars.mean():7.1f}  median={np.median(task_chars):7.0f}  p95={np.percentile(task_chars,95):.0f}',
    f'  history chars  μ={hist_chars.mean():7.1f}  median={np.median(hist_chars):7.0f}  p95={np.percentile(hist_chars,95):.0f}',
    f'  task    words  μ={task_words.mean():7.1f}  median={np.median(task_words):7.0f}  p95={np.percentile(task_words,95):.0f}',
    f'  history words  μ={hist_words.mean():7.1f}  median={np.median(hist_words):7.0f}  p95={np.percentile(hist_words,95):.0f}',
    '',
    '[vocabulary]',
    f'  |V_task|             = {len(vocab_task)}',
    f'  |V_history|          = {len(vocab_hist)}',
    f'  |V_task ∩ V_history| = {len(inter)}',
    f'  Jaccard(V_task, V_history) = {jaccard:.4f}',
    f'  JS divergence (unigram, log2) = {js:.4f}   (0=identical, 1=disjoint)',
    '',
    '[per-row TF-IDF cosine similarity]',
    f'  mean   = {sims.mean():.4f}',
    f'  median = {np.median(sims):.4f}',
    f'  p25/p75 = {np.percentile(sims,25):.4f} / {np.percentile(sims,75):.4f}',
    f'  share ≥ 0.5 = {(sims >= 0.5).mean()*100:.1f}%',
    f'  share ≥ 0.3 = {(sims >= 0.3).mean()*100:.1f}%',
    '',
    '[top-10 only-in-task tokens (by task freq)]',
    '  ' + ', '.join(w for w, _ in Counter({w: c for w, c in task_cnt.items() if w not in vocab_hist}).most_common(10)),
    '',
    '[top-10 only-in-history tokens (by history freq)]',
    '  ' + ', '.join(w for w, _ in Counter({w: c for w, c in hist_cnt.items() if w not in vocab_task}).most_common(10)),
]
summary = '\n'.join(lines)
(OUT / 'task_history_summary.txt').write_text(summary, encoding='utf-8')
print(summary)
print(f'\n[saved] {OUT}')
