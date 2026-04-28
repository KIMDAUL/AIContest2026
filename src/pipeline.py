"""3-stage pipeline (v2): op classifier + target_id ranker (DOM-aware features)
+ value heuristic.

Honest CV uses GroupKFold by site_token (test sites do not overlap with train).
Run from anywhere: python src/pipeline.py
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'data'

OP_MAP = {'CLICK': 0, 'TYPE': 1, 'SELECT': 2}
OP_INV = {v: k for k, v in OP_MAP.items()}
TAG_LIST = ['a', 'button', 'input', 'select', 'textarea', 'div', 'span', 'li',
            'option', 'label', 'p', 'img', 'form', 'h1', 'h2', 'h3', 'h4', 'ul']

# Tag → op affinity (used as cross-feature with op probabilities)
CLICK_TAGS = {'a', 'button', 'div', 'span', 'li', 'img', 'p', 'h1', 'h2', 'h3', 'h4'}
TYPE_TAGS = {'input', 'textarea'}
SELECT_TAGS = {'select'}

WORD_RE = re.compile(r'\w+')

OP_KEYWORDS_TYPE = ['type', 'enter', 'input', 'fill', 'write', 'name', 'email',
                    'address', 'search for', 'phone', 'message']
OP_KEYWORDS_SELECT = ['select', 'choose', 'pick', 'set the', 'option', 'dropdown']
OP_KEYWORDS_CLICK = ['click', 'go to', 'open', 'navigate', 'button', 'view']


# ────────────────────────────────────────────────────────────────────
# Loading & basic prep
# ────────────────────────────────────────────────────────────────────
def load():
    tr = pd.read_csv(DATA / 'train.csv')
    te = pd.read_csv(DATA / 'test.csv')
    tr['history'] = tr['history'].fillna('')
    te['history'] = te['history'].fillna('')
    tr['value'] = tr['value'].fillna('')
    tr['cands'] = tr['candidate_elements'].apply(json.loads)
    te['cands'] = te['candidate_elements'].apply(json.loads)
    return tr, te


# ────────────────────────────────────────────────────────────────────
# Stage 1 — op classifier
# ────────────────────────────────────────────────────────────────────
def row_features(row: pd.Series) -> dict:
    cands = row['cands']
    tags = [c['tag'] for c in cands]
    task_low = str(row['task']).lower()
    hist_low = str(row['history']).lower()
    feat = {
        'n_cands': len(cands),
        'task_len': len(row['task']),
        'task_words': len(WORD_RE.findall(row['task'])),
        'hist_len': len(row['history']),
        'hist_words': len(WORD_RE.findall(row['history'])),
        'task_has_dquote': int('"' in row['task']),
        'task_has_squote': int("'" in row['task']),
        'task_has_digit': int(bool(re.search(r'\d', row['task']))),
        'task_kw_type': sum(k in task_low for k in OP_KEYWORDS_TYPE),
        'task_kw_select': sum(k in task_low for k in OP_KEYWORDS_SELECT),
        'task_kw_click': sum(k in task_low for k in OP_KEYWORDS_CLICK),
        'hist_n_type': hist_low.count('type'),
        'hist_n_select': hist_low.count('select'),
        'hist_n_click': hist_low.count('click'),
    }
    for t in TAG_LIST:
        feat[f'cand_n_{t}'] = tags.count(t)
        feat[f'cand_has_{t}'] = int(t in tags)
    return feat


def build_op_features(df: pd.DataFrame, task_vec=None, hist_vec=None, fit=False):
    feats = pd.DataFrame([row_features(r) for _, r in df.iterrows()])
    if fit:
        task_vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                   stop_words='english', min_df=2)
        hist_vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 2),
                                   stop_words='english', min_df=2)
        task_tfidf = task_vec.fit_transform(df['task'].astype(str))
        hist_tfidf = hist_vec.fit_transform(df['history'].astype(str))
    else:
        task_tfidf = task_vec.transform(df['task'].astype(str))
        hist_tfidf = hist_vec.transform(df['history'].astype(str))
    X = hstack([csr_matrix(feats.values, dtype=np.float32), task_tfidf, hist_tfidf]).tocsr()
    return X, task_vec, hist_vec


def fit_op_classifier(X, y):
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                             num_leaves=63, n_jobs=-1, random_state=42,
                             verbosity=-1)
    clf.fit(X, y)
    return clf


def oof_op_proba(df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """OOF op-class probabilities on `df` using GroupKFold by site_token."""
    y = df['op'].map(OP_MAP).values
    out = np.zeros((len(df), 3), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(df, groups=df['site_token']):
        sub_tr = df.iloc[tr_idx].reset_index(drop=True)
        sub_va = df.iloc[va_idx].reset_index(drop=True)
        Xtr, tv, hv = build_op_features(sub_tr, fit=True)
        Xva, _, _ = build_op_features(sub_va, tv, hv)
        clf = fit_op_classifier(Xtr, y[tr_idx])
        out[va_idx] = clf.predict_proba(Xva)
    return out


# ────────────────────────────────────────────────────────────────────
# Stage 2 — target_id ranker (richer features)
# ────────────────────────────────────────────────────────────────────
HIST_STEP_RE = re.compile(r'Step\s*\d+:\s*\[([^\]]+)\]\s*([^>]*?)\s*->\s*(\w+)(?::\s*([^|]+))?', re.S)


def parse_history(h: str):
    """Return list of (tag, text, op, value) tuples from history string."""
    if not h or pd.isna(h):
        return []
    return [(m.group(1).strip(), m.group(2).strip(),
             m.group(3).strip(), (m.group(4) or '').strip())
            for m in HIST_STEP_RE.finditer(h)]


def explode_candidates(df: pd.DataFrame, with_label: bool) -> pd.DataFrame:
    rows = []
    for i, r in df.iterrows():
        hist = parse_history(r['history'])
        last = hist[-1] if hist else ('', '', '', '')
        hist_tags = ' '.join(t for t, _, _, _ in hist)
        hist_texts = ' '.join(tx for _, tx, _, _ in hist)
        hist_ops = ' '.join(o for _, _, o, _ in hist)
        for j, c in enumerate(r['cands']):
            rec = dict(row_id=i, id=r['id'],
                       candidate_id=c['candidate_id'],
                       tag=c['tag'], text=c['text'] or '', attrs=c['attrs'] or '',
                       cand_pos=j, n_cands=len(r['cands']),
                       task=r['task'], history=r['history'],
                       hist_n_steps=len(hist),
                       last_tag=last[0], last_text=last[1], last_op=last[2],
                       hist_tags=hist_tags, hist_texts=hist_texts, hist_ops=hist_ops)
            if with_label:
                rec['is_target'] = int(c['candidate_id'] == r['target_id'])
            rows.append(rec)
    return pd.DataFrame(rows)


def overlap_ratio(a: str, b: str) -> float:
    sa = set(WORD_RE.findall(str(a).lower()))
    sb = set(WORD_RE.findall(str(b).lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sb)


def token_coverage(task: str, text: str) -> float:
    """Fraction of task tokens present in element text+attrs."""
    sa = set(WORD_RE.findall(str(task).lower()))
    sb = set(WORD_RE.findall(str(text).lower()))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def per_row_norm(values: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rank, z-score, fraction-of-max) within each row group."""
    df = pd.DataFrame({'v': values, 'r': row_ids})
    g = df.groupby('r')['v']
    rank = g.rank(method='min', ascending=False).values
    n_per_row = df['r'].map(df.groupby('r').size()).values
    rank_norm = (rank - 1) / np.maximum(n_per_row - 1, 1)
    mean = df['r'].map(g.mean()).values
    std = df['r'].map(g.std()).fillna(1.0).values
    std = np.where(std < 1e-9, 1.0, std)
    z = (values - mean) / std
    mx = df['r'].map(g.max()).values
    frac_max = values / np.where(mx == 0, 1, mx)
    return rank_norm.astype(np.float32), z.astype(np.float32), frac_max.astype(np.float32)


def compute_tag_priors(cand_df: pd.DataFrame, train_op: pd.Series) -> dict:
    """For each op, returns dict[tag → P(target has this tag | op)] (Laplace-smoothed)."""
    # cand_df must have is_target column and row_id; train_op is op per row_id
    df = cand_df.copy()
    df['op'] = df['row_id'].map(dict(enumerate(train_op.values))) if not isinstance(train_op, dict) else df['row_id'].map(train_op)
    targets = df[df['is_target'] == 1]
    priors = {}
    alpha = 1.0
    for op in OP_MAP.keys():
        sub = targets[targets['op'] == op]
        total = len(sub)
        counts = sub['tag'].value_counts().to_dict()
        # smoothed
        all_tags = set(df['tag'].unique())
        n_tags = len(all_tags)
        priors[op] = {t: (counts.get(t, 0) + alpha) / (total + alpha * n_tags) for t in all_tags}
    return priors


def cand_features(cand_df: pd.DataFrame, op_proba_per_row: np.ndarray,
                  tag_priors: dict | None = None,
                  word_vec=None, char_vec=None, fit=False):
    """Return features for ranker.

    op_proba_per_row: (n_rows, 3) op class probabilities for each unique row,
    aligned with cand_df.row_id.
    """
    feat = pd.DataFrame(index=cand_df.index)
    for t in TAG_LIST:
        feat[f'tag_{t}'] = (cand_df['tag'] == t).astype(np.int8)
    feat['cand_pos'] = cand_df['cand_pos'].astype(np.int16)
    feat['cand_pos_rel'] = (cand_df['cand_pos'] / cand_df['n_cands'].clip(lower=1)).astype(np.float32)
    feat['n_cands'] = cand_df['n_cands'].astype(np.int16)
    feat['text_len'] = cand_df['text'].str.len().fillna(0).astype(np.int16)
    feat['attrs_len'] = cand_df['attrs'].str.len().fillna(0).astype(np.int16)
    feat['has_aria'] = cand_df['attrs'].str.contains('aria_label', na=False).astype(np.int8)
    feat['has_role'] = cand_df['attrs'].str.contains('role=', na=False).astype(np.int8)
    feat['has_alt'] = cand_df['attrs'].str.contains('alt=', na=False).astype(np.int8)
    feat['has_text'] = (cand_df['text'].str.len() > 0).astype(np.int8)

    feat['ovl_task_text'] = [overlap_ratio(t, x) for t, x in zip(cand_df['task'], cand_df['text'])]
    feat['ovl_task_attrs'] = [overlap_ratio(t, x) for t, x in zip(cand_df['task'], cand_df['attrs'])]
    feat['ovl_hist_text'] = [overlap_ratio(t, x) for t, x in zip(cand_df['history'], cand_df['text'])]
    feat['cov_task_text'] = [token_coverage(t, x) for t, x in zip(cand_df['task'], cand_df['text'])]
    feat['cov_task_attrs'] = [token_coverage(t, x) for t, x in zip(cand_df['task'], cand_df['attrs'])]

    # History-aware features
    feat['hist_n_steps'] = cand_df['hist_n_steps'].astype(np.int16)
    feat['ovl_lasttext_text'] = [overlap_ratio(a, b) for a, b in zip(cand_df['last_text'], cand_df['text'])]
    feat['ovl_histtext_text'] = [overlap_ratio(a, b) for a, b in zip(cand_df['hist_texts'], cand_df['text'])]
    # Task entity NOT in history → likely the next target
    cand_df_text_lower = cand_df['text'].str.lower().fillna('')
    cand_df_hist_lower = cand_df['hist_texts'].str.lower().fillna('')
    cand_df_task_lower = cand_df['task'].str.lower().fillna('')
    def remaining_overlap(task, hist, text):
        t = set(WORD_RE.findall(str(task)))
        h = set(WORD_RE.findall(str(hist)))
        e = set(WORD_RE.findall(str(text)))
        rem = t - h
        if not rem or not e:
            return 0.0
        return len(rem & e) / max(len(e), 1)
    feat['cov_task_minus_hist_text'] = [remaining_overlap(t, h, e)
                                        for t, h, e in zip(cand_df_task_lower,
                                                           cand_df_hist_lower,
                                                           cand_df_text_lower)]
    # Candidate's tag matches last action's tag
    feat['same_tag_as_last'] = (cand_df['tag'] == cand_df['last_tag']).astype(np.int8)
    # Last op was X
    for op in ['CLICK', 'TYPE', 'SELECT']:
        feat[f'last_op_{op}'] = (cand_df['last_op'] == op).astype(np.int8)

    elem_text = (cand_df['text'].fillna('') + ' ' + cand_df['attrs'].fillna('')).tolist()
    task_text = cand_df['task'].fillna('').tolist()
    hist_text = cand_df['history'].fillna('').tolist()

    if fit:
        word_vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                   stop_words='english', min_df=2)
        word_vec.fit(elem_text + task_text)
        char_vec = TfidfVectorizer(max_features=8000, analyzer='char_wb',
                                   ngram_range=(3, 5), min_df=2)
        char_vec.fit(elem_text + task_text)

    Et_w = normalize(word_vec.transform(elem_text))
    Tt_w = normalize(word_vec.transform(task_text))
    Ht_w = normalize(word_vec.transform(hist_text))
    feat['cos_w_task_elem'] = np.asarray(Et_w.multiply(Tt_w).sum(axis=1)).ravel().astype(np.float32)
    feat['cos_w_hist_elem'] = np.asarray(Et_w.multiply(Ht_w).sum(axis=1)).ravel().astype(np.float32)

    Et_c = normalize(char_vec.transform(elem_text))
    Tt_c = normalize(char_vec.transform(task_text))
    feat['cos_c_task_elem'] = np.asarray(Et_c.multiply(Tt_c).sum(axis=1)).ravel().astype(np.float32)

    # Per-row normalizations on the 3 cosine features
    rids = cand_df['row_id'].values
    for col in ['cos_w_task_elem', 'cos_w_hist_elem', 'cos_c_task_elem',
                'ovl_task_text', 'cov_task_text']:
        rk, z, fm = per_row_norm(feat[col].values, rids)
        feat[f'{col}_rank'] = rk
        feat[f'{col}_z'] = z
        feat[f'{col}_fmax'] = fm

    # Tag rarity within row (1 if only candidate of this tag) — needed before priors
    tag_counts = cand_df.groupby(['row_id', 'tag'])['tag'].transform('count').values
    feat['tag_count_in_row'] = tag_counts.astype(np.int16)
    feat['tag_unique_in_row'] = (tag_counts == 1).astype(np.int8)

    # Op-tag affinity: row-level op probabilities × tag indicator
    tag_arr = cand_df['tag'].values
    is_click = np.array([t in CLICK_TAGS for t in tag_arr], dtype=np.float32)
    is_type = np.array([t in TYPE_TAGS for t in tag_arr], dtype=np.float32)
    is_select = np.array([t in SELECT_TAGS for t in tag_arr], dtype=np.float32)
    p_click = op_proba_per_row[rids, OP_MAP['CLICK']]
    p_type = op_proba_per_row[rids, OP_MAP['TYPE']]
    p_select = op_proba_per_row[rids, OP_MAP['SELECT']]
    feat['op_p_click'] = p_click
    feat['op_p_type'] = p_type
    feat['op_p_select'] = p_select
    feat['op_tag_match'] = (p_click * is_click + p_type * is_type + p_select * is_select).astype(np.float32)

    # Empirical tag prior given op: P(target has tag t | op), mixed by op probabilities
    if tag_priors is not None:
        def lookup(tag, op):
            return tag_priors[op].get(tag, 1e-4)
        prior_click = np.array([lookup(t, 'CLICK') for t in tag_arr], dtype=np.float32)
        prior_type = np.array([lookup(t, 'TYPE') for t in tag_arr], dtype=np.float32)
        prior_select = np.array([lookup(t, 'SELECT') for t in tag_arr], dtype=np.float32)
        feat['tag_prior_click'] = prior_click
        feat['tag_prior_type'] = prior_type
        feat['tag_prior_select'] = prior_select
        feat['tag_prior_mix'] = (p_click * prior_click + p_type * prior_type
                                 + p_select * prior_select).astype(np.float32)
        feat['tag_prior_mix_norm'] = (feat['tag_prior_mix'].values
                                      / np.maximum(tag_counts, 1)).astype(np.float32)

    return feat, word_vec, char_vec


def fit_ranker(F_train: pd.DataFrame, y_train: np.ndarray, group_train: np.ndarray):
    rk = lgb.LGBMRanker(n_estimators=800, learning_rate=0.05, num_leaves=63,
                        n_jobs=-1, random_state=42, verbosity=-1,
                        objective='lambdarank',
                        label_gain=[0, 1])  # binary relevance
    rk.fit(F_train.values, y_train, group=group_train)
    return rk


# ────────────────────────────────────────────────────────────────────
# Stage 3 — value heuristic
# ────────────────────────────────────────────────────────────────────
QUOTE_RE = re.compile(r'["\'“”‘’]([^"\'“”‘’]{1,80})["\'“”‘’]')
EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
DATE_RE = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d+\b', re.I)
NUM_RE = re.compile(r'\b\d+\b')
CAPSEQ_RE = re.compile(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')


def predict_value(row: pd.Series) -> str:
    op = row['pred_op']
    if op == 'CLICK':
        return ''
    task = str(row['task'])
    if m := QUOTE_RE.search(task):
        return m.group(1)
    if op == 'TYPE':
        if m := EMAIL_RE.search(task):
            return m.group(0)
        if m := DATE_RE.search(task):
            return m.group(0)
        if m := NUM_RE.search(task):
            return m.group(0)
        caps = CAPSEQ_RE.findall(task)
        if caps:
            return caps[0]
        return ''
    if m := DATE_RE.search(task):
        return m.group(0)
    caps = CAPSEQ_RE.findall(task)
    if caps:
        return caps[0]
    return ''


# ────────────────────────────────────────────────────────────────────
# Validation (GroupKFold by site)
# ────────────────────────────────────────────────────────────────────
def validate(train: pd.DataFrame, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    op_accs, tgt_accs, val_accs, all3_accs = [], [], [], []
    fold = 0
    for tr_idx, va_idx in gkf.split(train, groups=train['site_token']):
        fold += 1
        tr = train.iloc[tr_idx].reset_index(drop=True)
        va = train.iloc[va_idx].reset_index(drop=True)

        # Stage 1: op — fit on outer-train; OOF op proba on outer-train (no leakage)
        Xtr, tv, hv = build_op_features(tr, fit=True)
        Xva, _, _ = build_op_features(va, tv, hv)
        ytr = tr['op'].map(OP_MAP).values
        yva = va['op'].map(OP_MAP).values
        op_proba_tr = oof_op_proba(tr, n_splits=5)  # for ranker training features
        clf = fit_op_classifier(Xtr, ytr)
        op_pred = clf.predict(Xva)
        op_proba_va = clf.predict_proba(Xva)
        op_acc = accuracy_score(yva, op_pred)

        # Stage 2: ranker (uses op probs + tag priors as features)
        tr_cand = explode_candidates(tr, with_label=True)
        va_cand = explode_candidates(va, with_label=False)
        tag_priors = compute_tag_priors(tr_cand, tr['op'])
        Ftr, wv, cv = cand_features(tr_cand, op_proba_tr, tag_priors, fit=True)
        Fva, _, _ = cand_features(va_cand, op_proba_va, tag_priors, wv, cv)
        group_tr = tr_cand.groupby('row_id').size().values
        rk = fit_ranker(Ftr, tr_cand['is_target'].values, group_tr)
        va_cand['score'] = rk.predict(Fva.values)
        top = (va_cand.sort_values(['row_id', 'score'], ascending=[True, False])
               .groupby('row_id').head(1))
        target_pred = top.set_index('row_id')['candidate_id']
        va = va.copy()
        va['pred_target'] = va.index.map(target_pred)
        tgt_acc = (va['pred_target'] == va['target_id']).mean()

        # Stage 3: value
        va['pred_op'] = pd.Series(op_pred).map(OP_INV).values
        va['pred_value'] = va.apply(predict_value, axis=1)
        val_acc = (va['pred_value'].fillna('') == va['value'].fillna('')).mean()

        all3 = ((va['pred_op'] == va['op']) &
                (va['pred_target'] == va['target_id']) &
                (va['pred_value'].fillna('') == va['value'].fillna(''))).mean()

        op_accs.append(op_acc); tgt_accs.append(tgt_acc); val_accs.append(val_acc); all3_accs.append(all3)
        print(f'  fold {fold}: op={op_acc:.4f}  target={tgt_acc:.4f}  '
              f'value={val_acc:.4f}  all3={all3:.4f}')

    print(f'CV mean: op={np.mean(op_accs):.4f}  target={np.mean(tgt_accs):.4f}  '
          f'value={np.mean(val_accs):.4f}  all3={np.mean(all3_accs):.4f}')


# ────────────────────────────────────────────────────────────────────
# Final fit + submission
# ────────────────────────────────────────────────────────────────────
def fit_and_submit(train: pd.DataFrame, test: pd.DataFrame):
    print('[final] fitting op classifier...')
    Xtr, tv, hv = build_op_features(train, fit=True)
    Xte, _, _ = build_op_features(test, tv, hv)
    ytr = train['op'].map(OP_MAP).values
    print('[final] computing OOF op probabilities for ranker training...')
    op_proba_train = oof_op_proba(train, n_splits=5)
    clf = fit_op_classifier(Xtr, ytr)
    op_pred_test = clf.predict(Xte)
    op_proba_test = clf.predict_proba(Xte)

    print('[final] fitting target ranker...')
    tr_cand = explode_candidates(train, with_label=True)
    te_cand = explode_candidates(test, with_label=False)
    tag_priors = compute_tag_priors(tr_cand, train['op'])
    Ftr, wv, cv = cand_features(tr_cand, op_proba_train, tag_priors, fit=True)
    Fte, _, _ = cand_features(te_cand, op_proba_test, tag_priors, wv, cv)
    group_tr = tr_cand.groupby('row_id').size().values
    rk = fit_ranker(Ftr, tr_cand['is_target'].values, group_tr)
    te_cand['score'] = rk.predict(Fte.values)
    top = (te_cand.sort_values(['row_id', 'score'], ascending=[True, False])
           .groupby('row_id').head(1))
    id_to_target = dict(zip(top['id'], top['candidate_id']))

    test = test.copy()
    test['pred_op'] = pd.Series(op_pred_test).map(OP_INV).values
    test['pred_target'] = test['id'].map(id_to_target)
    test['pred_value'] = test.apply(predict_value, axis=1)

    sub = pd.DataFrame({
        'id': test['id'],
        'op': test['pred_op'],
        'target_id': test['pred_target'],
        'value': test['pred_value'].fillna(''),
    })
    out_path = OUT / 'submission.csv'
    sub.to_csv(out_path, index=False)
    print(f'[final] wrote {out_path}  rows={len(sub)}')
    return sub


def main():
    train, test = load()
    print(f'train={len(train)}  test={len(test)}')
    print('=== CV (GroupKFold by site_token) ===')
    validate(train, n_splits=5)
    print('=== Final fit + submission ===')
    fit_and_submit(train, test)


if __name__ == '__main__':
    main()
