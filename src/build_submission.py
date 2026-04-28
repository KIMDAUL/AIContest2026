"""end-to-end heuristic baseline submission

train.csv 로 정확도 측정 → test.csv 예측 → submission.csv 작성

규칙
----
1. target_id : argmax_c Coverage(R, candidate_tokens(c))
               R = set(task_tokens) - set(history_tokens)  (전처리 후)
               동률은 candidate 등장 순(=DOM 순) 첫 번째.
               R 가 비면 Jaccard(task, candidate) 로 fallback.
2. op        : 선택된 candidate 의 tag → op 매핑
               input/textarea→TYPE, select→SELECT, button/a/link→CLICK
3. value     : op 별 분기
   - CLICK  : 빈 값 (NaN)
   - SELECT : candidate.attrs 의 options 중 task 에 등장하는 것 (가장 긴 매칭)
   - TYPE   : task 에서 candidate label/name 다음에 오는 구문 추출
"""
from pathlib import Path
import json, re, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT_DIR = ROOT  # submission.csv 는 프로젝트 루트에

# ---------- 전처리 ----------
DOMAIN_STOP = {'task', 'step', 'click', 'type', 'select', 'enter'}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
def tokenize(s):
    if not isinstance(s, str):
        s = '' if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return [t for t in (m.lower() for m in TOKEN_RE.findall(s))
            if t not in STOPWORDS and len(t) > 1]
def cand_text(c):
    return (c.get('text') or '') + ' ' + (c.get('attrs') or '')

# ---------- 점수 ----------
def coverage(R, C):
    if not R: return 0.0
    return len(R & C) / len(R)
def jaccard(A, B):
    if not A and not B: return 0.0
    return len(A & B) / len(A | B)

# ---------- 후보 점수화 + 선택 ----------
def pick_target(task, history, cand_list):
    if not cand_list:
        return None, None, 0
    R = set(tokenize(task)) - set(tokenize(history or ''))
    C_each = [set(tokenize(cand_text(c))) for c in cand_list]
    if R:
        scores = [coverage(R, C) for C in C_each]
    else:  # fallback
        T = set(tokenize(task))
        scores = [jaccard(T, C) for C in C_each]
    best = int(np.argmax(scores))
    return cand_list[best], scores[best], len(R)

# ---------- op 규칙 ----------
TAG2OP = {
    'input':    'TYPE',
    'textarea': 'TYPE',
    'select':   'SELECT',
    'button':   'CLICK',
    'a':        'CLICK',
    'link':     'CLICK',
}
def predict_op(cand):
    return TAG2OP.get((cand.get('tag') or '').lower(), 'CLICK')

# ---------- value 추출 ----------
ATTR_KV_RE = re.compile(r'(label|name|placeholder)=([^|]+?)(?=\s*\||$)', re.I)
OPT_RE     = re.compile(r'options=([^|]+?)(?=\s*\||$)', re.I)

def parse_options(attrs):
    m = OPT_RE.search(attrs or '')
    if not m: return []
    return [o.strip() for o in m.group(1).split('/') if o.strip()]

def candidate_labels(cand):
    """이 candidate 를 task 에서 가리키는 '라벨' 후보들 — 길이 내림차순"""
    labs = []
    txt = (cand.get('text') or '').strip()
    if txt: labs.append(txt)
    for m in ATTR_KV_RE.finditer(cand.get('attrs') or ''):
        v = m.group(2).strip()
        if v: labs.append(v)
        v2 = v.replace('_', ' ').strip()
        if v2 and v2 != v: labs.append(v2)
    # 중복 제거 + 길이 내림차순(긴 라벨이 더 구체적)
    seen = set(); out = []
    for l in labs:
        k = l.lower()
        if k in seen: continue
        seen.add(k); out.append(l)
    out.sort(key=len, reverse=True)
    return out

VALUE_TAIL_STRIP = re.compile(r'\s+(and|with|for|to|then)\b.*$', re.I)

def extract_value_type(task, cand):
    task = task or ''
    for lab in candidate_labels(cand):
        if not lab: continue
        m = re.search(rf'(?i)\b{re.escape(lab)}\b\s*[:]?\s*([^,\.\n]+)', task)
        if m:
            v = m.group(1).strip()
            v = VALUE_TAIL_STRIP.sub('', v).strip().strip(',.;')
            if v: return v
    return ''

def extract_value_select(task, cand):
    options = parse_options(cand.get('attrs') or '')
    if not options: return ''
    task_l = (task or '').lower()
    matched = [o for o in options if o.lower() in task_l]
    if matched:
        return max(matched, key=len)
    return ''  # task 에 옵션이 안 보이면 비워둠 (틀려도 다른 신호 없음)

def predict_value(task, cand, op):
    if op == 'CLICK': return np.nan
    if op == 'SELECT': return extract_value_select(task, cand)
    if op == 'TYPE':   return extract_value_type(task, cand)
    return ''

# ---------- 한 행에 대한 전체 예측 ----------
def predict_row(task, history, cand_list):
    target, score, rsz = pick_target(task, history, cand_list)
    if target is None:
        return {'target_id': '', 'op': 'CLICK', 'value': np.nan,
                'score': 0.0, 'rsz': rsz}
    op  = predict_op(target)
    val = predict_value(task, target, op)
    return {'target_id': target['candidate_id'], 'op': op, 'value': val,
            'score': score, 'rsz': rsz}

# ---------- 평가 ----------
def evaluate(df, name='train'):
    preds = []
    for _, row in df.iterrows():
        cands = json.loads(row['candidate_elements']) if isinstance(row['candidate_elements'], str) else []
        preds.append(predict_row(row.get('task'), row.get('history'), cands))
    p = pd.DataFrame(preds)
    p['true_target'] = df['target_id'].values
    p['true_op']     = df['op'].values
    p['true_value']  = df['value'].values

    # exact match (NaN==NaN 으로 처리)
    def eq(a, b):
        a = np.where(pd.isna(a), '', a.astype(str))
        b = np.where(pd.isna(b), '', b.astype(str))
        return a == b

    m_t  = eq(p['target_id'].values,    p['true_target'].values)
    m_o  = eq(p['op'].values,           p['true_op'].values)
    m_v  = eq(p['value'].values,        p['true_value'].values)
    m_all = m_t & m_o & m_v

    n = len(p)
    print(f'\n=== eval on {name} (n={n}) ===')
    print(f'  target_id  : {m_t.mean()*100:6.2f}%')
    print(f'  op         : {m_o.mean()*100:6.2f}%')
    print(f'  value      : {m_v.mean()*100:6.2f}%')
    print(f'  ALL match  : {m_all.mean()*100:6.2f}%')
    # 분해: target 맞은 행에서의 op/value 정확도 (각 모듈 상한 추정)
    if m_t.sum():
        print(f'  | given correct target_id:  op={m_o[m_t].mean()*100:.2f}%  value={m_v[m_t].mean()*100:.2f}%')
    # op 별 value 정확도
    for op_lbl in ['CLICK','TYPE','SELECT']:
        mm = (p['true_op'].values == op_lbl) & m_t
        if mm.sum():
            print(f'  | op={op_lbl:6s} correct-target rows={mm.sum():5d}  value={m_v[mm].mean()*100:.2f}%')
    return p

# ==========================================================================
# main
# ==========================================================================
if __name__ == '__main__':
    print('[load] train.csv')
    train = pd.read_csv(DATA / 'train.csv', encoding='utf-8', encoding_errors='replace')
    evaluate(train, 'train')

    print('\n[load] test.csv')
    test = pd.read_csv(DATA / 'test.csv', encoding='utf-8', encoding_errors='replace')
    print(f'test rows: {len(test)}')

    rows = []
    for _, r in test.iterrows():
        cands = json.loads(r['candidate_elements']) if isinstance(r['candidate_elements'], str) else []
        pred = predict_row(r.get('task'), r.get('history'), cands)
        rows.append({
            'id':        r['id'],
            'op':        pred['op'] if pred['op'] != 'CLICK' or pred['target_id'] else 'CLICK',
            'target_id': pred['target_id'],
            'value':     pred['value'],
        })
    sub = pd.DataFrame(rows, columns=['id','op','target_id','value'])
    # CLICK 일 때 value 는 비어야 함 (sample_submission 이 빈 셀)
    sub.loc[sub['op']=='CLICK', 'value'] = ''
    sub['value'] = sub['value'].fillna('')

    out_path = OUT_DIR / 'submission.csv'
    sub.to_csv(out_path, index=False, lineterminator='\n')
    print(f'\n[saved] {out_path}  shape={sub.shape}')
    print(sub.head())
    print('\nop distribution in submission:')
    print(sub['op'].value_counts())
