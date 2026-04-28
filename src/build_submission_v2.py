"""LightGBM ranker baseline (Stage B)

설계
----
- target_id : per-(row, candidate) 피처 → LGBMRanker(lambdarank) → argmax
- op        : 선택된 candidate 의 tag → op 매핑 (Stage A 와 동일)
- value     : op 별 규칙 (Stage A 와 동일)

피처 (per candidate)
- 토큰 집합: |R∩C|, |T∩C|, |R|, |T|, |C|
             coverage(R,C), coverage(T,C), coverage(C,R)
             jaccard(R,C), jaccard(T,C), dice(R,C)
- IDF 가중: cov_R_C_idf, cov_T_C_idf, idf_inter_RC
- attrs 구조: label/name/placeholder 토큰의 task 매칭
              options 중 task 에 등장하는 개수
- history 인접: candidate.text/label 이 history 에 이미 나왔는가
- DOM 위치: pos, pos_norm, n_candidates
- 컨텍스트: n_history_steps
- 태그: tag_idx (categorical)

학습/평가
- 80/20 (row 단위, 랜덤 시드 0) hold-out 으로 검증
- 최종 제출은 100% 재학습으로 생성
"""

from pathlib import Path
import json, re, math
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RNG_SEED = 0

# ============================== 전처리 ====================================
DOMAIN_STOP = {"task", "step", "click", "type", "select", "enter"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
ATTR_RE = re.compile(r"(\w+)=([^|]+?)(?=\s*\||$)", re.I)
OPT_RE = re.compile(r"options=([^|]+?)(?=\s*\||$)", re.I)


def tokenize(s):
    if not isinstance(s, str):
        s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return [
        t
        for t in (m.lower() for m in TOKEN_RE.findall(s))
        if t not in STOPWORDS and len(t) > 1
    ]


def cand_text(c):
    return (c.get("text") or "") + " " + (c.get("attrs") or "")


def get_attr(attrs, key):
    m = re.search(rf"\b{key}=([^|]+?)(?=\s*\||$)", attrs or "", re.I)
    return m.group(1).strip() if m else ""


def parse_options(attrs):
    m = OPT_RE.search(attrs or "")
    return [o.strip() for o in m.group(1).split("/") if o.strip()] if m else []


# ---------- cleaned_html 컨텍스트 파싱 ----------
COMPLETED_RE = re.compile(r"Completed:\s*([^<\n]+)", re.I)
STEP_RE = re.compile(r"step\s+(\d+)\s+of\s+(\d+)", re.I)
TYPE_ATTR_RE = re.compile(r"\btype=([A-Za-z]+)", re.I)
PANEL_RE = re.compile(
    r'<section[^>]*aria-label="current workflow panel"[^>]*>(.*?)</section>',
    re.I | re.S,
)
NAME_ATTR_HTML_RE = re.compile(r'\bname="([^"]+)"', re.I)
H1_RE = re.compile(r"<h1[^>]*>([^<]+)</h1>", re.I)
WORKFLOW_CTX_RE = re.compile(
    r'<aside[^>]*class="workflow-context"[^>]*>([^<]+)</aside>', re.I
)
INPUT_TYPE_VOCAB = [
    "text", "date", "email", "number", "checkbox", "radio",
    "password", "tel", "url", "time", "search",
]


def parse_html_context(html):
    """cleaned_html 에서 컨텍스트 한 묶음 추출.

    반환: (completed_set, cur_step, tot_step, html_len,
           panel_names, h1_tokens, ctx_tokens)
    - panel_names: <section aria-label="current workflow panel"> 안에 있는 name="X" 집합
    - h1_tokens   : <h1> 텍스트 토큰
    - ctx_tokens  : workflow-context aside 텍스트 토큰
    """
    if not isinstance(html, str):
        return set(), 0, 0, 0, set(), set(), set()
    completed = set()
    for m in COMPLETED_RE.finditer(html):
        for item in m.group(1).split(","):
            it = item.strip().lower()
            if it:
                completed.add(it)
    sm = STEP_RE.search(html)
    cur = int(sm.group(1)) if sm else 0
    tot = int(sm.group(2)) if sm else 0

    # current workflow panel 안의 name 집합
    panel_names = set()
    pm = PANEL_RE.search(html)
    if pm:
        panel_names = {n.lower() for n in NAME_ATTR_HTML_RE.findall(pm.group(1))}

    # h1 / workflow-context 텍스트 토큰
    h1_tokens = set()
    for m in H1_RE.finditer(html):
        h1_tokens |= set(tokenize(m.group(1)))
    ctx_tokens = set()
    for m in WORKFLOW_CTX_RE.finditer(html):
        ctx_tokens |= set(tokenize(m.group(1)))

    return completed, cur, tot, len(html), panel_names, h1_tokens, ctx_tokens


def input_type_index(attrs):
    m = TYPE_ATTR_RE.search(attrs or "")
    if not m:
        return len(INPUT_TYPE_VOCAB)
    tp = m.group(1).lower()
    try:
        return INPUT_TYPE_VOCAB.index(tp)
    except ValueError:
        return len(INPUT_TYPE_VOCAB)


def candidate_in_completed(cand, completed):
    if not completed:
        return 0.0
    txt = (cand.get("text") or "").strip().lower()
    if txt and txt in completed:
        return 1.0
    lab = get_attr(cand.get("attrs", ""), "label").strip().lower()
    if lab and lab in completed:
        return 1.0
    return 0.0


TAG_VOCAB = [
    "input",
    "textarea",
    "select",
    "button",
    "a",
    "link",
    "span",
    "div",
    "li",
    "section",
]


def tag_index(tag):
    t = (tag or "").lower()
    try:
        return TAG_VOCAB.index(t)
    except ValueError:
        return len(TAG_VOCAB)


# ============================== IDF =======================================
def compute_idf(token_lists):
    n = len(token_lists)
    df = Counter()
    for toks in token_lists:
        for t in set(toks):
            df[t] += 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


# ============================== 피처 ======================================
FEAT_NAMES = [
    "inter_RC",
    "inter_TC",
    "sz_R",
    "sz_T",
    "sz_C",
    "cov_R_C",
    "cov_T_C",
    "cov_C_R",
    "jac_RC",
    "jac_TC",
    "dice_RC",
    "cov_R_C_idf",
    "cov_T_C_idf",
    "idf_inter_RC",
    "label_match_T",
    "name_match_T",
    "ph_match_T",
    "options_in_task",
    "n_options",
    "text_in_history",
    "label_in_history",
    "pos",
    "n_cands",
    "pos_norm",
    "n_history_steps",
    "tag_idx",
    # cleaned_html 컨텍스트
    "is_in_completed",
    "n_completed",
    "html_current_step",
    "html_total_steps",
    "html_step_remaining",  # total - current
    "html_len_log",
    "input_type_idx",
    # workflow panel / 페이지 헤더 매칭
    "in_workflow_panel",
    "n_panel_names",
    "h1_match_C",
    "ctx_match_T",
]
CATEGORICAL = ["tag_idx", "input_type_idx"]


def feat_row(task, history, cands, idf, html_ctx=None):
    T = set(tokenize(task))
    H = set(tokenize(history))
    R = T - H
    R_idf_sum = sum(idf.get(t, 1.0) for t in R) if R else 0.0
    T_idf_sum = sum(idf.get(t, 1.0) for t in T) if T else 0.0
    n_steps = len(re.findall(r"Step \d+:", history)) if isinstance(history, str) else 0
    task_l = (task or "").lower() if isinstance(task, str) else ""
    history_l = (history or "").lower() if isinstance(history, str) else ""
    if html_ctx is None:
        completed, cur_step, tot_step, html_len = set(), 0, 0, 0
        panel_names, h1_tokens, ctx_tokens = set(), set(), set()
    else:
        (completed, cur_step, tot_step, html_len,
         panel_names, h1_tokens, ctx_tokens) = html_ctx
    html_len_log = math.log1p(html_len)
    step_remaining = max(0, tot_step - cur_step)
    # task ↔ workflow context 토큰 매칭 (행 단위, candidate 무관)
    ctx_match_T = (len(T & ctx_tokens) / len(ctx_tokens)) if ctx_tokens else 0.0

    rows = []
    n_c = len(cands)
    for pos, c in enumerate(cands):
        attrs = c.get("attrs") or ""
        text = c.get("text") or ""
        tag = (c.get("tag") or "").lower()
        C = set(tokenize(cand_text(c)))
        inter_RC = len(R & C)
        inter_TC = len(T & C)
        cov_R_C = inter_RC / len(R) if R else 0.0
        cov_T_C = inter_TC / len(T) if T else 0.0
        cov_C_R = inter_RC / len(C) if C else 0.0
        union_RC = R | C
        union_TC = T | C
        jac_RC = inter_RC / len(union_RC) if union_RC else 0.0
        jac_TC = inter_TC / len(union_TC) if union_TC else 0.0
        dice_RC = 2 * inter_RC / (len(R) + len(C)) if (R or C) else 0.0

        idf_inter_RC = sum(idf.get(t, 1.0) for t in (R & C))
        idf_inter_TC = sum(idf.get(t, 1.0) for t in (T & C))
        cov_R_C_idf = idf_inter_RC / R_idf_sum if R_idf_sum > 0 else 0.0
        cov_T_C_idf = idf_inter_TC / T_idf_sum if T_idf_sum > 0 else 0.0

        label_t = set(tokenize(get_attr(attrs, "label")))
        name_t = set(tokenize(get_attr(attrs, "name").replace("_", " ")))
        ph_t = set(tokenize(get_attr(attrs, "placeholder")))
        opts = parse_options(attrs)
        label_match_T = len(T & label_t) / len(label_t) if label_t else 0.0
        name_match_T = len(T & name_t) / len(name_t) if name_t else 0.0
        ph_match_T = len(T & ph_t) / len(ph_t) if ph_t else 0.0
        options_in_task = sum(1 for o in opts if o.lower() in task_l)

        text_in_h = 1.0 if text and text.lower() in history_l else 0.0
        label_in_h = 1.0
        lab_str = get_attr(attrs, "label")
        if not lab_str or lab_str.lower() not in history_l:
            label_in_h = 0.0

        pos_norm = pos / max(1, n_c - 1)

        is_completed = candidate_in_completed(c, completed)
        in_type_idx = input_type_index(attrs)
        # workflow panel 멤버십: candidate 의 name 이 panel 안의 name 집합에 있는가
        cand_name = get_attr(attrs, "name").strip().lower()
        in_panel = 1.0 if cand_name and cand_name in panel_names else 0.0
        # h1 토큰 ↔ candidate 토큰 매칭 (페이지 주제와 candidate 정합)
        h1_match_C = (len(C & h1_tokens) / len(h1_tokens)) if h1_tokens else 0.0

        rows.append(
            [
                inter_RC,
                inter_TC,
                len(R),
                len(T),
                len(C),
                cov_R_C,
                cov_T_C,
                cov_C_R,
                jac_RC,
                jac_TC,
                dice_RC,
                cov_R_C_idf,
                cov_T_C_idf,
                idf_inter_RC,
                label_match_T,
                name_match_T,
                ph_match_T,
                options_in_task,
                len(opts),
                text_in_h,
                label_in_h,
                pos,
                n_c,
                pos_norm,
                n_steps,
                tag_index(tag),
                # cleaned_html 컨텍스트
                is_completed,
                len(completed),
                cur_step,
                tot_step,
                step_remaining,
                html_len_log,
                in_type_idx,
                # workflow panel / 헤더
                in_panel,
                len(panel_names),
                h1_match_C,
                ctx_match_T,
            ]
        )
    return np.array(rows, dtype=float)


def build_dataset(df, idf):
    X_list, y_list, group_sizes, meta = [], [], [], []
    for _, r in df.iterrows():
        try:
            cands = (
                json.loads(r["candidate_elements"])
                if isinstance(r["candidate_elements"], str)
                else []
            )
        except Exception:
            cands = []
        if not cands:
            meta.append({"id": r.get("id"), "cand_ids": [], "cands": []})
            continue
        html_ctx = parse_html_context(r.get("cleaned_html"))
        Xi = feat_row(r.get("task"), r.get("history"), cands, idf, html_ctx)
        target = r.get("target_id") if "target_id" in df.columns else None
        labels = np.array([1.0 if c["candidate_id"] == target else 0.0 for c in cands])
        X_list.append(Xi)
        y_list.append(labels)
        group_sizes.append(len(cands))
        meta.append(
            {
                "id": r.get("id"),
                "cand_ids": [c["candidate_id"] for c in cands],
                "cands": cands,
                "task": r.get("task"),
                "history": r.get("history"),
            }
        )
    if X_list:
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
    else:
        X = np.zeros((0, len(FEAT_NAMES)))
        y = np.zeros(0)
    return X, y, np.array(group_sizes), meta


# ============================== op / value ================================
TAG2OP = {
    "input": "TYPE",
    "textarea": "TYPE",
    "select": "SELECT",
    "button": "CLICK",
    "a": "CLICK",
    "link": "CLICK",
}


def predict_op(cand):
    return TAG2OP.get((cand.get("tag") or "").lower(), "CLICK")


VALUE_TAIL = re.compile(r"\s+(and|with|for|to|then)\b.*$", re.I)
ATTR_KV = re.compile(r"(label|name|placeholder)=([^|]+?)(?=\s*\||$)", re.I)


def candidate_labels(cand):
    labs = []
    txt = (cand.get("text") or "").strip()
    if txt:
        labs.append(txt)
    for m in ATTR_KV.finditer(cand.get("attrs") or ""):
        v = m.group(2).strip()
        if v:
            labs.append(v)
        v2 = v.replace("_", " ").strip()
        if v2 and v2 != v:
            labs.append(v2)
    seen = set()
    out = []
    for l in labs:
        k = l.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(l)
    out.sort(key=len, reverse=True)
    return out


def extract_value_type(task, cand):
    task = task if isinstance(task, str) else ""
    for lab in candidate_labels(cand):
        if not lab:
            continue
        m = re.search(rf"(?i)\b{re.escape(lab)}\b\s*[:]?\s*([^,\.\n]+)", task)
        if m:
            v = m.group(1).strip()
            v = VALUE_TAIL.sub("", v).strip().strip(",.;")
            if v:
                return v
    return ""


def extract_value_select(task, cand):
    options = parse_options(cand.get("attrs") or "")
    if not options:
        return ""
    task_l = (task or "").lower()
    matched = [o for o in options if o.lower() in task_l]
    return max(matched, key=len) if matched else ""


def predict_value(task, cand, op):
    if op == "CLICK":
        return np.nan
    if op == "SELECT":
        return extract_value_select(task, cand)
    if op == "TYPE":
        return extract_value_type(task, cand)
    return ""


# ============================== 평가 ======================================
def eval_pred(meta_list, scores, group_sizes, df_true, name="val"):
    """ranker 점수 → argmax target_id → op/value → 정확도"""
    preds = []
    cursor = 0
    meta_iter = iter(meta_list)
    for size in group_sizes:
        m = next(meta_iter)
        s = scores[cursor : cursor + size]
        cursor += size
        best = int(np.argmax(s))
        cand = m["cands"][best]
        op = predict_op(cand)
        val = predict_value(m["task"], cand, op)
        preds.append(
            {"id": m["id"], "target_id": cand["candidate_id"], "op": op, "value": val}
        )
    # candidate_elements 가 비어 있어 학습 dataset 에서 빠진 행도 보정
    pred_ids = {p["id"] for p in preds}
    for _, r in df_true.iterrows():
        if r["id"] not in pred_ids:
            preds.append(
                {"id": r["id"], "target_id": "", "op": "CLICK", "value": np.nan}
            )
    p = pd.DataFrame(preds).set_index("id").loc[df_true["id"].values].reset_index()

    def eq(a, b):
        a = np.where(pd.isna(a), "", np.asarray(a, dtype=object).astype(str))
        b = np.where(pd.isna(b), "", np.asarray(b, dtype=object).astype(str))
        return a == b

    m_t = eq(p["target_id"].values, df_true["target_id"].values)
    m_o = eq(p["op"].values, df_true["op"].values)
    m_v = eq(p["value"].values, df_true["value"].values)
    m_all = m_t & m_o & m_v
    print(f"\n=== {name} (n={len(p)}) ===")
    print(f"  target_id : {m_t.mean()*100:6.2f}%")
    print(f"  op        : {m_o.mean()*100:6.2f}%")
    print(f"  value     : {m_v.mean()*100:6.2f}%")
    print(f"  ALL match : {m_all.mean()*100:6.2f}%")
    if m_t.sum():
        print(
            f"  | given correct target: op={m_o[m_t].mean()*100:.2f}%  value={m_v[m_t].mean()*100:.2f}%"
        )
    return p


# ============================== main ======================================
def main():
    print("[load] train")
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8", encoding_errors="replace")
    train = train[train["candidate_elements"].notna()].reset_index(drop=True)

    # IDF: train task + history 합본 기준
    print("[idf] computing on train task+history")
    docs = []
    for _, r in train.iterrows():
        docs.append(tokenize(r.get("task")) + tokenize(r.get("history")))
    idf = compute_idf(docs)
    print(f"  vocab size = {len(idf)}")

    # 80/20 split
    rng = np.random.default_rng(RNG_SEED)
    idx = np.arange(len(train))
    rng.shuffle(idx)
    n_val = len(train) // 5
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    df_trn = train.iloc[trn_idx].reset_index(drop=True)
    df_val = train.iloc[val_idx].reset_index(drop=True)
    print(f"  train={len(df_trn)}  val={len(df_val)}")

    print("[features] train/val")
    X_trn, y_trn, g_trn, meta_trn = build_dataset(df_trn, idf)
    X_val, y_val, g_val, meta_val = build_dataset(df_val, idf)
    print(f"  X_trn={X_trn.shape}  X_val={X_val.shape}")

    LR = 0.02
    NL = 31
    MIN_LEAF = 50
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=2000,
        learning_rate=LR,
        num_leaves=NL,
        min_data_in_leaf=MIN_LEAF,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1.0,
        label_gain=[0, 1],
        random_state=RNG_SEED,
        verbose=-1,
    )
    print("[fit] lambdarank on 80%")
    ranker.fit(
        X_trn,
        y_trn,
        group=g_trn,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        eval_at=[1, 3, 5],
        feature_name=FEAT_NAMES,
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100)],
    )

    print("[eval] hold-out")
    s_val = ranker.predict(X_val, num_iteration=ranker.best_iteration_)
    [meta_val_with_cands] = [m for m in [meta_val] if True]
    # build_dataset 의 meta 는 candidates 가 있는 행만 포함하므로 group 과 1:1
    meta_val_seq = [m for m in meta_val if m.get("cands")]
    eval_pred(meta_val_seq, s_val, g_val, df_val, name="val")

    print("[importance]")
    imp = pd.DataFrame(
        {
            "feat": FEAT_NAMES,
            "gain": ranker.booster_.feature_importance(importance_type="gain"),
            "split": ranker.booster_.feature_importance(importance_type="split"),
        }
    )
    print(imp.sort_values("gain", ascending=False).to_string(index=False))

    # ============================== retrain on 100% =======================
    print("\n[refit] on 100% train")
    X_all, y_all, g_all, meta_all = build_dataset(train, idf)
    # 100% 재학습은 best_iteration 의 1.1배까지 진행 (데이터 1/0.8 ≈ 1.25 배 늘었으므로)
    refit_iters = int((ranker.best_iteration_ or 500) * 1.1)
    final = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=refit_iters,
        learning_rate=LR,
        num_leaves=NL,
        min_data_in_leaf=MIN_LEAF,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1.0,
        label_gain=[0, 1],
        random_state=RNG_SEED,
        verbose=-1,
    )
    final.fit(
        X_all,
        y_all,
        group=g_all,
        feature_name=FEAT_NAMES,
        categorical_feature=CATEGORICAL,
    )

    # ============================== predict test ==========================
    print("\n[load] test")
    test = pd.read_csv(DATA / "test.csv", encoding="utf-8", encoding_errors="replace")
    print(f"test rows = {len(test)}")
    X_te, _, g_te, meta_te = build_dataset(test, idf)
    s_te = final.predict(X_te) if X_te.shape[0] else np.zeros(0)

    # 결과 행 단위 정리 — meta_te 는 candidates 가 있는 행만 포함
    rows = []
    cursor = 0
    seq = [m for m in meta_te if m.get("cands")]
    for m, size in zip(seq, g_te):
        s = s_te[cursor : cursor + size]
        cursor += size
        best = int(np.argmax(s))
        cand = m["cands"][best]
        op = predict_op(cand)
        val = predict_value(m["task"], cand, op)
        rows.append(
            {"id": m["id"], "op": op, "target_id": cand["candidate_id"], "value": val}
        )
    pred_df = pd.DataFrame(rows)
    # candidates 가 비어있는 test 행은 빈값
    full = test[["id"]].merge(pred_df, on="id", how="left")
    full["op"] = full["op"].fillna("CLICK")
    full["target_id"] = full["target_id"].fillna("")
    full.loc[full["op"] == "CLICK", "value"] = ""
    full["value"] = full["value"].fillna("")

    out = ROOT / "submission.csv"
    full[["id", "op", "target_id", "value"]].to_csv(
        out, index=False, lineterminator="\n"
    )
    print(f"\n[saved] {out}  shape={full.shape}")
    print(full.head())
    print("\nop distribution:")
    print(full["op"].value_counts())


if __name__ == "__main__":
    main()
