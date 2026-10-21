# -*- coding: utf-8 -*-
r"""
01c_comments_features.py — v3 DEAL-LEVEL (2026-10-20)
================================================================================
从 Zephyr deal comment 叙事文本中抽取 LM 金融情感词典特征与其他文本特征。

================================================================================
v2 -> v3 改动（重要）
================================================================================
[A1] 【默认输出改为 deal 级】COLLAPSE_TO_DEAL 默认值由 False 翻转为 True。
     原因：v2 输出 editorial 级（1,579,923 行 / 1,065,301 deal），
     下游 06 按 deal_num 做 1:1 left-merge 会把基表行数膨胀 1.58 倍，
     触发 "输出不是一行一 deal" 的 RuntimeError。
     翻转后 OUT_FILE 直接是一行一 deal，下游无需再聚合。

[A2] 双输出：
     OUT_RAW  = 01c_comments_features_raw.csv   (editorial 级，供诊断/断点续跑)
     OUT_FILE = 01c_comments_features.csv       (deal 级，下游直接使用)

[A3] 【聚合规则重写】按列语义分四类，不再笼统 "sum / max / mean"：
     SUM_COLS        计数类 -> sum      (LM 词频、事件次数、字符/词数、句子数)
     MAX_COLS        二值/存在类 -> max  (has_*, num_unique_reg)
     DENSITY_COLS    密度类 -> 聚合后重算 = count / wordcount
                      （禁止先算密度再平均：短评论会被等权放大，
                        例 10/1000=0.010 与 5/50=0.100 平均得 0.055，
                        正确应为 15/1050=0.0143，差 4 倍）
     EXTREME_COLS    日期/溢价 -> min/max，premium 再重算 avg=sum/count

[A4] premium 语义修正：avg_premium_pct 由 "mean" 改为 "sum/count"
     （原写法是对『每条 editorial 内部均值』再平均 = 平均的平均）

[A5] 新增 comment_n：该 deal 的评论条数。这是 deal 级必须保留的维度，
     也是下游重要的规模控制变量与权重依据。

[A6] 诊断同时报告 editorial 级与 deal 级分布，方便论文描述性统计引用。

[A7] 保留 v2 的全部性能与校验机制：分批、token 查表、断点续跑、
     token/regex 交叉校验、列名自适应、deal_num 规范化。

================================================================================
关于 reg_* 文本度量（沿用 v1 警告，不变）
================================================================================
  reg_event_count / num_unique_reg / has_phase2_investigation / has_reg_remedy
  是【文本正则】度量，仅覆盖西方反垄断机构，不含证券监管（CSRC/SEC/SEBI...），
  覆盖率 ~1.84%，远低于 01b 结构化 reg_* 的 ~12.8%。
  实证主度量请用 01b 的结构化 reg_*，此处仅作稳健性备选。
"""

import os
import re
import csv
import sys
import time
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# =====================================================================
# Config
# =====================================================================
BASE    = r"D:\MA"
CLEANED = os.path.join(BASE, "data", "cleaned")
MERGED  = os.path.join(BASE, "data", "merged")
os.makedirs(CLEANED, exist_ok=True)
os.makedirs(MERGED,  exist_ok=True)

SRC_DIR  = r"D:\zephyr\comment\cleaned_output"
SRC_GLOB = "MA_comment_*_cleaned.csv"

PARTS_DIR = os.path.join(CLEANED, "01c_parts")
OUT_RAW   = os.path.join(CLEANED, "01c_comments_features_raw.csv")   # editorial 级
OUT_FILE  = os.path.join(CLEANED, "01c_comments_features.csv")      # deal 级 ← 下游用这个
DIAG_FILE = os.path.join(CLEANED,  "01c_comments_diagnostics.txt")
os.makedirs(PARTS_DIR, exist_ok=True)

LM_DICT_PATH = os.path.join(BASE, "raw",
                            "Loughran-McDonald_MasterDictionary_1993-2021.csv")

# ---------- 处理参数 ----------
CHUNK_ROWS       = 50_000
RESUME           = False #RESUME = False 重跑一次（约 45 分钟），确认链路干净。之后切回 True。
VALIDATE_LM_MATCH= True
VALIDATE_N       = 2_000
LM_MATCH_MODE    = "token"
COLLAPSE_TO_DEAL = True      # ★ v3：默认聚合到 deal 级

# ---------- 输出开关 ----------
EMIT_LM_DENSITY  = True
EMIT_LM_PRESENCE = True
EMIT_COMPLEXITY  = False

# ---------- 聚合规则表（v3 新增，核心）----------
LM_CATS = ["pos", "neg", "uncertain", "litigious",
           "strongmodal", "weakmodal", "constrain"]

SUM_COLS = ([f"lm_{c}_count" for c in LM_CATS] +
            ["price_event_num", "rival_bidder_num", "reg_event_count",
             "comment_char_length", "comment_wordcount", "sentence_event_count"])

MAX_COLS = ([f"has_lm_{c}" for c in LM_CATS] +
            ["has_rumour", "has_target_reject", "has_unconditional_offer",
             "has_deal_complete", "has_goshop", "has_phase2_investigation",
             "has_reg_remedy", "has_debt_assumption", "num_unique_reg"])

DENSITY_COLS = [f"lm_{c}_density" for c in LM_CATS]   # 聚合后重算，不直接聚合

EXTREME_COLS = {                           # col -> min/max
    "dt_first":            "min",
    "dt_last":             "max",
    #"max_premium_pct":     "max",
    "avg_premium_pct":     "sum",          # 先汇总，再除以 price_event_num
}

KEEP_OBJECT = ["edit_date"]                 # object 列：取第一条

# ---------- 列名候选 ----------
ALIAS_DEAL = ["deal_num", "deal_number", "deal no", "deal no.",
              "Deal Number", "DealNumber", "dealnumber"]
ALIAS_DATE = ["edit_date", "editorial_date", "date_of_editorial",
              "Date of editorial", "date of editorial"]
ALIAS_TEXT = ["comments", "comment", "deal_comments", "deal_comment",
              "Deal comments", "Deal Comments"]

DROP_RAW_TEXT = ["editorial", "Deal rationale", "comments", "reg_entity_list",
                 "deal_comments", "Deal comments"]


# =====================================================================
# Log
# =====================================================================
_LOG, _T0 = [], time.time()

def log(msg=""):
    print(msg); _LOG.append(str(msg))

def banner(t):
    log("\n" + "=" * 82); log(t); log("=" * 82)


# =====================================================================
# 1. LM 词典
# =====================================================================
LM_SPEC = [
    ("pos",         "Positive"),
    ("neg",         "Negative"),
    ("uncertain",   "Uncertainty"),
    ("litigious",   "Litigious"),
    ("strongmodal", "Strong_Modal"),
    ("weakmodal",   "Weak_Modal"),
    ("constrain",   "Constraining"),
]

TOKEN_SETS, REGEX_PATS, LM_VOCAB_SIZES, LM_NONALPHA_WORDS = {}, {}, {}, {}

def build_word_re(word_list):
    if not word_list:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:{})\b".format("|".join(re.escape(w) for w in word_list)),
                      re.IGNORECASE)

def load_lm_dictionary():
    banner("STEP 1 — 载入 Loughran-McDonald 词典")
    lm = pd.read_csv(LM_DICT_PATH)
    lm["Word_lower"] = lm["Word"].astype(str).str.lower().str.strip()
    log(f"  词典总词数: {len(lm):,}")
    for out, col in LM_SPEC:
        if col not in lm.columns:
            raise KeyError(f"词典缺少列 {col}（实际列: {list(lm.columns)[:20]}）")
        words = [w for w in lm.loc[lm[col] != 0, "Word_lower"].dropna().tolist() if w]
        TOKEN_SETS[out] = set(words)
        REGEX_PATS[out] = build_word_re(words)
        LM_VOCAB_SIZES[out] = len(words)
        nonalpha = [w for w in words if not w.isalpha()]
        LM_NONALPHA_WORDS[out] = nonalpha
        log(f"  {out:<12s} vocab = {len(words):>6,}" +
            (f"  [含非纯字母 {len(nonalpha)} 词，token 模式无法命中] 例: {nonalpha[:5]}"
             if nonalpha else ""))
    return lm


# =====================================================================
# 2. 客观事实正则（沿用 v1，未改）
# =====================================================================
date_pat = re.compile(r"(\d{2}/\d{2}(?:/\d{2,4})?)")

reg_keywords = ["European Commission", "EU Commission", "CMA", "FTC", "DOJ",
               "CADE", "Fiscalia Nacional Economica", "SAMR", "Korea's FTC",
               "Federal Communications Commission", "Hart-Scott-Rodino",
               "Phase 1", "Phase 2"]
reg_pat = re.compile("|".join(re.escape(k) for k in reg_keywords), re.IGNORECASE)

price_pat   = re.compile(
    r"(?:EUR|USD|GBP|CHF|SEK|DKK|NOK|JPY|CNY|HKD|SGD|INR|KRW|AUD|CAD|\$|€|£|¥)\s*"
    r"[\d,.]+(?:\s*(?:billion|bn|million|mm|m)(?![a-z]))?", re.IGNORECASE)
premium_pat = re.compile(r"premium of\s*(\d+\.\d+)\s*per cent|premium.*?(\d+\.\d+)",
                         re.IGNORECASE)
rival_pat     = re.compile(r"rival|suitor|white knight|bid for|poised to bid",
                           re.IGNORECASE)
rumour_pat    = re.compile(r"reported that|rumour|speculation", re.IGNORECASE)
reject_pat    = re.compile(r"rejected", re.IGNORECASE)
unconditional_pat = re.compile(r"unconditional", re.IGNORECASE)
complete_pat  = re.compile(r"completed|closed the deal", re.IGNORECASE)
go_shop_pat   = re.compile(r"go-shop", re.IGNORECASE)
debt_fin_pat  = re.compile(r"loan|syndicated debt|assumption of debt", re.IGNORECASE)
remedy_pat    = re.compile(r"divest|sell off asset|remedy|concessions", re.IGNORECASE)

TOKEN_RE = re.compile(r"[a-z]+")


# =====================================================================
# 3. 工具函数
# =====================================================================
def parse_date(d_str):
    """欧洲式 DD/MM/YY 与 DD/MM/YYYY；两位数年份 <=30 -> 2000+，否则 1900+。"""
    try:
        parts = d_str.split("/")
        if len(parts) != 3:
            return None
        day, month, year_raw = int(parts[0]), int(parts[1]), int(parts[-1])
        year = (2000 + year_raw) if len(str(year_raw)) == 2 and year_raw <= 30 \
            else (1900 + year_raw) if len(str(year_raw)) == 2 else year_raw
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return datetime(year, month, day)
    except Exception:
        return None

def extract_all_dates(text):
    seen = set()
    for d in date_pat.findall(text):
        dt = parse_date(d)
        if dt is not None:
            seen.add(dt)
    return sorted(seen)

def count_match(text, pat):
    return len(pat.findall(text))

def has_match(text, pat):
    return 1 if pat.search(text) else 0

def get_premium_values(text):
    nums = []
    for t1, t2 in premium_pat.findall(text):
        for t in (t1, t2):
            if t and str(t).strip():
                nums.append(float(t))
    return nums

def count_lm(text, raw_text, cat):
    if LM_MATCH_MODE == "regex":
        return sum(1 for _ in REGEX_PATS[cat].finditer(raw_text))
    return sum(1 for t in TOKEN_RE.findall(text) if t in TOKEN_SETS[cat])


# =====================================================================
# 4. 特征抽取（editorial 行级）
# =====================================================================
def extract_features(raw_text):
    if raw_text is None or pd.isna(raw_text):
        return None
    raw_text = str(raw_text).strip()
    if len(raw_text) < 10:
        return None

    low = raw_text.lower()
    dates = extract_all_dates(raw_text)
    dt_min = dates[0] if dates else None
    dt_max = dates[-1] if dates else None
    total_days = (dt_max - dt_min).days if (dt_min and dt_max) else None

    prem_nums = get_premium_values(raw_text)
    tokens = TOKEN_RE.findall(low)
    word_total = len(tokens)
    reg_entities = list(set(reg_pat.findall(raw_text)))
    sentence_list = [s.strip() for s in raw_text.split(".") if len(s.strip()) > 5]
    lm = {cat: count_lm(low, raw_text, cat) for cat, _ in LM_SPEC}

    pres = {}
    if EMIT_LM_PRESENCE:
        pres = {f"has_lm_{cat}": int(lm[cat] > 0) for cat, _ in LM_SPEC}

    dens = {}
    if EMIT_LM_DENSITY:
        dens = {f"lm_{cat}_density": (lm[cat] / word_total if word_total > 0 else 0)
                for cat, _ in LM_SPEC}

    return {
        "dt_first": dt_min,
        "dt_last": dt_max,
        "total_timeline_days": total_days,
        "has_rumour": has_match(raw_text, rumour_pat),
        "has_target_reject": has_match(raw_text, reject_pat),
        "has_unconditional_offer": has_match(raw_text, unconditional_pat),
        "has_deal_complete": has_match(raw_text, complete_pat),
        "price_event_num": count_match(raw_text, price_pat),
        "max_premium_pct": max(prem_nums) if prem_nums else None,
        "avg_premium_pct": sum(prem_nums) / len(prem_nums) if prem_nums else None,
        "rival_bidder_num": count_match(raw_text, rival_pat),
        "has_goshop": has_match(raw_text, go_shop_pat),
        "reg_event_count": count_match(raw_text, reg_pat),
        "has_phase2_investigation": 1 if "phase 2" in low else 0,
        "has_reg_remedy": has_match(raw_text, remedy_pat),
        "reg_entity_list": reg_entities,
        "num_unique_reg": len(reg_entities),
        "has_debt_assumption": has_match(raw_text, debt_fin_pat),
        "comment_char_length": len(raw_text),
        "comment_wordcount": word_total,
        "sentence_event_count": len(sentence_list),
        **{f"lm_{cat}_count": lm[cat] for cat, _ in LM_SPEC},
        **pres, **dens,
    }


# =====================================================================
# 5. 列名解析
# =====================================================================
def resolve_cols(header):
    def pick(aliases):
        norm = {str(c).strip().lower(): c for c in header}
        for a in aliases:
            if a.strip().lower() in norm:
                return norm[a.strip().lower()]
        return None
    return pick(ALIAS_DEAL), pick(ALIAS_DATE), pick(ALIAS_TEXT)


# =====================================================================
# 6. 双模式交叉校验
# =====================================================================
def validate_lm_match(path, c_deal, c_date, c_text):
    banner("STEP 2 — LM 匹配算法交叉校验（token vs regex）")
    log(f"  样本: {VALIDATE_N:,} 行（来自第一个源文件）")
    smp = pd.read_csv(path, encoding="utf-8-sig", nrows=VALIDATE_N,
                      low_memory=False, engine="c", on_bad_lines="warn",
                      usecols=[c for c in (c_deal, c_date, c_text) if c])
    texts = smp[c_text].astype("object").tolist()

    log(f"\n  {'category':<14s} {'vocab':>8s} {'完全一致':>10s} {'一致率':>9s} "
        f"{'token均值':>11s} {'regex均值':>11s}")
    log("  " + "-" * 70)
    all_ok = True
    for cat, _ in LM_SPEC:
        tk, rg = [], []
        for t in texts:
            if t is None or pd.isna(t):
                tk.append(0); rg.append(0); continue
            raw = str(t).strip()
            if len(raw) < 10:
                tk.append(0); rg.append(0); continue
            low = raw.lower()
            tk.append(sum(1 for x in TOKEN_RE.findall(low) if x in TOKEN_SETS[cat]))
            rg.append(sum(1 for _ in REGEX_PATS[cat].finditer(raw)))
        tk, rg = np.array(tk), np.array(rg)
        same = int((tk == rg).sum())
        rate = same / max(len(tk), 1)
        flag = "" if rate >= 0.9999 else "  <-- 不一致"
        if rate < 0.9999:
            all_ok = False
        log(f"  {cat:<14s} {LM_VOCAB_SIZES[cat]:>8,} {same:>10,} {rate:>8.2%} "
            f"{tk.mean():>11.4f} {rg.mean():>11.4f}{flag}")

    if all_ok:
        log(f"\n  [OK] 两种算法在样本上逐行完全一致 -> 使用 LM_MATCH_MODE='{LM_MATCH_MODE}'")
    else:
        log(f"\n  [!!] 两种算法存在差异，差异来源通常为：")
        log(f"       - 词典中含非纯字母的词（token 模式无法命中）")
        log(f"       - 分词边界 vs \\b 边界对连字符/撇号的处理不同")
        for cat, ws in LM_NONALPHA_WORDS.items():
            if ws:
                log(f"       {cat}: 非纯字母词 {len(ws)} 个，例 {ws[:5]}")
    return all_ok


# =====================================================================
# 7. 单文件处理 -> 写 part（仍为 editorial 级）
# =====================================================================
def process_file(idx, path, total):
    name = os.path.basename(path)
    part_path = os.path.join(PARTS_DIR, f"01c_part_{idx:02d}.csv")
    banner(f"[{idx}/{total}] {name}")

    if RESUME and os.path.exists(part_path):
        n = sum(1 for _ in open(part_path, encoding="utf-8-sig", errors="ignore")) - 1
        log(f"  [SKIP] part 已存在 ({n:,} 行)，RESUME=True")
        return part_path, dict(file=name, rows=n, skipped=True)

    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    c_deal, c_date, c_text = resolve_cols(header)
    log(f"  列解析: deal='{c_deal}' | date='{c_date}' | text='{c_text}'")
    if c_deal is None or c_text is None:
        log(f"  [FATAL] 无法解析 deal/text 列。实际列: {header}")
        raise KeyError(name)

    usecols = [c for c in (c_deal, c_date, c_text) if c]
    dtype_map = {c_deal: "str", c_text: "str"}

    n_in = n_bad = n_short = n_written = 0
    t0 = time.time()
    first_write = True
    uniq, wordcounts = set(), []

    reader = pd.read_csv(path, encoding="utf-8-sig", engine="c",
                         on_bad_lines="warn", usecols=usecols,
                         dtype=dtype_map, chunksize=CHUNK_ROWS, low_memory=False)
    for ci, chunk in enumerate(reader, 1):
        n_in += len(chunk)
        dn = pd.to_numeric(chunk[c_deal], errors="coerce")
        bad = dn.isna() | (dn % 1 != 0) | (dn < 1e4) | (dn >= 1e13)
        n_bad += int(bad.sum())
        chunk = chunk.loc[~bad].copy()
        if chunk.empty:
            continue
        chunk["deal_num"] = dn.loc[~bad].astype("int64")

        feats, n_local_short = [], 0
        for t in chunk[c_text].tolist():
            f = extract_features(t)
            feats.append({} if f is None else f)
            if f is None:
                n_local_short += 1
        fdf = pd.DataFrame(feats, index=chunk.index)
        n_short += n_local_short

        keep = fdf["comment_wordcount"].notna()
        n_short += int((~keep).sum())
        fdf, chunk = fdf.loc[keep], chunk.loc[keep]
        if chunk.empty:
            continue

        out = pd.concat(
            [chunk[["deal_num"] + ([c_date] if c_date else [])].reset_index(drop=True),
             fdf.reset_index(drop=True)], axis=1)
        if c_date:
            out = out.rename(columns={c_date: "edit_date"})

        uniq.update(out["deal_num"].tolist())
        wordcounts.append(out["comment_wordcount"].to_numpy())

        out.to_csv(part_path, mode="w" if first_write else "a",
                   header=first_write, index=False, encoding="utf-8-sig")
        first_write = False
        n_written += len(out)
        log(f"    chunk {ci:>3d}: +{len(out):>7,}  (累计 {n_written:,}) "
            f"[{time.time()-t0:.0f}s]")

    wc = np.concatenate(wordcounts) if wordcounts else np.array([])
    stats = dict(file=name, rows=n_written, n_in=n_in, n_bad=n_bad,
                 n_short=n_short, uniq=len(uniq), skipped=False,
                 secs=time.time() - t0)
    log(f"  -> part: {n_written:,} 行 | unique deal_num {len(uniq):,} | "
        f"无效 deal_num {n_bad:,} | 短文本剔除 {n_short:,} | {stats['secs']:.0f}s")
    stats["_wc"] = wc
    return part_path, stats


# =====================================================================
# 8. append 分片 -> editorial 级汇总
# =====================================================================
def append_parts(part_paths):
    banner("STEP 4 — append 分片 -> editorial 级汇总")
    log(f"  {len(part_paths)} 个 part -> {OUT_RAW}")

    if os.path.exists(OUT_RAW):
        os.remove(OUT_RAW)

    total_rows = 0
    uniq_all = set()
    dup_rows = 0
    sums = {}
    wc_all, first = [], True

    for i, p in enumerate(part_paths, 1):
        if not os.path.exists(p):
            log(f"  [WARN] part 缺失，跳过: {p}")
            continue
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["deal_num"])
        total_rows += len(df)
        dup_rows += len(df) - df["deal_num"].nunique()
        uniq_all.update(df["deal_num"].astype("int64").tolist())

        for c in df.select_dtypes(include=[np.number]).columns:
            if c == "deal_num":
                continue
            sums[c] = sums.get(c, 0.0) + float(pd.to_numeric(df[c], errors="coerce").sum())
        if "comment_wordcount" in df.columns:
            wc_all.append(df["comment_wordcount"].to_numpy())

        df.to_csv(OUT_RAW, mode="w" if first else "a",
                  header=first, index=False, encoding="utf-8-sig")
        first = False
        del df
        log(f"    part {i:>2d}: 累计 {total_rows:,} 行")

    wc = np.concatenate(wc_all) if wc_all else np.array([])
    log(f"\n  editorial 级汇总：{total_rows:,} 行 | "
        f"{os.path.getsize(OUT_RAW)/1024/1024:.1f} MB")

    means = {c: v / max(total_rows, 1) for c, v in sums.items()}
    return total_rows, len(uniq_all), dup_rows, means, wc, sums


# =====================================================================
# 9. deal 级聚合（v3 重写）
# =====================================================================
def collapse_to_deal(sums_from_append):
    """按语义分四类聚合，输出一行一 deal。返回 (df_deal, deal_means)。"""
    banner("STEP 4c — 聚合到 deal 级")
    df = pd.read_csv(OUT_RAW, encoding="utf-8-sig", low_memory=False)
    n_raw = len(df)
    u_raw = df["deal_num"].nunique()
    log(f"  聚合前：{n_raw:,} 行 / {u_raw:,} deal（editorial 级）")

    # ---- 规则 1：计数类 sum ----
    agg = {c: "sum" for c in SUM_COLS if c in df.columns}
    # ---- 规则 2：二值/存在类 max ----
    agg.update({c: "max" for c in MAX_COLS if c in df.columns})
    # ---- 规则 3：密度类 先 sum 计数与词数，下面重算 ----
    for c in DENSITY_COLS:
        base = c.replace("_density", "_count")
        if base in df.columns:
            agg[base] = "sum"
    if "comment_wordcount" not in agg:
        agg["comment_wordcount"] = "sum"
    # ---- 规则 4：日期/溢价取极值 ----
    for c, fn in EXTREME_COLS.items():
        if c in df.columns:
            agg[c] = fn
    # ---- 规则 5：object 列取第一条 ----
    for c in KEEP_OBJECT:
        if c in df.columns:
            agg[c] = "first"

    g = df.groupby("deal_num", sort=False)
    out = g.agg(agg).reset_index()

    # ★ 评论条数：用 size()，不是假列名
    out["comment_n"] = g.size().reset_index(drop=True)

    # ---- 派生量重算 ----
    # 密度 = 词频合计 / 词数合计（禁止先算密度再平均）
    for c in LM_CATS:
        cnt = f"lm_{c}_count"
        if cnt in out.columns:
            wc = out["comment_wordcount"].replace(0, np.nan)
            out[f"lm_{c}_density"] = (out[cnt] / wc).fillna(0)

    # 时间跨度
    if {"dt_first", "dt_last"}.issubset(out.columns):
        d1 = pd.to_datetime(out["dt_first"], errors="coerce")
        d2 = pd.to_datetime(out["dt_last"],   errors="coerce")
        out["total_timeline_days"] = (d2 - d1).dt.days

    # ---- 清理原始文本列 ----
    out = out.drop(columns=[c for c in DROP_RAW_TEXT if c in out.columns],
                   errors="ignore")

    n_out = len(out)
    log(f"  聚合后：{n_out:,} 行 / {out['deal_num'].nunique():,} deal")
    log(f"  平均每条交易 {out['comment_n'].mean():.2f} 条评论 "
        f"(p50={out['comment_n'].median():.0f}, max={out['comment_n'].max():.0f})")

    if n_out != out["deal_num"].nunique():
        raise RuntimeError(f"[FATAL] 聚合后仍非一行一 deal：{n_out} vs "
                          f"{out['deal_num'].nunique()}")
    log(f"  [OK] 一行一 deal 校验通过")

    num = out.select_dtypes(include=[np.number]).drop(columns=["deal_num"], errors="ignore")
    deal_means = {c: float(num[c].mean()) for c in num.columns}
    return out, deal_means

# =====================================================================
# Main
# =====================================================================
def main():
    banner("01c_comments_features.py — v3 DEAL-LEVEL")
    log(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"src dir : {SRC_DIR}")
    log(f"pattern : {SRC_GLOB}")
    log(f"parts   : {PARTS_DIR}")
    log(f"out_raw : {OUT_RAW}")
    log(f"out     : {OUT_FILE}   <-- 下游 06 直接使用（deal 级）")
    log(f"mode    : LM_MATCH_MODE={LM_MATCH_MODE} | chunk={CHUNK_ROWS:,} | "
        f"resume={RESUME} | collapse={COLLAPSE_TO_DEAL}")

    files = sorted(glob.glob(os.path.join(SRC_DIR, SRC_GLOB)))
    if not files:
        log(f"\n[FATAL] 未找到文件: {os.path.join(SRC_DIR, SRC_GLOB)}")
        return
    log(f"\n  发现 {len(files)} 个源文件：")
    for f in files:
        log(f"    - {os.path.basename(f)}  ({os.path.getsize(f)/1024/1024:.1f} MB)")

    load_lm_dictionary()

    h0 = list(pd.read_csv(files[0], nrows=0, encoding="utf-8-sig").columns)
    c0 = resolve_cols(h0)
    if VALIDATE_LM_MATCH and c0[2]:
        validate_lm_match(files[0], c0[0], c0[1], c0[2])

    banner("STEP 3 — 逐文件抽取 -> part")
    parts, all_stats = [], []
    for i, p in enumerate(files, 1):
        pp, st = process_file(i, p, len(files))
        if pp:
            parts.append(pp)
        all_stats.append(st)

    total_rows, uniq, dup, means, wc, sums = append_parts(parts)

    # ---- 聚合 ----
    if COLLAPSE_TO_DEAL:
        out_deal, deal_means = collapse_to_deal(sums)
        out_deal.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        log(f"\n  [SAVED] {OUT_FILE}  ({len(out_deal):,} deals, "
            f"{os.path.getsize(OUT_FILE)/1024/1024:.1f} MB)")

    # ---- 诊断 ----
    banner("STEP 5 — 诊断报告")
    d = ["=== 01c COMMENTS FEATURES DIAGNOSTICS ===",
         f"generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"src files : {len(files)}",
         f"LM match  : {LM_MATCH_MODE}",
         f"collapse  : {COLLAPSE_TO_DEAL} (True = 输出 deal 级)", "",
         "--- per-file ---"]
    for s in all_stats:
        if s.get("skipped"):
            d.append(f"  {s['file']:<40s} (skipped, {s['rows']:,} rows)")
        else:
            d.append(f"  {s['file']:<40s} in={s['n_in']:>8,} out={s['rows']:>8,} "
                     f"bad_id={s['n_bad']:>6,} short={s['n_short']:>7,} "
                     f"uniq={s['uniq']:>8,} {s['secs']:.0f}s")
    d += ["", "--- totals (editorial level) ---",
          f"Total rows processed : {total_rows:,}",
          f"Unique deal_num      : {uniq:,}",
          f"Duplicate rows       : {dup:,}",
          f"Unit of observation  : {'ONE ROW PER DEAL' if total_rows == uniq else 'ONE ROW PER EDITORIAL'}"]
    if len(wc):
        d += ["",
              f"comment_wordcount (editorial): "
              f"p25={np.percentile(wc,25):.0f} p50={np.median(wc):.0f} "
              f"p75={np.percentile(wc,75):.0f} mean={wc.mean():.1f}"]
    d += ["", "--- LM presence rates (editorial 级均值) ---"]
    for c, _ in LM_SPEC:
        k = f"has_lm_{c}"
        if k in means:
            d.append(f"  {k:<24s} {means[k]:.2%}")
    d += ["", "--- LM mean counts (editorial 级均值) ---"]
    for c, _ in LM_SPEC:
        k = f"lm_{c}_count"
        if k in means:
            d.append(f"  {k:<24s} {means[k]:.4f}")

    if COLLAPSE_TO_DEAL:
        d += ["", "--- deal 级汇总（聚合后重算）---",
              f"Deals               : {len(out_deal):,}"]
        for c in ["comment_n", "comment_wordcount", "lm_neg_count",
                  "lm_uncertain_count", "lm_pos_count"]:
            if c in out_deal.columns:
                v = out_deal[c]
                d.append(f"  {c:<24s} mean={v.mean():.3f}  p50={v.median():.1f}  "
                         f"非零占比={float((v>0).mean()):.1%}")
        d += ["", "--- deal 级 LM 密度（重算：count/wordcount）---"]
        for c in LM_CATS:
            k = f"lm_{c}_density"
            if k in out_deal.columns:
                d.append(f"  {k:<24s} {out_deal[k].mean():.5f}")

    d += ["", "--- other features (editorial 级均值) ---"]
    for k in sorted(means):
        if not (k.startswith("has_lm_") or k.endswith("_count") or
                k.endswith("_density")):
            d.append(f"  {k:<32s} {means[k]:.4f}")
    d += ["", f"elapsed: {time.time()-_T0:.0f}s"]

    txt = "\n".join(d)
    log("\n" + txt)
    with open(DIAG_FILE, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    log(f"\nDiagnostics -> {DIAG_FILE}")
    log(f"\nScript 01c complete. ({time.time()-_T0:.0f}s)")


if __name__ == "__main__":
    main()