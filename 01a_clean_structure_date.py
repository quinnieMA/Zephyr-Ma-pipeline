# -*- coding: utf-8 -*-
r"""
01a_clean_structure_date.py — Structure & dates export (FIXED + DEDUP + CLIP)
===================================================================
Input  : D:\zephyr\structure_date\cleaned_output\MA_structure_date.csv
Output : D:\MA\data\cleaned\01_deal_structure_date.csv

===================================================================
FIX LIST
===================================================================
[FIX-1] 传闻占位符问题：输出三套判据 rumor_jfe / rumor_status / rumor_any。
[FIX-2] pay_mix_with_shares 恒为 0：单次遍历算完，加内部一致性校验。
[FIX-3] Excel 序列号闸门：只接受 20000..60000 才走序列号路径。
[FIX-4] 日期垃圾值清洗：lead 截到 [-30, 1825]，DaysToCompletion 截到 [0, 3650]。
        *修复点*：确保 parse_dates_robust 在计算差值前执行，防止裁剪失效。
[FIX-5] ln_days 覆盖率：改用 log1p(days)，必须在裁剪后执行。
[FIX-6] 完成状态口径：显式归一化 deal_status。
[FIX-7] SAMPLE MOMENTS 块。
[DEDUP] to_csv 前按 deal_num 智能去重并 assert 一行一 deal。
"""

import os
import sys
import time
import traceback
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# =====================================================================
# Module 1 — Config
# =====================================================================
BASE    = r"D:\MA"
ZEPHYR  = r"D:\zephyr"
CLEANED = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEANED, exist_ok=True)

WORK_LOG = os.path.join(CLEANED, "01a_structure_date.txt")
STRICT = False

RAW_FILES = {
    "structure_date": os.path.join(ZEPHYR, "structure_date", "cleaned_output", "MA_structure_date.csv"),
}

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

# --- 日期合理性闸门 ---
SERIAL_LO, SERIAL_HI = 20000, 60000          # 1954-01-01 .. 2064-05-xx
DT_LO = pd.Timestamp("1900-01-01")
DT_HI = pd.Timestamp("2035-12-31")

LEAD_LO, LEAD_HI = -30, 1825                 # 传闻提前天数合理范围
DAYS_LO, DAYS_HI = 0, 3650                   # 完成耗时合理范围

RUMOR_LEAD_MIN = 3                           # JFE 2021 口径：>= 3 天

# --- 完成状态口径 ---
STATUS_COMPLETED = ("completed", "completed assumed")


# ---------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------
_LOG_LINES = []
_SCRIPT_NAME = os.path.basename(__file__)
_RUN_STARTED = None
_LOG_INITIALISED = False


def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))


def _ensure_log_header():
    global _LOG_INITIALISED, _RUN_STARTED
    if _LOG_INITIALISED:
        return
    _RUN_STARTED = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(CLEANED, exist_ok=True)
    header = [
        "", "#" * 78,
        f"# 01a run — {_SCRIPT_NAME}",
        f"# started : {_RUN_STARTED}",
        f"# cwd     : {os.getcwd()}",
        f"# python  : {sys.version.split()[0]}",
        f"# pandas  : {pd.__version__}",
        "#" * 78,
    ]
    with open(WORK_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
    _LOG_INITIALISED = True


def flush_log():
    _ensure_log_header()
    if not _LOG_LINES:
        return
    with open(WORK_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG_LINES) + "\n")
    _LOG_LINES.clear()


def log_checkpoint(tag=""):
    flush_log()
    with open(WORK_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"--- checkpoint {tag} @ {time.strftime('%H:%M:%S')} ---\n")


def module_banner(num, title):
    log("\n" + "=" * 78)
    log(f"Module {num} — {title}")
    log("=" * 78)


# ---------------------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------------------
def clean_missing(df):
    """Replace missing-style tokens with NaN; strip strings."""
    try:
        cols = df.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        cols = df.select_dtypes(include=["object"]).columns
    for col in cols:
        df[col] = df[col].astype("object")
        df[col] = df[col].replace(MISSING_VALS, np.nan)
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
            df.loc[mask, col] = df.loc[mask, col].replace("", np.nan)
    return df


KEY_ALIASES = {
    "deal_num": ["deal_num", "deal number", "dealnumber", "deal id"],
}


def find_col(df, candidates):
    fold = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = fold.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None


def read_raw(file_key, label):
    fp = RAW_FILES[file_key]
    if not os.path.exists(fp):
        log(f"\n  x [MISSING INPUT] {label}: {fp}")
        if STRICT:
            raise FileNotFoundError(fp)
        return None
    df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    df = clean_missing(df)

    deal_col = find_col(df, KEY_ALIASES["deal_num"]) or "deal_num"
    if deal_col != "deal_num" and deal_col in df.columns:
        df = df.rename(columns={deal_col: "deal_num"})

    if "deal_num" in df.columns:
        n_before = int(df["deal_num"].isna().sum())
        df["deal_num"] = df["deal_num"].ffill()
        n_after = int(df["deal_num"].isna().sum())
        log(f"  [{label}] deal_num ffill: NaN {n_before:,} -> {n_after:,} "
            f"(filled {n_before - n_after:,})")
        df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")
    else:
        log(f"  x {label}: no deal_num column found")
        return None

    log(f"  [{label}] loaded: {len(df):,} rows | {df.shape[1]} cols | "
        f"unique deal_num: {df['deal_num'].nunique():,}")
    return df


# =====================================================================
# Helper Functions for Dates & Payments
# =====================================================================

def parse_dates_robust(series, label=""):
    """
    [FIX-3] Robust BvD date parsing with a serial-range gate.
    Only values inside [SERIAL_LO, SERIAL_HI] are treated as Excel serials.
    """
    s_num = pd.to_numeric(series, errors="coerce")
    ok_serial = s_num.between(SERIAL_LO, SERIAL_HI)

    dt_excel = pd.to_datetime(s_num.where(ok_serial), unit="D",
                              origin="1899-12-30", errors="coerce")
    dt_str = pd.to_datetime(series, errors="coerce")

    out = dt_excel.where(dt_excel.notna(), dt_str)

    n_bad = int(out.notna().sum() - out.between(DT_LO, DT_HI).sum())
    out = out.where(out.between(DT_LO, DT_HI))
    if label:
        log(f"    [{label}] parsed non-null {int(out.notna().sum()):,} | "
            f"out-of-range clipped {n_bad:,}")
    return out


def build_payment_vars(df):
    """
    [FIX-2] Build payment method dummies and classifications BEFORE dedup.
    """
    log("\n  -- Building Payment Method Variables --")
    if "deal_pay_method" not in df.columns:
        log("  [warn] deal_pay_method not found. Creating empty pay_* vars.")
        return pd.DataFrame(index=pd.Index(df["deal_num"].unique(), name="deal_num"))

    sub = df[["deal_num", "deal_pay_method"]].dropna(subset=["deal_pay_method"]).copy()
    if sub.empty:
        return pd.DataFrame(index=pd.Index(df["deal_num"].unique(), name="deal_num"))

    val_lower = sub["deal_pay_method"].astype(str).str.lower()

    sub["is_cash"]   = val_lower.str.contains(r"cash", na=False).astype(int)
    sub["is_shares"] = val_lower.str.contains(r"shares|equity|stock", na=False).astype(int)
    sub["is_debt"]   = val_lower.str.contains(r"debt|bonds|loan|note", na=False).astype(int)
    sub["is_other"]  = ((sub["is_cash"] == 0) &
                        (sub["is_shares"] == 0) &
                        (sub["is_debt"] == 0)).astype(int)

    flag_cols = ["is_cash", "is_shares", "is_debt", "is_other"]
    agg = sub.groupby("deal_num")[flag_cols].max().reset_index()
    agg = agg.rename(columns={c: c.replace("is_", "pay_has_") for c in flag_cols})

    has = ["pay_has_cash", "pay_has_shares", "pay_has_debt", "pay_has_other"]
    agg["pay_n_class"] = agg[has].sum(axis=1)

    # --- 一次性算完，口径统一 ---
    agg["pay_pure_cash"]   = ((agg["pay_n_class"] == 1) & (agg["pay_has_cash"] == 1)).astype(int)
    agg["pay_pure_shares"] = ((agg["pay_n_class"] == 1) & (agg["pay_has_shares"] == 1)).astype(int)
    agg["pay_pure_debt"]   = ((agg["pay_n_class"] == 1) & (agg["pay_has_debt"] == 1)).astype(int)
    agg["pay_pure_other"]  = ((agg["pay_n_class"] == 1) & (agg["pay_has_other"] == 1)).astype(int)

    is_mix = (agg["pay_n_class"] > 1).astype(int)
    agg["pay_mix"] = is_mix
    agg["pay_mix_cash_shares"] = (agg["pay_has_cash"] & agg["pay_has_shares"]).astype(int)
    agg["pay_mix_with_shares"] = (is_mix & agg["pay_has_shares"]).astype(int)
    agg["pay_mix_with_cash"]   = (is_mix & agg["pay_has_cash"]).astype(int)
    agg["pay_mix_with_debt"]   = (is_mix & agg["pay_has_debt"]).astype(int)

    def classify_pay(r):
        if r["pay_n_class"] == 0:
            return "Unknown"
        if r["pay_pure_cash"]:   return "Pure Cash"
        if r["pay_pure_shares"]: return "Pure Shares"
        if r["pay_pure_debt"]:   return "Pure Debt"
        if r["pay_pure_other"]:  return "Pure Other"
        if r["pay_mix_cash_shares"]: return "Mix: Cash+Shares"
        return "Mix: Other"

    agg["pay_class"] = agg.apply(classify_pay, axis=1)

    # --- [FIX-2] 内部一致性自检 ---
    n_cs = int(agg["pay_mix_cash_shares"].sum())
    n_ws = int(agg["pay_mix_with_shares"].sum())
    log(f"  [check] pay_mix_cash_shares={n_cs:,} | pay_mix_with_shares={n_ws:,}")
    if n_cs > 0 and n_ws == 0:
        log("  [!!] INCONSISTENT: cash+shares>0 but mix_with_shares==0")
    else:
        log("  [check] OK — mix_with_shares consistent with cash+shares")

    log(f"  Payment classification done. Deals with payment info: {len(agg):,}")
    log(f"  Pure Cash: {int(agg['pay_pure_cash'].sum()):,} | "
        f"Pure Shares: {int(agg['pay_pure_shares'].sum()):,} | "
        f"Mix: {int(agg['pay_mix'].sum()):,}")

    return agg.set_index("deal_num")


def report_coverage(df, label, cols=None):
    cols = list(df.columns) if cols is None else [c for c in cols if c in df.columns]
    log(f"\n  [{label}] COVERAGE DIAGNOSTICS — {len(df):,} rows")
    log(f"    {'Column':<45s} {'Non-Null':>10s} {'Pct':>7s}   {'Missing':>10s} {'Pct':>7s}")
    log("    " + "-" * 85)
    for c in cols:
        n = int(df[c].notna().sum())
        pct = n / len(df) * 100 if len(df) else 0.0
        m = len(df) - n
        mpct = 100.0 - pct
        log(f"    {c:<45s} {n:>10,} {pct:>6.2f}%   {m:>10,} {mpct:>6.2f}%")


def report_descriptives(df, label, cols=None):
    num_cols = df.select_dtypes(include=[np.number, "Int64"]).columns.tolist()
    if cols:
        num_cols = [c for c in cols if c in num_cols]
    if not num_cols:
        log(f"\n  [{label}] No numeric columns for descriptive stats.")
        return

    log(f"\n  [{label}] DESCRIPTIVE STATS — numeric ({len(num_cols)} columns)")
    log(f"    {'column':<35s} {'min':>15s} {'median':>15s} {'mean':>15s} "
        f"{'max':>15s} {'std':>15s}")
    log("    " + "-" * 110)
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            log(f"    {c:<35s} {'ALL MISSING':>15s}")
            continue
        mn, md, me, mx, sd = s.min(), s.median(), s.mean(), s.max(), s.std()
        fmt = lambda x: f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"
        log(f"    {c:<35s} {fmt(mn):>15s} {fmt(md):>15s} {fmt(me):>15s} "
            f"{fmt(mx):>15s} {fmt(sd):>15s}")


# =====================================================================
# Main Execution
# =====================================================================
def main():
    module_banner("M2.3+M6", "Structure & dates export -> 01_deal_structure_date.csv (FIXED+DEDUP+CLIP)")

    df_sd = read_raw("structure_date", "structure_date")
    if df_sd is None:
        log("\n  [skip] structure_date module aborted (no input).")
        flush_log()
        return

    df_sd = df_sd[df_sd["deal_num"].notna()].copy()
    log(f"\n  after dropna(deal_num): {len(df_sd):,} rows")
    log_checkpoint("after load")

    # -----------------------------------------------------------------
    # 0. deal_status 归一化（供传闻与完成判定共用）
    # -----------------------------------------------------------------
    if "deal_status" in df_sd.columns:
        df_sd["deal_status_clean"] = (df_sd["deal_status"].astype(str)
                                      .str.strip().str.lower())
        log("\n  -- deal_status distribution (normalized) --")
        vc = df_sd["deal_status_clean"].value_counts(dropna=False)
        for k, v in vc.items():
            log(f"    {str(k):<28s} {v:>12,}  {v/len(df_sd)*100:>6.2f}%")
    else:
        df_sd["deal_status_clean"] = np.nan
        log("  [warn] deal_status not found.")

    # -----------------------------------------------------------------
    # 1. PAYMENT METHOD DUMMIES & CLASSIFICATION (BEFORE DEDUP)
    # -----------------------------------------------------------------
    pay_vars_df = build_payment_vars(df_sd)

    # -----------------------------------------------------------------
    # 2. CATEGORICAL AGGREGATION (Standard fields)
    # -----------------------------------------------------------------
    CATEGORICAL_AGGR = ["deal_struct", "deal_fin", "deal_type"]
    log("\nCATEGORICAL AGG (struct, fin, type)")
    _agg_frames = []
    for _src in CATEGORICAL_AGGR:
        if _src not in df_sd.columns:
            continue
        _sub = df_sd.dropna(subset=[_src])
        if _sub.empty:
            continue
        _g = (_sub.groupby("deal_num")[_src]
              .apply(lambda x: "|".join(sorted(set(x.astype(str)))))
              .rename(_src + "_all").to_frame())
        _g[_src + "_n"] = _g[_src + "_all"].str.split("|").apply(len)
        _g = _g.reset_index()
        log(f"  [{_src}] {len(_g):,} deals | multi {int((_g[_src+'_n']>1).sum()):,}")
        _agg_frames.append(_g)

    _agg_all = None
    if _agg_frames:
        _agg_all = _agg_frames[0]
        for _f in _agg_frames[1:]:
            _agg_all = _agg_all.merge(_f, on="deal_num", how="outer")
        _agg_all["deal_num"] = pd.to_numeric(_agg_all["deal_num"], errors="coerce").astype("Int64")

    if "deal_pay_method" in df_sd.columns:
        _sub_pay = df_sd.dropna(subset=["deal_pay_method"])
        if not _sub_pay.empty:
            _g_pay = (_sub_pay.groupby("deal_num")["deal_pay_method"]
                      .apply(lambda x: "|".join(sorted(set(x.astype(str)))))
                      .rename("deal_pay_method_all").to_frame())
            _g_pay["deal_pay_method_n"] = _g_pay["deal_pay_method_all"].str.split("|").apply(len)
            _g_pay = _g_pay.reset_index()
            _g_pay["deal_num"] = pd.to_numeric(_g_pay["deal_num"], errors="coerce").astype("Int64")
            if _agg_all is not None:
                _agg_all = _agg_all.merge(_g_pay, on="deal_num", how="outer")
            else:
                _agg_all = _g_pay

    # -----------------------------------------------------------------
    # 2b. 日期：在去重前按 deal 聚合（取首个非空），避免 keep="first" 丢信息
    # -----------------------------------------------------------------
    DATE_COLS = ["announced_d", "completed_d", "withdrawn_d",
                 "assumed_comp_d", "rumour_d"]
    DATE_COLS = [c for c in DATE_COLS if c in df_sd.columns]

    _date_frames = []
    for _c in DATE_COLS:
        _s = df_sd[["deal_num", _c]].dropna(subset=[_c])
        if _s.empty:
            continue
        _g = _s.groupby("deal_num")[_c].first().rename(_c).to_frame().reset_index()
        _date_frames.append(_g)
    if _date_frames:
        _date_all = _date_frames[0]
        for _f in _date_frames[1:]:
            _date_all = _date_all.merge(_f, on="deal_num", how="outer")
        _date_all["deal_num"] = pd.to_numeric(_date_all["deal_num"], errors="coerce").astype("Int64")
        if _agg_all is not None:
            _agg_all = _agg_all.merge(_date_all, on="deal_num", how="outer")
        else:
            _agg_all = _date_all

    # Deduplicate main dataframe (preliminary)
    df_sd = df_sd.drop_duplicates(subset=["deal_num"], keep="first").copy()
    df_sd["deal_num"] = pd.to_numeric(df_sd["deal_num"], errors="coerce").astype("Int64")
    df_sd = df_sd.drop(columns=[c for c in DATE_COLS if c in df_sd.columns])

    # Merge aggregations
    if _agg_all is not None:
        df_sd = df_sd.merge(_agg_all, on="deal_num", how="left")

    # Merge Payment Dummies/Classes
    if not pay_vars_df.empty:
        df_sd = df_sd.join(pay_vars_df, on="deal_num")

    pay_dummy_cols = ["pay_has_cash", "pay_has_shares", "pay_has_debt", "pay_has_other",
                      "pay_n_class", "pay_pure_cash", "pay_pure_shares", "pay_pure_debt",
                      "pay_pure_other", "pay_mix", "pay_mix_cash_shares",
                      "pay_mix_with_shares", "pay_mix_with_cash", "pay_mix_with_debt"]
    for c in pay_dummy_cols:
        if c in df_sd.columns:
            df_sd[c] = df_sd[c].fillna(0).astype(int)

    df_sd["pay_unknown"] = (df_sd.get("pay_n_class", pd.Series(0, index=df_sd.index)) == 0).astype(int)

    log(f"  after aggregation + dedup + pay merge: {len(df_sd):,} rows")
    log_checkpoint("after aggregation")

    # -----------------------------------------------------------------
    # 3. DATE PROCESSING (必须放在算差值和裁剪之前！)
    # -----------------------------------------------------------------
    log("\n  -- Processing Dates (Parsing) --")
    for _c in DATE_COLS:
        if _c in df_sd.columns:
            df_sd[_c] = parse_dates_robust(df_sd[_c], label=_c)

    # -----------------------------------------------------------------
    # 3b. [FIX-1] 传闻：三套判据 + 占位符诊断 + [FIX-4] 裁剪
    # -----------------------------------------------------------------
    log("\n  -- Rumor block --")

    # (a) 占位符诊断：rumour_d == announced_d 的占比
    if "rumour_d" in df_sd.columns and "announced_d" in df_sd.columns:
        _eq = (df_sd["rumour_d"].notna() & df_sd["announced_d"].notna() &
               (df_sd["rumour_d"] == df_sd["announced_d"]))
        n_eq = int(_eq.sum())
        n_both = int((df_sd["rumour_d"].notna() & df_sd["announced_d"].notna()).sum())
        log(f"  [DIAG] rumour_d == announced_d : {n_eq:,} / {n_both:,} "
            f"({n_eq/max(n_both,1)*100:.2f}%)")
        if n_both > 0 and n_eq / n_both > 0.80:
            log("  [!!] rumour_d looks like a PLACEHOLDER filled with announced_d.")
            log("       Do NOT rely on the day-level criterion alone.")
        df_sd["rumour_eq_announced"] = _eq.astype(int)

    # (b) 提前天数计算 + [FIX-4] 裁剪
    if "rumour_d" in df_sd.columns and "announced_d" in df_sd.columns:
        df_sd["rumor_lead_days"] = (df_sd["announced_d"] - df_sd["rumour_d"]).dt.days
        
        # 强制裁剪
        _raw = df_sd["rumor_lead_days"]
        n_clip = int(_raw.notna().sum() - _raw.between(LEAD_LO, LEAD_HI).sum())
        df_sd["rumor_lead_days"] = _raw.where(_raw.between(LEAD_LO, LEAD_HI))
        
        log(f"  [FIX-4] lead clipped to [{LEAD_LO}, {LEAD_HI}] : {n_clip:,} rows")
        log(f"  rumor_lead_days non-null: {int(df_sd['rumor_lead_days'].notna().sum()):,}")
    else:
        log("  [warn] rumour_d / announced_d missing -> lead not computed.")
        df_sd["rumor_lead_days"] = np.nan

    log("\n  -- Rumor lead distribution (cleaned) --")
    _lead = df_sd["rumor_lead_days"]
    for _lab, _m in [("== 0", _lead == 0), ("<  0", _lead < 0),
                     (f">= {RUMOR_LEAD_MIN}", _lead >= RUMOR_LEAD_MIN),
                     (">= 30", _lead >= 30), (">= 90", _lead >= 90),
                     (">= 365", _lead >= 365), (">= 730", _lead >= 730),
                     (">= 1825", _lead >= 1825)]:
        log(f"    {_lab:<8s}: {int(_m.sum()):>10,}")

    # (c) 三套判据
    df_sd["rumor_jfe"] = (df_sd["rumor_lead_days"] >= RUMOR_LEAD_MIN).fillna(False).astype(int)

    if "deal_status_clean" in df_sd.columns:
        df_sd["has_rumor_status"] = (df_sd["deal_status_clean"]
                                     .str.contains("rumour|rumor", na=False)).astype(int)
    else:
        df_sd["has_rumor_status"] = 0

    df_sd["rumor_any"] = ((df_sd["rumor_jfe"] == 1) |
                          (df_sd["has_rumor_status"] == 1)).astype(int)

    log(f"\n  >> rumor_jfe  (lead >= {RUMOR_LEAD_MIN}d)        : "
        f"{int(df_sd['rumor_jfe'].sum()):,}  "
        f"({df_sd['rumor_jfe'].mean()*100:.2f}%)")
    log(f"  >> has_rumor_status                      : "
        f"{int(df_sd['has_rumor_status'].sum()):,}  "
        f"({df_sd['has_rumor_status'].mean()*100:.2f}%)")
    log(f"  >> rumor_any  (either)                   : "
        f"{int(df_sd['rumor_any'].sum()):,}  "
        f"({df_sd['rumor_any'].mean()*100:.2f}%)")

    log("\n  -- Cross-check: day-level vs status --")
    try:
        _ct = pd.crosstab(df_sd["has_rumor_status"], df_sd["rumor_jfe"],
                          margins=True)
        log(str(_ct))
    except Exception as e:
        log(f"    [warn] crosstab failed: {e}")

    # -----------------------------------------------------------------
    # 4. [FIX-6] 完成状态
    # -----------------------------------------------------------------
    log("\n  -- Completion --")
    if "deal_status_clean" in df_sd.columns:
        _st = df_sd["deal_status_clean"]
        df_sd["completed_flag"] = _st.str.startswith(STATUS_COMPLETED).astype(int)
        df_sd["completed_strict"] = (_st == "completed").astype(int)
        log(f"    completed_flag   (Completed*) : "
            f"{int(df_sd['completed_flag'].sum()):,} "
            f"({df_sd['completed_flag'].mean()*100:.2f}%)")
        log(f"    completed_strict (Completed)  : "
            f"{int(df_sd['completed_strict'].sum()):,} "
            f"({df_sd['completed_strict'].mean()*100:.2f}%)")
    else:
        df_sd["completed_flag"] = np.nan
        df_sd["completed_strict"] = np.nan

    # -----------------------------------------------------------------
    # 5. [FIX-4/5] DaysToCompletion 与 ln_days (先解析->再算差->再裁剪->最后log1p)
    # -----------------------------------------------------------------
    log("\n  -- DaysToCompletion & ln_days --")
    if "completed_d" in df_sd.columns and "announced_d" in df_sd.columns:
        # 此时 completed_d 和 announced_d 已经是可靠的 datetime 对象
        df_sd["DaysToCompletion"] = (df_sd["completed_d"] - df_sd["announced_d"]).dt.days
        
        # [FIX-4] 强制裁剪
        _d = df_sd["DaysToCompletion"]
        n_clip_d = int(_d.notna().sum() - _d.between(DAYS_LO, DAYS_HI).sum())
        df_sd["DaysToCompletion"] = _d.where(_d.between(DAYS_LO, DAYS_HI))
        log(f"  [FIX-4] DaysToCompletion clipped to [{DAYS_LO}, {DAYS_HI}] : {n_clip_d:,} rows")
    else:
        df_sd["DaysToCompletion"] = np.nan
        log("  [warn] completed_d / announced_d missing.")

    # [FIX-5] log1p：days=0 保留为 0，不再变 NaN (必须在裁剪之后做！)
    df_sd["ln_days"] = np.log1p(df_sd["DaysToCompletion"])
    log(f"  [FIX-5] ln_days = log1p(days), non-null: "
        f"{int(df_sd['ln_days'].notna().sum()):,} "
        f"({df_sd['ln_days'].notna().mean()*100:.2f}%)")

    log_checkpoint("after dates")

    # -----------------------------------------------------------------
    # 6. [FIX-7] SAMPLE MOMENTS
    # -----------------------------------------------------------------
    log("\n" + "=" * 78)
    log("  SAMPLE MOMENTS  (compare with paper: p_A=0.2563, p_D=0.6568)")
    log("=" * 78)
    _n = len(df_sd)
    log(f"    N                        : {_n:,}")
    for _k in ["rumor_jfe", "has_rumor_status", "rumor_any",
               "completed_flag", "completed_strict"]:
        if _k in df_sd.columns:
            log(f"    p({_k:<20s}): {df_sd[_k].mean():.4f}")
    if "announced_d" in df_sd.columns:
        log(f"    announced_d coverage     : "
            f"{df_sd['announced_d'].notna().mean()*100:.2f}%")
    log("")

    # =================================================================
    # 7. [DEDUP] 去重条件自检 + 智能去重 + Assert
    # =================================================================
    log("\n" + "=" * 78)
    log("  FINAL DEDUPLICATION CHECK (ONE ROW PER DEAL)")
    log("=" * 78)
    
    entity_cols = [c for c in df_sd.columns
                   if c.startswith(("tar_", "acq_", "ven_"))]
    log(f"\n  [DEDUP-CHECK] entity cols: {len(entity_cols)} -> {entity_cols[:10]}")
    
    if entity_cols:
        log("  [WARN] 存在实体列，不能按 deal_num 去重，跳过")
    else:
        n_before = len(df_sd)
        u_before = df_sd["deal_num"].nunique()
        log(f"  [DEDUP-CHECK] rows={n_before:,} | unique deal_num={u_before:,} | "
            f"多行={n_before - u_before:,}")
        
        # 按 deal_num 去重，保留填充最全的行
        tmp = df_sd.copy()
        tmp["_fill"] = tmp.notna().sum(axis=1)
        tmp["_ord"] = np.arange(len(tmp))
        tmp = tmp.sort_values(["deal_num", "_fill", "_ord"],
                              ascending=[True, False, True], kind="mergesort")
        df_sd = tmp.drop_duplicates(subset=["deal_num"], keep="first")
        df_sd = df_sd.drop(columns=["_fill", "_ord"]).reset_index(drop=True)
        
        log(f"  [DEDUP] {n_before:,} -> {len(df_sd):,} rows")
        
        assert df_sd["deal_num"].duplicated().sum() == 0, \
            f"[FATAL] structure_date 不是一行一 deal: {len(df_sd)} vs {df_sd['deal_num'].nunique()}"
        log("  [ASSERT] PASSED: exactly one row per deal_num.")

    # -----------------------------------------------------------------
    # 8. Reports & Save
    # -----------------------------------------------------------------
    report_coverage(df_sd, "01_deal_structure_date.csv")
    report_descriptives(df_sd, "01_deal_structure_date.csv")

    out_path = os.path.join(CLEANED, "01_deal_structure_date.csv")
    df_sd.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  Module 6: {len(df_sd):,} rows written -> {out_path}")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise