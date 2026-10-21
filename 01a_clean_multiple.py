# -*- coding: utf-8 -*-
r"""
01a_clean_multiple.py — Valuation multiples export (FIXED)
===================================================================
Input  : D:\zephyr\multiple\cleaned_output\MA_multiple.csv
Output : D:\MA\data\cleaned\01_deal_multiple.csv

===================================================================
FIX LIST（相对原版）
===================================================================
[FIX-1] 致命：keep="first" 去重静默丢值（同 value）。
        1,145,367 -> 1,066,572（丢 78,795 行）。
        直接影响 p_B：pre_rev_mul_ly 覆盖 11.12% 可能是被压出来的。
        修法：groupby(deal_num).first()。

[FIX-2] FIELD_MAP_MULTIPLE 映射目标改为 pre_* 前缀。
        原脚本映射表写的还是无前缀旧名（rev_mul_ly 等），而原始文件
        早已是 pre_* 前缀 -> 16/16 条全未命中，纯噪音。
        脚本靠"列名恰好已规范"跑通，映射表完全失效。

[FIX-3] MUL_KEEP 改用 pre_* 前缀（原脚本报 24 列 missing 噪音），
        并动态保留所有 pre_* 列，无论文件用哪套命名都能对上。

[FIX-4] 新增 [GAIN] 诊断 + p_B 直接打印（论文 p_B = 0.2081）。

[FIX-5] 新增描述统计 + 1/5/95/99 分位数（下游 5%/95% 缩尾直接用）。

[FIX-6] pre_rev_mul_ly <= 0 置 NaN（营收不可能为负）。
        其余倍数（EBITDA/EBIT/PE 类）分母可为负，只诊断不强行清洗。

[FIX-7] 与主表行数比对（1,066,572 vs 1,066,213，差 359）。
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

WORK_LOG = os.path.join(CLEANED, "01a_multiple.txt")
STRICT = False

RAW_CANDIDATES = {
    "multiple": [
        os.path.join(ZEPHYR, "multiple", "cleaned_output", "MA_multiple.csv"),
        os.path.join(ZEPHYR, "multiples", "cleaned_output", "MA_multiple.csv"),
        os.path.join(ZEPHYR, "valuation", "cleaned_output", "MA_multiple.csv"),
    ],
}

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

# --- [FIX-2] 映射目标统一为 pre_* 规范名 ---
FIELD_MAP_MULTIPLE = {
    "pre_rev_mul_ly":      ["ev / sales (ltm)", "price / sales (ltm)",
                            "revenue multiple ltm", "rev_mul_ly",
                            "pre_rev_mul_ly"],
    "pre_ebitda_mul_ly":   ["ev / ebitda (ltm)", "ebitda multiple ltm",
                            "ebitda_mul_ly", "pre_ebitda_mul_ly"],
    "pre_ebit_mul_ly":     ["ev / ebit (ltm)", "ebit multiple ltm",
                            "ebit_mul_ly", "pre_ebit_mul_ly"],
    "pre_pe_mul_ly":       ["price / earnings (ltm)", "p/e ltm",
                            "pe_mul_ly", "pre_pe_mul_ly"],
    "pre_book_mul_ly":     ["price / book value", "book multiple",
                            "book_mul_ly", "pre_book_mul_ly"],
}

# [FIX-3] 已知的目标列（pre_* 前缀）
MUL_KEEP = list(FIELD_MAP_MULTIPLE.keys())

# 主表基准
MAIN_N_EXPECT = 1066213

# 论文 p_B 参照
PAPER_PB = 0.2081


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
_LOG_LINES = []
_SCRIPT_NAME = os.path.basename(__file__)
_LOG_INITIALISED = False


def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))


def _ensure_log_header():
    global _LOG_INITIALISED
    if _LOG_INITIALISED:
        return
    os.makedirs(CLEANED, exist_ok=True)
    header = ["", "#" * 78,
              f"# 01a run — {_SCRIPT_NAME}",
              f"# started : {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"# cwd     : {os.getcwd()}", "#" * 78]
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
# Utilities
# ---------------------------------------------------------------------
def clean_missing(df):
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


def normalize_columns(df, field_map, label):
    fold = {str(c).strip().lower(): c for c in df.columns}
    renames, not_found = {}, []
    for std, candidates in field_map.items():
        if std in df.columns:
            continue
        hit = None
        for cand in candidates:
            if cand.lower() in fold:
                hit = fold[cand.lower()]
                break
        if hit is not None:
            renames[hit] = std
        else:
            not_found.append(std)
    if renames:
        df = df.rename(columns=renames)
        log(f"  [map:{label}] renamed {len(renames)} column(s)")
    if not_found:
        log(f"  [map:{label}] not found (NaN): {not_found}")
    return df


def read_raw(file_key, label):
    fp = None
    for cand in RAW_CANDIDATES[file_key]:
        if os.path.exists(cand):
            fp = cand
            break
    if fp is None:
        log(f"\n  x [MISSING INPUT] {label}: {RAW_CANDIDATES[file_key][0]}")
        if STRICT:
            raise FileNotFoundError(RAW_CANDIDATES[file_key][0])
        return None
    df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    df = clean_missing(df)
    fold = {str(c).strip().lower(): c for c in df.columns}
    dn = None
    for cand in ["deal_num", "deal number"]:
        if cand in fold:
            dn = fold[cand]
            break
    if dn is None:
        log(f"  x {label}: no deal_num")
        return None
    if dn != "deal_num":
        df = df.rename(columns={dn: "deal_num"})
    df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")
    log(f"  [{label}] loaded: {len(df):,} rows | {df.shape[1]} cols | "
        f"unique deal_num: {df['deal_num'].nunique():,}")
    return df


def aggregate_first(df, key="deal_num", label=""):
    n_before = len(df)
    base = df.drop_duplicates(subset=[key], keep="first").set_index(key)
    agg = df.groupby(key, sort=False).first()
    if label:
        log(f"\n  [AGG:{label}] rows {n_before:,} -> {len(agg):,} (unique {key})")
    return agg.reset_index(), base


def report_gain(agg, base, cols, label=""):
    log(f"\n  [GAIN] {label} — keep='first' vs groupby.first() 非空数")
    log(f"    {'column':<28s} {'keep-first':>12s} {'aggregated':>12s} {'gain':>12s}")
    log("    " + "-" * 68)
    for c in cols:
        if c not in agg.columns:
            continue
        n_agg = int(agg[c].notna().sum())
        n_base = int(base[c].notna().sum()) if (base is not None and c in base.columns) else 0
        log(f"    {c:<28s} {n_base:>12,} {n_agg:>12,} {n_agg - n_base:>+12,}")


def report_coverage(df, label, cols=None):
    cols = list(df.columns) if cols is None else [c for c in cols if c in df.columns]
    log(f"\n  [{label}] COVERAGE — {len(df):,} rows")
    log(f"    {'Column':<40s} {'Non-Null':>10s} {'Pct':>7s}   {'Missing':>10s} {'Pct':>7s}")
    log("    " + "-" * 80)
    for c in cols:
        n = int(df[c].notna().sum())
        pct = n / len(df) * 100 if len(df) else 0.0
        log(f"    {c:<40s} {n:>10,} {pct:>6.2f}%   {len(df)-n:>10,} {100-pct:>6.2f}%")


def report_descriptives(df, label, cols=None):
    num_cols = df.select_dtypes(include=[np.number, "Int64"]).columns.tolist()
    if cols:
        num_cols = [c for c in cols if c in num_cols]
    if not num_cols:
        log(f"\n  [{label}] no numeric columns.")
        return
    log(f"\n  [{label}] DESCRIPTIVE STATS + QUANTILES")
    log(f"    {'column':<24s} {'min':>11s} {'p1':>11s} {'p5':>11s} "
        f"{'median':>11s} {'p95':>11s} {'p99':>11s} {'max':>11s}")
    log("    " + "-" * 108)
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            log(f"    {c:<24s} {'ALL MISSING':>11s}")
            continue
        f = lambda x: f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"
        log(f"    {c:<24s} {f(s.min()):>11s} {f(s.quantile(.01)):>11s} "
            f"{f(s.quantile(.05)):>11s} {f(s.median()):>11s} "
            f"{f(s.quantile(.95)):>11s} {f(s.quantile(.99)):>11s} "
            f"{f(s.max()):>11s}")


# =====================================================================
# Main
# =====================================================================
def main():
    module_banner("M5", "Valuation multiples export -> 01_deal_multiple.csv (FIXED)")

    df = read_raw("multiple", "multiple")
    if df is None:
        log("  [skip] multiple module aborted.")
        flush_log()
        return

    df = df[df["deal_num"].notna()].copy()
    log(f"  after dropna(deal_num): {len(df):,} rows")

    # --- [FIX-2] 列名归一化到 pre_* ---
    df = normalize_columns(df, FIELD_MAP_MULTIPLE, "multiple")

    # --- [FIX-3] 动态保留所有 pre_* 列 ---
    pre_cols = [c for c in df.columns if c.startswith("pre_")]
    keep = [c for c in MUL_KEEP if c in df.columns]
    for c in pre_cols:
        if c not in keep:
            keep.append(c)
    log(f"  [FIX-3] keeping {len(keep)} multiple column(s): {keep}")

    # --- 数值转换 ---
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- [FIX-6] pre_rev_mul_ly <= 0 置 NaN ---
    if "pre_rev_mul_ly" in df.columns:
        n_bad = int((df["pre_rev_mul_ly"] <= 0).sum())
        df.loc[df["pre_rev_mul_ly"] <= 0, "pre_rev_mul_ly"] = np.nan
        log(f"  [FIX-6] pre_rev_mul_ly <= 0 -> NaN: {n_bad:,}")

    # 其余倍数只诊断
    for c in keep:
        if c == "pre_rev_mul_ly":
            continue
        n_neg = int((df[c] < 0).sum())
        if n_neg:
            log(f"    [diag] {c}: {n_neg:,} 负值（分母可为负，保留）")

    log_checkpoint("before aggregation")

    # --- [FIX-1] 聚合 ---
    df_agg, base = aggregate_first(df[["deal_num"] + keep], "deal_num", "multiple")

    # --- [FIX-4] GAIN + p_B ---
    report_gain(df_agg, base, keep, "multiple")
    df_agg["deal_num"] = pd.to_numeric(df_agg["deal_num"], errors="coerce").astype("Int64")

    report_coverage(df_agg, "01_deal_multiple.csv")
    report_descriptives(df_agg, "01_deal_multiple.csv")

    # --- [FIX-4] p_B 核心诊断 ---
    log("\n" + "=" * 78)
    log("  CORE MOMENT — p_B (估值可观测率)")
    log("=" * 78)
    if "pre_rev_mul_ly" in df_agg.columns:
        pb = df_agg["pre_rev_mul_ly"].notna().mean()
        log(f"    p_B (pre_rev_mul_ly 覆盖) = {pb:.4f}  "
            f"(n={int(df_agg['pre_rev_mul_ly'].notna().sum()):,} / {len(df_agg):,})")
        log(f"    论文 p_B                   = {PAPER_PB:.4f}")
        if pb < PAPER_PB * 0.7:
            log("    -> 选择比论文更严重，可观测性方程信息充足")
        elif pb > PAPER_PB * 1.3:
            log("    -> 选择比论文温和，注意 zeta2 识别不足")
        else:
            log("    -> 与论文量级接近，理想")
    log("")

    # --- [FIX-7] 主表比对 ---
    n = df_agg["deal_num"].nunique()
    log(f"  [vs main] unique deal_num: {n:,} "
        f"(structure_date: {MAIN_N_EXPECT:,}, diff={n - MAIN_N_EXPECT:+,})")

    out_path = os.path.join(CLEANED, "01_deal_multiple.csv")
    df_agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  Module 5: {len(df_agg):,} rows written -> {out_path}")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise