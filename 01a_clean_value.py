# -*- coding: utf-8 -*-
r"""
01a_clean_value.py — Deal value / stake export (FIXED)
===================================================================
Input  : D:\zephyr\value\cleaned_output\MA_value.csv
Output : D:\MA\data\cleaned\01_deal_value.csv

===================================================================
FIX LIST（相对原版）
===================================================================
[FIX-1] 致命：keep="first" 去重静默丢值。
        value 模块 1,144,984 -> 1,066,213（丢 78,771 行），这些文件是
        "一行一个属性"结构（多货币/多口径/多年份拆成多行），keep="first"
        只保留物理第一行，而第一行常为空行 -> 金额/持股被系统性低估。
        修法：改用 groupby(deal_num).first()，它自动跳过 NaN 取首个非空。

[FIX-2] 新增 [GAIN] 诊断：逐列对比 keep-first 与 groupby-first 的非空数，
        直接量化聚合带来的覆盖率提升。

[FIX-3] 金额 <= 0 置 NaN 并计数。overview 里 deal_value_ovw 的 min = -1679，
        负值会污染 deal_value>0 这个筛选（论文用它筛样本）。

[FIX-4] stake_acq_pct / stake_final_pct 限 (0, 100]，越界置 NaN。
        论文要求控股 >= 50%，这个筛选变量必须干净。

[FIX-5] currency strip + upper。

[FIX-6] 新增描述统计（原脚本只有 coverage，看不到 min/max，无法发现极端值）。

[FIX-7] ensure_key_columns 只查 deal_num。value 文件只有 deal_num，
        原脚本强查 8 个 entity key 全 missing，纯噪音。

[FIX-8] 新增 stake>=50 / deal_value>0 比例诊断 + 与主表行数比对，
        直接看论文筛选后还剩多少。

单位提醒：deal_val_usd 来自 "Deal value th USD"，单位是【千美元】。
         下游 ln_deal_value 用千美元还是美元只差常数 log(1000)，
         不影响任何斜率系数，故不做换算，仅在此标注。
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

WORK_LOG = os.path.join(CLEANED, "01a_value.txt")
STRICT = False

# 若实际子目录名不同，在此增删候选即可
RAW_CANDIDATES = {
    "value": [
        os.path.join(ZEPHYR, "value", "cleaned_output", "MA_value.csv"),
        os.path.join(ZEPHYR, "value", "cleaned_output", "MA_value.xlsx"),
        os.path.join(ZEPHYR, "deal_value", "cleaned_output", "MA_value.csv"),
    ],
}

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

# --- 字段名归一化：规范名 -> 可能的原始列名（小写比对） ---
FIELD_MAP_VALUE = {
    "deal_value":             ["deal value", "deal_value", "dealvalue"],
    "deal_val_usd":           ["deal value th usd", "deal value usd",
                               "deal_val_usd", "deal value (th usd)"],
    "deal_enterprise_value":  ["enterprise value", "deal_enterprise_value",
                               "ev", "enterprise value th usd"],
    "stake_acq_pct":          ["stake acquired %", "% stake acquired",
                               "stake_acq_pct", "stake acquired"],
    "stake_final_pct":        ["stake final %", "% stake final",
                               "stake_final_pct", "stake final"],
    "currency":               ["currency", "deal currency", "currency of deal"],
}

# 需要转成数值的列（规范名）
VALUE_NUM_COLS = ["deal_value", "deal_val_usd", "deal_enterprise_value",
                  "stake_acq_pct", "stake_final_pct"]

# [FIX-3] 金额列：<= 0 视为缺失
MONEY_COLS = ["deal_value", "deal_val_usd", "deal_enterprise_value"]

# [FIX-4] 持股列：必须在 (0, 100]
STAKE_COLS = ["stake_acq_pct", "stake_final_pct"]

# 主表（structure_date）行数基准，用于比对
MAIN_N_EXPECT = 1066213


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
    header = [
        "", "#" * 78,
        f"# 01a run — {_SCRIPT_NAME}",
        f"# started : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# cwd     : {os.getcwd()}",
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
    """把原始列名映射到规范名。已是规范名则直接保留。"""
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
        log(f"  [map:{label}] renamed {len(renames)} column(s): "
            f"{list(renames.items())[:6]}")
    if not_found:
        log(f"  [map:{label}] not found (will be NaN): {not_found}")
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

    # deal_num
    fold = {str(c).strip().lower(): c for c in df.columns}
    dn = None
    for cand in ["deal_num", "deal number", "dealnumber"]:
        if cand in fold:
            dn = fold[cand]
            break
    if dn is None:
        log(f"  x {label}: no deal_num column")
        return None
    if dn != "deal_num":
        df = df.rename(columns={dn: "deal_num"})
    df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")

    log(f"  [{label}] loaded: {len(df):,} rows | {df.shape[1]} cols | "
        f"unique deal_num: {df['deal_num'].nunique():,}")
    return df


# ---------------------------------------------------------------------
# [FIX-1] 聚合去重
# ---------------------------------------------------------------------
def aggregate_first(df, key="deal_num", label=""):
    """
    用 groupby.first() 取代 drop_duplicates(keep='first')。
    GroupBy.first() 自动跳过 NaN，取每组首个非空值。
    返回 (聚合后df, keep_first基线df) 以便做 GAIN 诊断。
    """
    n_before = len(df)
    base = df.drop_duplicates(subset=[key], keep="first").set_index(key)
    agg = df.groupby(key, sort=False).first()
    n_after = len(agg)
    if label:
        log(f"\n  [AGG:{label}] rows {n_before:,} -> {n_after:,} "
            f"(unique {key})")
    return agg.reset_index(), base


def report_gain(agg, base, cols, label=""):
    """[FIX-2] 逐列对比 keep-first 与 groupby-first 的非空数。"""
    log(f"\n  [GAIN] {label} — keep='first' vs groupby.first() 非空数")
    log(f"    {'column':<28s} {'keep-first':>12s} {'aggregated':>12s} {'gain':>12s}")
    log("    " + "-" * 68)
    for c in cols:
        if c not in agg.columns:
            continue
        n_agg = int(agg[c].notna().sum())
        n_base = int(base[c].notna().sum()) if (base is not None and c in base.columns) else 0
        log(f"    {c:<28s} {n_base:>12,} {n_agg:>12,} {n_agg - n_base:>+12,}")


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------
def report_coverage(df, label, cols=None):
    cols = list(df.columns) if cols is None else [c for c in cols if c in df.columns]
    log(f"\n  [{label}] COVERAGE DIAGNOSTICS — {len(df):,} rows")
    log(f"    {'Column':<40s} {'Non-Null':>10s} {'Pct':>7s}   {'Missing':>10s} {'Pct':>7s}")
    log("    " + "-" * 80)
    for c in cols:
        n = int(df[c].notna().sum())
        pct = n / len(df) * 100 if len(df) else 0.0
        m = len(df) - n
        log(f"    {c:<40s} {n:>10,} {pct:>6.2f}%   {m:>10,} {100 - pct:>6.2f}%")


def report_descriptives(df, label, cols=None):
    num_cols = df.select_dtypes(include=[np.number, "Int64"]).columns.tolist()
    if cols:
        num_cols = [c for c in cols if c in num_cols]
    if not num_cols:
        log(f"\n  [{label}] no numeric columns.")
        return
    log(f"\n  [{label}] DESCRIPTIVE STATS — numeric ({len(num_cols)})")
    log(f"    {'column':<28s} {'min':>14s} {'median':>14s} {'mean':>14s} "
        f"{'max':>14s} {'std':>14s}")
    log("    " + "-" * 104)
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            log(f"    {c:<28s} {'ALL MISSING':>14s}")
            continue
        f = lambda x: f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"
        log(f"    {c:<28s} {f(s.min()):>14s} {f(s.median()):>14s} "
            f"{f(s.mean()):>14s} {f(s.max()):>14s} {f(s.std()):>14s}")


# =====================================================================
# Main
# =====================================================================
def main():
    module_banner("M4", "Deal value / stake export -> 01_deal_value.csv (FIXED)")

    df = read_raw("value", "value")
    if df is None:
        log("  [skip] value module aborted (no input).")
        flush_log()
        return

    df = df[df["deal_num"].notna()].copy()
    log(f"  after dropna(deal_num): {len(df):,} rows")

    # --- 列名归一化 ---
    df = normalize_columns(df, FIELD_MAP_VALUE, "value")

    # --- 数值转换 ---
    for c in VALUE_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- [FIX-5] currency 归一化 ---
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.strip().str.upper()
        df.loc[df["currency"].isin(["NAN", "NONE"]), "currency"] = np.nan
        log(f"  [FIX-5] currency normalized, unique = {df['currency'].nunique():,}")

    # --- [FIX-3] 金额 <= 0 置 NaN ---
    log("\n  [FIX-3] 金额列 <= 0 -> NaN")
    for c in MONEY_COLS:
        if c not in df.columns:
            continue
        n_bad = int((df[c] <= 0).sum())
        df.loc[df[c] <= 0, c] = np.nan
        log(f"    {c:<28s} <=0 置 NaN: {n_bad:,}")

    # --- [FIX-4] 持股限 (0, 100] ---
    log("\n  [FIX-4] 持股列限 (0, 100]")
    for c in STAKE_COLS:
        if c not in df.columns:
            continue
        bad = (df[c] <= 0) | (df[c] > 100)
        n_bad = int(bad.sum())
        df.loc[bad, c] = np.nan
        log(f"    {c:<28s} 越界置 NaN: {n_bad:,}")

    log_checkpoint("before aggregation")

    # --- [FIX-1] 聚合去重 ---
    df_agg, base = aggregate_first(df, "deal_num", "value")

    # --- [FIX-2] GAIN 诊断 ---
    report_gain(df_agg, base, VALUE_NUM_COLS + ["currency"], "value")
    df_agg["deal_num"] = pd.to_numeric(df_agg["deal_num"], errors="coerce").astype("Int64")

    # --- 报告 ---
    report_coverage(df_agg, "01_deal_value.csv")
    report_descriptives(df_agg, "01_deal_value.csv")

    # --- [FIX-8] 论文筛选比例 ---
    log("\n  [FIX-8] 论文筛选条件的命中比例")
    log("    (论文: 非上市标的 + 控股 >= 50%)")
    for c in ["stake_acq_pct", "stake_final_pct"]:
        if c in df_agg.columns:
            ok = df_agg[c] >= 50
            log(f"    p({c} >= 50) : {ok.mean():.4f}  "
                f"(n={int(ok.sum()):,})")
    for c in ["deal_value", "deal_val_usd"]:
        if c in df_agg.columns:
            ok = df_agg[c] > 0
            log(f"    p({c} > 0)   : {ok.mean():.4f}  "
                f"(n={int(ok.sum()):,})")

    # --- 与主表比对 ---
    n = df_agg["deal_num"].nunique()
    log(f"\n  [vs main] unique deal_num: {n:,} "
        f"(structure_date: {MAIN_N_EXPECT:,}, diff={n - MAIN_N_EXPECT:+,})")

    # --- 保存 ---
    out_path = os.path.join(CLEANED, "01_deal_value.csv")
    df_agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  Module 4: {len(df_agg):,} rows written -> {out_path}")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise