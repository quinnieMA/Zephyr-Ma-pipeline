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

[FIX-4] stake_acq_pct / stake_final_pct / stake_init_pct 统一限 [0, 100]。
        0 是合法观测（从零收购 / 本次取得 0 股），保留不置 NaN；
        仅 < 0 与 > 100 越界置 NaN。论文要求控股 >= 50% 时请用 >= 50 筛，
        不要依赖"非缺失即 >=50"。

[FIX-5] currency strip + upper。

[FIX-6] 新增描述统计（原脚本只有 coverage，看不到 min/max，无法发现极端值）。

[FIX-7] ensure_key_columns 只查 deal_num。value 文件只有 deal_num，
        原脚本强查 8 个 entity key 全 missing，纯噪音。

[FIX-8] 新增 stake>=50 / deal_value>0 比例诊断 + 与主表行数比对，
        直接看论文筛选后还剩多少。

[FIX-9] 新增 stake_init_pct 字段：纳入 FIELD_MAP_VALUE / VALUE_NUM_COLS /
        STAKE_COLS，并在数值化前用正则抽取首个连续数字，把
        "Unknown minority" / "Unknown %" 等文本值安全转成 NaN，
        同时 0 / 55 / 100 等真实持股保持不变，最终与 stake_acq_pct、
        stake_final_pct 同为数值型。


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
    "stake_init_pct":         ["stake initial %", "% stake initial",
                               "stake_init_pct", "stake initial",
                               "stake init %", "initial stake %"],
    "currency":               ["currency", "deal currency", "currency of deal"],
}

# 需要转成数值的列（规范名）
VALUE_NUM_COLS = ["deal_value", "deal_val_usd", "deal_enterprise_value",
                  "stake_acq_pct", "stake_final_pct", "stake_init_pct"]

# [FIX-3] 金额列：<= 0 视为缺失
MONEY_COLS = ["deal_value", "deal_val_usd", "deal_enterprise_value"]

#   注意：acq / final 是“本次交易取得的/最终持股”，0 无经济含义，故不含 0；
#         init 是“交易前标的已存在的持股比例”，0 完全合法（从零收购），
STAKE_COLS = ["stake_acq_pct", "stake_final_pct", "stake_init_pct"]

# [FIX-9] 初始持股列：合法区间 [0, 100]。0 是真实观测值（从零收购），
#        只有 < 0 或 > 100 才视为缺失；文本类 Unknown % / Unknown minority
#        已在前面 extract+to_numeric 阶段转成 NaN。

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
    n_before = len(df)
    # keep-first 基线，供 [FIX-2] GAIN 诊断对比
    base = df.drop_duplicates(subset=[key], keep="first").set_index(key)
    # 本文件存在重复 deal_num（同一交易多行），先按 deal_num 排序，
    # 使非空的 stake 值排在 "Unknown %" 等文本行之前，再取首个非空值。
    df_agg = (
        df.sort_values(key)
          .groupby(key, as_index=False, dropna=False)
          .first()
    )
    log(f"  [FIX-1] aggregate_first({label or key}): "
        f"{n_before:,} rows -> {len(df_agg):,} rows "
        f"(unique {df_agg[key].nunique():,})")
    # 注意：本文件为「一行一组字段」结构，三列持股同处一行；
    # groupby.first() 取该 deal 首个非空值，不做跨字段填充。
    return df_agg, base


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def report_coverage(df, fname="01_deal_value.csv"):
    """[FIX-3] 覆盖率报告：每列非空数与占比。"""
    log(f"\n  [FIX-3] 覆盖率报告 [{fname}]  n={len(df):,}")
    rows = []
    for c in df.columns:
        n = int(df[c].notna().sum())
        rows.append((c, n, n / len(df) * 100))
    # 按缺失率从高到低，打印前 20 列
    for c, n, pct in sorted(rows, key=lambda x: x[2])[:20]:
        bar = "#" * int(pct / 5)
        log(f"    {c:<20s} {n:>9,} ({pct:5.1f}%) {bar}")


def report_descriptives(df, fname="01_deal_value.csv"):
    """[FIX-4] 描述性统计：数值列 min/p25/p50/mean/p75/p95/max/std。"""
    import numpy as _np
    log(f"\n  [FIX-4] 描述性统计 [{fname}]")
    num_cols = df.select_dtypes(include=["number", "Int64", "Float64"]).columns
    for c in num_cols:
        x = df[c].dropna()
        if len(x) == 0:
            continue
        try:
            log(f"    {c:<20s} n={len(x):>8,}  min={_np.percentile(x,0):>10.4g}  "
                f"p25={_np.percentile(x,25):>10.4g}  p50={_np.percentile(x,50):>10.4g}  "
                f"mean={x.mean():>10.4g}  p75={_np.percentile(x,75):>10.4g}  "
                f"p95={_np.percentile(x,95):>10.4g}  max={_np.percentile(x,100):>10.4g}  "
                f"std={x.std():>10.4g}")
        except Exception:
            pass


def report_gain(df_agg, base_df, cols, label):
    """[FIX-2] GAIN 诊断：逐列对比 groupby-first 与 keep-first 的非空数。"""
    log(f"\n  [FIX-2] GAIN 诊断（groupby-first vs keep-first）[{label}]")
    for c in cols:
        if c not in df_agg.columns:
            continue
        n_agg = int(df_agg[c].notna().sum())
        if c in base_df.columns:
            n_base = int(base_df[c].notna().sum())
            diff = n_agg - n_base
            tag = "GAIN" if diff > 0 else ("SAME" if diff == 0 else "LOSS")
        else:
            n_base, diff = 0, n_agg
            tag = "NEW"
        log(f"    {c:<20s} groupby:{n_agg:>9,}  keepfirst:{n_base:>9,}  "
            f"diff:{diff:+,.0f}  [{tag}]")


def main():
    module_banner(1, "读取原始字段表（含 stake_init_pct 候选列名）")

    # 1) 读取原始文件（多个 MA_value_*.csv 合并；一行一个 deal_num）
    df = read_raw("value", "value")
    if df is None or df.empty:
        log("[FATAL] 未读到任何 value 原始文件，请检查 RAW_CANDIDATES / BASE")
        flush_log()
        return

    # 2) 列名归一化：stake_init_pct 由此进入标准列名
    df = normalize_columns(df, FIELD_MAP_VALUE, "value")

    # 3) [FIX-1] 按 deal_num 去重：先排序让数值排在 Unknown 文本之前，取首个非空
    df_agg, base = aggregate_first(df, key="deal_num", label="value")

    # 4) 数值化：金额与持股列转数值，仅 <0 置缺失（0 全部保留）
    num_cols = [c for c in VALUE_NUM_COLS + STAKE_COLS if c in df_agg.columns]
    for c in num_cols:
        if c in STAKE_COLS:
            # 持股：先抽数字（把 Unknown % / Unknown majority 变 NaN），再限制 [0,100]
            df_agg[c] = (df_agg[c].astype(str)
                         .str.extract(r"(\d+(?:\.\d+)?)")[0]
                         .pipe(pd.to_numeric, errors="coerce"))
            mask_bad = (df_agg[c] < 0) | (df_agg[c] > 100)
            n_bad = int(mask_bad.sum())
            n_zero = int((df_agg[c] == 0).sum())
            df_agg.loc[mask_bad, c] = pd.NA
            log(f"    {c:<16s} 越界(<0或>100)置NaN: {n_bad:,} | 保留0: {n_zero:,}")
        else:
            df_agg[c] = pd.to_numeric(df_agg[c], errors="coerce")
            neg = df_agg[c] < 0
            df_agg.loc[neg, c] = pd.NA
            log(f"    {c:<16s} 负值置NaN: {int(neg.sum()):,}")

    # 5) deal_num 转 Int64
    df_agg["deal_num"] = pd.to_numeric(df_agg["deal_num"], errors="coerce").astype("Int64")

    # 6) 报告
    report_gain(df_agg, base, VALUE_NUM_COLS + ["currency", "stake_init_pct"], "value")
    report_coverage(df_agg, "01_deal_value.csv")
    report_descriptives(df_agg, "01_deal_value.csv")

    # [FIX-8] 论文筛选比例（非上市标的 + 控股 >= 50%）
    log("\n  [FIX-8] 论文筛选条件的命中比例")
    log("    (论文: 非上市标的 + 控股 >= 50%)")
    for c in ["stake_acq_pct", "stake_final_pct"]:
        if c in df_agg.columns:
            ok = df_agg[c] >= 50
            log(f"    p({c} >= 50) : {ok.mean():.4f}  (n={int(ok.sum()):,})")
    for c in ["deal_value", "deal_val_usd"]:
        if c in df_agg.columns:
            ok = df_agg[c] > 0
            log(f"    p({c} > 0)   : {ok.mean():.4f}  (n={int(ok.sum()):,})")

    # [FIX-10] 持股分布：0 / 100 / 缺失 占比（init 应含大量 0）
    log("\n  [FIX-10] 持股分布：0 / 100 / 缺失 占比（init 应含大量 0）")
    for c in ["stake_init_pct", "stake_acq_pct", "stake_final_pct"]:
        if c not in df_agg.columns:
            continue
        z = int((df_agg[c] == 0).sum())
        h = int((df_agg[c] == 100).sum())
        u = int(df_agg[c].isna().sum())
        log(f"    {c:<16s} ==0: {z:>9,} ({z / len(df_agg) * 100:5.2f}%) | "
            f"==100: {h:>9,} ({h / len(df_agg) * 100:5.2f}%) | "
            f"缺失: {u:>8,} ({u / len(df_agg) * 100:5.2f}%)")

    # 与主表比对
    n = df_agg["deal_num"].nunique()
    log(f"\n  [vs main] unique deal_num: {n:,} "
        f"(structure_date: {MAIN_N_EXPECT:,}, diff={n - MAIN_N_EXPECT:+,})")

    # 保存
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
