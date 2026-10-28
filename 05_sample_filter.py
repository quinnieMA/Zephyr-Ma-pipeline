# -*- coding: utf-8 -*-
r"""
05_sample_filter.py — v7
================================================================================
在 04_deal_firm_full.csv 基础上做 screening。只筛行，不改值。

基于 v4（能跑通的那一版），仅四处改动：
  [1] F5 (deal_type = Acquisition*) 恢复 —— v4 中该段被整段注释掉了
  [2] deal_year 改为 announced_d 优先（v4 用 completed_d 优先）
      原因：用完成日期定义交易年份，会让"年份可得性"与"是否完成"纠缠，
      未完成交易没有 completed_d -> 年份缺失 -> 被 F4 整批剔除。
      v4 实测：F4 一步 p_D 从 0.8729 跳到 0.9841。
  [3] moments 双口径：completed_strict（主，论文口径）+ completed_flag（宽，对比）
  [4] apply_filter 加 NA 守卫（BooleanArray 含 pd.NA 时 np.asarray(bool) 会报错）

筛选链：
  F1  acq_listed = 1                    [关]
  F2  tar_listed = 0                    [开]
  F3  drop public-offer structures      [开]
  F4  样本期 YEAR_START-YEAR_END        [开]
  F5  deal_type = Acquisition*          [开]  <- 本次恢复
  F6  stake >= 50%（缺失从 deal_type 插补）[开]
  F7  deal_value >= 1000 th USD         [开]
  F8  tar/acq SIC3 & country 均非空     [开]
  F9  deal_status = Completed 严格      [关 —— 主样本须保留未完成观测]
  F10 tar_overview_wordcount >= 20      [关 —— 需 tar_overview]
  F11 text_pool_rolling_count >= 10     [关 —— 需 tar_overview]
  F12 stock_traded_gdp
输出
    merged/05_deal_firm_filtered.csv
    merged/05_filter_log.txt
    merged/05_filter_steps.csv
"""

import os
import re
import numpy as np
import pandas as pd

# =====================================================================
# Config
# =====================================================================
BASE   = r"D:\MA"
MERGED = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)

IN_FILE  = os.path.join(MERGED, "04_deal_firm_full.csv")
OUT_FILE = os.path.join(MERGED, "05_deal_firm_filtered.csv")
LOG_FILE = os.path.join(MERGED, "05_filter_log.txt")
STEP_CSV = os.path.join(MERGED, "05_filter_steps.csv")

# ---------- 国家层面数据（股票市值/GDP 比率，合并入 05 输出） ----------
RAW_CTRY = os.path.join(BASE, "raw", "Country_level",
                        "world_stock_traded_gdp_ratio_1970_2025.csv")
ENABLE_COUNTRY_GDP_MERGE = True    # False 则跳过 STEP 1b

DEAL = "deal_num"

# ---------- 样本期 ----------
YEAR_START = 1997
YEAR_END   = 2024

# ---------- 年份口径 ----------
# "announced"（修复）| "completed"（v4 旧行为）
DEAL_YEAR_PRIORITY = "announced"

# ---------- 结果变量口径 ----------
COL_OUT_STRICT = "completed_strict"   # 主口径（仅 Completed）
COL_OUT_WIDE   = "completed_flag"     # 宽口径（含 Completed Assumed），仅对比

# ---------- 筛选开关 ----------
ENABLE_F1_ACQ_LISTED   = False
ENABLE_F2_TAR_UNLISTED = True
ENABLE_F3_PUBLIC_STRUCT= True
ENABLE_F4_PERIOD       = True
ENABLE_F5_DEAL_TYPE    = False     # [1] 
ENABLE_F6_STAKE50      = True
ENABLE_F7_VALUE_1M     = True
ENABLE_F8_SIC_CTRY     = True
ENABLE_F9_COMPLETED    = False
ENABLE_F10_WORDCOUNT   = False
ENABLE_F11_TEXTPOOL    = False
ENABLE_F12_STOCK_GDP   = True

MIN_STAKE_PCT  = 50.0
MIN_DEAL_VALUE = 1000.0            # th USD -> USD 1M
MIN_WORDCOUNT  = 20
MIN_TEXT_POOL  = 10

PUB_STRUCT = ["public takeover", "scheme of arrangement", "recommended bid",
              "hostile bid", "contested bid", "tender", "takeover"]

PAY_CLASS_MAP = {
    "Cash": "cash", "Cash Reserves": "cash",
    "Shares": "shares", "Third party shares": "shares",
    "Liabilities": "debt", "Converted Debt": "debt", "Bonds": "debt",
    "Deferred payment": "other", "Earn-out": "other",
    "Business assets": "other", "Dividend": "other",
    "Services": "other", "Other": "other", "Cash assumed": "other",
}

# ---------- 国家名 -> ISO2（keys 对齐 world_stock_traded_gdp 文件的列名） ----------
COUNTRY_MAP = {
    "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Bahrain": "BH",
    "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY", "Belgium": "BE",
    "Bermuda": "BM", "Botswana": "BW", "Brazil": "BR", "Bulgaria": "BG",
    "Canada": "CA", "Cayman_Islands": "KY", "Channel_Islands": "JE",
    "Chile": "CL", "China": "CN", "Colombia": "CO", "Costa_Rica": "CR",
    "Croatia": "HR", "Cyprus": "CY", "Czech_Republic": "CZ", "Denmark": "DK",
    "Ecuador": "EC", "Egypt": "EG", "Estonia": "EE", "Finland": "FI",
    "France": "FR", "Germany": "DE", "Ghana": "GH", "Greece": "GR",
    "Hong_Kong": "HK", "Hungary": "HU", "Iceland": "IS", "India": "IN",
    "Indonesia": "ID", "Iran": "IR", "Ireland": "IE", "Israel": "IL",
    "Italy": "IT", "Jamaica": "JM", "Japan": "JP", "Jordan": "JO",
    "Kazakhstan": "KZ", "Kenya": "KE", "Korea": "KR", "Kuwait": "KW",
    "Latvia": "LV", "Lebanon": "LB", "Lithuania": "LT", "Luxembourg": "LU",
    "Malaysia": "MY", "Malta": "MT", "Mauritius": "MU", "Mexico": "MX",
    "Montenegro": "ME", "Morocco": "MA", "Namibia": "NA", "Netherlands": "NL",
    "New_Zealand": "NZ", "Nigeria": "NG", "Norway": "NO", "Oman": "OM",
    "Pakistan": "PK", "Panama": "PA", "Papua_New_Guinea": "PG", "Paraguay": "PY",
    "Peru": "PE", "Philippines": "PH", "Poland": "PL", "Portugal": "PT",
    "Qatar": "QA", "Romania": "RO", "Russia": "RU", "Rwanda": "RW",
    "Saudi_Arabia": "SA", "Serbia": "RS", "Seychelles": "SC", "Singapore": "SG",
    "Slovakia": "SK", "Slovenia": "SI", "South_Africa": "ZA", "Spain": "ES",
    "Sri_Lanka": "LK", "Swaziland": "SZ", "Sweden": "SE", "Switzerland": "CH",
    "Tanzania": "TZ", "Thailand": "TH", "Trinidad_and_Tobago": "TT", "Tunisia": "TN",
    "Turkey": "TR", "Ukraine": "UA", "United_Arab_Emirates": "AE",
    "United_Kingdom": "GB", "US": "US", "Uruguay": "UY", "Venezuela": "VE",
    "Vietnam": "VN", "West_Bank_and_Gaza": "PS", "Zambia": "ZM", "Zimbabwe": "ZW",
}


# =====================================================================
# Logging
# =====================================================================
_log_lines = []
filter_summary = []   # (label, n_before, n_after, n_dropped)

def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))

def banner(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)

def apply_filter(df, keep_mask, label):
    # [4] NA 守卫：BooleanArray 含 pd.NA 时 np.asarray(..., bool) 会报错。
    #     NA 一律按「不满足条件」处理（年份未知 = 不在样本期内）。
    if isinstance(keep_mask, pd.Series):
        keep_mask = keep_mask.fillna(False).astype(bool).to_numpy()
    else:
        keep_mask = pd.Series(keep_mask).fillna(False).astype(bool).to_numpy()

    if len(keep_mask) != len(df):
        raise ValueError(f"[{label}] mask 长度 {len(keep_mask)} != 行数 {len(df)}")
    n_before = len(df)
    out = df[keep_mask].copy()
    n_after = len(out)
    n_drop = n_before - n_after
    log(f"  {label}")
    log(f"      dropped {n_drop:>8,}  ->  {n_after:>8,} remaining")
    filter_summary.append((label, n_before, n_after, n_drop))
    return out

def moments(df, tag):
    """[3] 双口径：pD_strict（主）+ pD_wide（对比）"""
    parts = []
    for col, name in [("rumor_jfe", "p_A"),
                      (COL_OUT_STRICT, "pD_strict"),
                      (COL_OUT_WIDE, "pD_wide"),
                      ("pre_rev_mul_ly", "p_B")]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if name == "p_B":
                parts.append(f"{name}={s.notna().mean():.4f}")
            else:
                parts.append(f"{name}={s.mean():.4f}")
        else:
            parts.append(f"{name}=NA")
    log(f"      [{tag}] " + "  ".join(parts))


# =====================================================================
# Helpers
# =====================================================================
def _as_str(s):
    return s.astype(str) if s.dtype.name == "category" else s

def _year_from_date(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    s = _as_str(df[col])
    return pd.to_numeric(s.str.slice(0, 4), errors="coerce")

def infer_stake_from_deal_type(s):
    if pd.isna(s):
        return np.nan
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", str(s))
    return max(float(p) for p in pcts) if pcts else np.nan


# =====================================================================
# Main
# =====================================================================
def main():
    banner("05_sample_filter.py — v7")
    log(f"input : {IN_FILE}")
    log(f"period: {YEAR_START}-{YEAR_END}   year from {DEAL_YEAR_PRIORITY}_d")
    log(f"outcome: {COL_OUT_STRICT}（主） / {COL_OUT_WIDE}（对比）")

    if not os.path.exists(IN_FILE):
        log(f"[FATAL] 找不到 {IN_FILE}")
        return

    # ---------- STEP 1 load ----------
    banner("STEP 1 — Load")
    df = pd.read_csv(IN_FILE, encoding="utf-8-sig", low_memory=False)
    INPUT_ROWS = len(df)
    log(f"Rows: {INPUT_ROWS:,} | Columns: {df.shape[1]}")

    assert "_row_id" in df.columns, "_row_id missing — re-run 04"
    assert df["_row_id"].nunique() == len(df), "_row_id not unique"
    assert df["_row_id"].notna().all(), "_row_id has NaN"
    log(f"_row_id check PASSED: 0 .. {df['_row_id'].max()}")
    log(f"deal_num: {df[DEAL].nunique():,} unique / {len(df):,} rows")

    # ---------- STEP 1b country-level stock-traded-GDP merge ----------
    banner("STEP 1b — Country-level stock-traded-GDP merge")
    if not ENABLE_COUNTRY_GDP_MERGE:
        log("  [SKIP] ENABLE_COUNTRY_GDP_MERGE = False")
    elif not os.path.exists(RAW_CTRY):
        log(f"  [SKIP] 找不到 {RAW_CTRY}（保留空列，不中断）")
        df["stock_traded_gdp_tar"] = np.nan
        df["stock_traded_gdp_acq"] = np.nan
    else:
        df_ctry_wide = pd.read_csv(RAW_CTRY)
        df_ctry_wide = df_ctry_wide.rename(columns={"Unnamed: 0": "year_date"})
        log(f"  country file: {RAW_CTRY}")
        log(f"  Raw shape: {df_ctry_wide.shape[0]:,} rows x {df_ctry_wide.shape[1]} cols | "
            f"year {df_ctry_wide['year_date'].iloc[0]} ~ {df_ctry_wide['year_date'].iloc[-1]}")
        df_ctry_wide["year"] = pd.to_numeric(
            df_ctry_wide["year_date"].astype(str).str[:4], errors="coerce").astype("Int64")
        country_cols = [c for c in df_ctry_wide.columns if c.endswith("_StockTraded_GDP")]
        df_long = df_ctry_wide[["year"] + country_cols].melt(
            id_vars="year", var_name="country_col", value_name="stock_traded_gdp")
        df_long["country_name"] = df_long["country_col"].str.replace(
            "_StockTraded_GDP", "", regex=False)
        df_long["country_code"] = df_long["country_name"].map(COUNTRY_MAP)
        n_unmapped = int(df_long["country_code"].isna().sum())
        if n_unmapped:
            log(f"  WARNING: {n_unmapped:,} country-year rows unmapped: "
                f"{df_long.loc[df_long['country_code'].isna(), 'country_name'].unique().tolist()}")
        df_long = df_long.dropna(subset=["country_code"])
        df_ctry = df_long[["year", "country_code", "stock_traded_gdp"]].copy()
        df_ctry["stock_traded_gdp"] = pd.to_numeric(df_ctry["stock_traded_gdp"], errors="coerce")
        log(f"  After reshape: {len(df_ctry):,} rows | "
            f"{df_ctry['country_code'].nunique():,} countries | {df_ctry['year'].nunique():,} years")
        for cc in ["CN", "GB", "US", "JP", "KR"]:
            log(f"    {cc}: {int((df_ctry['country_code'] == cc).sum()):,} year-rows")

        # M1 FIXER R1: completed_d_yr 优先，announced_d_yr 回退，再兜底 deal_year
        if "completed_d_yr" in df.columns:
            df["_merge_year"] = pd.to_numeric(df["completed_d_yr"], errors="coerce").astype("Int64")
            df["_year_source"] = "completed_d_yr"
        else:
            df["_merge_year"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
            df["_year_source"] = "none"
        if "announced_d_yr" in df.columns:
            _fb = df["_merge_year"].isna() & df["announced_d_yr"].notna()
            df.loc[_fb, "_merge_year"] = pd.to_numeric(
                df.loc[_fb, "announced_d_yr"], errors="coerce").astype("Int64")
            df.loc[_fb, "_year_source"] = "announced_d_yr"
        if df["_merge_year"].isna().any() and "deal_year" in df.columns:
            _fb2 = df["_merge_year"].isna() & df["deal_year"].notna()
            df.loc[_fb2, "_merge_year"] = df.loc[_fb2, "deal_year"].astype("Int64")
            df.loc[_fb2, "_year_source"] = "deal_year"
        log(f"  M1 合并年份来源: completed_d_yr="
            f"{int((df['_year_source'] == 'completed_d_yr').sum()):,} | "
            f"announced_d_yr={int((df['_year_source'] == 'announced_d_yr').sum()):,} | "
            f"deal_year={int((df['_year_source'] == 'deal_year').sum()):,} | "
            f"缺失 {int(df['_merge_year'].isna().sum()):,}")

        # merge keys
        df["_merge_cc_tar"] = df["tar_country_code"].astype(str).str.strip()
        df["_merge_cc_acq"] = df["acq_country_code"].astype(str).str.strip()

        # tar
        df_ctry_tar = df_ctry.rename(columns={
            "country_code": "_merge_cc_tar", "year": "_merge_year_tar",
            "stock_traded_gdp": "stock_traded_gdp_tar"})
        df_ctry_tar["_merge_year_tar"] = df_ctry_tar["_merge_year_tar"].astype("Int64")
        df["_merge_year_tar"] = df["_merge_year"]
        df = df.merge(df_ctry_tar[["_merge_cc_tar", "_merge_year_tar", "stock_traded_gdp_tar"]],
                      on=["_merge_cc_tar", "_merge_year_tar"], how="left")
        assert len(df) == INPUT_ROWS, "row count changed after tar gdp merge"

        # acq
        df_ctry_acq = df_ctry.rename(columns={
            "country_code": "_merge_cc_acq", "year": "_merge_year_acq",
            "stock_traded_gdp": "stock_traded_gdp_acq"})
        df_ctry_acq["_merge_year_acq"] = df_ctry_acq["_merge_year_acq"].astype("Int64")
        df["_merge_year_acq"] = df["_merge_year"]
        df = df.merge(df_ctry_acq[["_merge_cc_acq", "_merge_year_acq", "stock_traded_gdp_acq"]],
                      on=["_merge_cc_acq", "_merge_year_acq"], how="left")
        assert len(df) == INPUT_ROWS, "row count changed after acq gdp merge"

        # cleanup temp merge columns（保留 _year_source 供追溯）
        df = df.drop(columns=["_merge_year_tar", "_merge_cc_tar",
                              "_merge_year_acq", "_merge_cc_acq", "_merge_year"])
        n_tar = int(df["stock_traded_gdp_tar"].notna().sum())
        n_acq = int(df["stock_traded_gdp_acq"].notna().sum())
        log(f"  stock_traded_gdp_tar 匹配: {n_tar:,}/{INPUT_ROWS:,} "
            f"({n_tar/INPUT_ROWS*100:.1f}%)")
        log(f"  stock_traded_gdp_acq 匹配: {n_acq:,}/{INPUT_ROWS:,} "
            f"({n_acq/INPUT_ROWS*100:.1f}%)")
        log(f"  行数不变: {len(df):,} == {INPUT_ROWS:,}")

    # ---------- STEP 2 pre-filter vars ----------
    banner("STEP 2 — Pre-filter variable preparation [full dataset]")

    # 2.0 deal_year  [2] announced_d 优先
    y_comp = _year_from_date(df, "completed_d")
    y_ann  = _year_from_date(df, "announced_d")
    y_old  = pd.to_numeric(df["deal_year"], errors="coerce") \
             if "deal_year" in df.columns else pd.Series(np.nan, index=df.index)

    if DEAL_YEAR_PRIORITY == "announced":
        df["deal_year"] = y_ann.fillna(y_comp).fillna(y_old)
        log(f"deal_year = announced_d 优先，回退 completed_d   [修复口径]")
        log(f"  from announced_d {int(y_ann.notna().sum()):,} | "
            f"fallback completed_d {int((y_ann.isna() & y_comp.notna()).sum()):,}")
    else:
        df["deal_year"] = y_comp.fillna(y_ann).fillna(y_old)
        log(f"deal_year = completed_d 优先，回退 announced_d   [v4 旧行为]")
        log(f"  from completed_d {int(y_comp.notna().sum()):,} | "
            f"fallback announced_d {int((y_comp.isna() & y_ann.notna()).sum()):,}")

    df["deal_year"] = df["deal_year"].astype("Int64")
    log(f"  non-null {int(df['deal_year'].notna().sum()):,} | "
        f"still NaN {int(df['deal_year'].isna().sum()):,}")

    # 2.1 SIC
    df["tar_sic4"] = pd.to_numeric(df["tar_primary_sic_code"], errors="coerce")
    df["tar_sic3"] = (df["tar_sic4"] // 10).astype("Int64")
    df["tar_sic2"] = (df["tar_sic4"] // 100).astype("Int64")
    df["acq_sic4"] = pd.to_numeric(df["acq_primary_sic_code"], errors="coerce")
    df["acq_sic3"] = (df["acq_sic4"] // 10).astype("Int64")
    df["acq_sic2"] = (df["acq_sic4"] // 100).astype("Int64")
    log(f"tar_sic3 non-null: {int(df['tar_sic3'].notna().sum()):,}  "
        f"acq_sic3 non-null: {int(df['acq_sic3'].notna().sum()):,}")

    # 2.2 SIC rolling peer counts
    log("\nComputing SIC rolling peer counts...")
    d = df[["_row_id", "deal_year", "tar_sic3", "tar_sic2"]].dropna(
        subset=["deal_year", "tar_sic3"]).copy()
    d["deal_year"] = d["deal_year"].astype("int64")
    s3 = (d.groupby(["tar_sic3", "deal_year"]).size()
          .reset_index(name="n").sort_values(["tar_sic3", "deal_year"]))
    s3["sic3_rolling_count"] = s3.groupby("tar_sic3")["n"].cumsum() - 1
    d2 = d.dropna(subset=["tar_sic2"]).copy()
    s2 = (d2.groupby(["tar_sic2", "deal_year"]).size()
          .reset_index(name="n").sort_values(["tar_sic2", "deal_year"]))
    s2["sic2_rolling_count"] = s2.groupby("tar_sic2")["n"].cumsum() - 1
    d = d.merge(s3[["tar_sic3", "deal_year", "sic3_rolling_count"]],
                on=["tar_sic3", "deal_year"], how="left")
    d = d.merge(s2[["tar_sic2", "deal_year", "sic2_rolling_count"]],
                on=["tar_sic2", "deal_year"], how="left")
    df = df.merge(d[["_row_id", "sic3_rolling_count", "sic2_rolling_count"]],
                  on="_row_id", how="left")
    assert len(df) == INPUT_ROWS
    df["sic_benchmark_level"] = np.where(df["sic3_rolling_count"] >= 3, 3,
                                 np.where(df["sic2_rolling_count"] >= 3, 2, np.nan))
    log(f"  sic_benchmark_level=3: {int((df['sic_benchmark_level']==3).sum()):,}  "
        f"=2: {int((df['sic_benchmark_level']==2).sum()):,}")

    # 2.3 text peer pool（需 tar_overview）
    if "tar_overview" in df.columns:
        df["tar_overview_wordcount"] = (
            _as_str(df["tar_overview"]).str.split().str.len().fillna(0).astype(int))
        dt = (df.loc[df["tar_overview"].notna() & df["deal_year"].notna(),
                     ["_row_id", "deal_year"]]
              .sort_values(["deal_year", "_row_id"]).reset_index(drop=True).copy())
        dt["text_pool_rolling_count"] = dt.index
        df = df.merge(dt[["_row_id", "text_pool_rolling_count"]],
                      on="_row_id", how="left")
        assert len(df) == INPUT_ROWS
        log(f"  tar_overview_wordcount >= {MIN_WORDCOUNT}: "
            f"{int((df['tar_overview_wordcount']>=MIN_WORDCOUNT).sum()):,}")
    else:
        banner("[UNAVAILABLE] tar_overview not in 04")
        log("  tar_overview_wordcount / text_pool_rolling_count 无法计算。")
        log("  需用时：在 04 的 FORCE_KEEP_MASTER 加 'tar_overview' 后重跑 04。")
        df["tar_overview_wordcount"]  = np.nan
        df["text_pool_rolling_count"] = np.nan

    # ---------- STEP 2.5 listed diagnostics ----------
    banner("STEP 2.5 — Listed-status diagnostics (BEFORE filters)")
    a = pd.to_numeric(df["acq_listed"], errors="coerce") if "acq_listed" in df.columns else None
    t = pd.to_numeric(df["tar_listed"], errors="coerce") if "tar_listed" in df.columns else None
    for nm, v in [("acq_listed", a), ("tar_listed", t)]:
        if v is None:
            log(f"  [warn] {nm} not in 04"); continue
        log(f"\n{nm} distribution (input N={INPUT_ROWS:,}):")
        for val, cnt in v.value_counts(dropna=False).sort_index().items():
            log(f"    {str(val):<10s}: {cnt:>8,}  ({cnt/INPUT_ROWS*100:5.1f}%)")
    if a is not None and t is not None:
        log("\nCross-tab acq_listed (rows) x tar_listed (cols):")
        log(pd.crosstab(a, t, dropna=False, margins=True).to_string())
        n_unlist  = int((t == 0).sum())
        n_listacq = int((a == 1).sum())
        n_both    = int(((a == 1) & (t == 0)).sum())
        n_at      = int((a.notna() & t.notna()).sum())
        log(f"\n  >>> KEY SHARES (denominator = input N={INPUT_ROWS:,}) <<<")
        log(f"    Listed acquirer  (acq_listed=1) : {n_listacq:>8,}  {n_listacq/INPUT_ROWS*100:5.1f}%")
        log(f"    Unlisted target  (tar_listed=0) : {n_unlist:>8,}  {n_unlist/INPUT_ROWS*100:5.1f}%")
        log(f"    Listed acq AND unlisted tar     : {n_both:>8,}  {n_both/INPUT_ROWS*100:5.1f}%")
        if n_at > 0:
            log(f"    [both known N={n_at:,}] unlisted tar {n_unlist/n_at*100:5.1f}% | "
                f"listed acq+unlisted tar {n_both/n_at*100:5.1f}%")

    # ---------- STEP 3 filters ----------
    banner("STEP 3 — Sequential filters")
    log(f"\nInput: {INPUT_ROWS:,} rows")
    moments(df, "start")

    dt_col = next((c for c in ("deal_type", "deal_type_all") if c in df.columns), None)
    if dt_col is None:
        log("\n  [FATAL] 既无 deal_type 也无 deal_type_all")
    else:
        log(f"\n  [SOURCE] deal_type 使用列 = '{dt_col}'"
            + ("  (建议 04 白名单补 'deal_type')" if dt_col != "deal_type" else ""))

    if ENABLE_F1_ACQ_LISTED:
        df = apply_filter(df, pd.to_numeric(df["acq_listed"], errors="coerce") == 1,
                          "F1 (acq_listed = 1)")
        moments(df, "after F1")
    
        # 在 if ENABLE_F2_TAR_UNLISTED: 之前插入
    #_t = df["tar_listed"]
    #log(f"[F2-PRE] dtype={_t.dtype}  len={len(_t)}  "
        #f"n_nan={_t.isna().sum()}  "
        #f"n_eq0_raw={( _t == 0).sum()}  "
        #f"n_eq0_num={(pd.to_numeric(_t, errors='coerce') == 0).sum()}")
    #log(f"[F2-PRE] head={_t.head(10).tolist()}")

    if ENABLE_F2_TAR_UNLISTED:
        df = apply_filter(df, pd.to_numeric(df["tar_listed"], errors="coerce") == 0,
                          "F2 (tar_listed = 0)")
        moments(df, "after F2")

    if ENABLE_F3_PUBLIC_STRUCT:
        sc = next((c for c in ("deal_struct_all", "deal_struct") if c in df.columns), None)
        if sc is None:
            log("  [SKIP] F3: no deal_struct column")
        else:
            s = _as_str(df[sc]).fillna("").str.strip().str.lower()
            df = apply_filter(df, ~s.apply(lambda x: any(p in x for p in PUB_STRUCT)),
                              f"F3 (drop public-offer structures, src={sc})")
            moments(df, "after F3")

    if ENABLE_F4_PERIOD:
        y = pd.to_numeric(df["deal_year"], errors="coerce")
        df = apply_filter(df, (y >= YEAR_START) & (y <= YEAR_END),
                          f"F4 (sample period {YEAR_START}-{YEAR_END}, "
                          f"year from {DEAL_YEAR_PRIORITY}_d)")
        moments(df, "after F4")

    # [1] F5 恢复（v4 中该段被整段注释）
    if ENABLE_F5_DEAL_TYPE:
        if dt_col is None:
            log("  [SKIP] F5: no deal_type column")
        else:
            s = _as_str(df[dt_col]).fillna("").str.strip()
            df = apply_filter(df, s.str.startswith("Acquisition"),
                              f"F5 ({dt_col} starts with 'Acquisition')")
            moments(df, "after F5")

    if ENABLE_F6_STAKE50:
        if dt_col is None:
            log("  [SKIP] F6: no deal_type column")
        else:
            msk = df["stake_acq_pct"].isna()
            df["stake_acq_pct_imputed"] = np.nan
            df.loc[msk, "stake_acq_pct_imputed"] = df.loc[msk, dt_col].apply(
                infer_stake_from_deal_type)
            log(f"      stake 缺失 {int(msk.sum()):,}；从 {dt_col} 插补成功 "
                f"{int(df['stake_acq_pct_imputed'].notna().sum()):,}")
            stake_eff = pd.to_numeric(df["stake_acq_pct"], errors="coerce").fillna(
                df["stake_acq_pct_imputed"])
            df["stake_eff_pct"] = stake_eff
            df = apply_filter(df, stake_eff >= MIN_STAKE_PCT,
                              f"F6 (stake >= {MIN_STAKE_PCT}%, impute if missing)")
            moments(df, "after F6")

    if ENABLE_F7_VALUE_1M:
        df = apply_filter(df,
            pd.to_numeric(df["deal_value"], errors="coerce") >= MIN_DEAL_VALUE,
            f"F7 (deal_value >= {MIN_DEAL_VALUE} th USD = USD 1M)")
        moments(df, "after F7")

    if ENABLE_F8_SIC_CTRY:
        m = (df["tar_sic3"].notna() & df["acq_sic3"].notna() &
             df["tar_country_code"].notna() & df["acq_country_code"].notna())
        df = apply_filter(df, m, "F8 (tar/acq SIC3 & country all available)")
        moments(df, "after F8")

    if ENABLE_F9_COMPLETED:
        sc9 = "deal_status_clean" if "deal_status_clean" in df.columns else "deal_status"
        df = apply_filter(df,
            _as_str(df[sc9]).fillna("").str.strip().str.lower().eq("completed"),
            f"F9 ({sc9} == 'Completed' 严格)")
        moments(df, "after F9")

    if ENABLE_F10_WORDCOUNT:
        if "tar_overview" not in df.columns:
            log("  [SKIP] F10: tar_overview unavailable")
        else:
            df = apply_filter(df, df["tar_overview_wordcount"] >= MIN_WORDCOUNT,
                              f"F10 (tar_overview_wordcount >= {MIN_WORDCOUNT})")
            moments(df, "after F10")

    if ENABLE_F11_TEXTPOOL:
        if "tar_overview" not in df.columns:
            log("  [SKIP] F11: tar_overview unavailable")
        else:
            df = apply_filter(df, df["text_pool_rolling_count"] >= MIN_TEXT_POOL,
                              f"F11 (text_pool_rolling_count >= {MIN_TEXT_POOL})")
            moments(df, "after F11")
    # ---------- [NEW] F12 前诊断（可选） ----------
    _miss = df["stock_traded_gdp_tar"].isna() | df["stock_traded_gdp_acq"].isna()
    log(f"\n[DIAG] F8 后 stock/GDP 缺失: {int(_miss.sum()):,} / {len(df):,} "
        f"({_miss.mean()*100:.1f}%)")
    _tab = (df.assign(_miss=_miss).groupby("tar_country_code")["_miss"]
              .agg(["mean", "size"]).sort_values("mean", ascending=False))
    log("[DIAG] tar 国家缺失率 top15:\n" + _tab.head(15).to_string())
    # ---------- [NEW] end ----------
    
    # ---------- [NEW] F12: stock/GDP 缺失行 ----------
    if ENABLE_F12_STOCK_GDP:
        m = df["stock_traded_gdp_tar"].notna() & df["stock_traded_gdp_acq"].notna()
        df = apply_filter(df, m, "F12 (stock_traded_gdp tar & acq available)")
        moments(df, "after F12")
    # ---------- [NEW] end ----------

    # ---------- STEP 4 post-filter vars ----------
    banner("STEP 4 — Post-filter variable construction")
    # ---------- STEP 4 post-filter vars ----------
    banner("STEP 4 — Post-filter variable construction")

    bv = df["tar_sic3"].notna() & df["acq_sic3"].notna()
    df["cross_industry"] = np.where(
        bv, (df["tar_sic3"].astype("float") != df["acq_sic3"].astype("float")).astype(int),
        np.nan)
    log(f"cross_industry =1: {int((df['cross_industry']==1).sum()):,}  "
        f"(NaN {int(df['cross_industry'].isna().sum()):,})")

    dv = pd.to_numeric(df["deal_value"], errors="coerce")
    df["ln_deal_value"] = np.log(dv.where(dv > 0))
    log(f"ln_deal_value non-null: {int(df['ln_deal_value'].notna().sum()):,}")

    bc = df["tar_country_code"].notna() & df["acq_country_code"].notna()
    cb_new = np.where(bc, (_as_str(df["tar_country_code"]) !=
                           _as_str(df["acq_country_code"])).astype(int), np.nan)
    if "cross_border" in df.columns:
        diff = int((pd.to_numeric(df["cross_border"], errors="coerce") != cb_new).sum())
        log(f"cross_border 重算：与 04 原有值不一致 {diff:,} 行"
            + ("  (0 = 完全一致)" if diff == 0 else "  <- 注意来源差异"))
    df["cross_border"] = cb_new
    log(f"cross_border  =1: {int((df['cross_border']==1).sum()):,}")

    for side in ("tar", "acq"):
        c = f"{side}_incorp_d_year"
        if c in df.columns and "deal_year" in df.columns:
            df[f"{side}_age"] = pd.to_numeric(df["deal_year"], errors="coerce") \
                                - pd.to_numeric(df[c], errors="coerce")
            df.loc[df[f"{side}_age"] < 0, f"{side}_age"] = np.nan
    if "tar_age" in df.columns:
        log(f"tar_age non-null: {int(df['tar_age'].notna().sum()):,}  "
            f"acq_age non-null: {int(df['acq_age'].notna().sum()):,}")

    # ---- payment method ----
    PAY_SRC = "deal_pay_method_all"
    if PAY_SRC not in df.columns:
        log(f"\n  [SKIP] payment dummies: {PAY_SRC} not in 04")
    else:
        unmapped = set()
        def _to_classes(s):
            if pd.isna(s) or not str(s).strip():
                return frozenset()
            out = set()
            for v in (x.strip() for x in str(s).split("|")):
                if not v:
                    continue
                c = PAY_CLASS_MAP.get(v)
                if c is None:
                    unmapped.add(v)
                else:
                    out.add(c)
            return frozenset(out)
        ps = _as_str(df[PAY_SRC]).apply(_to_classes)
        if unmapped:
            log(f"\n  WARNING unmapped pay values: {sorted(unmapped)}")
        df["pay_has_cash"]   = ps.apply(lambda c: int("cash" in c))
        df["pay_has_shares"] = ps.apply(lambda c: int("shares" in c))
        df["pay_has_debt"]   = ps.apply(lambda c: int("debt" in c))
        df["pay_has_other"]  = ps.apply(lambda c: int("other" in c))
        df["pay_n_class"]    = ps.apply(len)
        df["pay_pure_cash"]   = ps.apply(lambda c: int(c == {"cash"}))
        df["pay_pure_shares"] = ps.apply(lambda c: int(c == {"shares"}))
        df["pay_pure_debt"]   = ps.apply(lambda c: int(c == {"debt"}))
        df["pay_pure_other"]  = ps.apply(lambda c: int(c == {"other"}))
        df["pay_mix"]         = ps.apply(lambda c: int(len(c) > 1))
        df["pay_unknown"]     = ps.apply(lambda c: int(len(c) == 0))
        df["pay_mix_cash_shares"] = ps.apply(lambda c: int(c == {"cash", "shares"}))
        df["pay_mix_with_shares"] = ps.apply(lambda c: int(len(c) > 1 and "shares" in c))
        df["pay_mix_with_cash"]   = ps.apply(lambda c: int(len(c) > 1 and "cash" in c))
        df["pay_mix_with_debt"]   = ps.apply(lambda c: int(len(c) > 1 and "debt" in c))
        df["pay_class"] = np.select(
            [df["pay_unknown"] == 1, df["pay_pure_cash"] == 1,
             df["pay_pure_shares"] == 1, df["pay_pure_debt"] == 1,
             df["pay_pure_other"] == 1],
            ["unknown", "cash", "shares", "debt", "other"], default="mix")
        chk = df[["pay_pure_cash", "pay_pure_shares", "pay_pure_debt",
                  "pay_pure_other", "pay_mix", "pay_unknown"]].sum(axis=1)
        if not (chk == 1).all():
            log("  [!!] payment dummies 不构成完备划分")
        log("\nPayment method distribution:")
        for k in ["cash", "shares", "debt", "other", "mix", "unknown"]:
            m = (df["pay_class"] == k)
            if m.sum():
                log(f"  {k:<9s} N={int(m.sum()):>7,} {m.mean():>6.1%}")

    # ---------- STEP 5 save ----------
    banner("STEP 5 — Save")
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    n_deals = df[DEAL].nunique()
    log(f"Saved -> {OUT_FILE}")
    log(f"  rows={len(df):,} | cols={df.shape[1]} | unique {DEAL}={n_deals:,}")

    steps, cum = [], 0
    for label, nb, na, nd in filter_summary:
        cum += nd
        steps.append(dict(step=label, n_before=nb, n_after=na, dropped=nd, cum_loss=cum))
    pd.DataFrame(steps).to_csv(STEP_CSV, index=False, encoding="utf-8-sig")
    log(f"Steps CSV -> {STEP_CSV}")

    # ---------- FINAL 变异检查 ----------
    banner("FINAL — 结果变量变异")
    log(f"  {'column':<20s} {'mean':>9s} {'N(1)':>10s} {'N(0)':>10s}")
    log("  " + "-" * 52)
    for c in (COL_OUT_STRICT, COL_OUT_WIDE, "rumor_jfe"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            n1 = int((v == 1).sum()); n0 = int((v == 0).sum())
            log(f"  {c:<20s} {v.mean():>9.4f} {n1:>10,} {n0:>10,}")
            if v.mean() > 0.95 or v.mean() < 0.05:
                log(f"      [!!] 几乎无变异，不宜作结果变量")
    log(f"\n  论文参照 p_D = 0.6568")

    # ---------- STEP 6 summary ----------
    banner("FILTER LOG SUMMARY")
    lt = [f"Input : 04_deal_firm_full.csv  {INPUT_ROWS:,} rows",
          f"Year  : {DEAL_YEAR_PRIORITY}_d 优先, period {YEAR_START}-{YEAR_END}", ""]
    cum = 0
    for label, nb, na, nd in filter_summary:
        cum += nd
        lt.append(f"{label[:52]:<52s}: dropped {nd:>7,}  -> {na:>7,} left  (cum {cum:>7,})")
    lt += ["", f"Final sample : {len(df):,} rows | {n_deals:,} unique deals"]
    log("\n".join(lt))
    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log_lines) + "\n")
    log(f"Filter log -> {LOG_FILE}")

    # ---------- STEP 7 validation ----------
    banner("STEP 7 — Validation")
    rem = [INPUT_ROWS] + [na for _, _, na, _ in filter_summary]
    mono = all(rem[i] >= rem[i + 1] for i in range(len(rem) - 1))
    log(f"Monotone decreasing : {'PASS' if mono else 'FAIL'}")
    log(f"Final N             : {len(df):,}")
    log(f"Retention           : {len(df)/INPUT_ROWS*100:.2f}%")
    log(f"Output size         : {os.path.getsize(OUT_FILE):,} bytes")
    log(f"unique {DEAL} == rows : {df[DEAL].nunique() == len(df)}")
    log("\nScript 05 complete (v7).")


if __name__ == "__main__":
    main()
