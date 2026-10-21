# -*- coding: utf-8 -*-
r"""
01a_clean_industry.py — Industry export (FIXED: composite PK deal_num+tar+acq)
====================================================
Input  : D:\zephyr\industry\cleaned_output\MA_industry.csv
Output : D:\MA\data\cleaned\01_deal_sic_industry.csv
FIX:
  1. Primary composite key: ["deal_num","tar_bvd_id_num","acq_bvd_id_num"]
  2. Fallback: if bvd id missing, use tar_name / acq_name for composite dedup
  3. DO NOT collapse multiple acquirers for same deal‑target pair
  4. Remove old single‑dimension tar_key dedup bug
"""
# Standalone version (2026-10-14): fully self‑contained — no imports from common_01a.py.
import os
import re
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
# Work log (txt) — every log line of every 01a script is mirrored here.
WORK_LOG = os.path.join(CLEANED, "01a_industry.txt")
# STRICT=True -> a missing input file raises instead of being skipped.
STRICT = False
# Run the (superseded) CoreNLP vendor‑identity script?  Set False to skip.
RUN_VENDOR_NER = True
# Single merged raw files (Zephyr "cleaned_output" edition).
RAW_FILES = {
    "overview":       os.path.join(ZEPHYR, "overview",       "cleaned_output", "MA_overview.csv"),
    "multiple":       os.path.join(ZEPHYR, "multiple",       "cleaned_output", "MA_multiple.csv"),
    "structure_date": os.path.join(ZEPHYR, "structure_date", "cleaned_output", "MA_structure_date.csv"),
    "value":          os.path.join(ZEPHYR, "value",          "cleaned_output", "MA_value.csv"),
    # ── NOT PROVIDED yet — placeholder paths; supply the files and re‑run ──
    "industry":       os.path.join(ZEPHYR, "industry",       "cleaned_output", "MA_industry.csv"),
    "info_source":    os.path.join(ZEPHYR, "info_source",    "cleaned_output", "MA_info_source.csv"),
}
MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]
def clean_missing(df):
    """Replace missing‑style tokens with NaN; strip strings (same as old 01a)."""
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
# ---------------------------------------------------------------------
# Module 1.1 — Logging (console + work‑log txt)
# ---------------------------------------------------------------------
_LOG_LINES = []
def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))
def flush_log():
    """Append buffered lines to the work‑log txt (crash‑safe: flushed at
    the end of every script and at key checkpoints)."""
    if not _LOG_LINES:
        return
    with open(WORK_LOG, "a", encoding="utf‑8") as fh:
        fh.write("\n".join(_LOG_LINES) + "\n")
    _LOG_LINES.clear()
def module_banner(num, title):
    log("\n" + "=" * 78)
    log(f"Module {num} — {title}")
    log("=" * 78)
def report_coverage(df, label, cols=None):
    """Per‑variable non‑null / missing diagnostics (missing % is what the
    user asked for); mirrors and extends the old print diagnostics."""
    cols = list(df.columns) if cols is None else [c for c in cols if c in df.columns]
    log(f"\n  [{label}] coverage — {len(df):,} rows")
    for c in cols:
        n = int(df[c].notna().sum())
        pct = n / len(df) * 100 if len(df) else 0.0
        log(f"    {c:<45s} non‑null {n:>9,} ({pct:6.2f}%)  "
            f"missing {len(df) - n:>9,} ({100.0 - pct:6.2f}%)")
def report_expected(df, label, expected):
    """List expected columns missing from the file (drives FIELD_MAP_* edits)."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log(f"  [expected] {label}: {len(missing)} expected column(s) missing -> "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}")
        log(f"    If your download renamed them, add 'new_name': 'old_name' "
            f"entries to FIELD_MAP_{label.upper()} and re‑run.")
    else:
        log(f"  [expected] {label}: all {len(expected)} expected columns present")
def report_sic_stats(df, label):
    """★ 新增：SIC/行业核心变量专项统计诊断 ★
    打印唯一值数、Top5分布、数值范围等，类似 value 脚本的数值转换诊断。"""
    sic_cols = [
        "tar_primary_sic_code", "tar_sic_codes", "tar_primary_naics_code",
        "acq_primary_sic_code", "acq_sic_codes", "ven_primary_sic_code",
    ]
    log(f"\n  [{label}] SIC/Industry statistics — {len(df):,} rows")
    log(f"  {'Column':<40s} {'Non‑null':>9s} {'Missing%':>9s} {'Unique':>8s} {'Top‑3 values'}")
    log(f"  {'‑'*40} {'‑'*9} {'‑'*9} {'‑'*8} {'‑'*40}")

    for c in sic_cols:
        if c not in df.columns:
            log(f"  {c:<40s} {'[NOT IN DF]':>9s}")
            continue

        n_notnull = int(df[c].notna().sum())
        n_total = len(df)
        miss_pct = (1 - n_notnull / n_total) * 100 if n_total else 0.0
        n_unique = int(df[c].nunique())

        # Top‑3 值及频次
        top_vals = ""
        if n_notnull > 0:
            vc = df[c].value_counts().head(3)
            top_vals = ", ".join([f"{str(v)[:12]}({cnt})" for v, cnt in zip(vc.index, vc.values)])

        log(f"  {c:<40s} {n_notnull:>9,} {miss_pct:>8.1f}% {n_unique:>8,}   {top_vals}")

    # 描述性字段覆盖率汇总
    desc_cols = ["tar_overview", "tar_trade_descr_en", "tar_busi_descr",
                 "acq_overview", "ven_overview"]
    log(f"\n  [{label}] Description fields coverage:")
    for c in desc_cols:
        if c in df.columns:
            n = int(df[c].notna().sum())
            pct = n / len(df) * 100 if len(df) else 0.0
            log(f"    {c:<40s} non‑null {n:>9,} ({pct:6.2f}%)")
# ---------------------------------------------------------------------
# Module 1.2 — Key normalisation (canonical names / dtypes across files)
# ---------------------------------------------------------------------
CANONICAL_KEYS = ["deal_num",
                  "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
                  "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
                  "ven_name", "ven_bvd_id_num"]
KEY_ALIASES = {
    "deal_num":         ["deal_num", "deal number", "dealnumber", "deal id"],
    "tar_name":         ["tar_name", "target name", "targetname"],
    "tar_bvd_id_num":   ["tar_bvd_id_num", "tar_bvd_id", "target bvd id number",
                         "targetbvdidnumber", "bvd id number"],
    "tar_orbis_id_num": ["tar_orbis_id_num", "tar_orbis_id", "target orbis id number",
                         "targetorbisidnumber", "orbis id number"],
    "acq_name":         ["acq_name", "acquirer name", "acquirorname"],
    "acq_bvd_id_num":   ["acq_bvd_id_num", "acq_bvd_id", "acquirer bvd id number",
                         "acquirorbvdidnumber"],
    "acq_orbis_id_num": ["acq_orbis_id_num", "acq_orbis_id", "acquirer orbis id number",
                         "acquirorbisidnumber"],
    "ven_name":         ["ven_name", "vendor name", "vendorname"],
    "ven_bvd_id_num":   ["ven_bvd_id_num", "ven_bvd_id", "vendor bvd id number",
                         "vendorbvdidnumber"],
}
# Old canonical column sets (kept identical to the original 01a so that
# downstream scripts 02..05 work without change).
INDUSTRY_KEEP = [
    "deal_num", "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    "tar_overview", "tar_trade_descr_en", "tar_busi_descr",
    "acq_overview", "ven_overview",
    "tar_primary_sic_code", "tar_sic_codes", "tar_primary_naics_code",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "acq_primary_sic_code", "acq_sic_codes",
    "ven_name", "ven_bvd_id_num", "ven_primary_sic_code",
]
# Columns that simply do not exist in the current Zephyr industry export.
# They are kept in INDUSTRY_KEEP (harmless: report_coverage filters by
# `if c in df.columns`) but excluded from the REQUIRED list used by
# report_expected, so the log stops claiming 6 phantom missing columns.
INDUSTRY_OPTIONAL = ["tar_trade_descr_en", "tar_busi_descr", "ven_overview",
                     "tar_sic_codes", "acq_sic_codes", "ven_primary_sic_code"]
INDUSTRY_REQUIRED = [c for c in INDUSTRY_KEEP if c not in INDUSTRY_OPTIONAL]
SD_KEEP = ["deal_num", "deal_type", "deal_status", "deal_struct", "deal_fin",
           "deal_pay_method", "deal_pay_method_all", "deal_pay_method_n",
           "deal_struct_all", "deal_struct_n", "deal_fin_all", "deal_fin_n",
           "deal_type_all", "deal_type_n",
           "announced_d", "completed_d", "withdrawn_d",
           "announced_d_yr", "completed_d_yr", "withdrawn_d_yr", "assumed_comp_d"]
VAL_KEEP = ["deal_num", "deal_value", "deal_enterprise_value", "deal_equity_value",
            "deal_modelled_enterprise_value", "stake_acq_pct", "stake_final_pct",
            "currency"]
def normalize_orbis_id(series):
    """Zero‑pad Orbis ID numbers to 9 digits (keeps leading zeros; str)."""
    def fix(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s in ("nan", "None", ""):
            return np.nan
        return s.zfill(9)
    return series.apply(fix)

def find_col(df, candidates):
    """First column whose name (case‑folded, stripped) matches a candidate."""
    fold = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = fold.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None
def ensure_key_columns(df, label, wanted=None):
    """Rename key columns to canonical names, normalise dtypes, log findings.
    Returns (df, found, missing)."""
    wanted = list(wanted) if wanted is not None else list(CANONICAL_KEYS)
    found, missing = [], []
    for key in wanted:
        if key in df.columns:
            found.append(key)
            continue
        hit = find_col(df, KEY_ALIASES.get(key, [key]))
        if hit is not None:
            df = df.rename(columns={hit: key})
            found.append(key)
        else:
            missing.append(key)
    if "deal_num" in df.columns:
        df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")
    for k in wanted:
        if k in df.columns and k != "deal_num":
            if k.endswith("_orbis_id_num"):
                df[k] = normalize_orbis_id(df[k])
            elif k.endswith("_bvd_id_num"):
                # BvD IDs: strip trailing ".0" (pandas reads numeric IDs as
                # float when the column has blanks), then keep clean string.
                df[k] = df[k].astype("object")
                m = df[k].notna()
                if m.any():
                    def _fix(val):
                        s = str(val).strip()
                        if s.endswith(".0"):
                            s = s[:-2]
                        return s if s not in ("nan", "None", "") else np.nan
                    df.loc[m, k] = df.loc[m, k].apply(_fix)
            else:
                df[k] = df[k].astype("object")
                m = df[k].notna()
                if m.any():
                    df.loc[m, k] = df.loc[m, k].astype(str).str.strip()
                    df.loc[m, k] = df.loc[m, k].replace(
                        {"nan": np.nan, "None": np.nan, "": np.nan})
    if found or missing:
        log(f"  [keys] {label}: found={found}  missing={missing}")
    return df, found, missing
# ---------------------------------------------------------------------
# Module 2 — Raw loader (single‑file edition)
# ---------------------------------------------------------------------
def apply_field_map(df, label, field_map):
    """Rename new (cleaned) columns -> old canonical names; log each rename
    and warn about expected new names that are absent."""
    for new_name, old_name in field_map.items():
        if new_name in df.columns:
            if old_name in df.columns and old_name != new_name:
                df = df.drop(columns=[old_name])
            df = df.rename(columns={new_name: old_name})
            log(f"  [map:{label}] {new_name!r} -> {old_name!r}")
        elif old_name not in df.columns:
            log(f"  [map:{label}] WARNING: expected column {new_name!r} "
                f"(-> {old_name!r}) NOT found in file")
    return df
def read_raw(file_key, label, field_map=None, expected=None):
    """Read one merged raw file, ffill deal_num (multi‑row‑per‑deal possible),
    clean missing, apply field map + key normalisation.  Returns df, or None
    when the file is missing (logged as [MISSING INPUT]; STRICT raises)."""
    fp = RAW_FILES[file_key]
    if not os.path.exists(fp):
        log(f"\n  ✗ [MISSING INPUT] {label}: {fp}")
        log(f"    Please provide this file (see the script docstring). Module skipped.")
        if STRICT:
            raise FileNotFoundError(fp)
        return None
    df = pd.read_csv(fp, encoding="utf‑8‑sig", low_memory=False)
    df.columns = df.columns.str.strip()
    df = clean_missing(df)
    deal_col = find_col(df, KEY_ALIASES["deal_num"]) or "deal_num"
    if deal_col in df.columns:
        df[deal_col] = df[deal_col].ffill()
    if field_map:
        df = apply_field_map(df, label, field_map)
    df, found, missing = ensure_key_columns(df, label)
    if "deal_num" not in df.columns:
        log(f"  ✗ {label}: no deal_num column found — module skipped")
        return None
    if expected:
        report_expected(df, label, expected)
    log(f"  [{label}] loaded: {len(df):,} rows | {df.shape[1]} cols | "
        f"unique deal_num: {df['deal_num'].nunique():,}")
    return df
# ---------------------------------------------------------------------
# FIELD_MAP_INDUSTRY — new (cleaned) column name -> old canonical name.
# The industry mapping was not provided; the file is assumed to already
# use the old canonical names (identity).  If your download renamed
# columns, add "new_name": "old_name" entries here and re‑run.
# ---------------------------------------------------------------------
FIELD_MAP_INDUSTRY = {}
def main():
    module_banner("M2.5+M4", "Industry export -> 01_deal_sic_industry.csv (FIX composite PK)")
    log("  NOTE: MUST RUN together with fixed 01a_clean_overview.py")
    log("        PRIMARY COMPOSITE KEY = (deal_num, tar_bvd_id_num, acq_bvd_id_num)")
    log("        overview  -> 01_deal_overview.csv      (country, regulatory, vendor)")
    log("        industry  -> 01_deal_sic_industry.csv  (SIC / NAICS, descriptions)")
    log("        Merge BOTH files on FULL composite key, NOT deal_num alone.")
    df_ind = read_raw("industry", "industry", FIELD_MAP_INDUSTRY, INDUSTRY_REQUIRED)
    if df_ind is None:
        flush_log()
        return
    # ★ 诊断点 1：加载后原始数据全量覆盖诊断 ★
    report_coverage(df_ind, "RAW industry (after load)", INDUSTRY_KEEP)
    report_sic_stats(df_ind, "RAW industry SIC stats")
    if os.path.exists(os.path.join(CLEANED, "01_deal_sic_industry.csv")):
        log("  ⚠ 01_deal_sic_industry.csv already exists — this run will OVERWRITE it.")

    # ── cascade SIC recovery ──
    n_acq_sic_missing_before = int(df_ind["acq_primary_sic_code"].isna().sum()) \
        if "acq_primary_sic_code" in df_ind.columns else 0
    if "acq_primary_sic_code" in df_ind.columns:
        _nunique_sic = df_ind.groupby("deal_num")["acq_primary_sic_code"].transform(
            lambda s: s.dropna().nunique())
        _unique_val = df_ind.groupby("deal_num")["acq_primary_sic_code"].transform(
            lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else np.nan)
        _m = df_ind["acq_primary_sic_code"].isna() & (_nunique_sic == 1)
        df_ind.loc[_m, "acq_primary_sic_code"] = _unique_val[_m]

        n_acq_sic_missing_after = int(df_ind["acq_primary_sic_code"].isna().sum())
        n_recovered = n_acq_sic_missing_before - n_acq_sic_missing_after
        log(f"\n  [cascade SIC] acq_primary_sic_code recovery: "
            f"filled {n_recovered:,} rows "
            f"(missing: {n_acq_sic_missing_before:,} -> {n_acq_sic_missing_after:,})")
    else:
        log("\n  [cascade SIC] acq_primary_sic_code not in columns — skipped")
    report_sic_stats(df_ind, "After cascade SIC recovery")

    # ── drop tar_name=NaN ──
    n_before = len(df_ind)
    if "tar_name" in df_ind.columns:
        df_ind = df_ind[df_ind["tar_name"].notna()].copy()
    log(f"\n  After drop tar_name=NaN: {len(df_ind):,} "
        f"(dropped {n_before - len(df_ind):,})")

    # ==================================================================

    # ── clean deal_num ──
    n_before_dealnum = len(df_ind)
    df_ind = df_ind[df_ind["deal_num"].notna()]
    df_ind["deal_num"] = df_ind["deal_num"].astype("Int64")
    log(f"  After drop deal_num=NaN: {len(df_ind):,} "
        f"(dropped {n_before_dealnum - len(df_ind):,})")

    # ── FIX: SIC codes arrive as floats (7374.0); cast to Int64 and derive sic2 ──
    for _src, _dst in [("tar_primary_sic_code", "tar_sic2"),
                       ("acq_primary_sic_code", "acq_sic2")]:
        if _src in df_ind.columns:
            _s = pd.to_numeric(df_ind[_src], errors="coerce")
            df_ind[_src] = _s.round().astype("Int64")
            df_ind[_dst] = np.floor(_s / 100.0).astype("Int64")
            df_ind.loc[_s.isna(), _dst] = pd.NA
            log(f"  [sic2] {_dst}: non‑null {int(df_ind[_dst].notna().sum()):,} "
                f"| unique {int(df_ind[_dst].nunique()):,}")

    # ── keep only INDUSTRY_KEEP columns (+ sic2) ──
    _keep_cols = [c for c in INDUSTRY_KEEP if c in df_ind.columns]
    for _extra in ["tar_sic2", "acq_sic2"]:
        if _extra in df_ind.columns and _extra not in _keep_cols:
            _keep_cols.append(_extra)
    df_ind = df_ind[_keep_cols].copy()

    # ===================== FIXED COMPOSITE DEDUP (T1_5key) =====================
    # 按 T1_5key 去重：deal_num + tar_name + acq_name + tar_bvd_id_num + acq_bvd_id_num
    # bvd 缺失时用 name 兜底（已含在键里），name 也缺失时用 "__NA__" 占位
    # 目的：保留多标的多行（tarA/acq、tarB/acq 键不同，保留为两行）；
    #       只合并五元组完全相同的纯重复行。
    n_before_dedup = len(df_ind)
    
    df_ind["_k_tar"] = (df_ind["tar_bvd_id_num"]
                        .fillna(df_ind["tar_name"]).fillna("__NA__").astype(str))
    df_ind["_k_acq"] = (df_ind["acq_bvd_id_num"]
                        .fillna(df_ind["acq_name"]).fillna("__NA__").astype(str))
    
    dedup_compound = ["deal_num", "_k_tar", "_k_acq", "tar_name", "acq_name"]
    df_ind = df_ind.drop_duplicates(subset=dedup_compound, keep="first").copy()
    df_ind = df_ind.drop(columns=["_k_tar", "_k_acq"])
    
    n_multi = int(df_ind.groupby("deal_num").size().gt(1).sum())
    log(f"  [DEDUP-T1_5key] {n_before_dedup:,} -> {len(df_ind):,} rows "
        f"(dropped {n_before_dedup - len(df_ind):,} pure duplicates)")
    log(f"  [DEDUP-T1_5key] unique deal_num: {df_ind['deal_num'].nunique():,} | "
        f"多标的 deal: {n_multi:,}（保留多行，未折叠）")
# ===========================================================================
    report_coverage(df_ind, "FINAL 01_deal_sic_industry.csv", INDUSTRY_KEEP)
    report_sic_stats(df_ind, "FINAL 01_deal_sic_industry.csv SIC stats")

    out_path_ind = os.path.join(CLEANED, "01_deal_sic_industry.csv")
    df_ind.to_csv(out_path_ind, index=False, encoding="utf‑8‑sig")
    log(f"\n  Saved -> {out_path_ind}  ({len(df_ind):,} rows x {df_ind.shape[1]} cols)")
    flush_log()
if __name__ == "__main__":
    main()
