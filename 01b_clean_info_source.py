# -*- coding: utf-8 -*-
r"""
01b_clean_info_source.py — Info-source export (FIXED)
==================================================================
Input  : D:\zephyr\info_source\cleaned_output\MA_info_source.csv
Output : D:\MA\data\cleaned\01b_deal_info_source_count.csv

==================================================================
FIX LIST（相对原版）
==================================================================
[FIX-1] 横幅仍写 "[NOT PROVIDED — placeholder]，但文件已提供并正常
        加载 191,841 行。横幅改为动态判定，不再误导。

[FIX-2] 191,841 行 vs 176,469 unique deal_num（15,372 行重复）。
        原版直接 groupby-sum，不区分"真多源"与"完全重复行"。
        修法：先 drop_duplicates() 全行去重，再聚合；并打印
              rows / after-exact-dedup / unique deals / multi-source deals
        四项，让伪影无所遁形。
        注：这与 01c 评论模块 (191,831 vs 176,459) 是同一批数据的
            两个视角，两边必须一起查。

[FIX-3] 致命：7 个 num_* 列全部 100% 非空 0 缺失是假象。
        本文件只覆盖 176,469 / 1,066,213 = 16.5% 的交易；
        fillna(0) 之后"不在导出中"与"在导出中但该类为 0"无法区分。
        这直接影响论文的稳健性检验（限定无顾问投稿记录的样本）。
        修法：不再盲目 fillna；输出 has_info_source=1 常量列，
              02 左连接后 fillna(0) 即得"是否在导出中"的标志。
              num_* 的 NaN 在连接后表示"不在导出中"，0 表示
              "在导出中但无此类来源"，二者可区分。

[FIX-4] 类别提取 "Category (detail)" 遇到无括号会静默丢失。
        修法：加兜底（无括号取整串），并做 strip / lower /
              连续空白压缩后匹配。

[FIX-5] 未见类别静默进 num_source_other。
        修法：打印完整 source_category 分布（映射前 top-30 +
              映射后各桶计数），未知类别一目了然。

[FIX-6] 删除 RUN_VENDOR_NER（docstring 自称 superseded，main 中未使用）。

[FIX-7] RAW_FILES 只保留 info_source（其余 5 项为死配置，且 industry
        被标为"未提供"但实际已提供）。

[FIX-8] ensure_key_columns 只查 deal_num（原版强查 8 个 key，
        7 个 missing，纯噪音）。

[FIX-9] 新增覆盖率诊断：16.5% vs 主表 1,066,213。

[FIX-10] 新增 num_source_total / num_distinct_categories /
         has_advisor_submission（论文稳健性直接可用）。

[FIX-11] docstring 标题 01a -> 01b（与文件名一致）。

[FIX-12] 类别列名用模糊查找（兼容 BvD 小写名与 cleaned 规范名）。
"""

import os
import re
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

WORK_LOG = os.path.join(CLEANED, "01b_info_source.txt")
STRICT = False

# [FIX-7] 只保留本模块真正需要的输入
RAW_FILES = {
    "info_source": os.path.join(ZEPHYR, "info_source", "cleaned_output",
                                "MA_info_source.csv"),
}

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

# 主表（structure_date）行数基准
MAIN_N_EXPECT = 1066213

# [FIX-5] 七个已知类别桶（归一后小写比对）
CANON_BUCKETS = {
    "num_Stock_Exchange":        ["stock exchange", "stock exchanges",
                                  "exchange"],
    "num_Website":               ["website", "web site", "websites"],
    "num_Company_Press_Release": ["company press release",
                                  "company press releases",
                                  "press release", "press releases"],
    "num_Electronic_Publication":["electronic publication",
                                  "electronic publications",
                                  "electronic publication (internet)"],
    "num_Advisor_Submission":    ["advisor submission", "adviser submission",
                                  "advisor submissions",
                                  "adviser submissions",
                                  "advisor submission (m&a)"],
    "num_Miscellaneous":         ["miscellaneous", "misc"],
}
BUCKET_COLS = list(CANON_BUCKETS.keys()) + ["num_source_other"]

# [FIX-12] 类别列 / 文档列的模糊查找候选
CAT_CANDIDATES = ["source_category", "categoryofsource", "category_of_source",
                  "category", "source type", "sourcetype", "type of source"]
DOC_CANDIDATES = ["source_document", "sourcedocumentation",
                  "source_document", "document", "sourcedoc"]

# BvD 小写名 -> 规范名（cleaned 导出已是规范名时为空操作）
LEGACY_RENAME = {
    "dealnumber": "deal_num", "categoryofsource": "source_category",
    "sourcedocumentation": "source_document", "targetname": "tar_name",
    "targetbvdidnumber": "tar_bvd_id_num",
    "targetorbisidnumber": "tar_orbis_id_num",
    "acquirorname": "acq_name", "acquirorbvdidnumber": "acq_bvd_id_num",
    "acquirororbisidnumber": "acq_orbis_id_num",
    "vendorname": "ven_name", "vendorbvdidnumber": "ven_bvd_id_num",
    "index": "src_index",
}

KEY_ALIASES = {"deal_num": ["deal_num", "deal number", "dealnumber", "deal id"]}


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
              f"# 01b run — {_SCRIPT_NAME}",
              f"# started : {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"# cwd     : {os.getcwd()}",
              f"# pandas  : {pd.__version__}", "#" * 78]
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
        return
    log(f"\n  [{label}] DESCRIPTIVE STATS — numeric ({len(num_cols)})")
    log(f"    {'column':<30s} {'min':>10s} {'median':>10s} {'mean':>10s} "
        f"{'max':>10s} {'sum':>12s}")
    log("    " + "-" * 84)
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            log(f"    {c:<30s} {'ALL MISSING':>10s}")
            continue
        log(f"    {c:<30s} {s.min():>10.0f} {s.median():>10.0f} "
            f"{s.mean():>10.2f} {s.max():>10.0f} {s.sum():>12,.0f}")


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


def find_col(df, candidates):
    fold = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = fold.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    return None


def norm_cat(text):
    """[FIX-4] 归一化类别串：strip / lower / 压缩空白。"""
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def extract_main_cat(text):
    """[FIX-4] 'Category (detail)' -> 'Category'；无括号则取整串。"""
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    m = re.match(r"(.*?)\s*\(", s)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return s


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
    # BvD 小写名 -> 规范名
    ren = {k: v for k, v in LEGACY_RENAME.items() if k in df.columns}
    if ren:
        df = df.rename(columns=ren)
        log(f"  [legacy rename] {len(ren)} column(s): {list(ren.items())[:5]}")
    # deal_num
    dn = find_col(df, KEY_ALIASES["deal_num"])
    if dn is None:
        log(f"  x {label}: no deal_num column found")
        return None
    if dn != "deal_num":
        df = df.rename(columns={dn: "deal_num"})
    df["deal_num"] = df["deal_num"].ffill()
    df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")
    log(f"  [{label}] loaded: {len(df):,} rows | {df.shape[1]} cols | "
        f"unique deal_num: {df['deal_num'].nunique():,}")
    return df


# =====================================================================
# Main
# =====================================================================
def main():
    fp = RAW_FILES["info_source"]
    provided = os.path.exists(fp)
    status = "" if provided else " [NOT PROVIDED — placeholder]"
    # [FIX-1] 横幅动态判定，不再写死 placeholder
    module_banner("M2.6+M8",
                  "Info-source export -> 01b_deal_info_source_count.csv" + status)

    if not provided:
        log(f"  [MISSING INPUT] {fp}")
        log("  Module skipped (no file).")
        flush_log()
        return

    df_src = read_raw("info_source", "info_source")
    if df_src is None:
        flush_log()
        return

    # 去掉垃圾列
    junk = [c for c in df_src.columns
            if c.lower().startswith("unnamed") or c.lower() in ("var1", "index")]
    if junk:
        df_src = df_src.drop(columns=junk)
        log(f"  dropped junk columns: {junk}")

    df_src = df_src[df_src["deal_num"].notna()].copy()

    # -----------------------------------------------------------------
    # [FIX-12] 定位类别列 / 文档列
    # -----------------------------------------------------------------
    cat_col = find_col(df_src, CAT_CANDIDATES)
    doc_col = find_col(df_src, DOC_CANDIDATES)
    log(f"\n  [cols] category column = {cat_col!r}")
    log(f"  [cols] document column = {doc_col!r}")
    if cat_col is None:
        log("  x no source-category column found. Module aborted.")
        log(f"    候选名: {CAT_CANDIDATES}")
        flush_log()
        return

    keep_cols = ["deal_num", cat_col] + ([doc_col] if doc_col else [])
    df_src = df_src[keep_cols].copy()

    # -----------------------------------------------------------------
    # [FIX-2] 重复结构诊断
    # -----------------------------------------------------------------
    n_raw = len(df_src)
    n_deals = df_src["deal_num"].nunique()

    df_dedup = df_src.drop_duplicates()
    n_dedup = len(df_dedup)

    multi = (df_dedup.groupby("deal_num").size() > 1).sum()

    log("\n" + "=" * 78)
    log("  [FIX-2] DUPLICATE STRUCTURE DIAGNOSTIC")
    log("=" * 78)
    log(f"    rows (raw)                       : {n_raw:,}")
    log(f"    rows after EXACT dedup           : {n_dedup:,}")
    log(f"    exact duplicate rows removed     : {n_raw - n_dedup:,}")
    log(f"    unique deal_num                  : {n_deals:,}")
    log(f"    deals with >1 DISTINCT record    : {multi:,}")
    log(f"    -> {'多源为主（正常）' if multi > (n_raw - n_dedup) else '完全重复行为主（数据伪影）'}")

    # -----------------------------------------------------------------
    # [FIX-4/5] 类别提取 + 分布
    # -----------------------------------------------------------------
    df_dedup["_cat_raw"] = df_dedup[cat_col].apply(extract_main_cat)
    df_dedup["_cat_norm"] = df_dedup["_cat_raw"].apply(norm_cat)

    log("\n  [FIX-5] source_category 分布（提取主类别后，top-30）")
    vc = df_dedup["_cat_raw"].value_counts(dropna=False)
    for k, v in vc.head(30).items():
        log(f"    {str(k)[:52]:<54s} {v:>10,}  {v/len(df_dedup)*100:>6.2f}%")
    if len(vc) > 30:
        log(f"    ... 共 {len(vc):,} 个不同类别")

    # ---- 映射到 7 个桶 ----
    def to_bucket(norm_val):
        if pd.isna(norm_val):
            return None
        for col, pats in CANON_BUCKETS.items():
            for p in pats:
                if norm_val == p or p in norm_val:
                    return col
        return "num_source_other"

    df_dedup["_bucket"] = df_dedup["_cat_norm"].apply(to_bucket)

    log("\n  [FIX-5] 映射后各桶计数")
    bc = df_dedup["_bucket"].value_counts()
    for col in BUCKET_COLS:
        v = int(bc.get(col, 0))
        log(f"    {col:<32s} {v:>10,}  {v/len(df_dedup)*100:>6.2f}%")

    # ---- 未识别类别清单 ----
    unk = df_dedup.loc[df_dedup["_bucket"] == "num_source_other", "_cat_raw"]
    if len(unk):
        log(f"\n    [warn] {len(unk):,} 条落入 num_source_other，其类别：")
        for k, v in unk.value_counts().head(15).items():
            log(f"      {str(k)[:50]:<52s} {v:>10,}")

    # -----------------------------------------------------------------
    # 聚合：每个 deal 每桶计数
    # -----------------------------------------------------------------
    df_dedup["_one"] = 1
    piv = (df_dedup.pivot_table(index="deal_num", columns="_bucket",
                                values="_one", aggfunc="sum", fill_value=0)
           .reset_index())
    for col in BUCKET_COLS:
        if col not in piv.columns:
            piv[col] = 0
        piv[col] = piv[col].astype(int)

    # [FIX-10] 衍生列
    piv["num_source_total"] = piv[BUCKET_COLS].sum(axis=1)
    piv["num_distinct_categories"] = (piv[BUCKET_COLS] > 0).sum(axis=1)
    piv["has_advisor_submission"] = (piv["num_Advisor_Submission"] > 0).astype(int)

    # [FIX-3] 关键：不再盲目 fillna
    #   本输出只含"在导出中"的 deal；02 左连接后：
    #     has_info_source.fillna(0) -> 1=在导出中, 0=不在导出中
    #     num_*.fillna(0)           -> 0=无此类来源（可含"不在导出中"）
    #   二者结合即可区分两种零。
    piv["has_info_source"] = 1

    piv["deal_num"] = pd.to_numeric(piv["deal_num"], errors="coerce").astype("Int64")

    log_checkpoint("after aggregation")

    # -----------------------------------------------------------------
    # [FIX-9] 覆盖率
    # -----------------------------------------------------------------
    log("\n" + "=" * 78)
    log("  [FIX-9] COVERAGE vs MAIN TABLE")
    log("=" * 78)
    cov = len(piv) / MAIN_N_EXPECT * 100
    log(f"    deals in info_source export : {len(piv):,}")
    log(f"    main table (structure_date): {MAIN_N_EXPECT:,}")
    log(f"    coverage                    : {cov:.2f}%")
    log(f"    NOT covered                 : {MAIN_N_EXPECT - len(piv):,} "
        f"({100 - cov:.2f}%)")
    log("")
    log("    [FIX-3] 02 左连接后请务必：")
    log("      has_info_source = has_info_source.fillna(0)")
    log("      -> 0 表示该 deal 不在信息源导出中（83.5% 的交易）")
    log("      num_* 的 0 表示无此类来源，与 has_info_source=0 含义不同")

    # -----------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------
    out_cols = (["deal_num"] + BUCKET_COLS +
                ["num_source_total", "num_distinct_categories",
                 "has_advisor_submission", "has_info_source"])
    piv = piv[out_cols]

    report_coverage(piv, "01b_deal_info_source_count.csv")
    report_descriptives(piv, "01b_deal_info_source_count.csv",
                        cols=BUCKET_COLS + ["num_source_total",
                                            "num_distinct_categories"])

    # -----------------------------------------------------------------
    # [FIX-10] 论文稳健性直接可用的数
    # -----------------------------------------------------------------
    log("\n" + "=" * 78)
    log("  [FIX-10] 论文稳健性：顾问投稿记录")
    log("=" * 78)
    n_adv = int(piv["has_advisor_submission"].sum())
    log(f"    has_advisor_submission = 1 : {n_adv:,} "
        f"({n_adv/len(piv)*100:.2f}% of covered deals)")
    log(f"    has_advisor_submission = 0 : {len(piv)-n_adv:,} "
        f"({(len(piv)-n_adv)/len(piv)*100:.2f}%)")
    log("")
    log("    注意：这里的 0 是在【已覆盖子样本】内的 0。")
    log("    若要复现论文'限定无顾问投稿记录'的稳健性检验，")
    log("    分母应为 has_info_source==1 的子样本，而非全样本。")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    out_path = os.path.join(CLEANED, "01b_deal_info_source_count.csv")
    piv.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  Module 8: {len(piv):,} deals written -> {out_path}")
    log(f"  columns ({len(out_cols)}): {out_cols}")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise