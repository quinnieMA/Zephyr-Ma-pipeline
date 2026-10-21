# -*- coding: utf-8 -*-
r"""
01b_clean_vendor_type.py — Vendor identity (FIXED v2, 2026-10-16)
=================================================================
Standalone（不 import common_01a.py），与旧版一致。

★ 源文件（RAW，Zephyr cleaned_output 直接 append 版）
------------------------------------------------------------------
    D:\zephyr\overview\cleaned_output\MA_overview.csv
        ↑ ven_name 的真正来源（原始列名 "Vendor name"）
    D:\zephyr\info_source\cleaned_output\MA_info_source.csv   (G-5, optional)
    D:\zephyr\value\cleaned_output\MA_value.csv               (G-5, optional)

★ 产出
------------------------------------------------------------------
    01a_deal_vendor_name.csv        ← 【本脚本自己产出，见 FIX-A】
    01b_deal_vendor_type.csv
    01b_deal_vendor_type_profile.csv
    01b_vendor_unmapped.csv         (diagnostic)
    D:\MA\data\vendor_lexicon.csv          (首跑生成，可手工编辑)
    D:\MA\data\cleaned\ner_cache.csv       (NER 缓存，支持断点续跑)

==================================================================
FIX LIST
==================================================================
[FIX-A] 致命：依赖不存在的中间文件。
        旧版 main 直接读 CLEANED/01a_deal_vendor_name.csv，不存在就 skip；
        我上一版原样照抄，等于从未真正跑起来。
        该文件的 ven_name 实际来自 RAW MA_overview.csv 的 "Vendor name" 列
        （见 KEY_ALIASES["ven_name"] = [..., "vendor name", ...]）。
        修法：本脚本自己读 RAW overview -> 提取 ven_name / ven_bvd_id_num
              -> groupby.first()（跳过 NaN）-> 写出 01a_deal_vendor_name.csv
              -> 再继续分类。自给自足，同时产出兼容命名的中间文件。

[FIX-B] 致命：NER 调用顺序错误（沿用上一版修复）。
        对全部唯一 vendor 名逐个跑 CoreNLP，但 docstring 声明
        lexicon > NER。修法：两阶段——先词典，只对未命中跑 NER，
        结果缓存到 ner_cache.csv 支持断点续跑。

[FIX-C] 致命：12.6% UNMAPPED 主体是后社会主义国家国有财产/私有化机构。
        top 原文：FEDERALNOE AGENTSTVO ... IMUSHCHESTVOM（俄联邦国资局）、
        FOND DERZHAVNOHO MAYNA UKRAYINY（乌克兰国有财产基金）、
        MINISTERSTWO SKARBU PANSTWA（波兰国库部）、VALSTYBES TURTO FONDAS、
        AVAS、GOSUDARSTVENNYI KOMITET PO IMUSHCHESTVU ...
        CoreNLP 是英文模型，对俄/乌/波兰语拉丁转写完全失效。
        修法：默认词典内置多语言 government 规则。
        论文含义：这批是转型经济体私有化交易，交易额中位数与信息环境
              与 corporate 显著不同，必须单独归类，不可混入 UNMAPPED。

[FIX-D] 集体占位名未归类：INDIVIDUALS / FOUNDERS / MINORITY SHAREHOLDERS /
        INVESTORS / EMPLOYEES / PROMOTERS -> shareholders_undisclosed；
        RAO EES ROSSII OAO -> corporate（OAO 后缀）。

[FIX-E] 无 NER 缓存 -> 中断即重跑几十万次。（见 FIX-B）

[FIX-F] UNMAPPED 无兜底：新增公司后缀启发式（OAO/PAO/OOO/LLC/GmbH...）
        -> corporate；政府关键词 -> government。

[FIX-G] 输出无法审计：新增 vendor_type_source（lexicon/ner/heuristic/
        unmapped）与 matched_pattern 两列。

[FIX-H] has_lm_neg 类比问题（vendor 版）：负向词表 6.6 倍于正向，
        短名几乎必然命中负向 -> 该代理近常数。此处为 vendor 分类，
        不涉及 LM；但保留"近常数代理不可单独定 lambda 上界"的原则，
        在 UNMAPPED 诊断里打印政府关键词残留。

[FIX-I] 覆盖率未报：打印 vendor 覆盖 vs 主表 1,066,213。

[FIX-J] 保留旧版原则：CoreNLP 启动失败时 raise（不静默降级为纯规则），
        由 ALLOW_LEXICON_ONLY 显式控制（默认 False）。
        保留旧版 RAW_FILES 六项；industry / info_source 实际已提供
        （旧注释标 "NOT PROVIDED" 已过期，此处更正）。
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

WORK_LOG = os.path.join(CLEANED, "01b_vendor_type.txt")

STRICT = False
RUN_VENDOR_NER = True

# CoreNLP
STANFORD_CORENLP_PATH = r"D:\MA\NLP\stanford-corenlp-4.5.7"
JAVA_BIN_PATH = r"C:\Program Files\Java\jdk1.8.0_202\bin"
ALLOW_LEXICON_ONLY = False
NER_LOG_EVERY = 5000

# 词典 / 缓存
LEXICON_FP   = os.path.join(BASE, "data", "vendor_lexicon.csv")
NER_CACHE_FP = os.path.join(CLEANED, "ner_cache.csv")

# [FIX-A] 本脚本自己产出的 vendor name 中间文件
VENDOR_NAME_OUT = os.path.join(CLEANED, "01a_deal_vendor_name.csv")

# 主表基准（structure_date 全集）
MAIN_N_EXPECT = 1066213

# RAW 源文件（六项均已在 cleaned_output 下提供）
RAW_FILES = {
    "overview":       os.path.join(ZEPHYR, "overview",       "cleaned_output", "MA_overview.csv"),
    "multiple":       os.path.join(ZEPHYR, "multiple",       "cleaned_output", "MA_multiple.csv"),
    "structure_date": os.path.join(ZEPHYR, "structure_date", "cleaned_output", "MA_structure_date.csv"),
    "value":          os.path.join(ZEPHYR, "value",          "cleaned_output", "MA_value.csv"),
    "industry":       os.path.join(ZEPHYR, "industry",       "cleaned_output", "MA_industry.csv"),
    "info_source":    os.path.join(ZEPHYR, "info_source",    "cleaned_output", "MA_info_source.csv"),
}

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]


# ---------------------------------------------------------------------
# Logging（沿用旧版）
# ---------------------------------------------------------------------
_LOG_LINES = []

def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))

def flush_log():
    if not _LOG_LINES:
        return
    with open(WORK_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG_LINES) + "\n")
    _LOG_LINES.clear()

def module_banner(num, title):
    log("\n" + "=" * 78)
    log(f"Module {num} — {title}")
    log("=" * 78)

def report_coverage(df, label, cols=None):
    cols = list(df.columns) if cols is None else [c for c in cols if c in df.columns]
    log(f"\n  [{label}] coverage — {len(df):,} rows")
    log(f"    {'Column':<45s} {'Non-Null':>10s} {'Pct':>7s}   {'Missing':>10s}")
    log("    " + "-" * 78)
    for c in cols:
        n = int(df[c].notna().sum())
        pct = n / len(df) * 100 if len(df) else 0.0
        log(f"    {c:<45s} {n:>10,} {pct:>6.2f}%   {len(df)-n:>10,}")

def report_expected(df, label, expected):
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log(f"  [expected] {label}: {len(missing)} expected col(s) missing -> "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}")
    else:
        log(f"  [expected] {label}: all {len(expected)} present")


# ---------------------------------------------------------------------
# Key normalisation（沿用旧版）
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

INDUSTRY_KEEP = [
    "deal_num", "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    "tar_overview", "tar_trade_descr_en", "tar_busi_descr",
    "acq_overview", "ven_overview",
    "tar_primary_sic_code", "tar_sic_codes", "tar_primary_naics_code",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "acq_primary_sic_code", "acq_sic_codes",
    "ven_name", "ven_bvd_id_num", "ven_primary_sic_code",
]
SD_KEEP = ["deal_num", "deal_type", "deal_status", "deal_struct", "deal_fin",
           "deal_pay_method", "deal_pay_method_all", "deal_pay_method_n",
           "deal_struct_all", "deal_struct_n", "deal_fin_all", "deal_fin_n",
           "deal_type_all", "deal_type_n",
           "announced_d", "completed_d", "withdrawn_d",
           "announced_d_yr", "completed_d_yr", "withdrawn_d_yr", "assumed_comp_d"]
VAL_KEEP = ["deal_num", "deal_value", "deal_enterprise_value", "deal_equity_value",
            "deal_modelled_enterprise_value", "stake_acq_pct", "stake_final_pct",
            "currency"]

# vendor 提取所需（用于 report_expected）
VENDOR_KEEP = ["deal_num", "ven_name", "ven_bvd_id_num"]


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

def normalize_orbis_id(series):
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
    fold = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = fold.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None

def ensure_key_columns(df, label, wanted=None):
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

def apply_field_map(df, label, field_map):
    for new_name, old_name in field_map.items():
        if new_name in df.columns:
            if old_name in df.columns and old_name != new_name:
                df = df.drop(columns=[old_name])
            df = df.rename(columns={new_name: old_name})
            log(f"  [map:{label}] {new_name!r} -> {old_name!r}")
        elif old_name not in df.columns:
            log(f"  [map:{label}] WARNING: expected {new_name!r} "
                f"(-> {old_name!r}) NOT found")
    return df

def read_raw(file_key, label, field_map=None, expected=None):
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
    if deal_col in df.columns:
        df[deal_col] = df[deal_col].ffill()      # 多标的交易
    if field_map:
        df = apply_field_map(df, label, field_map)
    df, found, missing = ensure_key_columns(df, label)
    if "deal_num" not in df.columns:
        log(f"  x {label}: no deal_num — skipped")
        return None
    if expected:
        report_expected(df, label, expected)
    log(f"  [{label}] loaded: {len(df):,} rows | {df.shape[1]} cols | "
        f"unique deal_num: {df['deal_num'].nunique():,}")
    return df


# =====================================================================
# [FIX-A] 自己从 RAW overview 提取 vendor name
# =====================================================================
def build_vendor_name_table():
    """
    读 RAW MA_overview.csv -> 提取 ven_name / ven_bvd_id_num
    -> groupby.first()（跳过 NaN，不用 keep="first"）
    -> 写出 01a_deal_vendor_name.csv
    """
    module_banner("STEP 1", "[FIX-A] Build vendor-name table from RAW overview")

    df = read_raw("overview", "overview", expected=VENDOR_KEEP)
    if df is None:
        log("  x cannot build vendor names: overview unavailable")
        return None

    if "ven_name" not in df.columns:
        log("  x 'ven_name' (Vendor name) not found in overview after "
            "alias mapping — check KEY_ALIASES['ven_name']")
        return None

    keep = ["deal_num", "ven_name"]
    if "ven_bvd_id_num" in df.columns:
        keep.append("ven_bvd_id_num")

    sub = df[keep].copy()
    sub = sub[sub["ven_name"].notna()]
    log(f"  rows with non-null ven_name: {len(sub):,} "
        f"(of {len(df):,})")

    if sub.empty:
        log("  x no vendor names found")
        return None

    # 多标的重复 -> groupby.first()（跳过 NaN）
    n_before = len(sub)
    sub["deal_num"] = pd.to_numeric(sub["deal_num"], errors="coerce").astype("Int64")
    out = sub.groupby("deal_num", sort=False).first().reset_index()
    out["deal_num"] = pd.to_numeric(out["deal_num"], errors="coerce").astype("Int64")
    log(f"  groupby.first(): {n_before:,} rows -> {len(out):,} deals")

    out.to_csv(VENDOR_NAME_OUT, index=False, encoding="utf-8-sig")
    log(f"  [WRITE] {VENDOR_NAME_OUT}")
    log(f"    rows={len(out):,} | cols={list(out.columns)}")

    n_deal = out["deal_num"].nunique()
    log(f"  [FIX-I] coverage vs main: {n_deal:,} / {MAIN_N_EXPECT:,} "
        f"= {n_deal/MAIN_N_EXPECT*100:.2f}%")

    return out


# =====================================================================
# 词典：[FIX-C][FIX-D] 多语言政府规则 + 集体占位
# =====================================================================
DEFAULT_LEXICON_ROWS = [
    # 10 undisclosed
    (10, "undisclosed", "undisclosed"),
    (10, "undisclosed", "not disclosed"),
    (10, "undisclosed", "not available"),
    (10, "undisclosed", "unknown"),
    (10, "undisclosed", "n.a."),
    (10, "undisclosed", "n.s."),
    (10, "undisclosed", "various"),
    (10, "undisclosed", "unnamed"),
    # 20 集体占位 [FIX-D]
    (20, "shareholders_undisclosed", "minority shareholders"),
    (20, "shareholders_undisclosed", "majority shareholders"),
    (20, "shareholders_undisclosed", "shareholders"),
    (20, "shareholders_undisclosed", "shareholder"),
    (20, "shareholders_undisclosed", "individuals"),
    (20, "shareholders_undisclosed", "individual"),
    (20, "shareholders_undisclosed", "founders"),
    (20, "shareholders_undisclosed", "founder"),
    (20, "shareholders_undisclosed", "promoters"),
    (20, "shareholders_undisclosed", "investors"),
    (20, "shareholders_undisclosed", "investor group"),
    (20, "shareholders_undisclosed", "employees"),
    (20, "shareholders_undisclosed", "employee"),
    (20, "shareholders_undisclosed", "private individuals"),
    # 30 government（多语言）[FIX-C]
    (30, "government", "agentstvo"),
    (30, "government", "agentsiya"),
    (30, "government", "agencija"),
    (30, "government", "agencja"),
    (30, "government", "privatisation"),
    (30, "government", "privatization"),
    (30, "government", "privatisatsiya"),
    (30, "government", "privatizatsiya"),
    (30, "government", "privatizacija"),
    (30, "government", "privatizaciju"),
    (30, "government", "privatizaci"),
    (30, "government", "privatisatsi"),
    (30, "government", "derzhavnoho mayna"),
    (30, "government", "derzhmaino"),
    (30, "government", "gosudarstvennym imushchestvom"),
    (30, "government", "gosudarstvennyi"),
    (30, "government", "goskomimushchestvo"),
    (30, "government", "komitet po imushchestvu"),
    (30, "government", "imushchestv"),
    (30, "government", "imushchestva"),
    (30, "government", "ministerstwo skarbu panstwa"),
    (30, "government", "skarbu panstwa"),
    (30, "government", "ministerstwo"),
    (30, "government", "ministerstvo"),
    (30, "government", "ministry of"),
    (30, "government", "ministry"),
    (30, "government", "ministere"),
    (30, "government", "federalnoe agentstvo"),
    (30, "government", "federalnoe"),
    (30, "government", "federal agency"),
    (30, "government", "state property fund"),
    (30, "government", "state property"),
    (30, "government", "national property"),
    (30, "government", "valstybes turto fondas"),
    (30, "government", "turto fondas"),
    (30, "government", "valorificarea activelor"),
    (30, "government", "autoritatea pentru"),
    (30, "government", "avas"),
    (30, "government", "apaps"),
    (30, "government", "republic of"),
    (30, "government", "republica"),
    (30, "government", "respublika"),
    (30, "government", "government of"),
    (30, "government", "municipality"),
    (30, "government", "municipal"),
    (30, "government", "city of"),
    (30, "government", "treasury"),
    (30, "government", "state of"),
    (30, "government", "national agency"),
    (30, "government", "fundusz"),
    # 40 insolvency
    (40, "insolvency", "insolven"),
    (40, "insolvency", "in liquidation"),
    (40, "insolvency", "liquidat"),
    (40, "insolvency", "bankrupt"),
    (40, "insolvency", "receiver"),
    (40, "insolvency", "administrator"),
    (40, "insolvency", "trustee in bankruptcy"),
    (40, "insolvency", "konkurs"),
    (40, "insolvency", "likvidat"),
    (40, "insolvency", "upadlosc"),
    # 50 management
    (50, "management", "management buyout"),
    (50, "management", "management buy-out"),
    (50, "management", "mbo"),
    (50, "management", "management team"),
    (50, "management", "management"),
    # 60 pe_vc
    (60, "pe_vc", "private equity"),
    (60, "pe_vc", "venture capital"),
    (60, "pe_vc", "venture"),
    (60, "pe_vc", "capital partners"),
    (60, "pe_vc", "equity partners"),
    (60, "pe_vc", "partners lp"),
    (60, "pe_vc", "partners llp"),
    (60, "pe_vc", "buyout fund"),
    (60, "pe_vc", "fund lp"),
    # 70 financial_inst
    (70, "financial_inst", "bank"),
    (70, "financial_inst", "banc"),
    (70, "financial_inst", "insurance"),
    (70, "financial_inst", "assurance"),
    (70, "financial_inst", "financial"),
    (70, "financial_inst", "securities"),
    (70, "financial_inst", "investment bank"),
    (70, "financial_inst", "asset management"),
    (70, "financial_inst", "pension"),
    # 90 corporate [FIX-D] OAO/PAO
    (90, "corporate", "oao"),
    (90, "corporate", "pao"),
    (90, "corporate", "zao"),
    (90, "corporate", "ooo"),
    (90, "corporate", "llc"),
    (90, "corporate", "ltd"),
    (90, "corporate", "limited"),
    (90, "corporate", "gmbh"),
    (90, "corporate", "sarl"),
    (90, "corporate", "srl"),
    (90, "corporate", "s.p.a"),
    (90, "corporate", "b.v."),
    (90, "corporate", "n.v."),
    (90, "corporate", "plc"),
    (90, "corporate", "inc."),
    (90, "corporate", "corporation"),
    (90, "corporate", "holdings"),
    (90, "corporate", "holding"),
    (90, "corporate", "industries"),
    (90, "corporate", "international"),
    (90, "corporate", "group"),
    (90, "corporate", "company"),
]

ORDERED_CATEGORIES = ["undisclosed", "shareholders_undisclosed", "government",
                      "insolvency", "management", "pe_vc", "financial_inst",
                      "family_person", "corporate"]


def load_or_create_lexicon():
    if os.path.exists(LEXICON_FP):
        lex = pd.read_csv(LEXICON_FP, encoding="utf-8-sig")
        lex.columns = [c.strip().lower() for c in lex.columns]
        if {"priority", "category", "pattern"}.issubset(set(lex.columns)):
            lex = lex.dropna(subset=["category", "pattern"])
            lex["priority"] = pd.to_numeric(lex["priority"], errors="coerce").fillna(99)
            lex["pattern"] = lex["pattern"].astype(str).str.strip().str.lower()
            lex = lex.sort_values("priority").reset_index(drop=True)
            log(f"  Loaded lexicon: {LEXICON_FP} ({len(lex)} rules)")
            return lex
        log("  [warn] lexicon bad columns; regenerating default.")
    lex = pd.DataFrame(DEFAULT_LEXICON_ROWS,
                       columns=["priority", "category", "pattern"])
    lex["pattern"] = lex["pattern"].str.lower()
    os.makedirs(os.path.dirname(LEXICON_FP), exist_ok=True)
    lex.to_csv(LEXICON_FP, index=False, encoding="utf-8-sig")
    log(f"  Generated default lexicon: {LEXICON_FP} ({len(lex)} rules)")
    log("    -> hand-editable: priority(small=first) / category / pattern")
    return lex


def classify_by_lexicon(names, lex):
    rules = list(zip(lex["priority"], lex["category"], lex["pattern"]))
    out = {}
    for nm in names:
        if nm is None:
            continue
        for _p, cat, pat in rules:
            if pat in nm:
                out[nm] = (cat, pat)
                break
    return out


# ---------------------------------------------------------------------
# [FIX-F] heuristic 兜底
# ---------------------------------------------------------------------
CORP_SUFFIX = re.compile(
    r"\b(oao|pao|zao|ooo|llc|ltd|limited|gmbh|ag|sa|srl|sarl|spa|bv|nv|"
    r"plc|inc|corp|corporation|as|a/s|ab|oy|kft|sro|s\.r\.o\.|co|holdings|"
    r"holding|group|industries|international|company)\b", re.I)

GOV_HINT = re.compile(
    r"\b(agentstvo|agentsiya|agencija|agencja|ministerstv|ministerstwo|"
    r"ministry|federalnoe|gosudarstven|imushchestv|turto|derzhavnoho|"
    r"skarbu|valorificarea|privatisat|privatiz|mayna|fondas)\b", re.I)


def heuristic_classify(nm):
    if nm is None:
        return None, None
    if GOV_HINT.search(nm):
        return "government", "heuristic:gov-token"
    if CORP_SUFFIX.search(nm):
        return "corporate", "heuristic:corp-suffix"
    return None, None


def norm_name(s):
    if pd.isna(s):
        return None
    t = str(s).strip().lower()
    t = re.sub(r"[^\w\s\.&,/-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


# =====================================================================
# [FIX-B][FIX-E] NER（仅未命中 + 缓存）
# =====================================================================
def start_corenlp():
    os.environ["PATH"] = JAVA_BIN_PATH + os.pathsep + os.environ.get("PATH", "")
    try:
        from stanfordcorenlp import StanfordCoreNLP
        nlp = StanfordCoreNLP(STANFORD_CORENLP_PATH, lang="en")
        log(f"  CoreNLP started: {STANFORD_CORENLP_PATH}")
        return nlp
    except Exception as e:
        log("\n" + "!" * 70)
        log(f"  x CoreNLP failed: {type(e).__name__}: {e}")
        log("!" * 70)
        return None

def load_ner_cache():
    if os.path.exists(NER_CACHE_FP):
        try:
            c = pd.read_csv(NER_CACHE_FP, encoding="utf-8-sig")
            if {"name", "ner_tag"}.issubset(set(c.columns)):
                log(f"  NER cache loaded: {len(c):,} entries")
                return dict(zip(c["name"].astype(str), c["ner_tag"].astype(str)))
        except Exception as e:
            log(f"  [warn] cache unreadable ({e}); fresh start")
    return {}

def save_ner_cache(cache):
    pd.DataFrame({"name": list(cache.keys()),
                  "ner_tag": list(cache.values())}).to_csv(
        NER_CACHE_FP, index=False, encoding="utf-8-sig")
    log(f"  NER cache saved ({len(cache):,})")

def run_ner_stage(unresolved, cache):
    todo = [n for n in unresolved if n not in cache]
    log(f"\n  [NER] unresolved : {len(unresolved):,}")
    log(f"  [NER] cached     : {len(unresolved)-len(todo):,}")
    log(f"  [NER] to process : {len(todo):,}")
    if not todo:
        log("  [NER] nothing to process (fully cached)")
        return cache

    nlp = start_corenlp()
    if nlp is None:
        if not ALLOW_LEXICON_ONLY:
            log("  [ABORT] NER unavailable and ALLOW_LEXICON_ONLY=False")
            flush_log()
            raise RuntimeError("CoreNLP unavailable")
        log("  [warn] lexicon-only fallback")
        return cache

    t0 = time.time()
    try:
        for i, nm in enumerate(todo, 1):
            try:
                tagged = nlp.ner(nm)
            except Exception:
                cache[nm] = "ERROR"
                continue
            tags = [t for _, t in tagged] if tagged else []
            if "PERSON" in tags:
                cache[nm] = "PERSON"
            elif "ORGANIZATION" in tags:
                cache[nm] = "ORGANIZATION"
            elif "LOCATION" in tags or "COUNTRY" in tags:
                cache[nm] = "LOCATION"
            else:
                cache[nm] = "OTHER"
            if i % NER_LOG_EVERY == 0:
                el = time.time() - t0
                log(f"    NER {i:,}/{len(todo):,} ({i/el:.1f}/s, "
                    f"eta {(len(todo)-i)/max(i/el,1e-9)/60:.1f} min)")
                save_ner_cache(cache)
    finally:
        try:
            nlp.close()
            log("  CoreNLP closed")
        except Exception:
            pass
        save_ner_cache(cache)
    return cache


# =====================================================================
# Main
# =====================================================================
def main():
    module_banner("M9", "Vendor identity (self-built vendor names + "
                        "lexicon -> NER -> heuristic) [FIXED v2]")
    if not RUN_VENDOR_NER:
        log("  skipped — RUN_VENDOR_NER=False")
        flush_log()
        return

    # ---- STEP 1 [FIX-A] 自己造 vendor name 表 ----
    df_vn = build_vendor_name_table()
    if df_vn is None:
        log("  x abort: vendor-name table unavailable")
        flush_log()
        return

    df = df_vn.copy()
    df["_nm"] = df["ven_name"].apply(norm_name)
    uniq = sorted(set(df["_nm"].dropna()))
    log(f"\n  Unique vendor names: {len(uniq):,}")

    resolved, source, pattern = {}, {}, {}

    # ---- STEP 2 词典 ----
    module_banner("STEP 2", "[FIX-B] Lexicon stage")
    lex = load_or_create_lexicon()
    lex_map = classify_by_lexicon(uniq, lex)
    for nm, (cat, pat) in lex_map.items():
        resolved[nm] = cat; source[nm] = "lexicon"; pattern[nm] = pat
    log(f"  lexicon resolved: {len(lex_map):,} / {len(uniq):,} "
        f"({len(lex_map)/max(len(uniq),1)*100:.2f}%)")
    unresolved = [n for n in uniq if n not in resolved]
    log(f"  remaining       : {len(unresolved):,}")

    # ---- STEP 3 NER（仅未命中）----
    module_banner("STEP 3", "[FIX-B][FIX-E] NER stage (unresolved only)")
    cache = {}
    if RUN_VENDOR_NER and unresolved:
        cache = run_ner_stage(unresolved, load_ner_cache())
        n_ner = 0
        for nm in unresolved:
            tag = cache.get(nm)
            if tag == "PERSON":
                resolved[nm] = "family_person"; source[nm] = "ner"
                pattern[nm] = "PERSON"; n_ner += 1
            elif tag == "ORGANIZATION":
                resolved[nm] = "corporate"; source[nm] = "ner"
                pattern[nm] = "ORGANIZATION"; n_ner += 1
        log(f"  NER resolved: {n_ner:,}")

    # ---- STEP 4 heuristic ----
    module_banner("STEP 4", "[FIX-F] Heuristic fallback")
    still = [n for n in uniq if n not in resolved]
    n_h = 0
    for nm in still:
        cat, pat = heuristic_classify(nm)
        if cat:
            resolved[nm] = cat; source[nm] = "heuristic"; pattern[nm] = pat
            n_h += 1
    log(f"  heuristic resolved: {n_h:,}")

    # ---- STEP 5 组装 [FIX-G] ----
    module_banner("STEP 5", "Assemble output")
    df["vendor_type"] = df["_nm"].map(resolved).fillna("UNMAPPED")
    df["vendor_type_source"] = df["_nm"].map(source).fillna("unmapped")
    df["matched_pattern"] = df["_nm"].map(pattern).fillna("")

    out = df.drop(columns=["_nm"]).copy()
    log("\n  Vendor-type distribution")
    for k, v in out["vendor_type"].value_counts().items():
        log(f"    {k:<28s} {v:>10,} ({v/len(out)*100:5.1f}%)")
    log("\n  Classification source [FIX-G]")
    for k, v in out["vendor_type_source"].value_counts().items():
        log(f"    {k:<28s} {v:>10,} ({v/len(out)*100:5.1f}%)")

    out_fp = os.path.join(CLEANED, "01b_deal_vendor_type.csv")
    out.to_csv(out_fp, index=False, encoding="utf-8-sig")
    log(f"\n  Saved -> {out_fp}")

    # ---- STEP 6 UNMAPPED 诊断 ----
    n_unm = int((out["vendor_type"] == "UNMAPPED").sum())
    log(f"\n  UNMAPPED: {n_unm:,} ({n_unm/len(out)*100:.2f}%)")
    if n_unm:
        unm = out[out["vendor_type"] == "UNMAPPED"]
        unm.to_csv(os.path.join(CLEANED, "01b_vendor_unmapped.csv"),
                   index=False, encoding="utf-8-sig")
        top = unm["ven_name"].value_counts().head(30)
        log("  top-30 unmapped:")
        for k, v in top.items():
            hint = " [GOV?]" if GOV_HINT.search(str(k)) else ""
            log(f"    {str(k)[:50]:<52s} {v:>8,}{hint}")
        n_gov = unm["ven_name"].astype(str).apply(
            lambda s: bool(GOV_HINT.search(s))).sum()
        log(f"\n  [FIX-H] 含政府关键词仍 UNMAPPED: {n_gov:,}")
        if n_gov:
            log("    -> 加入 vendor_lexicon.csv (category=government) 后重跑")

    # ---- STEP 7 G-5 profile ----
    module_banner("G-5", "Vendor type x information environment")
    prof = out.copy()
    isrc_fp = os.path.join(CLEANED, "01b_deal_info_source_count.csv")
    if os.path.exists(isrc_fp):
        isrc = pd.read_csv(isrc_fp, encoding="utf-8-sig", low_memory=False)
        dn = find_col(isrc, ["deal_num"])
        if dn:
            isrc = isrc.rename(columns={dn: "deal_num"})
            isrc["deal_num"] = pd.to_numeric(isrc["deal_num"], errors="coerce").astype("Int64")
            cols = [c for c in ["num_Advisor_Submission", "num_Stock_Exchange",
                                "num_Company_Press_Release"] if c in isrc.columns]
            if cols:
                prof = prof.merge(isrc[["deal_num"] + cols], on="deal_num", how="left")
                for c in cols:
                    prof[c] = prof[c].fillna(0)
                log(f"  merged {os.path.basename(isrc_fp)}")

    val_fp = os.path.join(CLEANED, "01_deal_value.csv")
    if not os.path.exists(val_fp):
        val_fp = os.path.join(CLEANED, "01a_deal_value.csv")
    if os.path.exists(val_fp):
        val = pd.read_csv(val_fp, encoding="utf-8-sig", low_memory=False)
        dn = find_col(val, ["deal_num"]); vc = find_col(val, ["deal_value", "deal_val_usd"])
        if dn and vc:
            val = val.rename(columns={dn: "deal_num", vc: "deal_value"})
            val["deal_num"] = pd.to_numeric(val["deal_num"], errors="coerce").astype("Int64")
            val["deal_value"] = pd.to_numeric(val["deal_value"], errors="coerce")
            prof = prof.merge(val[["deal_num", "deal_value"]], on="deal_num", how="left")
            log(f"  merged {os.path.basename(val_fp)}")

    agg = {}
    if "num_Advisor_Submission" in prof.columns:
        agg["share_advsub"] = prof.groupby("vendor_type")["num_Advisor_Submission"].apply(
            lambda s: (s > 0).mean())
    if "num_Stock_Exchange" in prof.columns:
        agg["share_exchange"] = prof.groupby("vendor_type")["num_Stock_Exchange"].apply(
            lambda s: (s > 0).mean())
    if "num_Company_Press_Release" in prof.columns:
        agg["share_press"] = prof.groupby("vendor_type")["num_Company_Press_Release"].apply(
            lambda s: (s > 0).mean())
    if "deal_value" in prof.columns:
        agg["median_deal_value"] = prof.groupby("vendor_type")["deal_value"].median()

    if agg:
        profile = pd.DataFrame(agg)
        profile.insert(0, "N", out["vendor_type"].value_counts())
        profile = profile.sort_values("N", ascending=False)
        log("\n  share_* = 组内比例；median_deal_value = 中位数")
        log(f"\n  {'vendor_type':<26s} {'N':>9s} " +
            " ".join(f"{c:>18s}" for c in profile.columns[1:]))
        log("  " + "-" * 92)
        for idx, row in profile.iterrows():
            cells = " ".join(
                f"{row[c]:>18.3f}" if isinstance(row[c], (float, np.floating))
                else f"{row[c]:>18,.0f}" for c in profile.columns[1:])
            log(f"  {str(idx):<26s} {int(row['N']):>9,} {cells}")
        pf_fp = os.path.join(CLEANED, "01b_deal_vendor_type_profile.csv")
        profile.to_csv(pf_fp, encoding="utf-8-sig")
        log(f"\n  Saved -> {pf_fp}")

    report_coverage(out, "01b_deal_vendor_type.csv")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise