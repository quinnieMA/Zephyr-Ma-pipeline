# -*- coding: utf-8 -*-
r"""
04_merge_firm_to_deal.py — v11 (2026-10-XX)
================================================================================
把 03 的 firm 表并入 02 的 deal master → 04_deal_firm_full.csv

原则：只做清晰的左连接 + 按论文 COLS_TO_SAVE 对齐字段名。
      不改任何数据，不用补丁硬拗，不静默丢行。

================================================================================
v4 -> v5：白名单改为【COLS_TO_SAVE 驱动】+ 字段名对齐
================================================================================
[FIX-7] 白名单来自论文的 COLS_TO_SAVE，逐条核对 02 / 03 实际字段。

[FIX-8] 关键发现：03 的财务字段是缩写命名，与论文名完全不同。
        若在 04 不 rename，后续 05/07/08 会全部 "NOT PRESENT"。
        例：
            论文 pre_deal_tar_ebitda_last_avail_yr  <- 03  pre_tar_ebitda_ly
            论文 post_deal_tar_pbt_1st_avail_yr     <- 03  post_tar_pft_before_tax_1st_avail_yr
            论文 pre_deal_acq_eq_last_avail_yr      <- 03  pre_acq_eq_ly
            论文 tar_incorp_d_year                  <- 03  tar_incorp_yr
        02 侧两处：
            论文 total_info_source                  <- 02  num_source_total
            论文 deal_value                         <- 02  deal_val_usd
        本脚本一次性 rename 成论文名，下游直接用论文名即可。

[FIX-9] 三类"找不到"的变量分组报告（不是 bug，是来源/阶段不同）：
        A) 需外部模块并入：advisor 28 列 / post 倍数 / vendor 财务 / 宏观 GDP
        B) 需派生（07/08）：tar_sic3 / ln_* / tar_roa / target_age / ...
        C) 命名不同已映射：见 [FIX-8]

[FIX-10] pre_np_mul_ly 是死列（1,066,572 全 0）-> 不进白名单
         *_implausible 标记列默认不进白名单（需要时开关打开）

[FIX-11] tar_listed / acq_listed / ven_listed / *_ipo_d 在 03 侧，不在 02 侧。

[FIX-12] 白名单列在文件里不存在时 WARN 并分组报告，不静默忽略。

================================================================================
v5 -> v11：适配 02/03 职责重划（实体键 & industry/overview 挪到 03）
================================================================================
[V11-1] 注释 entity_consistency_audit() 调用（02 不再含实体键）。
[V11-2] 新增 _rebalance_keep()：MASTER_KEEP 里在 02 不存在、但在 03 存在的列，
        自动挪到 FIRM_KEEP，无需手工枚举。

================================================================================
设计（不变）
================================================================================
[1] 主键 = deal_num 单键。两表均一行一 deal (02: 1,066,213 / 03: 999,614)。
[2] 不用 __NA__ 填充，不做强制 dedup；行数必须不变，否则 assert 报错。
    merge 用 validate="one_to_one" 硬校验。
[3] 内存：白名单 usecols + float64->float32 + 字符串转 category + merge 后 gc。
[4] 冲突列先报告再处理。

输出
    merged/04_deal_firm_full.csv
    merged/04_deal_firm_full.txt
"""

import os
import gc
import time
import traceback
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# Config
# =====================================================================
BASE   = r"D:\MA"
MERGED = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)

PATH_MASTER = os.path.join(MERGED, "02_deal_master.csv")
PATH_FIRM   = os.path.join(MERGED, "03_firm_deal_level.csv")
PATH_OUT    = os.path.join(MERGED, "04_deal_firm_full.csv")
PATH_LOG    = os.path.join(MERGED, "04_deal_firm_full.txt")

DEAL = "deal_num"

# =====================================================================
# [FIX-8] 02 侧字段名 rename（论文名 <- 02 名）
# =====================================================================
MASTER_RENAME = {
    "num_source_total": "total_info_source",
    "deal_val_usd":     "deal_value",
}

# =====================================================================
# [FIX-8] 03 侧字段名 rename（论文名 <- 03 名）
# =====================================================================
FIRM_RENAME = {
    # ---- target pre-deal ----
    "pre_tar_rev_ly":    "pre_deal_tar_rev_rev_last_avail_yr",
    "pre_tar_ebitda_ly": "pre_deal_tar_ebitda_last_avail_yr",
    "pre_tar_ebit_ly":   "pre_deal_tar_ebit_last_avail_yr",
    "pre_tar_pbt_ly":    "pre_deal_tar_pbt_last_avail_yr",
    "pre_tar_pat_ly":    "pre_deal_tar_pat_last_avail_yr",
    "pre_tar_np_ly":     "pre_deal_tar_np_last_avail_yr",
    "pre_tar_ta_ly":     "pre_deal_tar_ta_last_avail_yr",
    "pre_tar_na_ly":     "pre_deal_tar_na_last_avail_yr",
    "pre_tar_eq_ly":     "pre_deal_tar_eq_last_avail_yr",
    "pre_tar_cl_ly":     "pre_deal_tar_current_liabilities_last_avail_yr",
    "pre_tar_cap_ly":    "pre_deal_tar_cap",
    # ---- acquirer pre-deal ----
    "pre_acq_rev_ly":    "pre_deal_acq_rev_rev_last_avail_yr",
    "pre_acq_ebitda_ly": "pre_deal_acq_ebitda_last_avail_yr",
    "pre_acq_pat_ly":    "pre_deal_acq_pat_last_avail_yr",
    "pre_acq_ta_ly":     "pre_deal_acq_ta_last_avail_yr",
    "pre_acq_eq_ly":     "pre_deal_acq_eq_last_avail_yr",
    "pre_acq_ebit_ly":   "pre_deal_acq_ebit_last_avail_yr",
    "pre_acq_pbt_ly":    "pre_deal_acq_pbt_last_avail_yr",
    "pre_acq_np_ly":     "pre_deal_acq_np_last_avail_yr",
    "pre_acq_na_ly":     "pre_deal_acq_na_last_avail_yr",
    "pre_acq_cl_ly":     "pre_deal_acq_current_liabilities_last_avail_yr",
    "pre_acq_cap_ly":    "pre_deal_acq_cap",
    # ---- target post-deal ----
    "post_tar_op_rev_1st_avail_yr":        "post_deal_tar_rev_rev_1st_avail_yr",
    "post_tar_ebitda_1st_avail_yr":        "post_deal_tar_ebitda_1st_avail_yr",
    "post_tar_ebit_1st_avail_yr":          "post_deal_tar_ebit_1st_avail_yr",
    "post_tar_pft_before_tax_1st_avail_yr":"post_deal_tar_pbt_1st_avail_yr",
    "post_tar_pft_after_tax_1st_avail_yr": "post_deal_tar_pat_1st_avail_yr",
    "post_tar_net_pft_1st_avail_yr":       "post_deal_tar_np_1st_avail_yr",
    "post_tar_tot_assets_1st_avail_yr":    "post_deal_tar_ta_1st_avail_yr",
    "post_tar_net_assets_1st_avail_yr":    "post_deal_tar_na_1st_avail_yr",
    "post_tar_cur_liab_1st_avail_yr":      "post_deal_tar_current_liabilities_1st_avail_yr",
    "post_tar_sh_funds_1st_avail_yr":      "post_deal_tar_shareholder_funds_1st_avail_yr",
    "post_tar_mkt_cap_1st_avail_yr":       "post_deal_tar_mkt_cap_1st_avail_yr",
    # ---- acquirer post-deal ----
    "post_acq_op_rev_1st_avail_yr":         "post_deal_acq_rev_rev_1st_avail_yr",
    "post_acq_ebitda_1st_avail_yr":         "post_deal_acq_ebitda_1st_avail_yr",
    "post_acq_ebit_1st_avail_yr":           "post_deal_acq_ebit_1st_avail_yr",
    "post_acq_pft_before_tax_1st_avail_yr": "post_deal_acq_pbt_1st_avail_yr",
    "post_acq_pft_after_tax_1st_avail_yr":  "post_deal_acq_pat_1st_avail_yr",
    "post_acq_net_pft_1st_avail_yr":        "post_deal_acq_np_1st_avail_yr",
    "post_acq_tot_assets_1st_avail_yr":     "post_deal_acq_ta_1st_avail_yr",
    "post_acq_net_assets_1st_avail_yr":     "post_deal_acq_na_1st_avail_yr",
    "post_acq_sh_funds_1st_avail_yr":       "post_deal_acq_shareholder_funds_1st_avail_yr",
    "post_acq_mkt_cap_1st_avail_yr":        "post_deal_acq_mkt_cap_1st_avail_yr",
    # ---- firm age / listed ----
    "tar_incorp_yr": "tar_incorp_d_year",
    "acq_incorp_yr": "acq_incorp_d_year",
    # tar_listed / acq_listed / ven_listed 同名，无需 rename
}

# =====================================================================
# [FIX-7] 02 侧白名单（论文 COLS_TO_SAVE 中在 02 能找到的）
# =====================================================================
MASTER_KEEP = [
    # ---- Identifiers ----
    DEAL, "tar_sic2", "legal_origin_tar", "legal_origin_acq",
    # ---- Regulatory (Module E) ----
    "reg_body_count", "reg_country_count", "reg_antitrust", "reg_securities",
    "reg_state_assets", "reg_foreign_invest", "reg_financial",
    "reg_defense_tech", "reg_cross_national", "reg_common_law",
    "reg_civil_law", "reg_mixed_legal",
    "regulatory_bodies", "regulatory_countries",
    # ---- Comment text features (Module F) ----
    #"total_timeline_days", "has_rumour", "has_target_reject",
    #"has_phase2_investigation", "has_goshop", "num_unique_reg",
    #"comment_char_length", "comment_wordcount", "sentence_event_count",
    #"lm_pos_count", "lm_neg_count", "lm_uncertain_count",
    #"lm_litigious_count", "lm_strongmodal_count", "lm_weakmodal_count",
    #"lm_constrain_count",
    #"lm_pos_density", "lm_neg_density", "lm_uncertain_density",
    #"lm_litigious_density", "lm_strongmodal_density", "lm_weakmodal_density",
    #"lm_constrain_density",
    # ---- Info-source (Module K) ----
    "num_Stock_Exchange", "num_Website", "num_Company_Press_Release",
    "num_Electronic_Publication", "num_Advisor_Submission",
    "num_Miscellaneous", "num_source_other", "num_source_total",
    # ---- Dependent: raw pre multiples ----
    "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly",
    #恢复 8 个 multiple 倍数 (2026-10-20)
    "pre_pbt_mul_ly", "pre_pat_mul_ly", "pre_np_mul_ly",
    "pre_ta_mul_ly", "pre_na_mul_ly", "pre_cl_mul_ly",
    "pre_eq_mul_ly", "pre_cap_mul_ly",
    # ---- DaysToCompletion / ln_days ----
    "DaysToCompletion", "ln_days",
# ---- Rumor (01a) ----
    "rumor_jfe", "has_rumor_status", "rumor_lead_days", "rumor_lead_yr",
    "completed_flag", "completed_strict",     # [FIX] 补 completed_strict
    "deal_status", "deal_status_clean",       # [FIX] 补 deal_status_clean
    # ---- Deal-level controls ----
    "cross_border",
    # ---- Payment ----
    "deal_pay_method", "deal_pay_method_all", "deal_pay_method_n",
    "pay_has_cash", "pay_has_shares", "pay_has_debt", "pay_has_other",
    "pay_n_class",
    "pay_pure_cash", "pay_pure_shares", "pay_pure_debt", "pay_pure_other",
    "pay_mix", "pay_unknown",
    "pay_mix_cash_shares", "pay_mix_with_shares",
    "pay_mix_with_cash", "pay_mix_with_debt", "pay_class",
    # ---- structure ----
    "deal_struct_all", "deal_struct_n",
    "deal_fin_all", "deal_fin_n",
    "deal_type_all", "deal_type_n",
    # ---- deal characteristics ----
    "deal_status", "deal_val_usd",
    # ---- Country / legal ----
    "tar_country_code", "acq_country_code",
    "common_law_tar", "common_law_acq",
    # ---- SIC distance ----
    "classification_distance_tar", "classification_distance_acq",
    # ---- 日期（deal_year / 样本期 / 事件研究所必需）----
    "announced_d", "rumour_d", "completed_d", "withdrawn_d",
    "expected_comp_d", "postponed_d", "assumed_comp_d",
    # ---- SIC 原始码（tar_sic3 / acq_sic3 派生用）----
    "tar_primary_sic_code", "acq_primary_sic_code",
    # ---- 用户特别指定：股权（JFE 关键变量）----
    "stake_acq_pct", "stake_init_pct",
    # ---- 其他金额（备用）----
    "deal_eq_val_usd", "deal_ent_val_usd", "deal_tar_val_tot_usd",
    "stake_final_pct", "deal_ccy",
]

# =====================================================================
# [FIX-7][FIX-11] 03 侧白名单（论文名对应的 03 原始名）
# =====================================================================
FIRM_KEEP =[DEAL] +  list(FIRM_RENAME.keys()) + [
    "tar_listed", "acq_listed", "ven_listed",
    "tar_ipo_d", "acq_ipo_d", "ven_ipo_d",
    "ven_incorp_yr",
    # 实体键（原样保留，不 rename）
    "tar_bvd_id_num", "acq_bvd_id_num",
    "tar_orbis_id_num", "acq_orbis_id_num",
    "tar_name", "acq_name", "tar_key", "src_file",
]

# =====================================================================
# [FIX-9] 三类"找不到"的变量（仅用于报告，不参与读取）
# =====================================================================
NOT_IN_02_03_NEED_MODULE = [
    "num_tar_adv_acc", "num_tar_adv_asset_fin", "num_tar_adv_broker",
    "num_tar_adv_debt", "num_tar_adv_equity", "num_tar_adv_fa",
    "num_tar_adv_pr", "num_tar_adv_insurer", "num_tar_adv_law",
    "num_tar_adv_lead", "num_tar_adv_mezz", "num_tar_adv_other",
    "num_tar_adv_underwriter", "num_tar_adv_vcpe",
    "num_acq_adv_acc", "num_acq_adv_asset_fin", "num_acq_adv_broker",
    "num_acq_adv_debt", "num_acq_adv_equity", "num_acq_adv_fa",
    "num_acq_adv_pr", "num_acq_adv_insurer", "num_acq_adv_law",
    "num_acq_adv_lead", "num_acq_adv_mezz", "num_acq_adv_other",
    "num_acq_adv_underwriter", "num_acq_adv_vcpe",
    "tar_total_advisor", "acq_total_advisor",
    "post_rev_mul_fy", "post_ebitda_mul_fy", "post_ebit_mul_fy",
    "pre_deal_ven_rev_rev_last_avail_yr", "pre_deal_ven_ta_last_avail_yr",
    "stock_traded_gdp_tar", "stock_traded_gdp_acq",
]

NOT_IN_02_03_NEED_DERIVE = [
    "_row_id", "deal_year",
    "tar_sic3", "acq_sic3",
    "ln_mul_rev", "ln_mul_ebitda", "ln_mul_ebit",
    "ln_post_mul_rev", "ln_post_mul_ebitda", "ln_post_mul_ebit",
    "ln_text_rev", "ln_text_ebitda", "ln_text_ebit",
    "ln_sic_rev", "ln_sic_ebitda", "ln_sic_ebit",
    "ind_text_diversity", "ind_text_diversity_w",
    "ln_deal_value", "cross_industry", "cross_industry_alt",
    "tar_incorp_d_year", "acq_incorp_d_year",
    "target_age", "acquirer_age",
    "ln_tar_size", "tar_ebitda_margin", "tar_leverage", "tar_rev_growth",
    "tar_roa", "tar_post_ebitda_margin", "tar_post_roa",
    "ln_tar_post_size", "tar_post_leverage",
    "ln_rel_size", "acq_roa",
    "national_class",
    "tar_ta_for_t6", "tar_rev_for_t6", "tar_mktcap_for_t6",
    "tar_ev_for_t6", "tar_mb_for_t6",
    "sic_coverage_rate", "acq_tar_similarity",
    "lm_net_sentiment",
]

# =====================================================================
# 开关
# =====================================================================
KEEP_IMPLAUSIBLE   = False   # [FIX-10] *_implausible 标记列
KEEP_EXTRA_NUMERIC = False   # 白名单外的数值列是否也保留（False=严格白名单）
DOWNCAST_FLOAT     = True
FORCE_CATEGORY     = True    # 字符串列 -> category
CHUNK_SIZE         = 100_000
AUDIT_ONLY         = False   # True = 只核对白名单，不合并

# [FIX-9B] 极低成本的派生（不需要外部数据/参数）
DERIVE_IN_04   = True
DERIVE_ROW_ID  = True
DERIVE_YEAR    = True
DERIVE_NET_SENT= True
DERIVE_CROSSIND= True

KEEP_FIRM_ON_COLLIDE = False

# 报告用变量（论文名，rename 之后）
REPORT_VARS = [
    "deal_num", "deal_year", "deal_status",
    "tar_sic2", "tar_country_code", "acq_country_code",
    "rumor_jfe", "has_rumor_status", "completed_flag", "cross_border",
    "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly",
    #恢复 8 个 multiple 倍数 (2026-10-20)
    "pre_pbt_mul_ly", "pre_pat_mul_ly", "pre_np_mul_ly",
    "pre_ta_mul_ly", "pre_na_mul_ly", "pre_cl_mul_ly",
    "pre_eq_mul_ly", "pre_cap_mul_ly",
    "deal_value", "stake_init_pct", "stake_acq_pct",
    "total_info_source",
    "pre_deal_tar_rev_rev_last_avail_yr", "pre_deal_tar_ebitda_last_avail_yr",
    "pre_deal_tar_ta_last_avail_yr", "pre_deal_tar_eq_last_avail_yr",
    "pre_deal_acq_rev_rev_last_avail_yr", "pre_deal_acq_ta_last_avail_yr",
    "post_deal_tar_rev_rev_1st_avail_yr", "post_deal_tar_ebitda_1st_avail_yr",
    "post_deal_acq_rev_rev_1st_avail_yr", "post_deal_acq_ta_1st_avail_yr",
    "tar_incorp_d_year", "acq_incorp_d_year",
    "tar_listed", "acq_listed", "ven_listed",
    "tar_ipo_d", "acq_ipo_d", "ven_ipo_d",
    "classification_distance_tar", "classification_distance_acq",
]


# =====================================================================
# Log
# =====================================================================
_LINES = []
_T0 = time.time()

def log(msg=""):
    print(msg)
    _LINES.append(str(msg))

def flush_log():
    with open(PATH_LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")

def banner(t):
    log("\n" + "=" * 84)
    log(t)
    log("=" * 84)


# =====================================================================
# 列清单解析
# =====================================================================
def resolve_usecols(path, label, whitelist, rename_map):
    """读表头 -> 白名单过滤 -> 返回 (usecols 原始名, 命中, 未命中)"""
    hdr = [str(c).strip() for c in
           pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns]
    hset = set(hdr)
    wl = [c for c in dict.fromkeys(whitelist)]          # 去重保序

    hit  = [c for c in wl if c in hset]
    miss = [c for c in wl if c not in hset]

    usecols = list(hit)
    extra_num = []
    if KEEP_EXTRA_NUMERIC:
        smp = pd.read_csv(path, nrows=5_000, encoding="utf-8-sig",
                          low_memory=False, on_bad_lines="skip")
        smp.columns = [str(c).strip() for c in smp.columns]
        for c in hdr:
            if c in hset and c in usecols:
                continue
            if c in smp.columns and pd.api.types.is_numeric_dtype(smp[c].dtype):
                if (not KEEP_IMPLAUSIBLE) and c.endswith("_implausible"):
                    continue
                usecols.append(c); extra_num.append(c)

    log(f"\n  ── 白名单核对 [{label}] ──")
    log(f"    {os.path.basename(path)} | 文件列 {len(hdr)}")
    log(f"    白名单 {len(wl)} -> 命中 {len(hit)} | 未命中 {len(miss)}")
    if miss:
        log(f"\n    [WARN] 以下 {len(miss)} 列在文件中不存在：")
        for c in miss:
            tgt = rename_map.get(c)
            log(f"        - {c}" + (f"   (计划 rename -> {tgt})" if tgt else ""))
    else:
        log(f"    [OK] 白名单全部命中")
    if extra_num:
        log(f"    [EXTRA] 额外保留数值列 {len(extra_num)}")
    rename_hit = {k: v for k, v in rename_map.items() if k in hset}
    if rename_hit:
        log(f"\n    [RENAME] 将执行 {len(rename_hit)} 处字段名对齐：")
        for k, v in rename_hit.items():
            log(f"        {k:<42s} -> {v}")
    rename_miss = {k: v for k, v in rename_map.items() if k not in hset}
    if rename_miss:
        log(f"\n    [WARN] {len(rename_miss)} 处 rename 源列不存在：")
        for k, v in rename_miss.items():
            log(f"        {k:<42s} -> {v}   (源列缺失)")
    return usecols, hit, miss


def load_and_rename(path, label, usecols, rename_map):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                         usecols=usecols, on_bad_lines="warn")
    except MemoryError:
        log(f"    [!] OOM，降级分块读 (chunksize={CHUNK_SIZE:,})")
        parts = []
        for ch in pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                              usecols=usecols, chunksize=CHUNK_SIZE,
                              on_bad_lines="skip"):
            ch.columns = [str(c).strip() for c in ch.columns]
            parts.append(ch)
        df = pd.concat(parts, ignore_index=True)
        del parts; gc.collect()

    df.columns = [str(c).strip() for c in df.columns]
    if DEAL in df.columns:
        df[DEAL] = pd.to_numeric(df[DEAL], errors="coerce")

    # rename
    ren = {k: v for k, v in rename_map.items() if k in df.columns}
    if ren:
        df = df.rename(columns=ren)
        log(f"    [{label}] rename 执行 {len(ren)} 处")

    # 降精度
    n_down = 0
    if DOWNCAST_FLOAT:
        for c in df.columns:
            if c == DEAL:
                continue
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32"); n_down += 1
            elif df[c].dtype == "int64":
                df[c] = pd.to_numeric(df[c], downcast="integer"); n_down += 1

    # category
    n_cat, mb_before, mb_after = 0, 0.0, 0.0
    for c in [c for c in df.columns if df[c].dtype == object and c != DEAL]:
        mb_before += df[c].memory_usage(deep=True) / 1024 ** 2
        df[c] = df[c].astype("category")
        mb_after += df[c].memory_usage(deep=True) / 1024 ** 2
        n_cat += 1
    if n_cat:
        log(f"    [CATEGORY] {n_cat} 列: {mb_before:.0f} MB -> {mb_after:.0f} MB "
            f"(省 {mb_before-mb_after:.0f} MB)")

    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    log(f"    {len(df):,} rows x {df.shape[1]} cols | "
        f"unique {DEAL}: {df[DEAL].nunique():,} | mem={mem:,.0f} MB "
        f"(降精度 {n_down})")
    return df


# =====================================================================
# [ENTITY-AUDIT] 02 主行 vs 03 主行：行内实体一致性（纯审计，不改数据）
# =====================================================================
# 02 与 03 各自把多标的折叠成一行一 deal，但挑选主行的依据不同：
#   02 = 全列填充最多 + 原顺序；03 = 财务非空数 > tar_bvd > acq_bvd > 顺序。
# 因此同一 deal 的"主行"可能不是同一个 target。validate="one_to_one"
# 只能保证键唯一，发现不了这种错配。本审计独立读两个文件的实体键列
# （不进入主白名单输出），对重叠 deal 比对 target / acquirer 身份。
def entity_consistency_audit():
    banner("STEP 5c — 实体一致性审计（02 主行 vs 03 主行）")
    log("  读取列: deal_num + tar_bvd_id_num / tar_name / acq_bvd_id_num / acq_name")
    keys = ["deal_num", "tar_bvd_id_num", "tar_name",
            "acq_bvd_id_num", "acq_name"]
    dfs = {}
    for p, lab in [(PATH_MASTER, "02"), (PATH_FIRM, "03")]:
        try:
            hdr = [str(c).strip() for c in
                   pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns]
            use = [c for c in keys if c in hdr]
            d = pd.read_csv(p, usecols=use, low_memory=False,
                            encoding="utf-8-sig")
            d[DEAL] = pd.to_numeric(d[DEAL], errors="coerce")
            d = d.dropna(subset=[DEAL])
            d[DEAL] = d[DEAL].astype("int64")
            dfs[lab] = d
            log(f"  [{lab}] 实体键列可用 {len(use)} 个: {use}")
        except Exception as e:
            log(f"  [{lab}] 无法读取实体键: {e}")
            return
    m, f = dfs["02"], dfs["03"]
    j = m.merge(f, on=DEAL, how="inner", suffixes=("_02", "_03"))
    log(f"  重叠 deal: {len(j):,}")
    n_total_conf = 0
    for c in ["tar_bvd_id_num", "tar_name", "acq_bvd_id_num", "acq_name"]:
        a, b = f"{c}_02", f"{c}_03"
        if a not in j.columns or b not in j.columns:
            log(f"  {c:<18s}: 一方缺列，跳过")
            continue
        # Normalize missing to NaN FIRST (str "nan"/"" etc. are NOT NaN to
        # notna()), otherwise one-side-missing rows are counted as conflicts.
        av = j[a].astype(str).str.strip()
        av = av.mask(av.isin(["", "nan", "NaN", "None", "<NA>", "NaT"]))
        bv = j[b].astype(str).str.strip()
        bv = bv.mask(bv.isin(["", "nan", "NaN", "None", "<NA>", "NaT"]))
        both = av.notna() & bv.notna() & (av != bv)
        n = int(both.sum())
        if n:
            n_total_conf += n
            samp = list(zip(j.loc[both, DEAL].head(5),
                            j.loc[both, a].head(5), j.loc[both, b].head(5)))
            log(f"  [!!] {c:<18s}: {n:,} 笔不一致（02 主行 vs 03 主行）"
                f"样例 {samp}")
        else:
            log(f"  [OK] {c:<18s}: 一致（或一方缺失，不判冲突）")
    if n_total_conf:
        log(f"  [!!] 共 {n_total_conf:,} 处实体键不一致 —— 行内实体错配风险，"
            f"建议核对上游折叠规则或改用 02_entity_pairs / 03_firm_entity_pairs "
            f"在实体粒度对齐后再折叠")
    else:
        log("  [OK] 未发现 02/03 主行实体键冲突")


# =====================================================================
# [FIX-9B] 低成本派生
# =====================================================================
def derive(df):
    made = []
    if DERIVE_ROW_ID and "_row_id" not in df.columns:
        df["_row_id"] = np.arange(len(df), dtype="int64"); made.append("_row_id")

    if DERIVE_YEAR:
        src = None
        for c in ("announced_d", "completed_d"):
            if c in df.columns:
                src = c; break
        if src and "deal_year" not in df.columns:
            s = df[src]
            s = s.astype(str) if s.dtype.name == "category" else s
            df["deal_year"] = pd.to_numeric(
                s.str.slice(0, 4), errors="coerce").astype("Int32")
            made.append(f"deal_year <- {src}")

    if DERIVE_NET_SENT:
        a, b = "lm_pos_density", "lm_neg_density"
        if a in df.columns and b in df.columns and "lm_net_sentiment" not in df.columns:
            df["lm_net_sentiment"] = (pd.to_numeric(df[a], errors="coerce")
                                      - pd.to_numeric(df[b], errors="coerce"))
            made.append("lm_net_sentiment <- pos - neg")

    if DERIVE_CROSSIND:
        if "tar_sic2" in df.columns and "acq_sic2" in df.columns \
                and "cross_industry" not in df.columns:
            t = df["tar_sic2"]; a = df["acq_sic2"]
            df["cross_industry"] = np.where(t.notna() & a.notna(),
                                            (t != a).astype(float), np.nan)
            made.append("cross_industry <- tar_sic2 != acq_sic2")

    if made:
        log(f"\n    [DERIVE] 生成 {len(made)} 列：")
        for m in made:
            log(f"        {m}")
    return df


# =====================================================================
# 报告
# =====================================================================
def coverage_report(df, cols, denom=None):
    n = denom if denom else len(df)
    log(f"\n  覆盖率（分母 = {n:,}）")
    log(f"    {'column':<44s} {'non-null':>12s} {'pct':>9s} {'missing':>12s}")
    log("    " + "-" * 80)
    for c in cols:
        if c in df.columns:
            nn = int(df[c].notna().sum())
            log(f"    {c:<44s} {nn:>12,} {nn/max(n,1)*100:>8.2f}% {n-nn:>12,}")
        else:
            log(f"    {c:<44s} {'-- NOT PRESENT --':>12s}")


# =====================================================================
# Main
# =====================================================================
def main():
    banner("04_merge_firm_to_deal.py — v11（COLS_TO_SAVE 驱动 + 02/03 职责重划适配）")
    log(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  主键 = {DEAL}（单键；两表均一行一 deal）")
    log(f"  白名单 = 论文 COLS_TO_SAVE；03 财务列 rename 成论文名")
    log(f"  严格白名单 = {not KEEP_EXTRA_NUMERIC}")

    for p in (PATH_MASTER, PATH_FIRM):
        if not os.path.exists(p):
            log(f"\n  [FATAL] 找不到 {p}")
            flush_log(); return

    # [V11] 02/03 职责重划：MASTER_KEEP 里在 02 不存在的列，自动挪到 FIRM_KEEP
    def _rebalance_keep():
        global MASTER_KEEP, FIRM_KEEP
        hdr_m = set(str(c).strip() for c in
                    pd.read_csv(PATH_MASTER, nrows=0, encoding="utf-8-sig").columns)
        hdr_f = set(str(c).strip() for c in
                    pd.read_csv(PATH_FIRM, nrows=0, encoding="utf-8-sig").columns)
        moved = [c for c in MASTER_KEEP
                 if c != DEAL and c not in hdr_m and c in hdr_f]
        if moved:
            MASTER_KEEP = [c for c in MASTER_KEEP if c not in moved]
            FIRM_KEEP = list(dict.fromkeys(FIRM_KEEP + moved))
            log(f"\n  [REBALANCE] {len(moved)} 列从 MASTER_KEEP 挪到 FIRM_KEEP:")
            for c in moved:
                log(f"      - {c}")
        else:
            log("\n  [REBALANCE] 无需迁移（02/03 列分布已对齐）")
    _rebalance_keep()

    # ---- STEP 0 ----
    banner("STEP 0 — 白名单核对")
    us_m, hit_m, miss_m = resolve_usecols(PATH_MASTER, "02_deal_master",
                                          MASTER_KEEP, MASTER_RENAME)
    us_f, hit_f, miss_f = resolve_usecols(PATH_FIRM, "03_firm_deal_level",
                                          FIRM_KEEP, FIRM_RENAME)

    # ---- [FIX-9] 三类缺失报告 ----
    banner("STEP 0b — COLS_TO_SAVE 中不在 02/03 的变量（分组）")
    log(f"\n  A) 需外部模块/数据并入（{len(NOT_IN_02_03_NEED_MODULE)} 个）：")
    log(f"     - advisor 28 列 (Module J 手工) + tar/acq_total_advisor")
    log(f"     - post_rev/ebitda/ebit_mul_fy (post 倍数，02待提供)")
    log(f"     - pre_deal_ven_* (vendor 财务，03 无 vendor 财务模块)")
    log(f"     - stock_traded_gdp_tar/acq (宏观外部数据)")
    log(f"\n  B) 需派生（07/08 做，{len(NOT_IN_02_03_NEED_DERIVE)} 个）：")
    log(f"     - tar_sic3 / acq_sic3        <- tar(acq)_primary_sic_code")
    log(f"     - ln_mul_* / ln_text_* / ln_sic_*")
    log(f"     - ln_deal_value / ln_tar_size / ln_rel_size / ln_tar_post_size")
    log(f"     - tar_ebitda_margin / tar_leverage / tar_roa / tar_rev_growth")
    log(f"     - tar_post_* / acq_roa / target_age / acquirer_age")
    log(f"     - cross_industry_alt / national_class / sic_coverage_rate")
    log(f"     - acq_tar_similarity / ind_text_diversity(_w)")
    log(f"     - tar_ta_for_t6 / tar_rev_for_t6 / tar_mktcap_for_t6")
    log(f"       / tar_ev_for_t6 / tar_mb_for_t6")
    log(f"\n  C) 命名不同 —— 已在本脚本 rename，见上方 [RENAME]")

    if AUDIT_ONLY:
        log("\n  [AUDIT_ONLY=True] 核对完成，不执行合并。")
        flush_log(); return

    # ---- STEP 1 ----
    banner("STEP 1 — 加载 + rename")
    log(f"\n  [LOAD] 02_deal_master : usecols={len(us_m)}")
    dfm = load_and_rename(PATH_MASTER, "02", us_m, MASTER_RENAME)
    log(f"\n  [LOAD] 03_firm_deal_level : usecols={len(us_f)}")
    dff = load_and_rename(PATH_FIRM, "03", us_f, FIRM_RENAME)

    # ---- STEP 2 ----
    banner("STEP 2 — 键检查（不填充 / 不 dedup）")
    for nm, d in [("master", dfm), ("firm", dff)]:
        n_nan = int(d[DEAL].isna().sum())
        n_dup = int(d[DEAL].duplicated().sum())
        log(f"  [{nm}] deal_num: NaN={n_nan:,} | 重复={n_dup:,} | "
            f"unique={d[DEAL].nunique():,}")
        if n_nan:
            d.dropna(subset=[DEAL], inplace=True)
        if n_dup:
            log(f"      [FATAL] deal_num 有重复，不能 1:1 连接")
            raise RuntimeError(f"{nm} deal_num duplicated")
        d[DEAL] = d[DEAL].astype("int64")

    mset = set(dfm[DEAL])
    fset = set(dff[DEAL])
    base_rows = len(dfm)
    log(f"\n  base 行数 = {base_rows:,}（合并后必须不变）")

    # ---- STEP 3 ----
    banner("STEP 3 — 列名冲突（先报告）")
    m_cols = set(dfm.columns)
    collide = [c for c in dff.columns if c in m_cols and c != DEAL]
    if collide:
        log(f"\n  同名列 {len(collide)} 个（rename 之后）：")
        log(f"    {'column':<40s} {'master cov':>12s} {'firm cov':>12s} {'处理':>12s}")
        log("    " + "-" * 78)
        for c in collide:
            act = "保留 firm" if KEEP_FIRM_ON_COLLIDE else "保留 master"
            log(f"    {c:<40s} {int(dfm[c].notna().sum()):>12,} "
                f"{int(dff[c].notna().sum()):>12,} {act:>12s}")
        if KEEP_FIRM_ON_COLLIDE:
            dfm = dfm.drop(columns=collide)
        else:
            dff = dff.drop(columns=collide)
    else:
        log("\n  无同名列")

    new_cols = [c for c in dff.columns if c != DEAL]
    log(f"\n  将并入 {len(new_cols)} 个新列")

    # ---- STEP 4 ----
    banner("STEP 4 — 左连接（deal_num 单键）")
    log(f"  base={len(dfm):,} + firm={len(dff):,}")
    out = dfm.merge(dff, on=DEAL, how="left", validate="one_to_one")
    del dfm, dff
    gc.collect()
    log(f"  合并后 = {len(out):,} | "
        f"mem={out.memory_usage(deep=True).sum()/1024**2:,.0f} MB")
    if len(out) != base_rows:
        log(f"\n  [FATAL] 行数改变 {base_rows:,} -> {len(out):,}")
        log("          不执行强制 dedup（那会静默丢数据）")
        raise RuntimeError("row count changed")
    log("  [OK] 行数不变（validate='one_to_one' 已硬校验）")

    # ---- STEP 5 匹配率 ----
    banner("STEP 5 — 匹配率")
    n_hit = len(mset & fset)
    log(f"  02 deals     : {len(mset):,}")
    log(f"  03 deals     : {len(fset):,}")
    log(f"  交集（命中） : {n_hit:,} ({n_hit/max(len(mset),1)*100:.2f}%)")
    log(f"  02 有 03 无  : {len(mset-fset):,} "
        f"({len(mset-fset)/max(len(mset),1)*100:.2f}%)  <- 无财务数据，保留 NaN")
    if fset - mset:
        log(f"  03 有 02 无  : {len(fset-mset):,}  <- 不在全集，左连接丢弃")

    # 02/03 均已含实体键（04 从 03 侧读入 8 个实体键），恢复实体一致性审计
    entity_consistency_audit()

    # ---- STEP 5b 派生 ----
    if DERIVE_IN_04:
        banner("STEP 5b — 低成本派生")
        out = derive(out)

    # ---- STEP 6 ----
    banner("STEP 6 — FINAL COVERAGE（论文变量名）")
    coverage_report(out, REPORT_VARS, denom=len(out))

    n_rows, n_deals = len(out), out[DEAL].nunique()
    log(f"\n  rows={n_rows:,} | unique {DEAL}={n_deals:,}")
    assert n_rows == n_deals, "不是一行一 deal"
    log("  [OK] 一行一 deal")

    n_cat = sum(1 for c in out.columns if out[c].dtype.name == "category")
    log(f"\n  列构成: 共 {out.shape[1]} 列 | category {n_cat} | "
        f"数值 {out.shape[1]-n_cat-1} | 键 1")

    # ---- STEP 7 ----
    banner("STEP 7 — Save")
    out.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")
    sz = os.path.getsize(PATH_OUT) / 1024 ** 2
    log(f"  [SAVED] {PATH_OUT}")
    log(f"    rows={len(out):,} | cols={out.shape[1]} | {sz:.1f} MB")
    log(f"\n  done in {time.time()-_T0:.1f}s")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise