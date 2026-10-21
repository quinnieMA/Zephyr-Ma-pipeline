# -*- coding: utf-8 -*-
"""config.py — 06/07 真实列名 -> 论文变量名 映射 + 派生 + 筛选
v3: 修正 apply_sample 的 notna() 误删；区分硬/软控制变量
"""
import numpy as np
import pandas as pd

IN_07 = r"D:\MA\data\merged\07_full_sample.csv"

COL = dict(
    deal_num            = "deal_num",
    deal_year           = "deal_year",
    deal_status         = "deal_status",
    completed_strict    = "completed_strict",
    rumor              = "rumor_jfe",
    deal_value          = "deal_value",
    ln_deal_value       = "ln_deal_value",
    stake_eff_pct       = "stake_eff_pct",
    cross_border        = "cross_border",
    cross_industry      = "cross_industry",
    tar_age             = "tar_age",
    acq_age             = "acq_age",
    acq_listed          = "acq_listed",
    pay_has_shares      = "pay_has_shares",
    pay_mix             = "pay_mix",
    reg_antitrust       = "reg_antitrust",
    reg_securities      = "reg_securities",
    text_peer_sim       = "text_peer_sim_mean",
    n_peers_rev         = "n_peers_rev",
    comment_wordcount   = "comment_wordcount",
    lm_neg_density      = "lm_neg_density",
    lm_uncertain_density= "lm_uncertain_density",
    lm_litigious_density= "lm_litigious_density",
    lm_strongmodal_density = "lm_strongmodal_density",
    lm_weakmodal_density   = "lm_weakmodal_density",
    lm_constrain_density   = "lm_constrain_density",
    total_info_source   = "total_info_source",
    num_source_other    = "num_source_other",
    num_Stock_Exchange  = "num_Stock_Exchange",
    num_Electronic_Publication = "num_Electronic_Publication",
    num_Company_Press_Release  = "num_Company_Press_Release",
    # 06b 缩尾倍数（真实存在，覆盖率 34.2%，缺失 65.8%）
    multiple           = "pre_rev_mul_ly_y",
)

# 硬控制变量：缺失率 < 1%，可安全进主方程
HARD_CONTROLS = [
    "ln_deal_value", "cross_border", "cross_industry", "stake_eff_pct",
    "tar_age", "acq_age", "acq_listed",
    "pay_has_shares", "pay_mix", "reg_antitrust", "reg_securities",
]
# 软控制变量：高缺失，仅进辅助模型 / 稳健性
SOFT_CONTROLS = [
    "text_peer_sim", "n_peers_rev",
    "lm_neg_density", "lm_uncertain_density", "lm_litigious_density",
    "lm_strongmodal_density", "lm_weakmodal_density", "lm_constrain_density",
    "comment_wordcount",
]
CONTROLS = HARD_CONTROLS + SOFT_CONTROLS

# [FIX] 去掉 regulatory_countries（缺失 86.9%，且与 reg_antitrust 共线）
INSTRUMENTS = ["total_info_source", "num_source_other",
               "num_Stock_Exchange", "num_Electronic_Publication"]

YEAR_FE, MAX_YEAR_FE, SEED = True, 40, 20261020


def load_df(path=IN_07, usecols=None):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                       usecols=usecols)


def rename(df):
    out = df.copy()
    missing = []
    for k, v in COL.items():
        if v in out.columns:
            out = out.rename(columns={v: k})
        else:
            missing.append(f"{k}<-{v}")
    if missing:
        print(f"[config] {len(missing)} 列未找到: {missing[:6]}")
    return out


def build_derived(df):
    """派生列。倍数缺失是结构性（65.8%），保留 NaN，绝不 fillna"""
    out = df.copy()
    if "ln_deal_value" not in out.columns and "deal_value" in out.columns:
        dv = pd.to_numeric(out["deal_value"], errors="coerce")
        out["ln_deal_value"] = np.log1p(dv.clip(lower=0))
    if "multiple_ln" not in out.columns and "multiple" in out.columns:
        mv = pd.to_numeric(out["multiple"], errors="coerce")
        out["multiple_ln"] = np.log1p(mv.clip(lower=0))     # 缺失保留 NaN
    # 信息来源缺失哑变量（3 个信息来源变量缺失率一致 9.27%）
    if "total_info_source" in out.columns:
        out["has_info_source"] = out["total_info_source"].notna().astype(float)
    return out


def apply_sample(df, min_year=1997, max_year=2025):
    """[FIX v3] completed_strict 是 0/1，不能用 notna() 判断"""
    y = pd.to_numeric(df.get("deal_year"), errors="coerce")
    cs = pd.to_numeric(df.get("completed_strict"), errors="coerce")
    ok = y.between(min_year, max_year) & cs.notna()
    return df.loc[ok].reset_index(drop=True)