# -*- coding: utf-8 -*-
r"""
06a_merge_comment_vendor.py — v3 (2026-10-20)
================================================================================
在 05 screening 结果基础上并入两块外部数据：
  [A] 01c_comments_features.csv     —— LM 金融语调 / comment 文本特征（editorial 级）
  [B] 01b_deal_vendor_type.csv      —— 卖方类型（02 中因文件名歧义 disabled，挪到此处）
输入
  D:\MA\data\merged\05_deal_firm_filtered.csv     基表，一行一 deal
  D:\MA\data\cleaned\01c_comments_features.csv    1,579,923 editorial / 1,065,301 deal
  D:\MA\data\cleaned\01b_deal_vendor_type.csv     deal 级（显式指定，不走扫描）
输出
  D:\MA\data\merged\06a_merge_comment_vendor.csv
  D:\MA\data\merged\06a_merge_comment_vendor.txt
  D:\MA\data\merged\06a_merge_comment_vendor_steps.csv
================================================================================
★ v2 -> v3 改动：彻底移除同行文本估值倍数（06_text_peer_benchmarks.csv）合并逻辑 ★
================================================================================
[C1] 完全删除 [A] 06_text_peer_benchmarks.csv 所有相关代码与变量
     不再读取该文件、不再执行左连接、不再做benchmark与自身倍数对比统计；
     删掉所有 _p06 后缀、MUL_WIN、STEP4对比等全部peer相关逻辑。
[C2] 保留原有 comment + vendor_type 两套合并流程不变
     comment 合并规则不变：撞名默认加 _cmt 后缀，不覆盖旧列；
     vendor_type 依旧显式读取文件，全量并入，重名列加 _vt 后缀。
[C3] 只保留两步外部数据合并：
     Step1：读入05基表；Step2：合并comment文本特征；Step3：合并vendor_type；Step4：覆盖率诊断统计；Step5：保存输出。
================================================================================
★ 数据性质问题（非代码 bug）★
================================================================================
[P1] 保证最终输出严格一行一deal，deal_num唯一，程序末尾做唯一性校验。
================================================================================
"""
import os
import time
import traceback
from datetime import datetime
import pandas as pd
# ====================== 全局路径配置 ======================
BASE = r"D:\MA\data"
MERGED = os.path.join(BASE, "merged")
CLEAN = os.path.join(BASE, "cleaned")
# 输入文件
F_BASE = os.path.join(MERGED, "05_deal_firm_filtered.csv")
F_COMMENT = os.path.join(CLEAN, "01c_comments_features.csv")
F_VENDOR = os.path.join(CLEAN, "01b_deal_vendor_type.csv")
# 输出文件
OUT_FILE = os.path.join(MERGED, "06a_merge_comment_vendor.csv")
OUT_LOG = os.path.join(MERGED, "06a_merge_comment_vendor.txt")
OUT_STEPS = os.path.join(MERGED, "06a_merge_comment_vendor_steps.csv")
# 关键字段
DEAL = "deal_num"
ON_COLLISION = "suffix"  # 冲突策略：suffix=加后缀，overwrite=覆盖
# 日志与步骤记录
log_lines = []
steps = []
_T0 = time.time()
START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    log_lines.append(msg)
    print(msg)

def flush_log():
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

def banner(title: str):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)

def coverage_report(df):
    """生成和04脚本风格一致的变量覆盖率描述统计表"""
    banner("STEP 4 — FINAL COVERAGE DIAGNOSTICS")
    total_n = len(df)
    log(f"  分母（总行数） = {total_n:,}")
    log(f"  {'column':<48}{'non-null':>12}{'pct':>10}{'missing':>12}")
    log("-" * 82)
    # 优先输出核心变量，其余按原始顺序追加
    core_vars = [
        "deal_num", "deal_year", "deal_status",
        "tar_sic2", "tar_country_code", "acq_country_code",
        "rumor_jfe", "has_rumor_status", "completed_flag", "cross_border",
        "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly", 
        
        #恢复 8 个 multiple 倍数 (2026-10-20)
        "pre_pbt_mul_ly", "pre_pat_mul_ly", "pre_np_mul_ly",
        "pre_ta_mul_ly", "pre_na_mul_ly", "pre_cl_mul_ly",
        "pre_eq_mul_ly", "pre_cap_mul_ly",
    
        "deal_value",
    
        "stake_init_pct", "stake_acq_pct", "total_info_source",
        "pre_deal_tar_rev_rev_last_avail_yr",
        "pre_deal_tar_ebitda_last_avail_yr",
        "pre_deal_tar_ta_last_avail_yr",
        "pre_deal_tar_eq_last_avail_yr",
        "pre_deal_acq_rev_rev_last_avail_yr",
        "pre_deal_acq_ta_last_avail_yr",
        "post_deal_tar_rev_rev_1st_avail_yr",
        "post_deal_tar_ebitda_1st_avail_yr",
        "post_deal_acq_rev_rev_1st_avail_yr",
        "post_deal_acq_ta_1st_avail_yr",
        "tar_incorp_d_year", "acq_incorp_d_year",
        "tar_listed", "acq_listed", "ven_listed"
    ]
    # 过滤：存在于df的核心变量
    present_core = [v for v in core_vars if v in df.columns]
    # 剩余其他变量
    rest_vars = [c for c in df.columns if c not in present_core]
    all_cols = present_core + rest_vars

    for col in all_cols:
        nn = df[col].notna().sum()
        miss = total_n - nn
        pct = 100.0 * nn / total_n if total_n > 0 else 0.0
        log(f"  {col:<48}{nn:>12,}{pct:>9.2f}%{miss:>12,}")

    # 列类型汇总
    cat_cols = list(df.select_dtypes(include=["category", "object"]).columns)
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    log("\n" + "-" * 82)
    log(f"  rows={total_n:,} | unique {DEAL}={df[DEAL].nunique():,}")
    log(f"  [OK] 一行一 deal" if df[DEAL].nunique() == total_n else "[WARN] 存在重复 deal_num")
    log(f"  列构成: 共 {df.shape[1]} 列 | 类别 {len(cat_cols)} | 数值 {len(num_cols)} | 主键 1")

def main():
    banner("06_merge_comment_peer.py — v3")
    log(f"started: {START_TIME}")
    log(f"  主键 = {DEAL}（单键；基表一行一 deal）")
    log(f"  合并策略：左连接；列名冲突自动加后缀 _cmt / _vt")

    banner("STEP 1 — 读取基表 05_deal_firm_filtered.csv")
    df = pd.read_csv(F_BASE, encoding="utf-8-sig")
    log(f"  基表读取完成：rows={len(df):,}, cols={df.shape[1]}")
    log(f"  unique {DEAL} = {df[DEAL].nunique():,}")
    steps.append({"step": "load_base", "rows": len(df), "cols": df.shape[1]})

    # -------------------------- STEP 2 合并 comment 文本特征 --------------------------
    banner("STEP 2 — 合并 comment 语调特征 01c_comments_features.csv")
    df_cmt = pd.read_csv(F_COMMENT, encoding="utf-8-sig")
    log(f"  comment源表：rows={len(df_cmt):,}, unique {DEAL}={df_cmt[DEAL].nunique():,}")
    # 重名列加后缀 _cmt
    overlap_cols = [c for c in df_cmt.columns if c in df.columns and c != DEAL]
    rename_dict = {c: f"{c}_cmt" for c in overlap_cols}
    df_cmt = df_cmt.rename(columns=rename_dict)
    if overlap_cols:
        log(f"  重名列（加_cmt后缀）：{overlap_cols}")
    # 左连接
    df = df.merge(df_cmt, on=DEAL, how="left")
    log(f"  合并后：rows={len(df):,}, cols={df.shape[1]}")
    steps.append({"step": "merge_comment", "rows": len(df), "cols": df.shape[1]})

    # -------------------------- STEP 3 合并 vendor_type 卖方类型 --------------------------
    banner("STEP 3 — 合并卖方类型 01b_deal_vendor_type.csv")
    df_vt = pd.read_csv(F_VENDOR, encoding="utf-8-sig")
    log(f"  vendor源表：rows={len(df_vt):,}, unique {DEAL}={df_vt[DEAL].nunique():,}")
    log(f"  源表全部字段：{list(df_vt.columns)}")
    # 重名列加后缀 _vt
    overlap_cols_vt = [c for c in df_vt.columns if c in df.columns and c != DEAL]
    rename_vt = {c: f"{c}_vt" for c in overlap_cols_vt}
    df_vt = df_vt.rename(columns=rename_vt)
    if overlap_cols_vt:
        log(f"  重名列（加_vt后缀）：{overlap_cols_vt}")
    # 左连接
    df = df.merge(df_vt, on=DEAL, how="left")
    log(f"  合并后：rows={len(df):,}, cols={df.shape[1]}")
    steps.append({"step": "merge_vendor", "rows": len(df), "cols": df.shape[1]})

    # -------------------------- STEP4 覆盖率诊断统计（新增，和04风格对齐） --------------------------
    coverage_report(df)

    # -------------------------- STEP 5 保存结果 --------------------------
    banner("STEP 5 — Save")
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    log(f"  [SAVED] {OUT_FILE}")
    log(f"    rows={len(df):,} | cols={df.shape[1]} | "
        f"{os.path.getsize(OUT_FILE)/1024/1024:.1f} MB")
    log(f"    unique {DEAL} = {df[DEAL].nunique():,} "
        f"({'==' if df[DEAL].nunique()==len(df) else '!='} rows)")
    pd.DataFrame(steps).to_csv(OUT_STEPS, index=False, encoding="utf-8-sig")
    log(f"  [SAVED] {OUT_STEPS}")
    # 强制校验：必须一行一deal
    if df[DEAL].nunique() != len(df):
        raise RuntimeError("输出不是一行一 deal")
    log(f"\n  done in {time.time()-_T0:.0f}s")
    flush_log()
    log(f"  [SAVED] {OUT_LOG}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise
