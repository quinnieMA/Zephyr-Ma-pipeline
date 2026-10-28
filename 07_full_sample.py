# -*- coding: utf-8 -*-
"""
07_full_sample_v7.py  ——  已修复 Stata 变量名合规问题
================================================================================
合并 06a(comment+vendor) + 06b(text peer) -> 07_full_sample-JFE.csv / .dta
- 06a/06b 的 multiple 列全部保留 (_x=06a原始, _y=06b缩尾), 不缩尾, 只取 ln
- ln_*_size 与 ln_deal_value 等 ln 类变量不缩尾 (ln 变换已消除量纲/偏态)
- safe_ratio 加 min_den 保护, 防 EQ<=0 导致杠杆率爆炸
- 日志无条件刷盘 (finally)
- 【v7.1 修复】导出 .dta 前将所有列名转为 Stata 合法名:
    首字符必须为字母 -> '_year_source' 等下划线开头列加 's_' 前缀;
    非法字符替换为 '_'; 截断到 32 字符; 同时同步 value_labels 的 key。
================================================================================
"""
import os
import re
import time
import datetime
import traceback
import warnings

import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore")

BASE   = r"D:\MA\data"
MERGED = os.path.join(BASE, "merged")
F_06A = os.path.join(MERGED, "06a_merge_comment_vendor.csv")
F_06B = os.path.join(MERGED, "06b_text_peer.csv")

OUT_CSV       = os.path.join(MERGED, "07_full_sample-JFE.csv")
OUT_DTA       = os.path.join(MERGED, "07_full_sample-JFE.dta")
OUT_LOG       = os.path.join(MERGED, "07_full_sample_log-JFE.txt")
OUT_STEPS     = os.path.join(MERGED, "07_full_sample_steps-JFE.csv")
OUT_VAR_CHECK = os.path.join(MERGED, "07_var_check-JFE.csv")

DEAL = "deal_num"
WINSOR_LO, WINSOR_HI = 0.05, 0.95      # 当前启用; 改这里即可
P_CEIL = 1e9

# 仅剔除与 06a 语义重复的日期列; 倍数列一律保留
DROP_FROM_06B = {"deal_date", "deal_year", "announced_d", "completed_d", "rumour_d"}

# ln 类 / 比率类: 绝对不缩尾 (ln 变换已完成, 比率有天然上界)
NO_WINSOR = {
    "ln_tar_size", "ln_acq_size", "ln_tar_post_size", "ln_acq_post_size",
    "ln_rel_size", "ln_deal_value", "ln_days",
    "ln_mul_rev", "ln_mul_ebitda", "ln_mul_ebit", "ln_mul_pbt",
    "ln_mul_ta", "ln_mul_eq", "ln_mul_pat", "ln_mul_np", "ln_mul_na",
    "ln_mul_cl", "ln_mul_cap",
}

# ------------------------------------------------------------------ #
#  Stata 变量名合规化: 首字符必须为字母 (A-Za-z), 其余仅允许 字母/数字/_
# ------------------------------------------------------------------ #
_STATA_ILLEGAL = re.compile(r"[^A-Za-z0-9_]+")


def stata_varname(name):
    """把任意列名转成 Stata 合法变量名: 首字符非字母则加 's_' 前缀,
    中间非法字符替换为 '_', 并截断到 Stata 上限 32 字符。"""
    s = str(name)
    if not s[:1].isalpha():
        s = "s_" + s
    s = _STATA_ILLEGAL.sub("_", s).strip("_")
    if not s[:1].isalpha():          # 极端情况兜底
        s = "v_" + s
    return s[:32]


def build_rename_map(columns):
    """为一组列名构造 rename 字典, 并处理加前缀后可能产生的重名。"""
    out, used = {}, {}
    for c in columns:
        new = stata_varname(c)
        if new in used:
            used[new] += 1
            new = f"{new}_{used[new]}"
        else:
            used[new] = 0
        out[c] = new
    return out


# ------------------------------------------------------------------ #
#  通用工具函数 (原样保留)
# ------------------------------------------------------------------ #
log_lines, steps = [], []
_T0 = time.time()
START_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    log_lines.append(msg)
    print(msg)


def flush_log():
    try:
        with open(OUT_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"\n[LOG SAVED] {OUT_LOG}  ({len(log_lines)} lines)")
    except Exception as e:
        print(f"[LOG FAIL] {e}")


def pct(n, d):
    return f"{n/d*100:.1f}%" if d else "n/a"


def banner(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def add_step(name, n, note=""):
    steps.append({"step": name, "time_elapsed_sec": round(time.time()-_T0, 2),
                  "row_count": n, "note": note,
                  "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def winsorise(s, lo=None, hi=None):
    s = s.copy()
    if lo is not None:
        s = s.clip(lower=s.quantile(lo))
    if hi is not None:
        s = s.clip(upper=s.quantile(hi))
    return s


def safe_ratio(num, den, min_den=0.0):
    """min_den 保护: EQ<=0 时杠杆率不计算, 防爆炸"""
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return np.where(d.notna() & (d > min_den), n / d, np.nan)


def safe_get(df, col, warn=True):
    if col in df.columns:
        return df[col]
    if warn:
        log(f"    [MISS] {col} -> 全 NaN")
    return pd.Series(np.nan, index=df.index)


def parse_date_robust(s):
    if s is None:
        return pd.Series(pd.NaT, dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    num = pd.to_numeric(s, errors="coerce")
    ok = num.notna()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if ok.any():
        out.loc[ok] = pd.to_datetime(num.loc[ok], unit="D",
                                     origin="1899-12-30", errors="coerce")
    rest = ~ok & s.notna()
    if rest.any():
        out.loc[rest] = pd.to_datetime(s.loc[rest], errors="coerce",
                                       infer_datetime_format=True)
    return out


LEGAL_ORIGIN = {
    "AU": "English", "CA": "English", "GH": "English", "HK": "English",
    "IN": "English", "IE": "English", "JM": "English", "KE": "English",
    "MY": "English", "NZ": "English", "NG": "English", "PK": "English",
    "SG": "English", "ZA": "English", "TZ": "English", "UG": "English",
    "GB": "English", "US": "English", "ZW": "English", "BD": "English",
    "LK": "English", "ZM": "English", "KY": "English", "IL": "English",
    "BM": "English", "MT": "English", "PG": "English", "VG": "English",
    "MH": "English", "GI": "English", "VC": "English",
    "BE": "French", "BR": "French", "CL": "French", "CO": "French",
    "EG": "French", "FR": "French", "GR": "French", "ID": "French",
    "IT": "French", "JO": "French", "LB": "French", "LU": "French",
    "MX": "French", "MA": "French", "NL": "French", "PE": "French",
    "PH": "French", "PT": "French", "RO": "French", "ES": "French",
    "TH": "French", "TN": "French", "TR": "French", "VE": "French",
    "CN": "French", "RU": "French", "UA": "French", "VN": "French",
    "KZ": "French", "AE": "French", "SA": "French", "QA": "French",
    "KW": "French", "BH": "French", "DZ": "French", "OM": "French",
    "PA": "French", "CD": "French", "MN": "French", "BG": "French",
    "LT": "French", "BY": "French", "UY": "French", "UZ": "French",
    "AR": "French",
    "AT": "German", "CZ": "German", "DE": "German", "HU": "German",
    "JP": "German", "KR": "German", "SK": "German", "CH": "German",
    "TW": "German", "PL": "German", "HR": "German", "SI": "German",
    "RS": "German", "ME": "German", "BA": "German", "MK": "German",
    "EE": "German", "LV": "German",
    "DK": "Scandinavian", "FI": "Scandinavian", "IS": "Scandinavian",
    "NO": "Scandinavian", "SE": "Scandinavian",
}
CLASSIFICATION_DIST = {
    "US": 0, "CA": 0, "MX": 0,
    "GB": 1, "AU": 1, "NZ": 1, "IE": 1, "ZA": 1, "IN": 1,
    "SG": 1, "HK": 1, "MY": 1, "NG": 1, "KE": 1, "GH": 1,
    "IL": 1, "CY": 1, "MT": 1, "BM": 1, "KY": 1, "VG": 1,
    "DE": 2, "FR": 2, "IT": 2, "ES": 2, "NL": 2, "BE": 2,
    "AT": 2, "CH": 2, "SE": 2, "NO": 2, "DK": 2, "FI": 2,
    "PT": 2, "GR": 2, "LU": 2, "PL": 2, "CZ": 2, "HU": 2,
    "RO": 2, "SK": 2, "HR": 2, "RS": 2, "BG": 2, "LT": 2,
    "LV": 2, "EE": 2, "SI": 2, "MK": 2, "ME": 2, "BA": 2,
    "RU": 2, "TR": 2, "UA": 2, "BR": 2, "AR": 2, "CL": 2,
    "CO": 2, "PE": 2, "UY": 2, "SA": 2, "AE": 2, "QA": 2,
    "KW": 2, "BH": 2, "EG": 2, "MA": 2, "DZ": 2, "TN": 2,
    "JO": 2, "LB": 2, "KZ": 2, "UZ": 2, "BY": 2, "MN": 2,
    "JP": 3, "KR": 3, "TW": 3, "TH": 3, "ID": 3,
    "VN": 3, "PH": 3, "BD": 3, "PK": 3, "LK": 3,
    "CN": 4,
}
_VENDOR_OLD2NEW = {
    "family_person":            "individual_natural",
    "shareholders_undisclosed": "shareholder_group",
}


def main():
    banner("07_full_sample_v7.py  (v7.1: Stata 变量名合规化)")
    log(f"started : {START_TIME}")
    log(f"主键    : {DEAL}")
    log(f"06a     : {F_06A}")
    log(f"06b     : {F_06B}")
    log(f"日志文件: {OUT_LOG}")
    log(f"缩尾    : [{WINSOR_LO},{WINSOR_HI}]  不缩尾清单 {len(NO_WINSOR)} 个 ln/比率类")

    # ── STEP 1 ──
    banner("STEP 1 — 载入 06a")
    df = pd.read_csv(F_06A, low_memory=False, encoding="utf-8-sig")
    n06a = len(df)
    assert df[DEAL].nunique() == n06a
    if "_row_id" in df.columns:
        assert df["_row_id"].is_monotonic_increasing
    ven_in_06a = [c for c in df.columns
                  if "ven" in c.lower() or "vendor" in c.lower()]
    log(f"  rows={n06a:,}  cols={df.shape[1]}  unique_deal={df[DEAL].nunique():,}")
    log(f"  vendor 相关列({len(ven_in_06a)}): {ven_in_06a}")
    add_step("load_06a", n06a)

    # ── STEP 2 ──
    banner("STEP 2 — 载入 06b（仅剔除日期列 + deal_num 聚合）")
    if not os.path.exists(F_06B):
        raise FileNotFoundError(f"06b 缺失: {F_06B}")
    dfb = pd.read_csv(F_06B, low_memory=False, encoding="utf-8-sig")

    dropped = sorted([c for c in DROP_FROM_06B if c in dfb.columns])
    if dropped:
        dfb = dfb.drop(columns=dropped)
        log(f"  [剔除] 日期列({len(dropped)}): {dropped}")
    peer_cols = [c for c in dfb.columns if c != DEAL]
    n_raw, dup = len(dfb), dfb[DEAL].duplicated().sum()
    if dup > 0:
        log(f"  [聚合] 重复 deal_num={dup:,}，text_peer_sim_mean 取 median，"
            f"n_peers_*/bench_* 取 first  ->  {n_raw:,} -> {len(dfb):,}")
        agg = {c: ("median" if c == "text_peer_sim_mean" else "first")
               for c in peer_cols}
        dfb = dfb.groupby(DEAL, sort=False).agg(agg).reset_index()
    log(f"  rows={len(dfb):,}  peer_cols={peer_cols}")
    add_step("load_06b", len(dfb), f"raw={n_raw} dup={dup}")

    # ── STEP 3 ──
    banner("STEP 3 — 左连接 06a + 06b（保留 _x/_y 双列）")
    df = df.merge(dfb, on=DEAL, how="left", validate="one_to_one")
    assert df[DEAL].nunique() == len(df)
    log(f"  merged: {len(df):,} rows x {df.shape[1]} cols")

    x_cols = sorted([c for c in df.columns if c.endswith("_x")])
    y_cols = sorted([c for c in df.columns if c.endswith("_y")])
    log(f"  [INFO] _x={len(x_cols)} 个（06a 原始）  _y={len(y_cols)} 个（06b 缩尾），均保留")

    for c in peer_cols:
        if c not in df.columns:
            log(f"    [WARN] {c:<26} 未并入")
            continue
        nn = df[c].notna().sum()
        log(f"    {c:<26} non-null={nn:,} ({pct(nn,len(df))})")
    add_step("merge_06b", len(df))

    # ── STEP 4 倍数：_x/_y 均原样保留，只取 ln，不缩尾 ──
    banner("STEP 4 — 倍数处理（_x/_y 均不缩尾，只取 ln）")
    MUL_ALL = [c for c in df.columns if "_mul_" in c]
    MUL_ALL.sort(key=lambda x: (0 if x.startswith("pre_") else
                                1 if x.startswith("post_") else
                                2 if x.startswith("bench_") else 3, x))
    log(f"  识别到倍数列({len(MUL_ALL)}): {MUL_ALL}")

    for c in MUL_ALL:
        s = pd.to_numeric(df[c], errors="coerce")
        n_raw = s.notna().sum()
        if n_raw == 0:
            log(f"    [SKIP] {c:<28} 全空")
            continue
        if c.endswith("_x"):
            tag = "_x = 06a 原始"
        elif c.endswith("_y"):
            tag = "_y = 06b 缩尾"
        else:
            tag = "无后缀 = 06b 产出"
        log(f"    [keep] {c:<28} {tag:<16} n={n_raw:,} ({pct(n_raw,len(df))})  "
            f"p1={s.quantile(.01):.3f} p50={s.median():.3f} "
            f"p95={s.quantile(.95):.3f} p99={s.quantile(.99):.3f} "
            f"mean={s.mean():.3f}")

    # ln 倍数: 直接用实际存在的源列名, 生成 ln_{suf}{suffix}
    MUL_LN_MAP = [
        # (源列后缀_x, 源列后缀_y, 输出后缀 suf, 说明)
        ("pre_rev_mul_ly_x",   "pre_rev_mul_ly_y",   "rev",    "营收倍数"),
        ("pre_ebitda_mul_ly_x","pre_ebitda_mul_ly_y","ebitda", "EBITDA 倍数"),
        ("pre_ebit_mul_ly_x",  "pre_ebit_mul_ly_y",  "ebit",   "EBIT 倍数"),
        ("pre_pbt_mul_ly_x",   "pre_pbt_mul_ly_y",   "pbt",    "PBT 倍数"),
        ("pre_ta_mul_ly_x",    "pre_ta_mul_ly_y",    "ta",     "总资产倍数"),
        ("pre_eq_mul_ly_x",    "pre_eq_mul_ly_y",    "eq",     "净资产倍数"),
        ("pre_pat_mul_ly",     None,                 "pat",    "PAT 倍数(06b)"),
        ("pre_np_mul_ly",      None,                 "np",     "NP 倍数(06b)"),
        ("pre_na_mul_ly",      None,                 "na",     "NA 倍数(06b)"),
        ("pre_cl_mul_ly",      None,                 "cl",     "CL 倍数(06b)"),
        ("pre_cap_mul_ly",     None,                 "cap",    "CAP 倍数(06b)"),
    ]
    log(f"\n  --- ln 倍数（不缩尾）---")
    _gen_ln = []
    for xcol, ycol, suf, desc in MUL_LN_MAP:
        for src, suffix, tag in ((xcol, "_x", "06a原始"),
                                 (ycol, "_y", "06b缩尾")):
            if src is None or src not in df.columns:
                continue
            v = pd.to_numeric(df[src], errors="coerce")
            newc = f"ln_{suf}{suffix}"
            df[newc] = np.where(v > 0, np.log(v), np.nan)
            nn = df[newc].notna().sum()
            _gen_ln.append(newc)
            log(f"    {newc:<22} src={src:<24} {tag:<8} "
                f"valid={nn:,} ({pct(nn,len(df))})  "
                f"mean={df[newc].mean():.3f} sd={df[newc].std():.3f}")
    log(f"  [生成] 共 {len(_gen_ln)} 个 ln_ 倍数列: {_gen_ln}")
    add_step("log_multiples", len(df))

    # ── STEP 5 财务比率（ln 类不缩尾 + safe_ratio 保护） ──
    banner("STEP 5 — 财务比率 & 规模变量")
    tar_rev = pd.to_numeric(safe_get(df, "pre_deal_tar_rev_rev_last_avail_yr"),
                            errors="coerce")
    tar_eb  = pd.to_numeric(safe_get(df, "pre_deal_tar_ebitda_last_avail_yr"),
                            errors="coerce")
    tar_ta  = pd.to_numeric(safe_get(df, "pre_deal_tar_ta_last_avail_yr"),
                            errors="coerce")
    tar_eq  = pd.to_numeric(safe_get(df, "pre_deal_tar_eq_last_avail_yr"),
                            errors="coerce")
    tar_pat = pd.to_numeric(safe_get(df, "pre_deal_tar_pat_last_avail_yr"),
                            errors="coerce")

    df["tar_ebitda_margin"] = safe_ratio(tar_eb, tar_rev)
    df["tar_leverage"]      = safe_ratio(tar_ta - tar_eq, tar_ta, min_den=1e-6)
    df["tar_roa"]           = safe_ratio(tar_pat, tar_ta)
    df["ln_tar_size"]       = np.where(tar_ta > 0, np.log(tar_ta), np.nan)
    if "tar_rev_rev_yr__1" in df.columns:
        df["tar_rev_growth"] = safe_ratio(tar_rev - df["tar_rev_rev_yr__1"],
                                         df["tar_rev_rev_yr__1"])
    else:
        df["tar_rev_growth"] = np.nan
        log(f"    [MISS] tar_rev_growth: 缺 tar_rev_rev_yr__1")

    acq_rev = pd.to_numeric(safe_get(df, "pre_deal_acq_rev_rev_last_avail_yr"),
                            errors="coerce")
    acq_eb  = pd.to_numeric(safe_get(df, "pre_deal_acq_ebitda_last_avail_yr"),
                            errors="coerce")
    acq_ta  = pd.to_numeric(safe_get(df, "pre_deal_acq_ta_last_avail_yr"),
                            errors="coerce")
    acq_eq  = pd.to_numeric(safe_get(df, "pre_deal_acq_eq_last_avail_yr"),
                            errors="coerce")
    acq_pat = pd.to_numeric(safe_get(df, "pre_deal_acq_pat_last_avail_yr"),
                            errors="coerce")

    df["acq_ebitda_margin"] = safe_ratio(acq_eb, acq_rev)
    df["acq_leverage"]      = safe_ratio(acq_ta - acq_eq, acq_ta, min_den=1e-6)
    df["acq_roa"]           = safe_ratio(acq_pat, acq_ta)
    df["ln_acq_size"]       = np.where(acq_ta > 0, np.log(acq_ta), np.nan)
    df["ln_rel_size"]       = np.where((acq_ta > 0) & tar_ta.notna() & (tar_ta > 0),
                                       np.log(acq_ta / tar_ta), np.nan)

    pt_rev = pd.to_numeric(safe_get(df, "post_deal_tar_rev_rev_1st_avail_yr"),
                           errors="coerce")
    pt_eb  = pd.to_numeric(safe_get(df, "post_deal_tar_ebitda_1st_avail_yr"),
                           errors="coerce")
    pt_ta  = pd.to_numeric(safe_get(df, "post_deal_tar_ta_1st_avail_yr"),
                           errors="coerce")
    pt_pat = pd.to_numeric(safe_get(df, "post_deal_tar_pat_1st_avail_yr"),
                           errors="coerce")
    pt_eq  = pd.to_numeric(safe_get(df, "post_deal_tar_eq_1st_avail_yr"),
                           errors="coerce")

    df["tar_post_ebitda_margin"] = safe_ratio(pt_eb, pt_rev)
    df["tar_post_roa"]           = safe_ratio(pt_pat, pt_ta)
    df["ln_tar_post_size"]       = np.where(pt_ta > 0, np.log(pt_ta), np.nan)
    df["tar_post_leverage"]      = safe_ratio(pt_ta - pt_eq, pt_ta, min_den=1e-6)

    pa_rev = pd.to_numeric(safe_get(df, "post_deal_acq_rev_rev_1st_avail_yr"),
                           errors="coerce")
    pa_eb  = pd.to_numeric(safe_get(df, "post_deal_acq_ebitda_1st_avail_yr"),
                           errors="coerce")
    pa_ta  = pd.to_numeric(safe_get(df, "post_deal_acq_ta_1st_avail_yr"),
                           errors="coerce")
    pa_pat = pd.to_numeric(safe_get(df, "post_deal_acq_pat_1st_avail_yr"),
                           errors="coerce")
    pa_eq  = pd.to_numeric(
        safe_get(df, "post_deal_acq_shareholder_funds_1st_avail_yr"),
        errors="coerce")

    df["acq_post_ebitda_margin"] = safe_ratio(pa_eb, pa_rev)
    df["acq_post_roa"]           = safe_ratio(pa_pat, pa_ta)
    df["ln_acq_post_size"]       = np.where(pa_ta > 0, np.log(pa_ta), np.nan)
    df["acq_post_leverage"]      = safe_ratio(pa_ta - pa_eq, pa_ta, min_den=1e-6)

    win_cols = [c for c in [
        "tar_ebitda_margin", "tar_leverage", "tar_rev_growth", "tar_roa",
        "acq_ebitda_margin", "acq_leverage", "acq_roa",
        "tar_post_ebitda_margin", "tar_post_roa", "tar_post_leverage",
        "acq_post_ebitda_margin", "acq_post_roa", "acq_post_leverage"
    ] if c in df.columns and c not in NO_WINSOR]
    for c in win_cols:
        b = (df[c].min(), df[c].quantile(.01), df[c].median(),
             df[c].quantile(.99), df[c].max())
        df[c] = winsorise(df[c], lo=WINSOR_LO, hi=WINSOR_HI)
        a = (df[c].min(), df[c].quantile(.01), df[c].median(),
             df[c].quantile(.99), df[c].max())
        log(f"    [winsor] {c:<24} before min={b[0]:.3f} p1={b[1]:.3f} "
            f"p50={b[2]:.3f} p99={b[3]:.3f} max={b[4]:.3f}")
        log(f"             after  min={a[0]:.3f} p1={a[1]:.3f} "
            f"p50={a[2]:.3f} p99={a[3]:.3f} max={a[4]:.3f}")
    log(f"  winsorised {len(win_cols)} cols @ [{WINSOR_LO},{WINSOR_HI}] "
        f"(已排除 {len(NO_WINSOR & set(df.columns))} 个 ln/比率类)")
    for c in ["tar_ebitda_margin", "tar_leverage", "tar_rev_growth", "tar_roa",
              "acq_ebitda_margin", "acq_leverage", "acq_roa", "ln_rel_size",
              "tar_post_leverage", "acq_post_leverage"]:
        if c in df.columns:
            log(f"    {c:<24} valid={df[c].notna().sum():,} "
                f"({pct(df[c].notna().sum(), len(df))})")
    add_step("financial_ratios", len(df))

    # ── STEP 6 firm age ──
    banner("STEP 6 — firm age")
    dy      = pd.to_numeric(safe_get(df, "deal_year"), errors="coerce")
    tar_inc = pd.to_numeric(safe_get(df, "tar_incorp_d_year"), errors="coerce")
    acq_inc = pd.to_numeric(safe_get(df, "acq_incorp_d_year"), errors="coerce")
    df["target_age"]   = dy - tar_inc
    df["acquirer_age"] = dy - acq_inc
    df.loc[df["target_age"]   < 0, "target_age"]   = np.nan
    df.loc[df["acquirer_age"] < 0, "acquirer_age"] = np.nan
    log(f"  target_age   valid={df['target_age'].notna().sum():,}   "
        f"({pct(df['target_age'].notna().sum(), len(df))})")
    log(f"  acquirer_age valid={df['acquirer_age'].notna().sum():,}   "
        f"({pct(df['acquirer_age'].notna().sum(), len(df))})")
    add_step("firm_age", len(df))

    # ── STEP 7 DaysToCompletion ──
    banner("STEP 7 — DaysToCompletion & ln_days")
    ad = parse_date_robust(safe_get(df, "announced_d", warn=False))
    cd = parse_date_robust(safe_get(df, "completed_d", warn=False))
    ac = parse_date_robust(safe_get(df, "assumed_comp_d", warn=False))
    log(f"  announced_d    parsed OK: {ad.notna().sum():,} / {len(ad):,}")
    log(f"  completed_d    parsed OK: {cd.notna().sum():,} / {len(cd):,}")
    log(f"  assumed_comp_d parsed OK: {ac.notna().sum():,} / {len(ac):,}")

    off = (pd.to_numeric(safe_get(df, "assumed_comp_d", warn=False), errors="coerce")
           - pd.to_numeric(safe_get(df, "announced_d", warn=False), errors="coerce"))
    mech = off.between(700, 760) & cd.isna()
    log(f"  730-day placeholder -> NaT: {int(mech.sum()):,}")
    ac.loc[mech] = pd.NaT
    zero = (cd.notna() & ad.notna() & (cd == ad))
    log(f"  zero-duration placeholder -> NaT: {int(zero.sum()):,}")
    cd.loc[zero] = pd.NaT

    end = cd.fillna(ac)
    df["DaysToCompletion"] = (end - ad).dt.days
    df.loc[df["DaysToCompletion"] <= 0, "DaysToCompletion"] = np.nan
    df["ln_days"] = np.where(df["DaysToCompletion"].notna(),
                             np.log(df["DaysToCompletion"]), np.nan)
    log(f"  DaysToCompletion valid={df['DaysToCompletion'].notna().sum():,} "
        f"({pct(df['DaysToCompletion'].notna().sum(), len(df))})  "
        f"p50={df['DaysToCompletion'].median():.0f} "
        f"p90={df['DaysToCompletion'].quantile(.9):.0f}")
    log(f"  ln_days         valid={df['ln_days'].notna().sum():,}")
    add_step("days_to_completion", len(df))

    # ── STEP 8 legal origin ──
    banner("STEP 8 — legal origin / classification distance")
    tar_cc = safe_get(df, "tar_country_code").astype(str).str.upper()
    acq_cc = safe_get(df, "acq_country_code").astype(str).str.upper()
    df["legal_origin_tar"] = tar_cc.map(LEGAL_ORIGIN)
    df["legal_origin_acq"] = acq_cc.map(LEGAL_ORIGIN)
    df["common_law_tar"] = (df["legal_origin_tar"] == "English").astype("Int64")
    df["common_law_acq"] = (df["legal_origin_acq"] == "English").astype("Int64")
    df["classification_distance_tar"] = tar_cc.map(CLASSIFICATION_DIST)
    df["classification_distance_acq"] = acq_cc.map(CLASSIFICATION_DIST)
    naics_col = next((c for c in ["tar_primary_naics_code",
                                  "tar_primary_nace_code",
                                  "tar_primary_sic_code"] if c in df.columns), None)
    if naics_col:
        df["national_class"] = safe_get(df, naics_col).notna().astype("Int64")
        log(f"  national_class 基于 {naics_col}")
    else:
        df["national_class"] = pd.NA
        log(f"  [WARN] 无 naics/nace/sic 列")
    log(f"  common_law_tar=1: "
        f"{int(df['common_law_tar'].sum(skipna=True)):,} "
        f"({pct(df['common_law_tar'].sum(skipna=True), len(df))})")
    log(f"  common_law_acq=1: "
        f"{int(df['common_law_acq'].sum(skipna=True)):,} "
        f"({pct(df['common_law_acq'].sum(skipna=True), len(df))})")
    add_step("legal_origin", len(df))

    # ── STEP 9 vendor ──
    banner("STEP 9 — vendor dummies（数据已在 06a 并入）")
    if "vendor_type" not in df.columns:
        raise KeyError(f"[FATAL] df 无 vendor_type。06a ven 列: {ven_in_06a}")
    vt = (df["vendor_type"].astype(str).str.strip().replace("nan", np.nan)
          .replace(_VENDOR_OLD2NEW))
    actual_types = sorted(vt.dropna().unique().tolist())
    log(f"  vendor_type 非空: {vt.notna().sum():,} / {len(df):,} "
        f"({pct(vt.notna().sum(), len(df))})")
    log(f"  取值({len(actual_types)}): {actual_types}")
    mapped = set(_VENDOR_OLD2NEW.get(t, t) for t in actual_types)
    for t in actual_types:
        t_norm = _VENDOR_OLD2NEW.get(t, t)
        col = f"ven_is_{t_norm}"
        if col not in df.columns:
            df[col] = (vt == t_norm).astype("Int64")
            df.loc[vt.isna(), col] = pd.NA
    if "ven_unmapped" not in df.columns:
        df["ven_unmapped"] = (~vt.isna() & ~vt.isin(mapped)).astype("Int64")
        df.loc[vt.isna(), "ven_unmapped"] = pd.NA
    for c in [x for x in df.columns if x.startswith("ven_is_")]:
        log(f"    {c:<26} =1: {int(df[c].sum(skipna=True)):,} "
            f"({pct(df[c].sum(skipna=True), len(df))})")
    add_step("vendor_dummies", len(df))

    # ── STEP 10 ln_deal_value ──
    banner("STEP 10 — ln_deal_value")
    dvv = pd.to_numeric(safe_get(df, "deal_value"), errors="coerce")
    df["ln_deal_value"] = np.where(dvv > 0, np.log(dvv), np.nan)
    log(f"  ln_deal_value valid={df['ln_deal_value'].notna().sum():,} "
        f"({pct(df['ln_deal_value'].notna().sum(), len(df))})  "
        f"p1={df['ln_deal_value'].quantile(.01):.2f} "
        f"p50={df['ln_deal_value'].median():.2f} "
        f"p99={df['ln_deal_value'].quantile(.99):.2f}")
    add_step("ln_deal_value", len(df))

    # ── STEP 11 Table-6 别名 ──
    banner("STEP 11 — Table-6 别名变量")
    df["tar_ta_for_t6"]  = tar_ta
    df["tar_rev_for_t6"] = tar_rev
    df["tar_mktcap_for_t6"] = pd.to_numeric(
        safe_get(df, "deal_equity_value"), errors="coerce")
    df["tar_ev_for_t6"] = pd.to_numeric(
        safe_get(df, "deal_modelled_enterprise_value"), errors="coerce")
    df["tar_mb_for_t6"] = np.where((tar_eq != 0) & tar_eq.notna(),
                                   df["tar_mktcap_for_t6"] / tar_eq, np.nan)
    for c in ["tar_ta_for_t6", "tar_rev_for_t6", "tar_mktcap_for_t6",
              "tar_ev_for_t6", "tar_mb_for_t6"]:
        log(f"    {c:<22} non-null={df[c].notna().sum():,} "
            f"({pct(df[c].notna().sum(), len(df))})")
    add_step("table6_aliases", len(df))

    # ── STEP 12 comment dummy ──
    banner("STEP 12 — has_comment")
    if "has_lm_litigious" in df.columns:
        df["has_comment"] = df["has_lm_litigious"].notna().astype("Int64")
        log(f"  has_comment=1: "
            f"{int(df['has_comment'].sum(skipna=True)):,} "
            f"({pct(df['has_comment'].sum(skipna=True), len(df))})")
    else:
        df["has_comment"] = pd.NA
        log(f"  [WARN] has_lm_litigious 缺失")

    # ── STEP 13 SIC-peer 占位 ──
    banner("STEP 13 — SIC-peer 占位 (NaN)")
    for e in ["ln_sic_rev", "ln_sic_ebitda", "ln_sic_ebit",
              "cross_industry_alt", "peer_overlap_frac",
              "ind_text_diversity", "ind_text_diversity_w",
              "sic_coverage_rate", "acq_tar_similarity"]:
        if e not in df.columns:
            df[e] = np.nan

    # ── STEP 14 白名单核对 ──
    banner("STEP 14 — 白名单核对")
    COLS_TO_SAVE = build_whitelist(df)

    # ── STEP 15 Stata 导出 ──
    banner("STEP 15 — 导出 Stata .dta")
    df_save = df[[c for c in COLS_TO_SAVE if c in df.columns]].copy()

    # 15a: 列名 -> Stata 合法名 (_year_source -> s_year_source 等)
    rename_map = build_rename_map(df_save.columns)
    n_renamed = sum(1 for k, v in rename_map.items() if k != v)
    df_save = df_save.rename(columns=rename_map)
    log(f"  [STATA NAMES] 重命名 {n_renamed} 列（首字符非字母/含非法字符）")
    for k, v in list(rename_map.items())[:6] if n_renamed else []:
        if k != v:
            log(f"      {k}  ->  {v}")
    if n_renamed > 6:
        log(f"      ... 共 {n_renamed} 列, 详见日志")

    # 15b: value_labels 的 key 必须与写出时的变量名一致
    value_labels = {}
    for col in ["common_law_tar", "common_law_acq", "national_class"]:
        if col in rename_map:
            df_save[rename_map[col]] = df_save[rename_map[col]].astype("float64")
            value_labels[rename_map[col]] = {0.0: "No / Civil", 1.0: "Yes / Common"}
    for col in df_save.columns:
        if str(df_save[col].dtype) in ("Int64", "Int32", "boolean"):
            df_save[col] = df_save[col].astype("float64")

    pyreadstat.write_dta(df_save, OUT_DTA, version=15,
                         variable_value_labels=value_labels)
    log(f"  dta: {OUT_DTA}  shape={df_save.shape}")

    # ── STEP 16 保存 ──
    banner("STEP 16 — 保存 CSV + 汇总")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"  csv: {OUT_CSV}  rows={len(df):,}  "
        f"{os.path.getsize(OUT_CSV)/1024/1024:.1f} MB")
    pd.DataFrame(steps).to_csv(OUT_STEPS, index=False, encoding="utf-8-sig")

    banner("FINAL — 关键变量覆盖率")
    key = ["ln_rev_x", "ln_rev_y", "ln_ebitda_x", "ln_ebitda_y",
           "ln_ebit_x", "ln_ebit_y", "ln_pbt_x", "ln_pbt_y",
           "ln_ta_x", "ln_ta_y", "ln_eq_x", "ln_eq_y",
           "ln_text_rev", "ln_text_ebitda", "ln_text_ebit",
           "text_peer_sim_mean",
           "ln_days", "ln_deal_value", "ln_tar_size", "ln_acq_size",
           "ln_rel_size", "tar_ebitda_margin", "acq_ebitda_margin",
           "tar_post_leverage", "acq_post_leverage",
           "target_age", "acquirer_age", "common_law_tar",
           "tar_ta_for_t6", "tar_rev_for_t6"]
    for c in key:
        if c in df.columns:
            nn = df[c].notna().sum()
            log(f"  {c:<24} {nn:>10,}  ({pct(nn, len(df))})")
        else:
            log(f"  {c:<24} {'MISSING':>10}")
    log(f"\n  done in {time.time()-_T0:.1f}s")


def build_whitelist(df):
    base = [
        DEAL, "_row_id", "deal_year",
        "tar_sic3", "tar_sic2", "acq_sic3",
        "legal_origin_tar", "legal_origin_acq",
        "reg_body_count", "reg_country_count", "reg_antitrust", "reg_securities",
        "reg_state_assets", "reg_foreign_invest", "reg_financial", "reg_defense_tech",
        "reg_cross_national", "reg_common_law", "reg_civil_law", "reg_mixed_legal",
        "regulatory_bodies", "regulatory_countries",
        "total_timeline_days", "has_rumour", "has_target_reject",
        "has_phase2_investigation", "has_goshop", "num_unique_reg",
        "comment_char_length", "comment_wordcount", "sentence_event_count",
        "lm_pos_count", "lm_neg_count", "lm_uncertain_count", "lm_litigious_count",
        "lm_strongmodal_count", "lm_weakmodal_count", "lm_constrain_count",
        "lm_pos_density", "lm_neg_density", "lm_uncertain_density",
        "lm_litigious_density", "lm_strongmodal_density",
        "lm_weakmodal_density", "lm_constrain_density", "lm_net_sentiment",
        "num_Stock_Exchange", "num_Website", "num_Company_Press_Release",
        "num_Electronic_Publication", "num_Advisor_Submission",
        "num_Miscellaneous", "num_source_other", "total_info_source",
        "pre_deal_tar_rev_rev_last_avail_yr", "pre_deal_tar_ebitda_last_avail_yr",
        "pre_deal_tar_ebit_last_avail_yr", "pre_deal_tar_pbt_last_avail_yr",
        "pre_deal_tar_pat_last_avail_yr", "pre_deal_tar_np_last_avail_yr",
        "pre_deal_tar_ta_last_avail_yr", "pre_deal_tar_na_last_avail_yr",
        "pre_deal_tar_eq_last_avail_yr", "pre_deal_tar_current_liabilities_last_avail_yr",
        "pre_deal_acq_rev_rev_last_avail_yr", "pre_deal_acq_ebitda_last_avail_yr",
        "pre_deal_acq_pat_last_avail_yr", "pre_deal_acq_ta_last_avail_yr",
        "pre_deal_acq_eq_last_avail_yr",
        "post_deal_tar_rev_rev_1st_avail_yr", "post_deal_tar_ebitda_1st_avail_yr",
        "post_deal_tar_ebit_1st_avail_yr", "post_deal_tar_pbt_1st_avail_yr",
        "post_deal_tar_pat_1st_avail_yr", "post_deal_tar_np_1st_avail_yr",
        "post_deal_tar_ta_1st_avail_yr", "post_deal_tar_na_1st_avail_yr",
        "post_deal_tar_current_liabilities_1st_avail_yr",
        "post_deal_acq_rev_rev_1st_avail_yr", "post_deal_acq_ebitda_1st_avail_yr",
        "post_deal_acq_ebit_1st_avail_yr", "post_deal_acq_pbt_1st_avail_yr",
        "post_deal_acq_pat_1st_avail_yr", "post_deal_acq_np_1st_avail_yr",
        "post_deal_acq_ta_1st_avail_yr", "post_deal_acq_na_1st_avail_yr",
        "post_deal_acq_shareholder_funds_1st_avail_yr",
        "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly",
        "pre_pbt_mul_ly", "pre_ta_mul_ly", "pre_eq_mul_ly",
        "pre_pat_mul_ly", "pre_np_mul_ly", "pre_na_mul_ly",
        "pre_cl_mul_ly", "pre_cap_mul_ly",
        "post_rev_mul_fy", "post_ebitda_mul_fy", "post_ebit_mul_fy",
        "ln_rev_x", "ln_ebitda_x", "ln_ebit_x", "ln_pbt_x",
        "ln_ta_x", "ln_eq_x", "ln_pat_x", "ln_np_x",
        "ln_na_x", "ln_cl_x", "ln_cap_x",
        "ln_rev_y", "ln_ebitda_y", "ln_ebit_y", "ln_pbt_y",
        "ln_ta_y", "ln_eq_y", "ln_pat_y", "ln_np_y",
        "ln_na_y", "ln_cl_y", "ln_cap_y",
        "ln_post_rev_x", "ln_post_ebitda_x", "ln_post_ebit_x",
        "ln_post_rev_y", "ln_post_ebitda_y", "ln_post_ebit_y",
        "ln_text_rev", "ln_text_ebitda", "ln_text_ebit",
        "ln_text_pbt", "ln_text_ta", "ln_text_eq",
        "ln_sic_rev", "ln_sic_ebitda", "ln_sic_ebit",
        "DaysToCompletion", "ln_days",
        "rumor_jfe", "has_rumor_status", "rumor_lead_days", "rumor_lead_yr",
        "completed_flag", "completed_strict",
        "tar_incorp_d_year", "acq_incorp_d_year", "target_age", "acquirer_age",
        "cross_industry", "cross_industry_alt", "ln_deal_value", "cross_border",
        "deal_pay_method", "deal_pay_method_all", "deal_pay_method_n",
        "pay_has_cash", "pay_has_shares", "pay_has_debt", "pay_has_other",
        "pay_n_class", "pay_pure_cash", "pay_pure_shares", "pay_pure_debt",
        "pay_pure_other", "pay_mix", "pay_unknown", "pay_mix_cash_shares",
        "pay_mix_with_shares", "pay_mix_with_cash", "pay_mix_with_debt", "pay_class",
        "deal_struct_all", "deal_struct_n", "deal_fin_all", "deal_fin_n",
        "deal_type_all", "deal_type_n", "deal_value", "deal_status",
        "ln_tar_size", "tar_ebitda_margin", "tar_leverage", "tar_rev_growth",
        "tar_roa", "tar_post_ebitda_margin", "tar_post_roa",
        "ln_tar_post_size", "tar_post_leverage",
        "ln_acq_size", "acq_ebitda_margin", "acq_leverage", "acq_roa",
        "ln_rel_size", "ln_acq_post_size", "acq_post_ebitda_margin",
        "acq_post_roa", "acq_post_leverage",
        "tar_country_code", "acq_country_code",
        #JFE 关键变量
        "stock_traded_gdp_tar", "stock_traded_gdp_acq",
        
        "common_law_tar", "common_law_acq",
        "tar_ta_for_t6", "tar_rev_for_t6", "tar_mktcap_for_t6",
        "tar_ev_for_t6", "tar_mb_for_t6",
        "tar_listed", "acq_listed",
        "sic_coverage_rate", "acq_tar_similarity",
        "classification_distance_tar", "classification_distance_acq",
        "tar_name", "acq_name", "ven_name",
        "text_peer_sim_mean", "n_peers_rev", "n_peers_ebitda", "n_peers_ebit",
        "n_peers_pbt", "n_peers_ta", "n_peers_eq",
        "bench_rev", "bench_ebitda", "bench_ebit", "bench_pbt", "bench_ta", "bench_eq",
        "has_comment",
        
        # ---- 用户特别指定：股权（JFE 关键变量）----
        "stake_acq_pct", "stake_init_pct",
        "deal_n_tar","deal_n_acq",


    ]

    # 动态追加: 白名单之外的新列（去重，保序）
    seen = set(base)
    extra = [c for c in df.columns
             if c not in seen and c != DEAL]
    seen.update(extra)
    allc = base + extra

    records = []
    for c in allc:
        ok = c in df.columns
        nn = int(df[c].notna().sum()) if ok else 0
        records.append({"variable": c, "exist": ok, "non_null": nn,
                        "coverage_pct": nn / len(df) * 100 if ok else 0,
                        "note": "available" if ok else "MISSING"})
        tag = "[EXIST]" if ok else "[MISS] "
        log(f"  {tag} {c:<48} {nn:>10,}  ({pct(nn, len(df))})")
    pd.DataFrame(records).to_csv(OUT_VAR_CHECK, index=False, encoding="utf-8-sig")
    log(f"  var_check: {OUT_VAR_CHECK}")
    return allc


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL ERROR]\n" + traceback.format_exc())
        flush_log()
        raise
    finally:
        flush_log()
