# -*- coding: utf-8 -*-
"""
v48_lambda_bootstrap_final_v3.py
================================
修复 v2 的两个问题：
1. simulate_4eq_4 的 Z 必须随 bootstrap 副本变化，不可使用主样本 Z_res。
2. 聚类粒度从 (国家, SIC首位, 年份) 改为 (国家, 年份)：
   原 71805 个簇几乎等于逐笔，不构成真正的簇抽样。

说明：
- 结构方程的 Xb1/Xb2/Xb3/Xb4 已在 fit_nuisance 中按副本重算，
  本版把它们显式回写到 A，避免与主样本混淆。
- 主样本轨迹仍用主样本 Z_res，不受 bootstrap 影响。
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 100)

_T0 = time.time()
MASTER_SEED = 20240924
DATA_PATH = r"d:\MA\data\merged\07_full_sample-JFE.csv"
OUT_DIR = r"D:\MA\output"
OUT_TRAJECTORY = os.path.join(OUT_DIR, "lambda_trajectory.csv")
OUT_PLOT = os.path.join(OUT_DIR, "kappa_lambda_bounds.png")

SET_A = [
    ("tar_age",          "target_age",            "log1p", True),
    ("acq_age",          "acquirer_age",          "log1p", True),
    ("deal_n_acq",       "deal_n_acq",            "asis",   True),
    ("Toehold",          "stake_init_pct",        "pct",    True),
    ("RelSize",          "ln_rel_size",           "asis",   True),
    ("Local",            "cross_border",          "negate", False),
    ("SameInd",          "cross_industry",        "negate", False),
    ("TarEbitdaMargin",  "tar_ebitda_margin",     "asis",   True),
    ("TarLeverage",      "tar_leverage",          "asis",   True),
    ("TarROA",           "tar_roa",               "asis",   True),
    ("AcqEbitdaMargin",  "acq_ebitda_margin",     "asis",   True),
    ("AcqLeverage",      "acq_leverage",          "asis",   True),
    ("AcqROA",           "acq_roa",               "asis",   True),
    ("DealValue",        "ln_deal_value",         "asis",   False),
    ("TotalInfoSrc",     "total_info_source",     "log1p",  True),
    ("TarStockGDP",      "stock_traded_gdp_tar",  "asis",   True),
]

TEXT_POS = "lm_pos_density"
TEXT_NEG = "lm_neg_density"
TEXT_UNC = "lm_uncertain_density"


def _to_num(arg, col=None):
    if col is not None:
        arg = arg[col]
    if isinstance(arg, pd.Series):
        arr = pd.to_numeric(arg, errors="coerce").values
    else:
        arr = np.asarray(arg, dtype=float)
    return arr.ravel() if arr.ndim != 1 else arr


def _series(df, raw):
    if raw not in df.columns:
        raise KeyError("缺少字段：%s" % raw)
    return pd.to_numeric(df[raw], errors="coerce")


def _log1p(s):
    return np.log1p(s.clip(lower=0))


def _pct(s):
    return s / 100.0


def _negate(s):
    return 1.0 - s


def _corr(x, y):
    x, y = _to_num(x), _to_num(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xo, yo = x[m] - x[m].mean(), y[m] - y[m].mean()
    sx, sy = xo.std(), yo.std()
    if sx <= 0 or sy <= 0:
        return np.nan
    return float((xo * yo).mean() / (sx * sy))


def _cov(x, y):
    x, y = _to_num(x), _to_num(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return 0.0
    xo, yo = x[m] - x[m].mean(), y[m] - y[m].mean()
    if not (xo.std() > 0 and yo.std() > 0):
        return 0.0
    return float((xo * yo).mean())


def _skew(x):
    x = _to_num(x)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 0.0
    x = x - x.mean()
    sd = x.std()
    if sd <= 0:
        return 0.0
    return float(((x / sd) ** 3).mean())


def _rng(stream):
    return np.random.RandomState(MASTER_SEED + stream)


def _ols(y, X):
    y = _to_num(y)
    Xm = np.asarray(X, dtype=float).reshape(-1, X.shape[-1])
    ok = np.isfinite(y) & np.isfinite(Xm).all(axis=1)
    k = Xm.shape[1]
    if ok.sum() <= k:
        return (np.zeros(k), np.ones(k), np.eye(k),
                ["b%d" % i for i in range(k)])
    yo, Xo = y[ok], Xm[ok]
    coef, resid, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    resid = yo - Xo @ coef
    dof = max(Xo.shape[0] - k, 1)
    sig2 = max(float(np.var(resid, ddof=0)), 0.0) * Xo.shape[0] / dof
    try:
        vcv = float(sig2) * np.linalg.inv(Xo.T @ Xo)
    except np.linalg.LinAlgError:
        vcv = np.linalg.pinv(Xo.T @ Xo) * float(sig2)
    se = np.sqrt(np.maximum(np.diag(vcv), 0.0))
    return coef.ravel(), se.ravel(), vcv, ["b%d" % i for i in range(len(coef))]


def _probit_coef(y, X):
    try:
        from statsmodels.discrete.discrete_model import Probit
        from statsmodels.api import add_constant
        y = np.asarray(y).ravel().astype(float)
        Xm = np.asarray(X).reshape(-1, X.shape[-1]).astype(float)
        ok = np.isfinite(y) & np.isfinite(Xm).all(axis=1)
        if ok.sum() <= Xm.shape[1] + 2:
            return np.zeros(Xm.shape[1])
        res = Probit(y[ok], add_constant(Xm[ok]), cov_type="HC1").fit(disp=False)
        return res.params[1:].astype(float)
    except Exception:
        return np.zeros(X.shape[1])


def build_errors_0304(n, lam, rng, sigma=1.0):
    F1 = rng.standard_normal(n)
    F2 = rng.standard_normal(n)
    u1 = rng.standard_normal(n)
    u2 = rng.standard_normal(n)
    u3 = rng.standard_normal(n)
    u4 = rng.standard_normal(n)

    a1 = np.sqrt(0.499)
    denom = np.sqrt(max(1.0 - 0.499, 1e-9))
    b1 = lam / denom
    c2 = np.sqrt(1.0 - 0.499)

    s1 = np.sqrt(max(1.0 - a1 ** 2 - b1 ** 2, 1e-9))
    s2 = np.sqrt(max(1.0 - 0.499, 1e-9))
    s3 = np.sqrt(max(1.0 - c2 ** 2, 1e-9))

    e1 = (a1 * F1 + b1 * F2 + s1 * u1) * sigma
    e2 = (np.sqrt(0.499) * F1 + s2 * u2) * sigma
    e3 = (c2 * F2 + s3 * u3) * sigma
    e4 = u4 * sigma
    return np.column_stack([e1, e2, e3, e4])


def load_data(path=DATA_PATH):
    print("[载入] %s" % path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    print("[载入] 当前样本行数 = %d" % len(df))
    if "deal_num" in df.columns:
        df = df.sort_values("deal_num", kind="mergesort").reset_index(drop=True)
        print("[锁序] 按 deal_num 升序，%d 行" % len(df))
    req = ["rumor_jfe", "tar_country_code", "tar_primary_sic_code",
           "deal_year", "ln_rev_y",
           TEXT_POS, TEXT_NEG, TEXT_UNC,
           "completed_strict"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError("数据文件缺少必要字段：%s" % miss)
    return df


def build_controls(df):
    controls, missing = [], []
    for name, raw, trans, mk in SET_A:
        s = _series(df, raw).copy()
        if trans == "log1p":
            s = _log1p(s)
        elif trans == "pct":
            s = _pct(s)
        elif trans == "negate":
            s = _negate(s)
        med = float(s.median())
        df["c_%s" % name] = s.fillna(med)
        controls.append("c_%s" % name)
        if mk:
            df["c_%s_miss" % name] = s.isna().astype(float)
            missing.append("c_%s_miss" % name)
        print("[ctrl] %-17s raw=%-22s 中位数=%9.4f 缺失=%6d (%5.1f%%)"
              % (name, raw, med, int(s.isna().sum()), 100 * s.isna().mean()))
    names = controls + missing
    X = df[names].values.astype(float)
    print("[ctrl] K_base = %d（控制变量 %d + 缺失指示 %d）"
          % (len(names), len(controls), len(missing)))
    return names, X, len(controls)


def _year_series(df):
    s = df["deal_year"]
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64")
    return pd.to_datetime(s, errors="coerce").dt.year.astype("Int64")


def build_leak_iv(df, ALL_base):
    print("[IV] 构建 Leak-IV：目标国 × 行业首位 × 年份，剔除交易自身")
    country = df["tar_country_code"].astype(str).fillna("0")
    sic = df["tar_primary_sic_code"].astype(str).fillna("0")
    year = _year_series(df)
    sic1 = sic.str[0].replace({"": "0", "nan": "0", "None": "0"})
    sic1 = sic1.where(sic1.str.isdigit(), "0")

    R = _to_num(df, "rumor_jfe")
    key = pd.DataFrame({
        "key": country + "|" + sic1,
        "year": year.astype(int),
        "R": pd.Series(R, index=df.index),
    }).dropna(subset=["key", "year", "R"])

    agg = key.groupby(["key", "year"], as_index=False).agg(
        mean_R=("R", "mean"), n=("R", "count"))
    m = key.merge(agg, on=["key", "year"], how="left")
    Z = (m["mean_R"] * m["n"] - m["R"]) / (m["n"] - 1)
    Z = Z.fillna(m["mean_R"]).values.astype(float)

    Xb = df[ALL_base].values.astype(float)
    fe_blocks = []
    for c in ["tar_country_code", "tar_primary_sic_code", "deal_year"]:
        d = pd.get_dummies(df[c].astype(str), prefix="iv", drop_first=True)
        keep = d.columns[d.mean() > 0.001]
        fe_blocks.append(d[keep].values.astype(float))

    Q = np.column_stack([Xb] + fe_blocks)
    ok = np.isfinite(Z) & np.isfinite(Q).all(axis=1)
    Qo, Zo = Q[ok], Z[ok]
    coef, *_ = np.linalg.lstsq(Qo, Zo, rcond=None)
    Z_res = np.zeros_like(Z)
    Z_res[ok] = Zo - Qo @ coef
    sd = float(np.std(Z_res[ok]))
    if sd > 0:
        Z_res[ok] = (Z_res[ok] - Z_res[ok].mean()) / sd
    print("[IV] Leak-IV 残差化完成：Z_res sd=%.4f" % sd)
    X_aug = np.column_stack([Xb, Z_res[:, None], (Z_res ** 2)[:, None]])
    print("[IV] X_aug = (%d, %d)" % X_aug.shape)
    return ALL_base, X_aug, Z_res


def build_text(df):
    pos = pd.to_numeric(df[TEXT_POS], errors="coerce").values.astype(float)
    neg = pd.to_numeric(df[TEXT_NEG], errors="coerce").values.astype(float)
    unc = pd.to_numeric(df[TEXT_UNC], errors="coerce").values.astype(float)
    M = pos - neg - unc
    used = np.isfinite(pos) & np.isfinite(neg) & np.isfinite(unc)
    print("[TEXT] M = pos - neg - uncertain")
    print("[TEXT] M 可用 = %d / %d (%.2f%%)"
          % (used.sum(), len(df), 100 * used.mean()))
    return M


def text_pi_diagnostics(A):
    df, M = A["df"], A["M"]
    lm_cols = [
        "lm_pos_density",
        "lm_neg_density",
        "lm_uncertain_density",
        "lm_litigious_density",
        "lm_weakmodal_density",
        "lm_strongmodal_density",
        "lm_constrain_density",
    ]
    missing = [c for c in lm_cols if c not in df.columns]
    if missing:
        raise KeyError("缺少 LM 文本字段：%s" % missing)

    X = df[lm_cols].apply(pd.to_numeric, errors="coerce")
    S = X.dropna()
    cols = list(S.columns)

    triples = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            for k in range(j + 1, len(cols)):
                a, b, c = cols[i], cols[j], cols[k]
                rab = S[a].corr(S[b], method="spearman")
                rac = S[a].corr(S[c], method="spearman")
                rbc = S[b].corr(S[c], method="spearman")
                if abs(rbc) > 1e-8:
                    triples.append(rab * rac / rbc)

    pi_vals = np.asarray([x for x in triples if np.isfinite(x)])
    if pi_vals.size == 0:
        pi_vals = np.array([np.nan])

    print("=" * 88)
    print("[TEXT-PI] Spearman π 诊断")
    print("=" * 88)
    print("[TEXT-PI] 七列均非缺失样本：%d / %d (%.2f%%)"
          % (len(S), len(df), 100 * len(S) / len(df)))
    print("[TEXT-PI] 有效三元组数           = %d" % len(pi_vals))
    print("[TEXT-PI] pi 中位数              = %.6f" % np.nanmedian(pi_vals))
    print("[TEXT-PI] pi 均值                = %.6f" % np.nanmean(pi_vals))
    print("[TEXT-PI] pi 标准差              = %.6f" % np.nanstd(pi_vals))

    q025 = np.nanquantile(pi_vals, 0.025)
    q975 = np.nanquantile(pi_vals, 0.975)
    print("[TEXT-PI] pi2 [2.5%%, 97.5%%]    = [%.6f, %.6f]" % (q025, q975))
    print("[TEXT-PI] |pi2| > 1 的比例       = %.4f%%"
          % (100.0 * (np.abs(pi_vals) > 1).mean()))

    print("[TEXT-PI] m_R（来自 nuisance）   = %.6f" % float(A["m_R"]))
    print("[TEXT-PI] m_P（来自 nuisance）   = %.6f" % float(A["m_P"]))
    print("[TEXT-PI] tilde_lambda = m_R*m_P = %.6f" % float(A["lambda_text"]))
    print("[TEXT-PI] pi^2 估计不稳定，lambda 采用 m_R*m_P，不除以 pi^2。")

    return {
        "pi_median": float(np.nanmedian(pi_vals)),
        "pi_mean": float(np.nanmean(pi_vals)),
        "pi_sd": float(np.nanstd(pi_vals)),
        "n_triples": int(len(pi_vals)),
        "tilde_lambda": float(A["lambda_text"]),
        "m_R": float(A["m_R"]),
        "m_P": float(A["m_P"]),
    }


def estimate_pi(df):
    return float(_to_num(df, "completed_strict").mean())


def build_B(df):
    return np.isfinite(_to_num(df, "ln_rev_y")).astype(float)


def fit_nuisance(A):
    X = A["X"]
    R = A["R"]
    B_var = A["B_var"]
    logP = A["logP"]
    M = A["M"]

    print("[nuisance] 强制重算：当前样本行数 = %d" % len(X))
    X1 = np.column_stack([np.ones(len(X)), X])
    A["Xf"] = X1
    print("[nuisance] X1 维度：%d 列，shape = %s"
          % (X1.shape[1], X1.shape))
    print("[nuisance] D 字段 = %s，覆盖率 = %.4f (%d/%d)"
          % (A["D_field"], A["D"].mean(),
             int(A["D"].sum()), len(X)))
    print("[nuisance] D.mean() 用作重算证据：%.6f" % A["D"].mean())

    A["bR"] = _probit_coef(R, X1)

    Xiv = np.column_stack([X1, A["Z_iv"][:, None]])
    phi_all = _ols(R, Xiv)
    A["phi_iv"] = float(phi_all[0][-1])
    A["phi_se"] = max(float(phi_all[1][-1]), 1e-6)
    A["F_first"] = float(A["phi_iv"] ** 2 / max(A["phi_se"] ** 2, 1e-12))

    X1R = np.column_stack([X1, R[:, None]])
    bD_coef, bD_se, bD_vcv, _ = _ols(A["D"], X1R)
    if len(bD_coef) != X1.shape[1] + 1:
        raise ValueError("bD 未包含 R")
    A["bD"] = bD_coef
    A["gamma_fixed"] = float(bD_coef[-1])
    A["gamma_se"] = float(bD_se[-1])
    A["gamma_cov_Xb2"] = float(bD_vcv[-1, :-1] @ X1.T @ X1 @ bD_coef[:-1])
    print("DEBUG bD=", bD_coef[-3:])

    X1RB = np.column_stack([X1, R[:, None]])
    bB_coef, bB_se, _, _ = _ols(B_var, X1RB)
    if len(bB_coef) != X1.shape[1] + 1:
        raise ValueError("bB 未包含 R")
    A["bB"] = bB_coef
    A["zeta1_from_bB"] = float(bB_coef[-1])
    print("DEBUG bB=", bB_coef[-3:])

    mv = np.isfinite(logP) & (B_var > 0)
    if mv.sum() > X1.shape[1] + 10:
        Xr = np.column_stack([X1[mv], R[mv][:, None]])
        bRk, seRk, _, _ = _ols(logP[mv], Xr)
        A["kappa_hat"] = float(bRk[-1])
        A["kappa_se"] = float(seRk[-1])
        A["sigma_fixed"] = float(np.std(logP[mv] - Xr @ bRk))
        A["bR_kappa"] = bRk
    else:
        A["kappa_hat"], A["kappa_se"] = 0.0, 1.0
        A["sigma_fixed"] = 1.0
        A["bR_kappa"] = np.zeros(X1.shape[1] + 1)

    A["rho_fixed"] = 0.0

    if mv.sum() > X1.shape[1] + 10:
        A["bP"] = _ols(logP[mv], X1[mv])[0].ravel()
    else:
        A["bP"] = np.zeros(X1.shape[1], dtype=float)

    used = np.isfinite(M)
    A["m_R"] = _corr(R[used], M[used])
    mvp = np.isfinite(logP) & (B_var > 0) & np.isfinite(M)
    if mvp.sum() > 2:
        cP = logP[mvp] - np.nanmean(logP[mvp])
        A["m_P"] = _cov(cP, M[mvp]) / max(
            np.std(cP) * np.std(M[mvp]), 1e-12)
    else:
        A["m_P"] = 0.0

    A["lambda_text"] = float(A["m_R"] * A["m_P"])

    # 关键：显式写出当前样本（副本）的 Xb1..Xb4
    A["Xb1"] = X1 @ A["bR"]
    A["Xb2"] = X1 @ A["bD"][:-1] + A["gamma_fixed"] * R
    A["Xb3"] = X1 @ A["bP"]
    A["Xb4"] = X1 @ A["bB"][:-1] + A["zeta1_from_bB"] * R

    print("[nuisance] phi=%.4f (se=%.4f) | F_first=%.2f"
          % (A["phi_iv"], A["phi_se"], A["F_first"]))
    print("[nuisance] gamma(D) = %.6f (se=%.6f)"
          % (A["gamma_fixed"], A["gamma_se"]))
    print("[nuisance] zeta1(B, 简约式) = %.4f" % A["zeta1_from_bB"])
    print("[nuisance] sigma=%.4f | rho=%.4f"
          % (A["sigma_fixed"], A["rho_fixed"]))
    print("[nuisance] kappa_hat=%.4f (se=%.4f)"
          % (A["kappa_hat"], A["kappa_se"]))
    print("[nuisance] m_R=%.6f m_P=%.6f tilde_lambda=%.6f"
          % (A["m_R"], A["m_P"], A["lambda_text"]))
    return A


def simulate_4eq_4(theta, Xb1, Xb2, Xb3, Xb4, U, gamma, sigma, rho,
                   Z=None, phi=0.0, zeta1=0.0):
    kap, z1, zeta2, lam = theta
    n = Xb1.shape[0]
    l0 = float(Xb3.mean())
    E = build_errors_0304(n, lam, _rng(7), sigma=sigma)
    e1, e2, e3, e4 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]

    # 修复：Z 必须是当前样本/副本长度
    if Z is None:
        Z_use = np.zeros(n)
    else:
        Z_use = np.asarray(Z, dtype=float).reshape(-1)
        if Z_use.shape[0] != n:
            raise ValueError(
                "Z 长度 %d 与 Xb1 长度 %d 不一致，"
                "bootstrap 中必须传入副本自己的 Z_iv" % (Z_use.shape[0], n))
    linR = Xb1 + phi * Z_use + e1
    R_s = (linR > 0).astype(float)

    linD = Xb2 + gamma * R_s + e2
    D_s = (linD > 0).astype(float)

    logP_s = Xb3 + kap * R_s + sigma * e3

    linB = Xb4 + z1 * R_s + zeta2 * (logP_s - l0) + e4
    B_s = (linB > 0).astype(float)
    return R_s, D_s, logP_s, B_s


def estimate_structural_4(A, l0, gamma, sigma, rho, rng=None,
                          MC_II=5, maxfev=2000, Z=None, phi=None,
                          lambda_text=None, lambda_weight=1.0):
    if rng is None:
        rng = _rng(4)
    X = A["X"]
    R = A["R"]
    B_var = A["B_var"]
    logP = A["logP"]
    M = A["M"]

    kh = A["kappa_hat"]
    kap_lb, kap_ub = kh - 0.5, kh + 0.5
    z1_hat = float(A["bB"][-1]) if len(A["bB"]) > 0 else 0.0
    z1_se = 0.05
    z1_lb, z1_ub = z1_hat - 3 * z1_se, z1_hat + 3 * z1_se
    lam_c = lambda_text if lambda_text is not None else A["lambda_text"]
    lam_lb, lam_ub = lam_c - 0.02, lam_c + 0.02
    z2_lb, z2_ub = -2.0, 2.0

    mv = np.isfinite(logP) & (B_var > 0)
    aux_real = np.array([
        float(B_var.mean()),
        float(_corr(B_var, R)),
        float(np.nanmean(logP[mv])) if mv.sum() else 0.0,
        float(kh),
        float(_skew(logP[mv])) if mv.sum() > 10 else 0.0,
    ])

    def sim_aux(theta):
        kap, z1, zeta2, lam = theta
        auxs = []
        for _ in range(MC_II):
            U = rng.standard_normal((len(X), 6))
            _, _, logPs, Bs = simulate_4eq_4(
                theta, A["Xb1"], A["Xb2"], A["Xb3"], A["Xb4"],
                U, gamma, sigma, rho, Z=Z, phi=phi, zeta1=z1)
            mvs = np.isfinite(logPs) & (Bs > 0)
            auxs.append(np.array([
                float(np.mean(Bs)),
                float(_corr(Bs, R)),
                float(np.nanmean(logPs[mvs])) if mvs.sum() else 0.0,
                float(kh),
                float(_skew(logPs[mvs])) if mvs.sum() > 10 else 0.0,
            ]))
        return np.nanmean(auxs, axis=0)

    def penalty(theta):
        kap, z1, z2, lam = theta
        p = 0.0
        if kap < kap_lb: p += 1e6 * (kap_lb - kap) ** 2
        if kap > kap_ub: p += 1e6 * (kap - kap_ub) ** 2
        if z1 < z1_lb:   p += 1e6 * (z1_lb - z1) ** 2
        if z1 > z1_ub:   p += 1e6 * (z1 - z1_ub) ** 2
        if z2 < z2_lb:   p += 1e6 * (z2_lb - z2) ** 2
        if z2 > z2_ub:   p += 1e6 * (z2 - z2_ub) ** 2
        if lam < lam_lb: p += 1e6 * (lam_lb - lam) ** 2
        if lam > lam_ub: p += 1e6 * (lam - lam_ub) ** 2
        p += lambda_weight * (lam - lam_c) ** 2
        return p

    def obj(theta):
        aux_m = sim_aux(theta)
        f0 = float(np.sum((aux_real - aux_m) ** 2))
        return f0 + penalty(theta), f0, aux_m

    starts = [
        ("A正溢价",  0.25, 0.07, 0.05, lam_c),
        ("B零溢价",  0.00, 0.07, 0.05, lam_c),
        ("C负溢价", -0.25, 0.07, 0.05, lam_c),
    ]
    results = []
    for tag, k0, z10, z20, l0 in starts:
        x0 = np.clip([k0, z10, z20, l0],
                     [kap_lb, z1_lb, z2_lb, lam_lb],
                     [kap_ub, z1_ub, z2_ub, lam_ub])
        try:
            r = minimize(lambda x: obj(x)[0], x0, method="Nelder-Mead",
                         options={"maxfev": maxfev,
                                   "xatol": 1e-4,
                                   "fatol": 1e-4})
            ft, f0, _ = obj(r.x)
            best = r.x
        except Exception as e:
            print("[起点异常][%s] %s" % (tag, e))
            ft, f0, best = np.inf, np.inf, x0
        results.append({"tag": tag, "th": best, "f1": ft, "f0": f0,
                        "kappa": float(best[0])})
        print("  [%s] |r|²+pen=%.6f | kappa=%+.4f zeta1=%+.4f "
              "zeta2=%+.4f lambda=%+.6f"
              % (tag, ft, best[0], best[1], best[2], best[3]))

    results.sort(key=lambda d: d["f1"] if np.isfinite(d["f1"]) else np.inf)
    sel = results[0]
    print("[多起点] 采用 %s：kappa=%+.4f" % (sel["tag"], sel["kappa"]))
    return {
        "kappa": sel["kappa"],
        "zeta1": float(sel["th"][1]),
        "zeta2": float(sel["th"][2]),
        "lam": float(sel["th"][3]),
        "f1": sel["f1"],
        "f0": sel["f0"],
        "kappa_A": results[0]["kappa"],
        "kappa_B": results[1]["kappa"],
        "kappa_C": results[2]["kappa"],
        "F_first": A["F_first"],
    }


def lambda_kappa_trajectory(A, l0, gamma, sigma, rho,
                            Z, phi, lambda_text, lambda_weight,
                            MC_II=5, maxfev=2000):
    rng = _rng(6)
    X = A["X"]
    R = A["R"]
    B_var = A["B_var"]
    logP = A["logP"]
    M = A["M"]

    tilde_lambda = float(lambda_text)
    lam_grid = np.arange(-0.05, 0.05 + 1e-9, 0.01)

    kap_lb, kap_ub = A["kappa_hat"] - 0.5, A["kappa_hat"] + 0.5
    z1_hat = float(A["bB"][-1]) if len(A["bB"]) > 0 else 0.0
    z1_se = 0.05
    z1_lb, z1_ub = z1_hat - 3 * z1_se, z1_hat + 3 * z1_se
    z2_lb, z2_ub = -2.0, 2.0

    mv = np.isfinite(logP) & (B_var > 0)
    aux_real = np.array([
        float(B_var.mean()),
        float(_corr(B_var, R)),
        float(np.nanmean(logP[mv])) if mv.sum() else 0.0,
        float(A["kappa_hat"]),
        float(_skew(logP[mv])) if mv.sum() > 10 else 0.0,
    ])

    def sim_aux_fixed(kap, z1, zeta2, lam):
        auxs = []
        for _ in range(MC_II):
            U = rng.standard_normal((len(X), 6))
            _, _, logPs, Bs = simulate_4eq_4(
                np.array([kap, z1, zeta2, lam]),
                A["Xb1"], A["Xb2"], A["Xb3"], A["Xb4"],
                U, gamma, sigma, rho, Z=Z, phi=phi, zeta1=z1)
            mvs = np.isfinite(logPs) & (Bs > 0)
            auxs.append(np.array([
                float(np.mean(Bs)),
                float(_corr(Bs, R)),
                float(np.nanmean(logPs[mvs])) if mvs.sum() else 0.0,
                float(A["kappa_hat"]),
                float(_skew(logPs[mvs])) if mvs.sum() > 10 else 0.0,
            ]))
        return np.nanmean(auxs, axis=0)

    def pen_fixed(kap, z1, zeta2, lam):
        p = 0.0
        if kap < kap_lb: p += 1e6 * (kap_lb - kap) ** 2
        if kap > kap_ub: p += 1e6 * (kap - kap_ub) ** 2
        if z1 < z1_lb:   p += 1e6 * (z1_lb - z1) ** 2
        if z1 > z1_ub:   p += 1e6 * (z1 - z1_ub) ** 2
        if zeta2 < z2_lb: p += 1e6 * (z2_lb - zeta2) ** 2
        if zeta2 > z2_ub: p += 1e6 * (zeta2 - z2_ub) ** 2
        p += lambda_weight * (lam - tilde_lambda) ** 2
        return p

    rows = []
    print("=" * 92)
    print("[轨迹] κ–λ 识别轨迹：固定 lambda，只优化 kappa/zeta1/zeta2")
    print("[轨迹] lambda 网格：%s"
          % [float(f"{x:.4f}") for x in lam_grid])
    print("=" * 92)

    for lam in lam_grid:
        lam = float(lam)

        def obj3(theta, lam=lam):
            kap, z1, zeta2 = theta
            aux_m = sim_aux_fixed(kap, z1, zeta2, lam)
            f0 = float(np.sum((aux_real - aux_m) ** 2))
            return f0 + pen_fixed(kap, z1, zeta2, lam)

        x0 = np.array([
            float(np.clip(A["kappa_hat"], kap_lb, kap_ub)),
            float(np.clip(z1_hat, z1_lb, z1_ub)),
            0.05,
        ])
        r = minimize(obj3, x0, method="Nelder-Mead",
                     options={"maxfev": maxfev,
                               "xatol": 1e-4,
                               "fatol": 1e-4})
        kap_opt, z1_opt, z2_opt = r.x
        aux_m = sim_aux_fixed(kap_opt, z1_opt, z2_opt, lam)
        r2 = float(np.sum((aux_real - aux_m) ** 2))
        kap_opt = float(np.clip(kap_opt, kap_lb, kap_ub))

        rows.append((lam, kap_opt, r2))
        print("[轨迹] lam=%+.6f, kappa=%+.6f, |r|²=%+.6f"
              % (lam, kap_opt, r2))

    traj = pd.DataFrame(rows, columns=["lam", "kappa", "r2"])
    traj["lam"] = traj["lam"].round(6).replace(0, 0.0)

    if not getattr(lambda_kappa_trajectory, "_saved", False):
        traj.to_csv(OUT_TRAJECTORY, index=False)
        print("[轨迹] 已写入 %s" % OUT_TRAJECTORY)
        print("[轨迹] 二维表：")
        print(traj.to_string(index=False))
        print("[轨迹] κ 网格范围 = [%+.6f, %+.6f]"
              % (traj.kappa.min(), traj.kappa.max()))

        fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
        ax.plot(traj.lam, traj.kappa, marker="o", linewidth=2,
                color="#1f4e79", label=r"$\kappa(\lambda)$")
        ax.axvline(tilde_lambda, color="#c00000", linestyle="--",
                   linewidth=1.8,
                   label=r"$\widetilde{\lambda}=%.6f$" % tilde_lambda)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\kappa$")
        ax.set_title("Kappa vs Lambda Identification Bounds", fontsize=14)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_PLOT, dpi=200)
        plt.close(fig)
        print("[轨迹] 图已写入 %s" % OUT_PLOT)
        lambda_kappa_trajectory._saved = True

    ibest = int(traj.r2.idxmin())
    return float(traj.loc[ibest, "lam"]), traj


# ============================================================
# Block Bootstrap
# ============================================================
def _block_bootstrap_index(n, block_len, rng):
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        b = min(block_len, n - i)
        start = rng.randint(0, n - b + 1)
        idx[i:i + b] = np.arange(start, start + b)
        i += b
    return idx[:n]


def _make_boot_dict(nuisance, idx, l0):
    return {
        "df": nuisance["df"].iloc[idx].reset_index(drop=True),
        "X": nuisance["X"][idx].copy(),
        "R": nuisance["R"][idx].copy(),
        "D": nuisance["D"][idx].copy(),
        "B_var": nuisance["B_var"][idx].copy(),
        "logP": nuisance["logP"][idx].copy(),
        "M": nuisance["M"][idx].copy(),
        "D_field": nuisance["D_field"],
        "N": int(len(idx)),
        "K": nuisance["X"].shape[1],
        "ALL": nuisance["ALL"],
        "Z_iv": nuisance["Z_iv"][idx].copy(),
        "l0": float(l0),
    }


def bootstrap_kappa_trajectory(
    df, X_cols, nuisance, traj_func,
    Z, phi, lambda_text, lambda_weight,
    B=500, block_len=20, mc_per=2, maxfev=1200, rng_base=11
):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    print("=" * 92)
    print("[BOOT] 参数 block-bootstrap：B=%d，block_len=%d"
          % (B, block_len))
    print("[BOOT] 每次重算 nuisance + 固定 lambda 轨迹")
    print("=" * 92)

    for b in range(B):
        rng_b = np.random.RandomState(MASTER_SEED + rng_base + b)
        idx = _block_bootstrap_index(len(df), block_len, rng_b)

        Ab = _make_boot_dict(nuisance, idx, nuisance["l0"])
        Ab = fit_nuisance(Ab)

        _, traj_b = traj_func(
            Ab, Ab["l0"], Ab["gamma_fixed"], Ab["sigma_fixed"],
            Ab["rho_fixed"], Z=Ab["Z_iv"], phi=phi,
            lambda_text=Ab["lambda_text"],
            lambda_weight=lambda_weight,
            MC_II=mc_per, maxfev=maxfev
        )
        traj_b = traj_b.copy()
        traj_b["b"] = b
        traj_b["lambda_text_boot"] = Ab["lambda_text"]
        rows.append(traj_b)

        if (b + 1) % 50 == 0 or b == 0:
            print("[BOOT] 已完成 %d / %d" % (b + 1, B))

    boot_all = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "bootstrap_block_kappa_trajectory.csv")
    boot_all.to_csv(out_path, index=False)
    print("[BOOT] 明细写入 %s" % out_path)

    summary = (boot_all.groupby("lam")["kappa"]
                       .agg(kappa_mean="mean",
                            kappa_median="median",
                            kappa_p025=lambda x: np.percentile(x, 2.5),
                            kappa_p975=lambda x: np.percentile(x, 97.5),
                            kappa_sd="std",
                            n="count")
                       .reset_index())
    summ_path = os.path.join(OUT_DIR, "bootstrap_block_kappa_summary.csv")
    summary.to_csv(summ_path, index=False)
    print("[BOOT] 汇总写入 %s" % summ_path)
    print(summary.to_string(index=False))
    return boot_all, summary


# ============================================================
# Cluster Bootstrap：默认 (国家, 年份)
# ============================================================
def _cluster_bootstrap_index(df, cluster_cols, rng):
    cl = df[list(cluster_cols)].astype(str).agg("|".join, axis=1)
    clusters = np.unique(cl.values)
    pick = clusters[rng.choice(len(clusters), size=len(clusters),
                                replace=True)]
    mask = np.isin(cl.values, pick)
    idx, = np.where(mask)
    return idx


def bootstrap_cluster_kappa_trajectory(
    df, X_cols, nuisance, traj_func,
    Z, phi, lambda_text, lambda_weight,
    cluster_cols=("tar_country_code", "deal_year"),
    B=500, mc_per=2, maxfev=1200, rng_base=31,
    min_obs=5000, min_clusters=30
):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    cl = df[list(cluster_cols)].astype(str).agg("|".join, axis=1)
    n_clusters = cl.nunique()
    print("=" * 92)
    print("[BOOT-CLUSTER] 簇 bootstrap：B=%d" % B)
    print("[BOOT-CLUSTER] 聚类变量 = %s" % list(cluster_cols))
    print("[BOOT-CLUSTER] 当前聚类数 = %d（务必远小于 N=%d）"
          % (n_clusters, len(df)))
    print("[BOOT-CLUSTER] 最小观测要求 = %d" % min_obs)
    print("[BOOT-CLUSTER] 最小簇数要求 = %d" % min_clusters)
    if n_clusters >= len(df) * 0.5:
        print("[BOOT-CLUSTER][警告] 簇数接近样本量，"
              "这不是真正的簇抽样！请换更粗的聚类层级。")
    print("=" * 92)

    if n_clusters < min_clusters:
        raise RuntimeError(
            "簇数 %d < 最小要求 %d，无法做簇 bootstrap" % (
                n_clusters, min_clusters))

    done = 0
    attempts = 0
    while done < B and attempts < B * 20:
        attempts += 1
        rng_b = np.random.RandomState(MASTER_SEED + rng_base + attempts)
        idx = _cluster_bootstrap_index(df, cluster_cols, rng_b)
        if len(idx) < min_obs:
            continue

        Ab = _make_boot_dict(nuisance, idx, nuisance["l0"])
        Ab = fit_nuisance(Ab)

        _, traj_b = traj_func(
            Ab, Ab["l0"], Ab["gamma_fixed"], Ab["sigma_fixed"],
            Ab["rho_fixed"], Z=Ab["Z_iv"], phi=phi,
            lambda_text=Ab["lambda_text"],
            lambda_weight=lambda_weight,
            MC_II=mc_per, maxfev=maxfev
        )
        traj_b = traj_b.copy()
        traj_b["b"] = done
        traj_b["n_cluster"] = n_clusters
        traj_b["lambda_text_boot"] = Ab["lambda_text"]
        rows.append(traj_b)
        done += 1

        if done % 50 == 0 or done == 1:
            print("[BOOT-CLUSTER] 已完成 %d / %d" % (done, B))

    if done < B:
        raise RuntimeError("只完成 %d / %d 个簇 bootstrap 副本" % (done, B))

    boot_all = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(OUT_DIR,
                            "bootstrap_cluster_kappa_trajectory.csv")
    boot_all.to_csv(out_path, index=False)
    print("[BOOT-CLUSTER] 明细写入 %s" % out_path)

    summary = (boot_all.groupby("lam")["kappa"]
                       .agg(kappa_mean="mean",
                            kappa_median="median",
                            kappa_p025=lambda x: np.percentile(x, 2.5),
                            kappa_p975=lambda x: np.percentile(x, 97.5),
                            kappa_sd="std",
                            n="count")
                       .reset_index())
    summ_path = os.path.join(OUT_DIR,
                             "bootstrap_cluster_kappa_summary.csv")
    summary.to_csv(summ_path, index=False)
    print("[BOOT-CLUSTER] 汇总写入 %s" % summ_path)
    print(summary.to_string(index=False))
    return boot_all, summary


# ============================================================
# main
# ============================================================
def main():
    print("=" * 92)
    print("[ENV] Python : %s" % sys.version.replace("\n", " "))
    print("[ENV] MASTER_SEED : %d" % MASTER_SEED)
    print("[ENV] DATA_PATH : %s" % DATA_PATH)
    print("[ENV] 版本 : v48_lambda_bootstrap_final_v3")
    print("=" * 92)

    df = load_data()
    ALL_base, X_base, _ = build_controls(df)
    ALL_aug, X_aug, Z_res = build_leak_iv(df, ALL_base)

    B_var = build_B(df)
    M = build_text(df)
    R = _to_num(df, "rumor_jfe")

    if "completed_strict" in df.columns:
        Dc = _to_num(df, "completed_strict")
        d_field = "completed_strict"
    else:
        Dc = _to_num(df, "completed")
        d_field = "completed"

    logP = _to_num(df, "ln_rev_y")

    A = {
        "df": df,
        "X": X_aug,
        "R": R,
        "D": Dc,
        "B_var": B_var,
        "logP": logP,
        "M": M,
        "D_field": d_field,
        "N": len(df),
        "K": X_aug.shape[1],
        "ALL": ALL_base,
        "Z_iv": Z_res,
    }

    A["pi_hat"] = estimate_pi(df)
    mv = np.isfinite(logP) & (B_var > 0)
    print("[B] 可观测率 = %.4f（finite(ln_rev_y)=%d/%d）"
          % (B_var.mean(), int(mv.sum()), len(df)))

    A = fit_nuisance(A)
    A["l0"] = float(np.nanmean(logP[mv])) if mv.sum() else 0.0
    pi_diag = text_pi_diagnostics(A)

    print("[CHECK] TEXT-PI tilde_lambda = %.6f" % pi_diag["tilde_lambda"])
    print("[CHECK] nuisance tilde_lambda = %.6f" % A["lambda_text"])
    print("[CHECK] 两处 tilde_lambda 相等：%s"
          % np.isclose(pi_diag["tilde_lambda"], A["lambda_text"]))

    g, gs = A["gamma_fixed"], A["gamma_se"]
    print("[GAMMA] gamma_fixed = %.4f (se=%.4f), t = %.2f"
          % (g, gs, g / max(gs, 1e-6)))

    # LPM 诊断
    X1_now = np.column_stack([np.ones(len(A["X"])), A["X"]])
    Phat = X1_now @ A["bD"][:-1] + A["gamma_fixed"] * A["R"]
    print("[LPM] Phat min = %.4f, max = %.4f, mean = %.4f"
          % (Phat.min(), Phat.max(), Phat.mean()))
    print("[LPM] Phat < 0 比例 = %.4f%%" % (100 * (Phat < 0).mean()))
    print("[LPM] Phat > 1 比例 = %.4f%%" % (100 * (Phat > 1).mean()))

    # 主样本轨迹：使用主样本 Z_res
    res = estimate_structural_4(
        A, A["l0"], A["gamma_fixed"], A["sigma_fixed"],
        A["rho_fixed"], rng=_rng(4), MC_II=5, maxfev=2000,
        Z=Z_res, phi=A["phi_iv"],
        lambda_text=A["lambda_text"], lambda_weight=1.0)

    lambda_best, traj = lambda_kappa_trajectory(
        A, A["l0"], A["gamma_fixed"], A["sigma_fixed"],
        A["rho_fixed"], Z=Z_res, phi=A["phi_iv"],
        lambda_text=A["lambda_text"], lambda_weight=1.0,
        MC_II=5, maxfev=2000)

    print("=" * 92)
    print("SUMMARY: kappa=%+.4f zeta1=%+.4f zeta2=%+.4f lambda=%+.6f"
          % (res["kappa"], res["zeta1"], res["zeta2"], res["lam"]))
    print("SUMMARY: tilde_lambda=%+.6f | phi_iv=%+.4f | gamma(D)=%+.4f"
          % (A["lambda_text"], A["phi_iv"], A["gamma_fixed"]))
    print("SUMMARY: F_first=%.2f | N=%d" % (res["F_first"], A["N"]))
    print("SUMMARY: 轨迹保守 lambda（r2 最小）= %+.6f" % lambda_best)
    print("=" * 92)

    # nuisance 快照
    nuisance_current = {
        "df": df,
        "X": A["X"],
        "R": A["R"],
        "D": A["D"],
        "B_var": A["B_var"],
        "logP": A["logP"],
        "M": A["M"],
        "D_field": A["D_field"],
        "ALL": A["ALL"],
        "Z_iv": A["Z_iv"],
        "l0": A["l0"],
    }

    # ---- Block bootstrap ----
    boot_block, summ_block = bootstrap_kappa_trajectory(
        df=df,
        X_cols=ALL_base,
        nuisance=nuisance_current,
        traj_func=lambda_kappa_trajectory,
        Z=Z_res,
        phi=A["phi_iv"],
        lambda_text=A["lambda_text"],
        lambda_weight=1.0,
        B=10,                 # 先冒烟
        block_len=20,
        mc_per=2,
        maxfev=1200,
        rng_base=11
    )

    valid_block = boot_block[
        (boot_block.lam >= boot_block.lambda_text_boot - 0.02) &
        (boot_block.lam <= boot_block.lambda_text_boot + 0.02)
    ]
    print("=" * 92)
    print("[BOOT][论文识别集合] block：每个副本 λ ∈ [λ̃_b − 0.02, λ̃_b + 0.02]")
    print("[BOOT] block kappa 2.5%%  = %+.6f"
          % valid_block.kappa.quantile(0.025))
    print("[BOOT] block kappa 中位数 = %+.6f"
          % valid_block.kappa.median())
    print("[BOOT] block kappa 97.5%% = %+.6f"
          % valid_block.kappa.quantile(0.975))
    print("[BOOT] block 95%% CI 包含 0：%s"
          % (valid_block.kappa.quantile(0.025) < 0
             < valid_block.kappa.quantile(0.975)))
    print("=" * 92)

    # ---- Cluster bootstrap（国家 × 年份）----
    boot_cluster, summ_cluster = bootstrap_cluster_kappa_trajectory(
        df=df,
        X_cols=ALL_base,
        nuisance=nuisance_current,
        traj_func=lambda_kappa_trajectory,
        Z=Z_res,
        phi=A["phi_iv"],
        lambda_text=A["lambda_text"],
        lambda_weight=1.0,
        cluster_cols=("tar_country_code", "deal_year"),
        B=10,                 # 先冒烟
        mc_per=2,
        maxfev=1200,
        rng_base=31,
        min_obs=5000,
        min_clusters=30
    )

    valid_cluster = boot_cluster[
        (boot_cluster.lam >= boot_cluster.lambda_text_boot - 0.02) &
        (boot_cluster.lam <= boot_cluster.lambda_text_boot + 0.02)
    ]
    print("=" * 92)
    print("[BOOT-CLUSTER][论文识别集合] "
          "每个副本 λ ∈ [λ̃_b − 0.02, λ̃_b + 0.02]")
    print("[BOOT-CLUSTER] kappa 2.5%%  = %+.6f"
          % valid_cluster.kappa.quantile(0.025))
    print("[BOOT-CLUSTER] kappa 中位数 = %+.6f"
          % valid_cluster.kappa.median())
    print("[BOOT-CLUSTER] kappa 97.5%% = %+.6f"
          % valid_cluster.kappa.quantile(0.975))
    print("[BOOT-CLUSTER] 95%% CI 包含 0：%s"
          % (valid_cluster.kappa.quantile(0.025) < 0
             < valid_cluster.kappa.quantile(0.975)))
    print("=" * 92)

    print("SUMMARY: 全部完成，耗时 = %.1fs" % (time.time() - _T0))


if __name__ == "__main__":
    main()