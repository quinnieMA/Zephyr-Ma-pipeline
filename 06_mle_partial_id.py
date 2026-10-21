# -*- coding: utf-8 -*-
r"""
06_mle_partial_id.py  —  v1 (2026-10-17)
================================================================================
复现目标：Alperovych, Cumming, Czellar & Groh (JFE, forthcoming)
          "M&A Rumors about Unlisted Firms"  (SSRN 3823312)
          EFMA 工作稿版："M&A RUMORS: WHY SELLERS HATE THEM"

输入 : D:\MA\data\merged\05_deal_firm_filtered.csv   (05 screening 后的样本)
输出 : D:\MA\data\merged\06_mle_partial_id_results.csv
       D:\MA\data\merged\06_partial_id_bounds.csv
       D:\MA\data\merged\06_mle_partial_id_log.txt

================================================================================
模型设定（★ 两处为推断，需你确认 ★）
================================================================================
【推断 1】MLE = 联立方程 / bivariate probit（全信息极大似然）
    潜变量形式：
        y1* = Z'γ + u1,   y1 = 1[y1* > 0]      ← RUMOR
        y2* = β·y1 + X'δ + u2,  y2 = 1[y2* > 0] ← COMPLETION
        (u1, u2) ~ N(0, 0, 1, 1, ρ)
    ρ ≠ 0 即为 rumor 内生（论文：rumor 可能是刻意散布的）。
    β 与 AME 由全信息 MLE 估计。
    需要至少一个排除限制（进入 Z 但不进入 X）。默认用 total_info_source。
    ★ 该排除限制的合理性由研究者论证，代码不替你判断。

【推断 2】部分识别 = 丢掉参数化假设，用线性规划求 sharp bounds
    潜在类型 (D(0), D(1)) ∈ {0,1}²，记 p_jk = P(D(0)=j, D(1)=k)
    选择 s_jk = P(A=1 | type jk) 完全自由（最坏情况选择）
    可观测 4 个 cell：
        O11 = P(A=1, D=1) = p01·s01 + p11·s11
        O10 = P(A=1, D=0) = p00·s00 + p10·s10
        O01 = P(A=0, D=1) = p10(1-s10) + p11(1-s11)
        O00 = P(A=0, D=0) = p00(1-s00) + p01(1-s01)
    令 a_jk = p_jk·s_jk ∈ [0, p_jk]，则约束对 (p, a) 全线性。
    目标 ATE = P(D(1)=1) - P(D(0)=1) = (p01+p11) - (p10+p11) = p01 - p10。
    → 线性规划，8 个变量，4 个等式约束。
    可选约束：
        MTR（rumor 为 deal-breaker，D(1) ≤ D(0)）→ p01 = 0
    ★ 与解析 Manski worst-case 界交叉验证（代码内自检，必须一致）。

================================================================================
结构
================================================================================
PART 0  样本矩 vs 论文参照值
PART 1  单方程 probit（基准）
PART 2  Bivariate probit 全信息 MLE（解析梯度）
PART 3  部分识别：线性规划 sharp bounds（+ Manski 解析界交叉验证）
PART 4  推断：bootstrap
PART 5  可检验含义：MLE 点估计是否落在识别集内 + 保存

================================================================================
注意：不使用 LM 文本距离变量（ind_text_diversity 等尚未生成）。
      默认控制变量与工具变量均为非文本变量。
"""
# -*- coding: utf-8 -*-
r"""
06_mle_partial_id.py — v2 (2026-10-17)
================================================================================
v1 -> v2 修复
================================================================================
[FIX-1] 结果变量换 completed_strict（严格口径）
    v1 用 completed_flag（宽口径，含 Assumed/Expected 等状态），
    最终样本 p_D = 0.9729，结果变量几乎无变异。
    现默认 COL_OUT = "completed_strict"，并同时报告宽口径作对比。

[FIX-2] 排除限制 bug：total_info_source 同时进了 CONTROLS 和 INSTRUMENTS
    -> 两个方程都在，模型实际不可识别，只剩函数形式硬撑。
    已从 CONTROLS 移除，只保留在 INSTRUMENTS。

[FIX-3] 收敛失败（nit=400 达上限）
    - maxiter 提高到 1500
    - 多起点重启（rho 初值 0, ±0.4, ±0.7），取 nll 最优
    - 用 OPG（outer-product-of-gradients）标准误替代 bootstrap
      （解析梯度已有，一次计算即得；196k 样本 bootstrap 不现实）

[FIX-4] 新增结果变量变异预警：p_D > 0.95 明确告警。

其余设定不变：
    MLE = bivariate probit 联立方程（推断，需确认）
    部分识别 = 类型概率线性规划 sharp bounds（+ Manski 解析界交叉验证）
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog, brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# =====================================================================
BASE   = r"D:\MA"
MERGED = os.path.join(BASE, "data", "merged")

IN_FILE  = os.path.join(MERGED, "05_deal_firm_filtered.csv")
OUT_RES  = os.path.join(MERGED, "06_mle_partial_id_results.csv")
OUT_BND  = os.path.join(MERGED, "06_partial_id_bounds.csv")
LOG_FILE = os.path.join(MERGED, "06_mle_partial_id_log.txt")

DEAL = "deal_num"

# ---- 变量 ----
COL_TREAT = "rumor_jfe"
COL_OUT   = "completed_strict"    # [FIX-1] 严格口径
COL_OUT_ALT = "completed_flag"    # 宽口径，仅对比

# ---- 控制变量（不含 total_info_source —— [FIX-2]）----
CONTROLS = [
    "ln_deal_value", "cross_border", "cross_industry", "stake_eff_pct",
    "tar_age", "acq_age", "acq_listed",
    "pay_has_shares", "pay_mix", "reg_antitrust",
]

# ---- 排除限制（只进 rumor 方程）----
INSTRUMENTS = ["total_info_source"]

MAKE_AGE    = True
YEAR_FE     = True
MAX_YEAR_FE = 40

# ---- 估计选项 ----
SUBSAMPLE_N  = 0          # 0=全样本；试跑设 30000
MLE_MAXITER  = 1500       # [FIX-3]
RHO_STARTS   = [0.0, 0.4, -0.4, 0.7, -0.7]
BOOSTRAP_LP  = 1000
BVN_GL_NODES = 24
RHO_CAP      = 0.995
SEED         = 20261017
ALPHA        = 0.05

PAPER_REF = {"p_A": 0.2563, "p_D": 0.6568, "p_B": 0.2081}


# =====================================================================
_L = []
_T0 = time.time()

def log(msg=""):
    print(msg); _L.append(str(msg))

def banner(t):
    log("\n" + "=" * 82); log(t); log("=" * 82)

def flush_log():
    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_L) + "\n")


# =====================================================================
_GL_X, _GL_W = np.polynomial.legendre.leggauss(BVN_GL_NODES)

def _bvn_pdf(h, k, rho):
    om = np.clip(1.0 - rho * rho, 1e-12, None)
    q = (h * h - 2.0 * rho * h * k + k * k) / (2.0 * om)
    return np.exp(-q) / (2.0 * np.pi * np.sqrt(om))

def bvn_cdf(h, k, rho, chunk=50_000):
    h = np.asarray(h, float); k = np.asarray(k, float)
    rho = np.clip(np.asarray(rho, float), -RHO_CAP, RHO_CAP)
    n = h.shape[0]; out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        hs, ks, rs = h[s:e], k[s:e], rho[s:e]
        base = norm.cdf(hs) * norm.cdf(ks)
        t = 0.5 * rs[:, None] * (_GL_X[None, :] + 1.0)
        om = np.clip(1.0 - t * t, 1e-12, None)
        q = (hs[:, None] ** 2 - 2.0 * t * hs[:, None] * ks[:, None]
             + ks[:, None] ** 2) / (2.0 * om)
        dens = np.exp(-q) / (2.0 * np.pi * np.sqrt(om))
        out[s:e] = base + (0.5 * rs) * (dens * _GL_W[None, :]).sum(axis=1)
    return np.clip(out, 1e-300, 1.0)


# =====================================================================
def probit_mle(X, y, maxiter=300):
    X = np.asarray(X, float); y = np.asarray(y, float)
    q = 2.0 * y - 1.0
    def negll(b):
        return -np.sum(norm.logcdf(q * (X @ b)))
    def grad(b):
        z = q * (X @ b)
        return -(X.T @ (q * np.exp(norm.logpdf(z) - norm.logcdf(z))))
    b0 = np.zeros(X.shape[1])
    b0[0] = norm.ppf(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    r = minimize(negll, b0, jac=grad, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, ftol=1e-12))
    return r.x, r

def probit_ame(X, b, idx):
    X1 = X.copy(); X1[:, idx] = 1.0
    X0 = X.copy(); X0[:, idx] = 0.0
    return float(np.mean(norm.cdf(X1 @ b) - norm.cdf(X0 @ b)))


# =====================================================================
def biprobit_negll_grad(theta, X1, y1, X2, y2, k1):
    g, d, ath = theta[:k1], theta[k1:-1], theta[-1]
    rho = np.tanh(ath)
    a = X1 @ g; b = X2 @ d
    q1 = 2.0 * y1 - 1.0; q2 = 2.0 * y2 - 1.0
    w1 = q1 * a; w2 = q2 * b; rs = q1 * q2 * rho

    P = bvn_cdf(w1, w2, rs)
    nll = -np.sum(np.log(P))

    om = np.clip(1.0 - rs * rs, 1e-12, None); sq = np.sqrt(om)
    dF1 = norm.pdf(w1) * norm.cdf((w2 - rs * w1) / sq)
    dF2 = norm.pdf(w2) * norm.cdf((w1 - rs * w2) / sq)
    dFr = _bvn_pdf(w1, w2, rs)
    inv = 1.0 / P
    g_gam = -(X1.T @ (inv * dF1 * q1))
    g_del = -(X2.T @ (inv * dF2 * q2))
    g_ath = -np.sum(inv * dFr * q1 * q2) * (1.0 - rho * rho)
    return nll, np.concatenate([g_gam, g_del, [g_ath]])

def biprobit_scores(theta, X1, y1, X2, y2, k1):
    """逐观测 score，用于 OPG 标准误"""
    g, d, ath = theta[:k1], theta[k1:-1], theta[-1]
    rho = np.tanh(ath)
    a = X1 @ g; b = X2 @ d
    q1 = 2.0 * y1 - 1.0; q2 = 2.0 * y2 - 1.0
    w1 = q1 * a; w2 = q2 * b; rs = q1 * q2 * rho
    P = bvn_cdf(w1, w2, rs)
    om = np.clip(1.0 - rs * rs, 1e-12, None); sq = np.sqrt(om)
    dF1 = norm.pdf(w1) * norm.cdf((w2 - rs * w1) / sq)
    dF2 = norm.pdf(w2) * norm.cdf((w1 - rs * w2) / sq)
    dFr = _bvn_pdf(w1, w2, rs)
    inv = 1.0 / P
    s1 = (inv * dF1 * q1)[:, None] * X1
    s2 = (inv * dF2 * q2)[:, None] * X2
    s3 = (inv * dFr * q1 * q2 * (1.0 - rho * rho))[:, None]
    return np.column_stack([s1, s2, s3])

def fit_biprobit(X1, y1, X2, y2, verbose=True):
    """[FIX-3] 多起点重启，取 nll 最优"""
    k1 = X1.shape[1]
    b1, _ = probit_mle(X1, y1)
    b2, _ = probit_mle(X2, y2)
    best = None
    for r0 in RHO_STARTS:
        theta0 = np.concatenate([b1, b2, [np.arctanh(np.clip(r0, -0.99, 0.99))]])
        try:
            res = minimize(biprobit_negll_grad, theta0,
                           args=(X1, y1, X2, y2, k1), jac=True,
                           method="L-BFGS-B",
                           options=dict(maxiter=MLE_MAXITER, ftol=1e-12, gtol=1e-8))
        except Exception as e:
            if verbose:
                log(f"    start rho={r0:+.1f}: failed ({e})")
            continue
        if verbose:
            log(f"    start rho={r0:+.1f}: nll={res.fun:,.2f} "
                f"converged={res.success} nit={res.nit}")
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError("all starts failed")
    k2 = X2.shape[1]
    out = dict(gamma=best.x[:k1], delta=best.x[k1:k1 + k2],
               rho=float(np.tanh(best.x[-1])), theta=best.x,
               nll=float(best.fun), success=bool(best.success), nit=int(best.nit))
    # OPG 标准误
    G = biprobit_scores(best.x, X1, y1, X2, y2, k1)
    try:
        V = np.linalg.inv(G.T @ G)
        se = np.sqrt(np.clip(np.diag(V), 0, None))
    except Exception:
        se = np.full(len(best.x), np.nan)
    out["se"] = se
    return out

def biprobit_ame(X1, X2, gamma, delta, rho):
    a = X1 @ gamma
    X2a = X2.copy(); X2a[:, 1] = 1.0
    X2b = X2.copy(); X2b[:, 1] = 0.0
    b1 = X2a @ delta; b0 = X2b @ delta
    pa = np.clip(norm.cdf(a), 1e-12, 1 - 1e-12)
    P11 = bvn_cdf(a, b1, np.full(len(a), rho))
    P01 = bvn_cdf(a, b0, np.full(len(a), rho))
    return float(np.mean(P11 / pa - (norm.cdf(b0) - P01) / (1.0 - pa)))


# =====================================================================
IDX = {"p00": 0, "p01": 1, "p10": 2, "p11": 3,
       "a00": 4, "a01": 5, "a10": 6, "a11": 7}

def _A_eq():
    A = np.zeros((4, 8))
    A[0, IDX["a01"]] = 1; A[0, IDX["a11"]] = 1
    A[1, IDX["a00"]] = 1; A[1, IDX["a10"]] = 1
    A[2, IDX["p10"]] = 1; A[2, IDX["a10"]] = -1
    A[2, IDX["p11"]] = 1; A[2, IDX["a11"]] = -1
    A[3, IDX["p00"]] = 1; A[3, IDX["a00"]] = -1
    A[3, IDX["p01"]] = 1; A[3, IDX["a01"]] = -1
    return A

def _A_ub():
    rows = []
    for p, a in [("p00", "a00"), ("p01", "a01"), ("p10", "a10"), ("p11", "a11")]:
        r = np.zeros(8); r[IDX[a]] = 1.0; r[IDX[p]] = -1.0
        rows.append(r)
    return np.array(rows)

def partial_id_bounds(cells, mtr=False):
    A_eq = _A_eq()
    b_eq = np.array([cells["O11"], cells["O10"], cells["O01"], cells["O00"]])
    A_ub = _A_ub(); b_ub = np.zeros(A_ub.shape[0])
    bounds = [(0, 1)] * 8
    if mtr:
        bounds[IDX["p01"]] = (0.0, 0.0)
        bounds[IDX["a01"]] = (0.0, 0.0)
    c = np.zeros(8); c[IDX["p01"]] = 1.0; c[IDX["p10"]] = -1.0
    rmin = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")
    rmax = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")
    if not (rmin.success and rmax.success):
        raise RuntimeError(f"LP failed: {rmin.message} | {rmax.message}")
    return float(rmin.fun), float(-rmax.fun), dict(
        manski_lo=-(cells["O01"] + cells["O10"]),
        manski_up=cells["O11"] + cells["O00"])

def imbens_manski_c(d, s_u, alpha=ALPHA):
    f = lambda c: norm.cdf(c + d / max(s_u, 1e-12)) - norm.cdf(-c) - (1 - alpha)
    try:
        return float(brentq(f, 0.0, 20.0))
    except Exception:
        return float(norm.ppf(1 - alpha / 2))


# =====================================================================
def main():
    banner("06_mle_partial_id.py — v2")
    log(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"treat   : {COL_TREAT}")
    log(f"outcome : {COL_OUT}  (严格口径；对比 {COL_OUT_ALT})")
    log(f"排除限制: {INSTRUMENTS}  (已从 CONTROLS 移除)")

    df = pd.read_csv(IN_FILE, encoding="utf-8-sig", low_memory=False)
    log(f"\n  rows={len(df):,} | cols={df.shape[1]}")

    if MAKE_AGE:
        for side in ("tar", "acq"):
            c = f"{side}_incorp_d_year"
            if c in df.columns and "deal_year" in df.columns:
                df[f"{side}_age"] = pd.to_numeric(df["deal_year"], errors="coerce") \
                                    - pd.to_numeric(df[c], errors="coerce")
                df.loc[df[f"{side}_age"] < 0, f"{side}_age"] = np.nan

    if SUBSAMPLE_N and SUBSAMPLE_N < len(df):
        rng = np.random.default_rng(SEED)
        df = df.iloc[rng.choice(len(df), SUBSAMPLE_N, replace=False)].copy()
        log(f"  [SUBSAMPLE] n={len(df):,}")

    missing = [c for c in (COL_OUT, COL_OUT_ALT, COL_TREAT) if c not in df.columns]
    if missing:
        log(f"\n  [FATAL] 05 输出缺少: {missing}")
        log(f"          请先跑 05 v5（它会从 02 补并 completed_strict）")
        flush_log(); return

    y1 = pd.to_numeric(df[COL_TREAT], errors="coerce")
    y2 = pd.to_numeric(df[COL_OUT], errors="coerce")
    ok = y1.notna() & y2.notna()
    log(f"  A/D 均非缺失: {int(ok.sum()):,} / {len(df):,}")

    # ---- PART 0 ----
    banner("PART 0 — 样本矩 vs 论文参照")
    p_A = float(y1[ok].mean())
    p_D = float(y2[ok].mean())
    p_Dw = float(pd.to_numeric(df[COL_OUT_ALT], errors="coerce")[ok].mean())
    p_B = float(df.loc[ok, "pre_rev_mul_ly"].notna().mean()) \
          if "pre_rev_mul_ly" in df.columns else float("nan")
    log(f"  {'moment':<30s} {'sample':>10s} {'paper':>10s} {'diff':>10s}")
    log("  " + "-" * 64)
    for nm, v, r in [(f"p_A ({COL_TREAT})", p_A, PAPER_REF["p_A"]),
                     (f"p_D ({COL_OUT} 严)", p_D, PAPER_REF["p_D"]),
                     (f"p_D ({COL_OUT_ALT} 宽)", p_Dw, PAPER_REF["p_D"]),
                     ("p_B (pre_rev_mul_ly)", p_B, PAPER_REF["p_B"])]:
        log(f"  {nm:<30s} {v:>10.4f} {r:>10.4f} {v-r:>+10.4f}")

    # ---- [FIX-4] 变异预警 ----
    banner("结果变量变异检查")
    for nm, v in [(COL_OUT, p_D), (COL_OUT_ALT, p_Dw)]:
        n1 = int((pd.to_numeric(df[nm], errors="coerce")[ok] == 1).sum())
        n0 = int((pd.to_numeric(df[nm], errors="coerce")[ok] == 0).sum())
        log(f"  {nm:<18s} mean={v:.4f}  N(1)={n1:,}  N(0)={n0:,}")
        if v > 0.95:
            log(f"      [!!] {nm} 完成率 >95%，结果变量几乎无变异。")
            log(f"          请确认 05 已用 completed_strict 且年份口径已修。")
    if p_D > 0.95:
        log("\n  [STOP] 结果变量无变异，后续估计无意义。先修样本。")
        flush_log(); return

    df = df.loc[ok].reset_index(drop=True)
    y1 = y1.loc[ok].to_numpy(float)
    y2 = y2.loc[ok].to_numpy(float)
    n = len(df)
    log(f"\n  分析样本 N = {n:,}")

    # ---- 设计矩阵 ----
    use_c = [c for c in CONTROLS if c in df.columns]
    use_z = [c for c in INSTRUMENTS if c in df.columns]
    miss_c = [c for c in CONTROLS if c not in df.columns]
    log(f"\n  控制变量 ({len(use_c)}): {use_c}")
    if miss_c:
        log(f"  [WARN] 缺失: {miss_c}")
    log(f"  排除限制 ({len(use_z)}): {use_z}")
    if "total_info_source" in use_c:
        log("  [!!] total_info_source 仍在 CONTROLS 中，排除限制失效！")
    if not use_z:
        log("  [FATAL] 无排除限制")
        flush_log(); return

    ctrl = df[use_c].apply(pd.to_numeric, errors="coerce")
    ctrl = ctrl.fillna(ctrl.median(numeric_only=True))
    if YEAR_FE and "deal_year" in df.columns:
        yr = pd.to_numeric(df["deal_year"], errors="coerce")
        top = yr.value_counts().head(MAX_YEAR_FE).index.tolist()
        ye = pd.get_dummies(yr.where(yr.isin(top)), prefix="y").astype(float)
        ye = ye.loc[:, ye.columns[:MAX_YEAR_FE]]
        ctrl = pd.concat([ctrl, ye], axis=1)
        log(f"  年份 FE: {ye.shape[1]} 个")
    C = ctrl.to_numpy(float)
    Zx = df[use_z].apply(pd.to_numeric, errors="coerce")
    Zx = Zx.fillna(Zx.median(numeric_only=True)).to_numpy(float)

    X2 = np.column_stack([np.ones(n), y1, C])
    X1 = np.column_stack([np.ones(n), Zx, C])
    k1 = X1.shape[1]
    log(f"  X1 (rumor eq) : {X1.shape[1]} 列")
    log(f"  X2 (outcome)  : {X2.shape[1]} 列")

    # ---- PART 1 ----
    banner("PART 1 — 单方程 probit（基准）")
    b_p, _ = probit_mle(X2, y2)
    ame_p = probit_ame(X2, b_p, idx=1)
    log(f"  coefficient on {COL_TREAT} : {b_p[1]:+.4f}")
    log(f"  AME                        : {ame_p:+.4f}")

    # ---- PART 2 ----
    banner("PART 2 — Bivariate probit 全信息 MLE")
    t0 = time.time()
    fit = fit_biprobit(X1, y1, X2, y2)
    rho = fit["rho"]; beta = fit["delta"][1]
    se_beta = fit["se"][k1 + 1]; se_rho = fit["se"][-1]
    ame_b = biprobit_ame(X1, X2, fit["gamma"], fit["delta"], rho)
    log(f"\n  converged        : {fit['success']}  (nit={fit['nit']}, "
        f"maxiter={MLE_MAXITER})")
    log(f"  β (rumor->comp)  : {beta:+.4f}   se={se_beta:.4f}   "
        f"z={beta/se_beta if se_beta else float('nan'):+.2f}")
    log(f"  ρ                : {rho:+.4f}   se={se_rho:.4f}   "
        f"z={rho/se_rho if se_rho else float('nan'):+.2f}")
    log(f"  结构 AME         : {ame_b:+.4f}")
    log(f"  logL             : {fit['nll']:,.2f}")
    log(f"  [{time.time()-t0:.0f}s]")
    log(f"\n  参考：单方程 probit AME = {ame_p:+.4f}")
    if abs(rho) < 0.05:
        log("  ρ 接近 0：在该样本上内生性证据弱，单方程 probit 或已足够。")
    else:
        log("  ρ 明显非零：rumor 内生，单方程 probit 有偏。")

    # ---- PART 3 ----
    banner("PART 3 — 部分识别 sharp bounds")
    A1 = (y1 == 1)
    cells = dict(O11=float(np.mean(A1 & (y2 == 1))),
                 O10=float(np.mean(A1 & (y2 == 0))),
                 O01=float(np.mean(~A1 & (y2 == 1))),
                 O00=float(np.mean(~A1 & (y2 == 0))))
    log(f"\n  O11={cells['O11']:.6f}  O10={cells['O10']:.6f}  "
        f"O01={cells['O01']:.6f}  O00={cells['O00']:.6f}  "
        f"sum={sum(cells.values()):.6f}")

    results = {}
    for tag, mtr in [("worst-case (无假设)", False), ("MTR: D(1)<=D(0)", True)]:
        lo, up, det = partial_id_bounds(cells, mtr=mtr)
        results[tag] = (lo, up, det)
        log(f"\n  ── {tag} ──")
        log(f"    ATE ∈ [{lo:+.4f}, {up:+.4f}]   宽度 {up-lo:.4f}")
        if not mtr:
            ml, mu = det["manski_lo"], det["manski_up"]
            agree = abs(lo - ml) < 1e-6 and abs(up - mu) < 1e-6
            log(f"    解析 Manski 界: [{ml:+.4f}, {mu:+.4f}]  "
                f"交叉验证 {'PASS' if agree else 'FAIL'}")

    # ---- PART 4 ----
    banner("PART 4 — Bootstrap（仅部分识别）")
    rng = np.random.default_rng(SEED)
    los, ups, los_m, ups_m = [], [], [], []
    for _ in range(BOOSTRAP_LP):
        idx = rng.integers(0, n, n)
        y1b, y2b = y1[idx], y2[idx]
        Ab = (y1b == 1)
        cb = dict(O11=float(np.mean(Ab & (y2b == 1))),
                  O10=float(np.mean(Ab & (y2b == 0))),
                  O01=float(np.mean(~Ab & (y2b == 1))),
                  O00=float(np.mean(~Ab & (y2b == 0))))
        l, u, _ = partial_id_bounds(cb, False); los.append(l); ups.append(u)
        lm, um, _ = partial_id_bounds(cb, True); los_m.append(lm); ups_m.append(um)
    los, ups = np.array(los), np.array(ups)
    los_m, ups_m = np.array(los_m), np.array(ups_m)

    def report(nm, l0, u0, sl, su):
        d = max(u0 - l0, 0.0); c = imbens_manski_c(d, su)
        log(f"\n  ── {nm} ──")
        log(f"    set : [{l0:+.4f}, {u0:+.4f}]")
        log(f"    se  : {sl:.4f} / {su:.4f}")
        log(f"    IM2004 (c={c:.3f}) : [{l0-c*sl:+.4f}, {u0+c*su:+.4f}]")
        return l0 - c * sl, u0 + c * su

    ci_w = report("worst-case", *results["worst-case (无假设)"][:2],
                  los.std(ddof=1), ups.std(ddof=1))
    ci_m = report("MTR", *results["MTR: D(1)<=D(0)"][:2],
                  los_m.std(ddof=1), ups_m.std(ddof=1))

    # ---- PART 5 ----
    banner("PART 5 — 可检验含义 & 保存")
    log(f"\n  MLE 结构 AME = {ame_b:+.4f}")
    for tag in ["worst-case (无假设)", "MTR: D(1)<=D(0)"]:
        lo, up, _ = results[tag]
        log(f"  {tag:<24s} [{lo:+.4f}, {up:+.4f}] -> "
            f"{'INSIDE' if lo <= ame_b <= up else 'OUTSIDE（拒绝该设定）'}")

    pd.DataFrame([
        dict(model="probit_naive", param="beta", est=b_p[1]),
        dict(model="probit_naive", param="AME", est=ame_p),
        dict(model="biprobit", param="beta", est=beta, se=se_beta),
        dict(model="biprobit", param="rho", est=rho, se=se_rho),
        dict(model="biprobit", param="AME", est=ame_b),
        dict(model="partial_id", param="set_lo_worst", est=results["worst-case (无假设)"][0]),
        dict(model="partial_id", param="set_up_worst", est=results["worst-case (无假设)"][1]),
        dict(model="partial_id", param="set_lo_mtr", est=results["MTR: D(1)<=D(0)"][0]),
        dict(model="partial_id", param="set_up_mtr", est=results["MTR: D(1)<=D(0)"][1]),
    ]).to_csv(OUT_RES, index=False, encoding="utf-8-sig")
    log(f"\n  [SAVED] {OUT_RES}")

    pd.DataFrame([
        dict(restriction="worst_case", stat="set_lower", value=results["worst-case (无假设)"][0]),
        dict(restriction="worst_case", stat="set_upper", value=results["worst-case (无假设)"][1]),
        dict(restriction="worst_case", stat="ci_lower", value=ci_w[0]),
        dict(restriction="worst_case", stat="ci_upper", value=ci_w[1]),
        dict(restriction="MTR", stat="set_lower", value=results["MTR: D(1)<=D(0)"][0]),
        dict(restriction="MTR", stat="set_upper", value=results["MTR: D(1)<=D(0)"][1]),
        dict(restriction="MTR", stat="ci_lower", value=ci_m[0]),
        dict(restriction="MTR", stat="ci_upper", value=ci_m[1]),
    ]).to_csv(OUT_BND, index=False, encoding="utf-8-sig")
    log(f"  [SAVED] {OUT_BND}")
    log(f"\n  done in {time.time()-_T0:.0f}s")
    flush_log()


if __name__ == "__main__":
    main()