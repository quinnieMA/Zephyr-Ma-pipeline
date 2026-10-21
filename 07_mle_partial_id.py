# -*- coding: utf-8 -*-
"""
ii_v28.py — 四方程联立 + 间接推断 + Monte Carlo 验证
================================================================================
系统：
  (1) R* = X b1 + e1            R = 1[R*>0]
  (2) D* = X b2 + γ R + e2      D = 1[D*>0]
  (3) logP = X b3 + κ R + σ e3  P 仅在 D=1 且 B=1 时观测
  (4) B* = X b4 + ζ1 R + ζ2 P + e4,  B = 1[B*>0]
  (e1,e2,e3,e4) ~ N(0, Σ)

估计：间接推断 (Gourieroux et al. 1993; Smith et al. 1993)
  辅助模型 = 结构模型的受限可处理版本 (Calvet & Czellar 2015)
  θ* = argmin_θ (m_data - E_θ[m̂])' W (m_data - E_θ[m̂])
================================================================================
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, r"D:\MA\data")
from config import (load_df, rename, build_derived, apply_sample,
                    HARD_CONTROLS, INSTRUMENTS, YEAR_FE, MAX_YEAR_FE, SEED)

OUT_II  = r"D:\MA\data\merged\ii_v28_results.csv"
OUT_MC  = r"D:\MA\data\merged\ii_v28_monte_carlo.csv"
LOG_OUT = r"D:\MA\data\merged\ii_v28_log.txt"

S_SIM      = 20
N_SIM      = 30000
MC_SAMPLES = 30
MC_N       = 20000
MAXITER    = 150
SEED       = 20261020
ALPHA      = 0.05

# 结构参数真值（MC 用，仅作偏差参照）
THETA0 = dict(gamma=0.15, kappa=0.10, zeta1=-0.20, zeta2=0.30,
              rho=0.05, lam=0.05, rOL=0.15, r23=-0.10)

_L = []; _T0 = time.time()
def log(m=""):
    print(m, flush=True); _L.append(str(m))
def banner(t):
    log("\n" + "=" * 74); log(t); log("=" * 74)
def flush():
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_L) + "\n")


def make_sigma(rho, lam, rOL, r23):
    """4x4 相关矩阵；r14=rOL*lam, r24=rOL*r23 是因子结构推论，非自由参数"""
    r14, r24 = rOL * lam, rOL * r23
    return np.array([[1, rho, lam, r14],
                     [rho, 1, r23, r24],
                     [lam, r23, 1, rOL],
                     [r14, r24, rOL, 1]])


def _lege(n, a, b):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * (b - a) * w


def _add1(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def _ols(X, y):
    return np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]


def _logit_fit(X, y, maxiter=200):
    X = np.asarray(X, float); y = np.asarray(y, float)
    q = 2.0 * y - 1.0
    def nll(b):
        z = X @ b
        return -float(np.sum(norm.logcdf(q * z)))
    def grad(b):
        z = X @ b
        return -(X.T @ (q * norm.pdf(z) / norm.cdf(q * z).clip(1e-12, None)))
    b0 = np.zeros(X.shape[1])
    b0[0] = norm.ppf(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    return minimize(nll, b0, jac=grad, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-10)).x


def auxiliary(data, Xc, Xt, Xb):
    """data: DataFrame（真实数据）或 dict（simulate 返回）"""
    if isinstance(data, pd.DataFrame):
        R = np.asarray(data["rumor"], float)
        D = np.asarray(data["completed_strict"], float)
        P = pd.to_numeric(data.get("multiple_ln"), errors="coerce").values
    else:
        R, D, P = data["R"], data["D"], data["P"]

    m = []
    m.append(_logit_fit(Xc, R)[:6])                       # (1) R*: 前6项

    X2 = _add1(np.column_stack([Xc, R]))
    b2 = _logit_fit(X2, D)
    m.append(np.array([b2[Xc.shape[1] + 1]]))             # (2) γ

    ok3 = np.isfinite(P) & (D == 1)
    if ok3.sum() > 100:
        X3 = _add1(np.column_stack([Xt[ok3], R[ok3]]))
        m.append(np.array([_ols(X3, P[ok3])[-1]]))        # (3) κ
    else:
        m.append(np.array([np.nan]))

    ok4 = np.isfinite(P) & (D == 1)
    if ok4.sum() > 100:
        X4 = _add1(np.column_stack([Xb[ok4], R[ok4], P[ok4]]))
        m.append(_logit_fit(X4, np.ones(ok4.sum(), float))[-2:])  # (4) ζ1, ζ2
    else:
        m.append(np.array([np.nan, np.nan]))

    return np.concatenate([np.array(x).ravel() for x in m])

def simulate(theta, Xc, Xt, Xb, n=None, seed=None):
    rng = np.random.default_rng(seed)
    n = Xc.shape[0] if n is None else n
    S = make_sigma(theta["rho"], theta["lam"], theta["rOL"], theta["r23"])
    L = np.linalg.cholesky(S)
    e = (L @ rng.standard_normal((4, n))).T          # (n,4) 一次生成
    R = (Xc @ theta["b1"] + e[:, 0] > 0).astype(float)
    D = (Xc @ theta["b2"] + theta["gamma"] * R + e[:, 1] > 0).astype(float)
    logP = Xt @ theta["b3"] + theta["kappa"] * R + e[:, 2]
    B = (Xb @ theta["b4"] + theta["zeta1"] * R
         + theta["zeta2"] * logP + e[:, 3] > 0).astype(float)
    # [FIX] 返回数组，不构造 DataFrame
    return dict(R=R, D=D, P=np.where((D == 1) & (B == 1), logP, np.nan),
                B=B), None

# =====================================================================
#  关键修正：自由参数统一拼成向量，优化完再拆回 dict
# =====================================================================
def pack(theta, sh):
    """theta: dict -> 一维数组"""
    return np.concatenate([
        theta["b1"].ravel(), theta["b2"].ravel(), theta["b3"].ravel(),
        theta["b4"].ravel(),
        np.array([theta["gamma"], theta["kappa"], theta["zeta1"],
                  theta["zeta2"], theta["rho"], theta["lam"],
                  theta["rOL"], theta["r23"]]),
    ])


def unpack(a, sh):
    """一维数组 -> theta dict"""
    k1, k2, k3, k4 = sh
    i = 0
    th = {}
    for name, k in [("b1", k1), ("b2", k2), ("b3", k3), ("b4", k4)]:
        th[name] = a[i:i + k].copy(); i += k
    for name in ["gamma", "kappa", "zeta1", "zeta2",
                 "rho", "lam", "rOL", "r23"]:
        th[name] = float(a[i]); i += 1
    return th

    
def indirect_inference(df, Xc, Xt, Xb, theta_init, S=S_SIM,
                       seed=SEED, verbose=True):
    m_data = auxiliary(df, Xc, Xt, Xb)          # DataFrame 路径
    sh = (Xc.shape[1], Xc.shape[1], Xt.shape[1], Xb.shape[1])
    theta0 = pack(theta_init, sh)
    W = np.eye(len(m_data))

    def Q(a):
        th = unpack(a, sh)
        ms = np.zeros_like(m_data)
        for s in range(S):
            d, _ = simulate(th, Xc, Xt, Xb, n=len(df), seed=seed + s)  # dict 路径
            ms += auxiliary(d, Xc, Xt, Xb)
        ms /= S
        d = m_data - ms
        return float(d.T @ W @ d)

    r = minimize(Q, theta0, method="Nelder-Mead",
                 options=dict(maxiter=MAXITER, xatol=1e-5, fatol=1e-5))
    return unpack(r.x, sh), r.fun


def monte_carlo(theta0, Xc, Xt, Xb, n_mc=MC_SAMPLES, seed=SEED + 7):
    banner("PART D — Monte Carlo 验证")
    rng = np.random.default_rng(seed)
    names = ["gamma", "kappa", "zeta1", "zeta2", "rho", "lam", "rOL", "r23"]
    est = {k: [] for k in names}
    sh = (Xc.shape[1], Xc.shape[1], Xt.shape[1], Xb.shape[1])
    t0 = time.time()
    for mc in range(n_mc):
        sd = seed + mc * 137
        d, _ = simulate(theta0, Xc, Xt, Xb, n=MC_N, seed=sd)
        try:
            ti = dict(theta0)
            for k in names:
                ti[k] = theta0[k] * (1 + rng.standard_normal() * 0.15)
            th, _ = indirect_inference(d, Xc, Xt, Xb, ti, S=15,
                                       seed=sd, verbose=False)
            for k in names:
                est[k].append(th.get(k, np.nan))
        except Exception:
            for k in names:
                est[k].append(np.nan)
        if (mc + 1) % 10 == 0:
            log(f"    MC {mc+1}/{n_mc} [{time.time()-t0:.0f}s]")
    return est


def main():
    banner("ii_v28.py — 四方程间接推断 + MC 验证")
    log(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = apply_sample(build_derived(rename(load_df())))
        # main() 里，df 建好之后立刻抽子样
    S_SIM      = 8      # 20 -> 8   （速度 ×2.5）
    N_SIM      = 8000   # 152,755 -> 8,000（速度 ×19）
    MAXITER    = 60     # 150 -> 60
    MC_SAMPLES = 20
    MC_N       = 5000
    
    R = pd.to_numeric(df["rumor"], errors="coerce")
    D = pd.to_numeric(df["completed_strict"], errors="coerce")
    ok = R.notna() & D.notna()
    df = df.loc[ok].reset_index(drop=True)
    log(f"  N = {len(df):,}  p_A={R[ok].mean():.4f}  p_D={D[ok].mean():.4f}")

    use_c = [c for c in HARD_CONTROLS if c in df.columns]
    use_z = [c for c in INSTRUMENTS if c in df.columns and c not in use_c]

    C = (df[use_c].apply(pd.to_numeric, errors="coerce")
         .fillna(df[use_c].median(numeric_only=True)).to_numpy(float))
    Zraw = (df[use_z].apply(pd.to_numeric, errors="coerce")
            .fillna(0).to_numpy(float))
    if YEAR_FE and "deal_year" in df.columns:
        yr = pd.to_numeric(df["deal_year"], errors="coerce")
        top = yr.value_counts().head(MAX_YEAR_FE).index
        ye = pd.get_dummies(yr.where(yr.isin(top)), prefix="y").astype(float)
        C = np.column_stack([C, ye.to_numpy(float)])
        Z = np.column_stack([Zraw, np.ones((len(df), 1))])   # 年份 FE 只进 C
    else:
        Z = np.column_stack([Zraw, np.ones((len(df), 1))])
    Xc = _add1(np.column_stack([Z, C]))
    Xt = _add1(C)
    Xb = _add1(C)
    log(f"  Xc={Xc.shape}  Xt={Xt.shape}  Xb={Xb.shape}")

    P = pd.to_numeric(df.get("multiple_ln"), errors="coerce").values
    ok3 = np.isfinite(P) & (D.values == 1)
    log(f"  倍数子样本 (D=1 且有倍数): {int(ok3.sum()):,}")

    banner("PART A — 辅助模型")
    m_data = auxiliary(df, Xc, Xt, Xb)
    log(f"  m_data 维度 = {m_data.shape}")

    theta_init = dict(
        b1=np.zeros(Xc.shape[1]), b2=np.zeros(Xc.shape[1]),
        b3=np.zeros(Xt.shape[1]), b4=np.zeros(Xb.shape[1]),
        **{k: THETA0[k] for k in ["gamma", "kappa", "zeta1", "zeta2",
                                  "rho", "lam", "rOL", "r23"]})
    log(f"  自由参数维度 = {pack(theta_init, (Xc.shape[1],)*2 + (Xt.shape[1],)*2).shape}")
    log(f"  θ0 = {THETA0}")

    banner("PART B — 间接推断")
    t0 = time.time()
    theta_hat, Qmin = indirect_inference(df, Xc, Xt, Xb, theta_init)
    log(f"\n  {'param':<8s} {'est':>10s} {'θ0':>10s} {'diff':>10s}")
    for k in ["gamma", "kappa", "zeta1", "zeta2", "rho", "lam", "rOL", "r23"]:
        v, v0 = theta_hat.get(k), THETA0[k]
        log(f"  {k:<8s} {v:>+10.4f} {v0:>+10.4f} {v-v0:>+10.4f}")
    log(f"  Q(θ̂) = {Qmin:.6f}  [{time.time()-t0:.0f}s]")

    banner("PART C — Monte Carlo")
    est = monte_carlo(theta_init, Xc, Xt, Xb)
    rows = []
    for k in THETA0:
        e = np.array([x for x in est[k] if not np.isnan(x)])
        bias = e.mean() - THETA0[k]
        rmse = np.sqrt(((e - THETA0[k]) ** 2).mean())
        se = e.std(ddof=1)
        cp = float(np.mean(
            (THETA0[k] - 1.96 * se <= e) &
            (e <= THETA0[k] + 1.96 * se)
        ))

        rows.append(dict(param=k, true=THETA0[k], bias=bias, rmse=rmse,
                         mean=e.mean(), std=se, coverage=cp, n=len(e)))
        log(f"  {k:<8s} true={THETA0[k]:+.4f} mean={e.mean():+.4f} "
            f"bias={bias:+.4f} rmse={rmse:.4f} coverage={cp:.3f}")
    pd.DataFrame(rows).to_csv(OUT_MC, index=False, encoding="utf-8-sig")
    log(f"\n  [SAVED] {OUT_MC}")

    pd.DataFrame([dict(gamma=theta_hat.get("gamma"), kappa=theta_hat.get("kappa"),
                       rho=theta_hat.get("rho"), rOL=theta_hat.get("rOL"),
                       Q_min=Qmin)]).to_csv(OUT_II, index=False,
                                            encoding="utf-8-sig")
    log(f"\n  [SAVED] {OUT_II}")
    log(f"\n  done in {time.time()-_T0:.0f}s")
    flush()


if __name__ == "__main__":
    main()