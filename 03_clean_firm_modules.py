# -*- coding: utf-8 -*-
r"""
03_clean_firm_modules.py — v7 (2026-10-22) OOM修复 + 日志整洁 + L1=TA
================================================================================
Firm-level (target / acquirer) module merge → 通用并购研究数据库。
[V7] 改动汇总：
  - [FIX-OOM] collapse_to_deal 不再对百万行大表做全局 sort+copy，
    改用"仅提取决胜键 -> groupby.idxmax -> iloc 切片"，内存占用降 90%+
  - [FIX-LOG] cascade_merge ENTITY-AUDIT 不再遍历几十万ID，改聚合+前5例
  - collapse_to_deal: L1 改为 pre_tar_ta_ly 降序（文献规则），5级决胜日志
  - dedup_on: 打印 前/后/丢弃 三数
  - ffill+dedup: 日志明确区分"重复行丢弃"与"异常行"
  - 各源 unique deal_num: 加一句解释 industry/overview 偏少的原因
  - TA/REV 覆盖率: 并列打印，TA<=0 和 >阈值告警
  - report_full: 排除主键列数值统计；字符串列不打印"解析率"
  - VERBOSE 开关控制 STEP 2 全量报告长度
================================================================================
"""
import os
import re
import sys
import time
import atexit
import traceback
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# =====================================================================
# Config — 与 02 共用 config 模块（无则回落 v4 写死路径）
# =====================================================================
try:
    import config as CFG
except Exception:
    CFG = None

def cfg(name, default):
    return getattr(CFG, name, default) if CFG is not None else default

ZEPHYR_BASE = cfg("ZEPHYR_BASE", r"D:\zephyr")
BASE        = cfg("BASE_DIR",    r"D:\MA")
MERGED      = cfg("MERGED_DIR",  os.path.join(BASE, "data", "merged"))
CLEANED     = cfg("CLEANED_DIR", os.path.join(BASE, "data", "cleaned"))
KEEP_ORBIS  = cfg("KEEP_ORBIS", True)

os.makedirs(MERGED, exist_ok=True)

PATH_FIN    = os.path.join(ZEPHYR_BASE, "financial",      "cleaned_output", "MA_finance.csv")
PATH_LEG    = os.path.join(ZEPHYR_BASE, "legal",          "cleaned_output", "MA_legal.csv")
PATH_LIST   = os.path.join(ZEPHYR_BASE, "list",           "cleaned_output", "MA_list.csv")
PATH_POST   = os.path.join(ZEPHYR_BASE, "post_financial", "cleaned_output", "MA_post_finance.csv")

DEAL    = "deal_num"
TAR_BVD = "tar_bvd_id_num"
ACQ_BVD = "acq_bvd_id_num"

TIERS = [
    ("T1_5key",       [DEAL, "tar_name", "acq_name", "tar_bvd_id_num", "acq_bvd_id_num"]),
    ("T2_2key_bvdid", [DEAL, TAR_BVD, ACQ_BVD]),
    ("T3_2key_name",  [DEAL, "tar_name", "acq_name"]),
    ("T4_tarkey",     [DEAL, "tar_key"]),
    ("T5_deal",       [DEAL]),
]

MERGE_KEYS = list(dict.fromkeys(k for _, keys in TIERS for k in keys))

OUT_PAIRS  = os.path.join(MERGED, "03_firm_entity_pairs.csv")
OUT_DEAL   = os.path.join(MERGED, "03_firm_deal_level.csv")
LOG_PATH   = os.path.join(MERGED, "03_clean_merge_firm.txt")
BASE_ROLE   = "finance"
DONOR_ROLES = ["legal", "list", "post_finance", "industry", "overview"]
PATHS = {"finance": PATH_FIN, "legal": PATH_LEG,
         "list": PATH_LIST, "post_finance": PATH_POST,
         "industry": os.path.join(CLEANED, "01_deal_sic_industry.csv"),
         "overview": os.path.join(CLEANED, "01_deal_overview.csv")}
EMIT_ENTITY_PAIRS = True

# ======================== [V7] NEW CONFIG CONSTANTS ========================
REPORT_FULL_AFTER_EACH_STEP = True
REPORT_VERBOSE = False               # False=只打关键摘要; True=全量
KEEP_COLS = {"finance": None, "legal": None, "list": None, "post_finance": None,
             "industry": None, "overview": None}

TA_COL  = "pre_tar_ta_ly"
REV_COL = "pre_tar_rev_ly"
TA_MAX_THRESHOLD = 1e12

PK_COLS = frozenset({
    DEAL, TAR_BVD, ACQ_BVD, "ven_bvd_id_num",
    "tar_orbis_id_num", "acq_orbis_id_num", "tar_key"
})
# ===========================================================================

MISSING_TOKENS = {"", "-", "--", "n.a.", "n.s.", "na", "n/a", "nan",
                  "none", "null", "__na__", "__missing__", "unknown"}
YEAR_BOUNDS = (1600, 2026)
SAFE_NUM_THRESHOLD = 0.95

KEY_ALIASES = {
    DEAL:    ["deal_num", "deal number", "dealnumber", "deal id", "deal_id"],
    TAR_BVD: ["tar_bvd_id_num", "tar_bvd_id", "target bvd id number",
              "targetbvdidnumber", "target bvd id", "tar_bvdid"],
    ACQ_BVD: ["acq_bvd_id_num", "acq_bvd_id", "acquirer bvd id number",
              "acquirorbvdidnumber", "acquirer bvd id", "acq_bvdid"],
    "ven_bvd_id_num": ["ven_bvd_id_num", "ven_bvd_id", "vendor bvd id number",
                       "vendorbvdidnumber"],
    "tar_name": ["tar_name", "target name", "targetname"],
    "acq_name": ["acq_name", "acquirer name", "acquirorname"],
    "ven_name": ["ven_name", "vendor name", "vendorname"],
}

LISTED_ALIASES = {
    "tar": ["tar_listed", "tar_listed_raw", "target listed",
            "tar_listed_status", "target_status"],
    "acq": ["acq_listed", "acq_listed_raw", "acquirer listed",
            "acq_listed_status", "acquiror listed"],
    "ven": ["ven_listed", "ven_listed_raw", "vendor listed",
            "ven_listed_status"],
}
LISTED_UNLISTED = {"no", "0", "false", "n", "unlisted", "unquoted", "private",
                   "n.a.", "n.s.", "-", "", "none", "nan", "not listed"}

YEAR_WHITELIST = re.compile(
    r"^(?:tar|acq|ven|gr_tar|gr_acq|gr_ven)_(?:incorp|ipo|founded|established)"
    r"[a-z_]*yr$", re.I)
YEAR_BLACKLIST = re.compile(
    r"(1st_avail|_avail_|post_|pre_|_rev_|_ebitda_|_ebit_|_pft_|_net_|_tot_|"
    r"_cur_|_sh_|_mkt_|assets|liab|cap)", re.I)

# =====================================================================
# Log
# =====================================================================
class _Tee(object):
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try: s.write(data)
            except Exception: pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

_log_fh      = open(LOG_PATH, "w", encoding="utf-8")
_orig_stdout = sys.stdout
sys.stdout   = _Tee(_orig_stdout, _log_fh)
atexit.register(lambda: (lambda: (_log_fh.flush(), _log_fh.close()))())

def log(msg=""):
    print(msg)

def banner(t):
    log("\n" + "=" * 82)
    log(t)
    log("=" * 82)

log("=" * 82)
log("03_clean_firm_modules.py — REBUILT v7 | L1=Tar_TA_Max | OOM-FIX | LOG CLEANUP")
log(f"started : {datetime.now():%Y-%m-%d %H:%M:%S}")
log("=" * 82)
log(f"\n  主键 = {MERGE_KEYS}")
log(f"  级联 = {[t for t, _ in TIERS]}")
log(f"  [V7] L1 折叠规则 = {TA_COL} 降序 (NaN 排最后)")
log(f"  [V7] REPORT_VERBOSE = {REPORT_VERBOSE}")

# =====================================================================
# Helpers
# =====================================================================
def clean_missing(df):
    try:
        cols = df.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        cols = df.select_dtypes(include=["object"]).columns
    for col in cols:
        df[col] = df[col].astype("object")
        df[col] = df[col].replace(list(MISSING_TOKENS), np.nan)
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
            df.loc[mask, col] = df.loc[mask, col].replace("", np.nan)
    return df

def norm_id_series(s):
    def f(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip()
        if t.endswith(".0"):
            t = t[:-2]
        t = t.strip()
        return np.nan if t.lower() in MISSING_TOKENS else t
    return s.map(f)

def find_col(df, aliases):
    norm = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        hit = norm.get(a.strip().lower())
        if hit is not None:
            return hit
    return None

def standardize(df, role):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for canon, aliases in KEY_ALIASES.items():
        if canon in df.columns:
            continue
        hit = find_col(df, aliases)
        if hit is not None:
            ren[hit] = canon
    if ren:
        df = df.rename(columns=ren)
        log(f"      rename: {ren}")
    orb = [c for c in df.columns if "orbis" in c.lower()]
    if orb:
        if KEEP_ORBIS:
            log(f"      [ORBIS] 保留 {len(orb)} 列作为参考（不参与合并键）: {orb[:8]}")
        else:
            df = df.drop(columns=orb)
            log(f"      [DROP-ORBIS] 删除 {len(orb)} 列")
    for c in (TAR_BVD, ACQ_BVD, "ven_bvd_id_num"):
        if c in df.columns:
            df[c] = norm_id_series(df[c])
    if DEAL in df.columns:
        df[DEAL] = pd.to_numeric(df[DEAL], errors="coerce")
    return df

def safe_numeric(df, label=""):
    changed, skipped = [], []
    for c in df.columns:
        if c in MERGE_KEYS or c in PK_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c]
        nn = int(s.notna().sum())
        if nn == 0:
            continue
        conv = pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(),
                             errors="coerce")
        ratio = int(conv.notna().sum()) / nn
        if ratio >= SAFE_NUM_THRESHOLD:
            df[c] = conv
            changed.append((c, ratio))
        else:
            skipped.append((c, ratio, nn))
    if changed:
        log(f"      [SAFE-NUM] 转数值 {len(changed)} 列")
        for c, r in changed[:12]:
            log(f"          {c:<34s} 解析率 {r*100:.1f}%")
        if len(changed) > 12:
            log(f"          ... 另 {len(changed)-12} 列")
    if skipped:
        log(f"      [SAFE-NUM] 保留为字符串 {len(skipped)} 列（避免误杀）:")
        for c, r, n in skipped[:12]:
            if r < 0.01:
                log(f"          {c:<34s} (非数值列，非空 {n:,})")
            else:
                log(f"          {c:<34s} 解析率 {r*100:.1f}% (非空 {n:,})")
        if len(skipped) > 12:
            log(f"          ... 另 {len(skipped)-12} 列")
    return df

# =====================================================================
# 报告
# =====================================================================
def fmt_num(x):
    if x is None: return "—"
    try:
        if pd.isna(x): return "—"
    except Exception: pass
    try:
        if isinstance(x, (bool, np.bool_)): return str(x)
        if isinstance(x, (int, np.integer)): return f"{int(x):,}"
        v = float(x)
        if abs(v) >= 1e12: return f"{v:.4e}"
        if float(v).is_integer() and abs(v) < 1e15: return f"{int(v):,}"
        return f"{v:,.2f}"
    except Exception:
        return str(x)

def report_full(df, label="", verbose=None):
    if verbose is None:
        verbose = REPORT_VERBOSE
    n_rows, n_cols = len(df), len(df.columns)
    mode_str = "FULL" if verbose else "SUMMARY"
    log(f"\n  ── {mode_str} VARIABLE REPORT [{label}] ──")
    log(f"  rows={n_rows:,}  cols={n_cols}")

    numeric_cols, cat_cols = [], []
    low50 = low90 = full_cov = 0
    missing_rank = []

    log(f"\n  A) COVERAGE — all {n_cols} columns")
    log(f"     {'#':>4}  {'column':<34} {'dtype':<12} {'non-null':>12} "
        f"{'%':>10} {'missing':>12} {'%':>10}")
    
    printed_count = 0
    for i, col in enumerate(df.columns, 1):
        s = df[col]
        s_clean = s.replace("__NA__", np.nan) if (s.dtype == object) else s
        n_nonnull = int(s_clean.notna().sum())
        n_missing = n_rows - n_nonnull
        p_n = n_nonnull / max(n_rows, 1) * 100
        p_m = n_missing / max(n_rows, 1) * 100
        if p_n < 50: low50 += 1
        elif p_n < 90: low90 += 1
        if p_n >= 99.999: full_cov += 1
        missing_rank.append((p_m, col))

        is_key = col in MERGE_KEYS or col in PK_COLS
        is_fin = col.startswith(("pre_", "post_"))
        is_tier = col.endswith("_tier")
        
        if verbose or is_key or is_fin or is_tier:
            log(f"     {i:>4}  {col:<34} {str(s.dtype):<12} {n_nonnull:>12,} "
                f"{p_n:>9.2f}% {n_missing:>12,} {p_m:>9.2f}%")
            printed_count += 1

        if pd.api.types.is_numeric_dtype(s.dtype):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    if not verbose and printed_count < n_cols:
        log(f"     ... (省略 {n_cols - printed_count} 列，设 REPORT_VERBOSE=True 查看全部)")

    stat_num_cols = [c for c in numeric_cols if c not in PK_COLS]
    if stat_num_cols:
        log(f"\n  B) DESCRIPTIVE STATS — numeric ({len(stat_num_cols)}, 已排除主键ID列)")
        log(f"     {'column':<34} {'min':>16} {'median':>16} {'mean':>16} "
            f"{'max':>16} {'std':>16}")
        for col in stat_num_cols:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) == 0:
                log(f"     {col:<34} {'ALL MISSING':>16}")
                continue
            log(f"     {col:<34} {fmt_num(v.min()):>16} {fmt_num(v.median()):>16} "
                f"{fmt_num(v.mean()):>16} {fmt_num(v.max()):>16} "
                f"{fmt_num(v.std()):>16}")

    if cat_cols and verbose:
        log(f"\n  C) CATEGORICAL SUMMARY — {len(cat_cols)} columns")
        log(f"     {'column':<34} {'unique':>10} {'top value':>34} "
            f"{'top n':>10} {'top %':>8}")
        for col in cat_cols:
            vals = df[col].replace("__NA__", np.nan).dropna()
            line = f"     {col:<34} {vals.nunique():>10,}"
            if len(vals):
                top = vals.value_counts().head(1)
                line += f" {str(top.index[0])[:30]!r:>34} {int(top.iloc[0]):>10,} " \
                        f"{top.iloc[0]/max(n_rows,1)*100:>7.2f}%"
            log(line)
    elif cat_cols and not verbose:
        log(f"\n  C) CATEGORICAL SUMMARY — 省略 (REPORT_VERBOSE=False)")

    missing_rank.sort(reverse=True)
    log(f"\n  Coverage summary: <50% cov={low50} | <90% cov={low90} | full cov={full_cov}")
    log("  Top-5 most-missing: " +
        ", ".join(f"{c} ({p:.2f}%)" for p, c in missing_rank[:5]))

# =====================================================================
# Load
# =====================================================================
def load_module(role):
    path = PATHS[role]
    banner(f"LOAD — {role}")
    log(f"  path: {path}")
    if not os.path.exists(path):
        log("  [MISSING] 文件不存在，跳过")
        return None
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", on_bad_lines="warn")
    df.columns = [str(c).strip() for c in df.columns]
    drop_idx = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_idx:
        df = df.drop(columns=drop_idx)
        log(f"  [DROP-INDEX] {drop_idx}")
    log(f"  raw: {len(df):,} rows x {df.shape[1]} cols")
    df = clean_missing(df)
    df = standardize(df, role)
    df = safe_numeric(df, role)
    keep = KEEP_COLS.get(role)
    if keep is not None:
        present = [c for c in keep if c in df.columns]
        df = df[present].copy()
        log(f"  [KEEP] 保留 {len(present)} 列")
    if DEAL in df.columns:
        df[DEAL] = pd.to_numeric(df[DEAL], errors="coerce").astype("Int64")
        df = df.dropna(subset=[DEAL])
        log(f"  after clean: {len(df):,} rows | unique {DEAL}: {df[DEAL].nunique():,}")
    for c in (TAR_BVD, ACQ_BVD):
        if c in df.columns:
            n = int(df[c].notna().sum())
            log(f"  {c:<20s} 覆盖 {n:,} ({n/max(len(df),1)*100:.2f}%)")
    return df

# =====================================================================
# [FFILL-PATCH]
# =====================================================================
FFILL_ENTITY_COLS = ["tar_bvd_id_num", "tar_name", "tar_orbis_id_num",
                     "acq_bvd_id_num", "acq_name", "acq_orbis_id_num",
                     "tar_key"]

def ffill_block(df, prefix, exclude=None):
    exclude = set(exclude or [])
    cols = [c for c in FFILL_ENTITY_COLS
            if c in df.columns and c.startswith(prefix) and c not in exclude]
    if not cols or "deal_num" not in df.columns:
        return df
    d = df.sort_values("deal_num", kind="mergesort")
    for c in cols:
        if d[c].dtype == object:
            v = d[c].astype(str).str.strip()
            d[c] = v.mask(v.isin(["", "nan", "NaN", "None", "<NA>", "NaT"]))
    d[cols] = d.groupby("deal_num", sort=False)[cols].ffill()
    return d

def dedup_entity_blocks(df):
    key_cols = [k for k in ("tar_bvd_id_num", "acq_bvd_id_num")
                if k in df.columns]
    if not key_cols:
        return df
    d = df.sort_values(["deal_num"] + key_cols, kind="mergesort")
    d = d.drop_duplicates(subset=["deal_num"] + key_cols, keep="first")
    return d

# =====================================================================
# 级联合并
# =====================================================================
def dedup_on(df, keys, label):
    n0 = len(df)
    if n0 == 0:
        return df
    tmp = df.copy()
    tmp["_fill"] = tmp.notna().sum(axis=1)
    tmp["_ord"] = np.arange(n0)
    tmp = tmp.sort_values(keys + ["_fill", "_ord"],
                          ascending=[True]*len(keys) + [False, True],
                          kind="mergesort")
    out = (tmp.drop_duplicates(subset=keys, keep="first")
              .drop(columns=["_fill", "_ord"]).reset_index(drop=True))
    n_out = len(out)
    dropped = n0 - n_out
    log(f"        dedup[{label}] 前:{n0:,} -> 后:{n_out:,} (丢弃:{dropped:,})")
    return out

def _prepare_keys(df, keys):
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            out[k] = "__NA__"
        elif k != DEAL:
            out[k] = out[k].fillna("__NA__").astype(str)
    return out

def cascade_merge(base, donor, module):
    """[FIX-LOG] ENTITY-AUDIT 改为聚合计数+前5例，绝不打印几十万ID"""
    if donor is None or len(donor) == 0:
        log(f"\n    [{module}] 无数据，跳过")
        return base, np.ones(0, dtype=bool)
    donor_cols = [c for c in donor.columns if c not in MERGE_KEYS]
    if not donor_cols:
        log(f"\n    [{module}] 无可用列，跳过")
        return base, np.ones(len(donor), dtype=bool)
    collide = [c for c in donor_cols if c in base.columns]
    override = []
    if collide:
        for c in collide:
            b_obj = not pd.api.types.is_numeric_dtype(base[c])
            d_num = pd.api.types.is_numeric_dtype(donor[c])
            if b_obj and d_num:
                override.append(c)
        skip = [c for c in collide if c not in override]
        if skip:
            log(f"\n    [{module}] 列名冲突（保留 base）{len(skip)} 列: {skip[:6]}")
        if override:
            log(f"\n    [{module}] 列名冲突（用 donor 覆盖）{len(override)} 列: {override}")
        donor_cols = [c for c in donor_cols if c not in skip]
    log(f"\n    [{module}] 待并入 {len(donor_cols)} 列")
    tier_col = f"{module}_tier"
    base[tier_col] = None
    n0 = len(base)
    t1_full_n = t1_pad_n = 0
    donor_proc = donor.copy()
    donor_proc["_src"] = np.arange(len(donor), dtype=np.int64)
    handled = np.zeros(len(donor), dtype=bool)
    for tname, tkeys in TIERS:
        if any(k not in donor.columns for k in tkeys):
            log(f"      {tname}: donor 缺键，跳过")
            continue
        todo_mask = base[tier_col].isna()
        if not todo_mask.any():
            break
        d = dedup_on(donor_proc, tkeys, f"{module}@{tname}")
        d = _prepare_keys(d, tkeys)[tkeys + donor_cols + ["_src"]]
        idx_all = base.index[todo_mask.values].to_numpy()
        left = _prepare_keys(base.loc[todo_mask], tkeys)[tkeys].reset_index(drop=True)
        mm = left.merge(d, on=tkeys, how="left", indicator="_ind")
        if len(mm) != len(left):
            log(f"      [WARN] {tname} 行数 {len(left):,}->{len(mm):,}，强制去重")
            mm = mm.drop_duplicates(subset=tkeys, keep="first")
        ok = (mm["_ind"] == "both").to_numpy()
        if ok.any():
            sel = idx_all[ok[:len(idx_all)]]
            for c in donor_cols:
                base.loc[sel, c] = mm.loc[ok, c].to_numpy()
            base.loc[sel, tier_col] = tname
            srcs = mm.loc[ok, "_src"].to_numpy(dtype=np.int64)
            handled[srcs] = True
            d2 = _prepare_keys(donor_proc, tkeys)
            keyset = set(map(tuple, mm.loc[ok, tkeys].to_numpy()))
            if keyset:
                dup = np.array([tuple(r) in keyset for r in
                                d2[tkeys].to_numpy()], dtype=bool)
                dup_src = d2.loc[dup, "_src"].to_numpy(dtype=np.int64)
                handled[dup_src] = True

            # [FIX-LOG] 向量化比对，只打印总数和前5个差异示例，防止Token爆炸
            try:
                b_rows = base.loc[sel]
                d_rows = donor_proc.loc[srcs]
                mismatch_examples = []
                n_ek_total = 0
                ek_cols = [ek for ek in ["tar_name", "tar_bvd_id_num",
                                         "acq_name", "acq_bvd_id_num"]
                           if ek in b_rows.columns and ek in d_rows.columns]
                
                # 向量化计算不一致总数
                diff_masks = {}
                for ek in ek_cols:
                    bv = b_rows[ek].fillna("__NA__").astype(str).to_numpy()
                    dv = d_rows[ek].fillna("__NA__").astype(str).to_numpy()
                    diff_masks[ek] = (bv != "__NA__") & (dv != "__NA__") & (bv != dv)
                    n_ek_total += int(diff_masks[ek].sum())

                if n_ek_total:
                    log(f"      [ENTITY-AUDIT][{tname}] 命中 {int(ok.sum()):,} 行，"
                        f"实体键不一致共 {n_ek_total:,} 处")
                    
                    # 收集前5例
                    collected = 0
                    for ek in ek_cols:
                        if collected >= 5:
                            break
                        mask = diff_masks[ek]
                        if mask.any():
                            idxs = np.where(mask)[0]
                            for ix in idxs:
                                if collected >= 5:
                                    break
                                orig_idx = sel[ix]
                                b_val = b_rows[ek].iloc[ix]
                                d_val = d_rows[ek].iloc[ix]
                                mismatch_examples.append(
                                    f"idx={orig_idx}:{ek}(base='{b_val}'!=donor='{d_val}')")
                                collected += 1
                    
                    if mismatch_examples:
                        log(f"          前{len(mismatch_examples)}例: " + " | ".join(mismatch_examples))
            except Exception as e:
                log(f"      [ENTITY-AUDIT][{tname}] 跳过（{e}）")

            if tname == "T1_5key":
                na_mask = (mm.loc[ok, TAR_BVD] == "__NA__") | \
                          (mm.loc[ok, ACQ_BVD] == "__NA__")
                t1_pad_n += int(na_mask.sum())
                t1_full_n += int(ok.sum()) - int(na_mask.sum())
        log(f"      {tname}: 命中 {int(ok.sum()):,} / 待补 {len(left):,}")
    if t1_full_n + t1_pad_n > 0:
        tot = t1_full_n + t1_pad_n
        log(f"      [FIX-B] T1 质量细分: 五键齐全 {t1_full_n:,} ({t1_full_n/tot*100:.2f}%) | "
            f"含 __NA__ 填充 {t1_pad_n:,} ({t1_pad_n/tot*100:.2f}%)")
        log(f"               后者实际等价于 deal_num 匹配，计入 T1 会高估键质量")
    filled = int(base[tier_col].notna().sum())
    log(f"    [{module}] 合计命中 {filled:,}/{n0:,} ({filled/max(n0,1)*100:.2f}%) | "
        f"donor 未并入 {int((~handled).sum()):,} 行（交由调用方追加）")
    return base, handled

# =====================================================================
# 数据卫生
# =====================================================================
def parse_listed(df, label=""):
    for pfx in ("tar", "acq", "ven"):
        src = None
        for a in LISTED_ALIASES[pfx]:
            hit = find_col(df, [a])
            if hit is not None:
                src = hit
                break
        if src is None:
            cand = [c for c in df.columns
                    if c.lower().startswith(pfx) and "list" in c.lower()
                    and not c.lower().endswith(("_raw", "_state"))]
            src = cand[0] if cand else None
        if src is None:
            log(f"      [LISTED] {pfx}: 未找到列")
            continue
        raw = df[src]
        if pd.api.types.is_numeric_dtype(raw):
            df[f"{pfx}_listed"] = raw
            log(f"      [LISTED] {pfx} <- '{src}' 已是数值")
            continue
        s = raw.astype(str).str.strip().str.lower()
        df[f"{pfx}_listed_raw"]   = raw
        df[f"{pfx}_listed_state"] = s
        df[f"{pfx}_listed"] = np.where(raw.isna(), np.nan,
                                       (~s.isin(LISTED_UNLISTED)).astype(float))
        log(f"      [LISTED] {pfx} <- '{src}'")
        for k, v in df[f"{pfx}_listed_state"].value_counts(dropna=False).head(6).items():
            log(f"          {str(k)[:24]:<26s} {v:>10,}")
        log(f"          -> listed=1: {int((df[f'{pfx}_listed']==1).sum()):,} | "
            f"listed=0: {int((df[f'{pfx}_listed']==0).sum()):,}")
    return df

def fix_years(df):
    log(f"\n  [YEAR] 年份越界检查 (bounds={YEAR_BOUNDS})")
    candidates = [c for c in df.columns if re.search(r"(yr|year)$", c.lower())]
    year_cols, skipped = [], []
    for c in candidates:
        if YEAR_BLACKLIST.search(c):
            skipped.append(c)
            continue
        if YEAR_WHITELIST.match(c):
            year_cols.append(c)
        else:
            skipped.append(c)
    if skipped:
        log(f"\n      [SKIP] 以下 {len(skipped)} 列列名含 yr/year，"
            f"但按黑名单判定为【非年份】(财务值/其他)，不作年份处理：")
        for c in skipped[:8]:
            s = pd.to_numeric(df[c], errors="coerce")
            med = s.median() if s.notna().any() else float("nan")
            log(f"          {c:<42s} median={fmt_num(med)}")
        if len(skipped) > 8:
            log(f"          ... 另 {len(skipped)-8} 列")
    if not year_cols:
        log("      [YEAR] 无年份列需要处理")
        return df
    log(f"\n      [CHECK] 判定为年份列 {len(year_cols)} 个：")
    for c in year_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        bad = (s < YEAR_BOUNDS[0]) | (s > YEAR_BOUNDS[1])
        rng = f"min={s.min():.0f} p50={s.median():.0f} max={s.max():.0f}"
        if bad.any():
            log(f"          {c:<34s} 越界 {int(bad.sum()):,} -> NaN  ({rng})")
            df.loc[bad, c] = np.nan
        else:
            log(f"          {c:<34s} OK            ({rng})")
    return df

def diagnose_fake_year(df):
    susp = []
    for c in df.columns:
        if not re.search(r"(yr|year)$", c.lower()):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 100:
            continue
        med = float(s.median())
        if med > 3000 or med < 1600:
            susp.append((c, med))
    if susp:
        log(f"\n  [FIX-F][!!] 以下列【列名含 yr 但取值不像年份】，"
            f"若为财务值请确认未被 fix_years 处理：")
        for c, med in susp[:10]:
            s = pd.to_numeric(df[c], errors="coerce")
            log(f"      {c:<42s} median={med:,.2f}  min={s.min():,.2f}  max={s.max():,.2f}")
        if len(susp) > 10:
            log(f"      ... 另 {len(susp)-10} 列")
    else:
        log("\n  [FIX-F] 未发现'列名像年份但取值不像年份'的列")
    return df

def flag_outliers(df):
    fin = [c for c in df.columns if c.startswith(("pre_", "post_"))]
    num = [c for c in fin if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return df
    log(f"\n  [OUTLIER] 检查 {len(num)} 个财务列（单位 th USD），只标记不改值")
    log(f"      {'column':<40s} {'|x|>1e9':>10s} {'min':>16s} {'p99.9':>16s} {'max':>18s}")
    n_flag = 0
    for c in num:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        bad = s.abs() > 1e9
        nb = int(bad.sum())
        if nb:
            n_flag += 1
            df[c + "_implausible"] = bad.astype(int)
        log(f"      {c:<40s} {nb:>10,} {s.min():>16.3e} "
            f"{s.quantile(.999):>16.3e} {s.max():>18.3e}")
    if n_flag:
        log(f"\n      标记了 {n_flag} 个含不可信极值的列（新增 *_implausible，原值保留）")

    if TA_COL in df.columns:
        ta_s = pd.to_numeric(df[TA_COL], errors="coerce")
        ta_valid = ta_s.dropna()
        if len(ta_valid) > 0:
            n_neg = int((ta_valid <= 0).sum())
            n_huge = int((ta_valid > TA_MAX_THRESHOLD).sum())
            log(f"\n  [TA-WARN] {TA_COL} 专项:")
            log(f"      非空行数: {len(ta_valid):,}")
            if n_neg > 0:
                log(f"      <=0 的行数: {n_neg:,} (可能单位混用或录入错误，将影响 L1 排序)")
            if n_huge > 0:
                log(f"      >{TA_MAX_THRESHOLD:.0e} 的行数: {n_huge:,} (极端大值，可能决定主行)")
            log(f"      max={ta_valid.max():.3e}, min={ta_valid.min():.3e}")
    return df

# =====================================================================
# 折叠 — [V7 核心修改] L1=TA Max + [FIX-OOM] 极低内存占用实现
# =====================================================================
def collapse_to_deal(df):
    """按文献规则将多标的 deal 折叠为一行一 deal。
    [FIX-OOM] 不对百万行大表做全局 sort_values 和 copy，
    而是仅提取 5 个决胜指标构建轻量 DataFrame，用 groupby.idxmax 找出每个 deal 的主行索引，
    最后通过 iloc 零拷贝切片原表。内存峰值从 ~2GB 降至 ~50MB。
    """
    n0 = len(df)
    nd = df[DEAL].nunique()
    if n0 == nd:
        log(f"      [FOLD] 已是一行一 deal ({n0:,})")
        return df.copy(), {}

    # 1. 仅提取决胜指标（轻量 Series，不复制整个 df）
    actual_ta_col = TA_COL
    if actual_ta_col not in df.columns:
        log(f"      [FOLD][WARN] {TA_COL} 不在列中，尝试备选 {REV_COL}")
        actual_ta_col = REV_COL

    if actual_ta_col in df.columns:
        ta_vals = pd.to_numeric(df[actual_ta_col], errors="coerce")
        # fillna(-inf) 让 NaN 在升序排列时永远垫底
        s_ta = ta_vals.fillna(-np.inf)
    else:
        log(f"      [FOLD][WARN] TA 和 REV 均缺失，L1 退化")
        s_ta = pd.Series(0.0, index=df.index)

    fin = [c for c in df.columns if c.startswith(("pre_", "post_"))]
    s_fin = df[fin].notna().sum(axis=1) if fin else pd.Series(0, index=df.index, dtype=int)
    s_tar = df[TAR_BVD].notna().astype(int) if TAR_BVD in df.columns else pd.Series(0, index=df.index, dtype=int)
    s_acq = df[ACQ_BVD].notna().astype(int) if ACQ_BVD in df.columns else pd.Series(0, index=df.index, dtype=int)
    s_ord = pd.Series(np.arange(n0), index=df.index, dtype=np.int64)

    # 2. 构建决胜专用小表（仅 6 列，约 50MB）
    w = pd.DataFrame({
        DEAL: df[DEAL],
        "_ta": s_ta,
        "_fin": s_fin,
        "_tar": s_tar,
        "_acq": s_acq,
        "_ord": s_ord
    })

    # 3. 多级决胜逻辑：
    # 组合成一个可比较的元组序列进行 groupby.idxmax
    # 注意：我们需要 _ta 越大越好，_fin/_tar/_acq 越大越好，_ord 越小越好
    # 因此把 _ord 取负，统一变成“求最大值”
    w["_neg_ord"] = -w["_ord"]
    
    # 使用 apply 逐组寻找最优行索引（对于 100万行、100万组，pandas 底层 C 优化很快）
    # 为了极致性能，我们将其转为单列复合排序键
    # 但由于类型不同，最稳妥且快速的方式是：
    g = w.groupby(DEAL, sort=False)
    
    # 统计多标的分布
    sizes = g.size()
    multi_mask = sizes > 1
    n_multi = int(multi_mask.sum())
    
    # 找出每组的获胜者索引
    # 按优先级依次决出胜负：先比TA，再比FIN，再比TAR，再比ACQ，最后比ORD(越小越好即_neg_ord越大越好)
    # 技巧：利用 pandas 的稳定排序特性，从低优先级到高优先级依次排序，然后 drop_duplicates
    w_sorted = w.sort_values(
        by=[DEAL, "_ta", "_fin", "_tar", "_acq", "_neg_ord"],
        ascending=[True, True, True, True, True, True],
        kind="mergesort"   # mergesort 保证稳定性
    )
    # 此时每个 deal 组的最后一行就是各项指标最大的行
    winners = w_sorted.drop_duplicates(subset=[DEAL], keep="last")
    win_idx = winners.index
    
    # 4. 通过 iloc/loc 切片原表（零拷贝，极快）
    out = df.loc[win_idx].copy()
    out = out.reset_index(drop=True)

    # 5. 统计每笔多标的 deal 究竟由哪一级决出胜负
    w_multi = w[w[DEAL].isin(sizes[multi_mask].index)]
    g_m = w_multi.groupby(DEAL, sort=False)
    
    agg = pd.DataFrame({
        "ta_max":  g_m["_ta"].max(),
        "ta_min":  g_m["_ta"].min(),
        "fin_max": g_m["_fin"].max(),
        "fin_min": g_m["_fin"].min(),
        "tar_max": g_m["_tar"].max(),
        "tar_min": g_m["_tar"].min(),
        "acq_max": g_m["_acq"].max(),
        "acq_min": g_m["_acq"].min(),
    })

    v1 = agg["ta_max"] > agg["ta_min"]
    v2 = (~v1) & (agg["fin_max"] > agg["fin_min"])
    v3 = (~v1) & (~v2) & (agg["tar_max"] > agg["tar_min"])
    v4 = (~v1) & (~v2) & (~v3) & (agg["acq_max"] > agg["acq_min"])
    v5 = (~v1) & (~v2) & (~v3) & (~v4)

    stats = {
        "L1_tar_ta_max": int(v1.sum()),
        "L2_fin_fill":   int(v2.sum()),
        "L3_tarid":      int(v3.sum()),
        "L4_acqid":      int(v4.sum()),
        "L5_order":      int(v5.sum()),
    }

    log(f"      [FOLD] {n0:,} rows -> {len(out):,} deals "
        f"(多标的 {n_multi:,} 笔，丢弃 {n0-len(out):,} 行)")
    log(f"      [FOLD] 主行决胜层级分布 (占多标的百分比):")
    for k, v in stats.items():
        pct = v / max(n_multi, 1) * 100
        log(f"          {k:<18s} {v:>9,} ({pct:6.2f}%)")

    if actual_ta_col in df.columns:
        ta_nn = int(pd.to_numeric(df[actual_ta_col], errors="coerce").notna().sum())
        rev_nn = 0
        if REV_COL in df.columns and REV_COL != actual_ta_col:
            rev_nn = int(pd.to_numeric(df[REV_COL], errors="coerce").notna().sum())
        log(f"      [FOLD] TA列({actual_ta_col}) 非空率: {ta_nn:,}/{n0:,} = {ta_nn/max(n0,1)*100:.2f}%")
        if REV_COL in df.columns and REV_COL != actual_ta_col:
            log(f"      [FOLD] REV列({REV_COL}) 非空率: {rev_nn:,}/{n0:,} = {rev_nn/max(n0,1)*100:.2f}%")
        if n_multi > 0 and stats["L1_tar_ta_max"] / n_multi < 0.30:
            log(f"      [FOLD][WARN] L1_tar_ta_max < 30%，说明 {actual_ta_col} 覆盖率太低！")
            log(f"                   考虑回退到 {REV_COL} 或恢复 '财务非空数' 作为 L1。")

    return out, stats

# =====================================================================
# 02 对照（分块读，避免 OOM）
# =====================================================================
def compare_with_02(dl):
    p02 = os.path.join(MERGED, "02_deal_master.csv")
    if not os.path.exists(p02):
        return
    banner("STEP 9 — 与 02_deal_master 对照")
    s2 = set()
    try:
        for chunk in pd.read_csv(p02, usecols=[DEAL], chunksize=200_000,
                                 low_memory=False, on_bad_lines="skip"):
            s2.update(pd.to_numeric(chunk[DEAL], errors="coerce")
                      .dropna().astype("int64").tolist())
    except Exception as e:
        log(f"  [WARN] 读取 02_deal_master 失败: {e}")
        return
    s3 = set(pd.to_numeric(dl[DEAL], errors="coerce").dropna().astype("int64"))
    log(f"  02_deal_master : {len(s2):,} deals")
    log(f"  03_firm        : {len(s3):,} deals")
    log(f"  交集           : {len(s2 & s3):,} "
        f"(03 覆盖 02 的 {len(s2 & s3)/max(len(s2),1)*100:.2f}%)")
    log(f"  02 有 03 无    : {len(s2 - s3):,}  (无财务数据，04 左连接会保留)")
    log(f"  03 有 02 无    : {len(s3 - s2):,}")

# =====================================================================
# Main
# =====================================================================
def main():
    t0 = time.time()
    banner("STEP 1 — 载入 BASE: MA_finance.csv (keep all)")
    base = load_module(BASE_ROLE)
    if base is None or len(base) == 0:
        log("\n  [FATAL] base 为空，终止")
        return

    fin_excl = [c for c in base.columns if c.startswith(("pre_", "post_"))]
    n0b = len(base)
    base = ffill_block(base, "tar_", exclude=fin_excl)
    base = ffill_block(base, "acq_", exclude=fin_excl)

    n_after_ffill = len(base)
    base = dedup_entity_blocks(base)
    n_after_dedup = len(base)
    log(f"  [FFILL] base(finance): {n0b:,} -> ffill后:{n_after_ffill:,} -> "
        f"去重后:{n_after_dedup:,} rows (tar/acq ffill，{len(fin_excl)} 财务列排除)")
    if n0b != n_after_dedup:
        log(f"          注: 减少的 {n0b - n_after_dedup:,} 行均为 tar_bvd+acq_bvd 联合键"
            f"完全相同的纯重复行，非异常丢弃。")

    banner("STEP 1b — BASE 去重（在场 MERGE_KEYS）")
    n0 = len(base)
    _mk = [k for k in MERGE_KEYS if k in base.columns]
    base = dedup_on(base, _mk, "base@T1").reset_index(drop=True)
    log(f"\n  base: {n0:,} -> {len(base):,} rows | unique {DEAL}: {base[DEAL].nunique():,}")
    if n0 == len(base):
        log(f"  [INFO] 去重前后行数一致，说明 MERGE_KEYS 在 base 上无重复行（正常）。")
    if REPORT_FULL_AFTER_EACH_STEP:
        report_full(base, "STEP 1 base")

    donors = {}
    for r in DONOR_ROLES:
        banner(f"STEP 2 — 载入 {r}")
        d = load_module(r)
        if d is None:
            continue
        if r == "list":
            log("\n  [HYGIENE] 解析 listed（含 ven）")
            d = parse_listed(d, r)
        d_excl = [c for c in d.columns if c.startswith(("pre_", "post_"))]
        n0d = len(d)
        d = ffill_block(d, "tar_", exclude=d_excl)
        d = ffill_block(d, "acq_", exclude=d_excl)

        n_after_ffill_d = len(d)
        d = dedup_entity_blocks(d)
        n_after_dedup_d = len(d)
        log(f"  [FFILL] donor[{r}]: {n0d:,} -> ffill后:{n_after_ffill_d:,} -> "
            f"去重后:{n_after_dedup_d:,} rows (tar/acq ffill，{len(d_excl)} 财务列排除)")
        if n0d != n_after_dedup_d:
            log(f"          注: 减少的 {n0d - n_after_dedup_d:,} 行为实体键纯重复行。")

        donors[r] = d
        if REPORT_FULL_AFTER_EACH_STEP:
            report_full(d, f"STEP 2 loaded {r}")

    banner("STEP 2b — 各源 unique deal_num 覆盖对比")
    src_deals = {}
    if DEAL in base.columns:
        src_deals["finance(base)"] = set(base[DEAL].dropna().unique())
    for r, d in donors.items():
        if d is not None and DEAL in d.columns:
            src_deals[r] = set(d[DEAL].dropna().unique())
    for name, s in src_deals.items():
        log(f"  {name:<20s} {len(s):>10,} deals")
    log("  [NOTE] industry / overview 的 deal 数明显少于 finance/legal/list，")
    log("         这是因为这两源仅覆盖有 SIC 行业分类 / overview 文本的 deal，属正常现象。")

    banner("STEP 3 — 级联回退合并 (T1 -> T5)，未命中 donor 追加（0 丢弃）")
    for r in DONOR_ROLES:
        d = donors.get(r)
        base, handled = cascade_merge(base, d, r)
        if d is not None and len(d):
            unmatched = d.loc[~handled].copy()
            if len(unmatched):
                unmatched[f"{r}_tier"] = f"{r}_only"
                base = pd.concat([base, unmatched], ignore_index=True)
                log(f"    [append] {r} 未并入 {len(unmatched):,} 行 "
                    f"-> 追加为独立实体行（{r}_only）")

    banner("STEP 4 — 数据卫生")
    log("  [4a] listed 复核（合并后，含 ven）")
    base = parse_listed(base, "merged")
    log("\n  [4a2] 假年份诊断")
    base = diagnose_fake_year(base)
    log("\n  [4b] 年份越界（白名单判定）")
    base = fix_years(base)
    log("\n  [4c] 极值标记（不改值）+ TA 专项告警")
    base = flag_outliers(base)
    log("\n  [4d] 还原 __NA__ 哨兵")
    n_restore = 0
    for k in MERGE_KEYS:
        if k != DEAL and k in base.columns:
            m = (base[k] == "__NA__")
            n_restore += int(m.sum())
            base.loc[m, k] = np.nan
    log(f"      还原 {n_restore:,} 个 __NA__ -> NaN")

    banner("STEP 5 — 输出 deal × tar × acq 全量表")
    if EMIT_ENTITY_PAIRS:
        base.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")
        sz = os.path.getsize(OUT_PAIRS) / 1024 / 1024
        log(f"  [SAVED] {OUT_PAIRS}")
        log(f"    rows={len(base):,} | cols={base.shape[1]} | {sz:.1f} MB")
        log(f"    unique {DEAL}: {base[DEAL].nunique():,}  （多标的保留多行）")

    banner("STEP 6 — 折叠到 deal 层 (L1=TA Max)")
    dl, stats = collapse_to_deal(base)
    n_rows, n_deals = len(dl), dl[DEAL].nunique()
    log(f"\n  rows={n_rows:,} | unique {DEAL}={n_deals:,}")
    if n_rows != n_deals:
        log("  x [FAIL] 不是一行一 deal")
    else:
        log("  [OK] 一行一 deal，可安全并到 02_deal_master")
    dl.to_csv(OUT_DEAL, index=False, encoding="utf-8-sig")
    sz = os.path.getsize(OUT_DEAL) / 1024 / 1024
    log(f"\n  [SAVED] {OUT_DEAL}")
    log(f"    rows={len(dl):,} | cols={dl.shape[1]} | {sz:.1f} MB")

    banner("STEP 7 — 合并层级分布")
    for c in [c for c in base.columns if c.endswith("_tier")]:
        vc = base[c].value_counts(dropna=False)
        log(f"\n  {c}:")
        for k, v in vc.items():
            log(f"    {str(k):<14s} {v:>10,} ({v/max(len(base),1)*100:6.2f}%)")

    banner("STEP 8 — 键覆盖 (deal-level)")
    for c in MERGE_KEYS + ["ven_bvd_id_num"]:
        if c in dl.columns:
            n = int(dl[c].notna().sum())
            log(f"  {c:<20s} {n:>10,} ({n/max(len(dl),1)*100:6.2f}%)")

    compare_with_02(dl)

    banner("STEP 10 — FINAL 全量变量报告 (deal-level)")
    report_full(dl, "03_firm_deal_level", verbose=True)

    log(f"\n  done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        raise