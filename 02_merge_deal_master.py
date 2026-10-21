# -*- coding: utf-8 -*-
r"""
02_merge_deal_master.py — v11‑deal‑only (2026‑10‑21)
==================================================================
把 01 系列清洗产物合并为 master。
[V11] 02 只做 deal 级合并（structure + multiple + value + info_source），
      实体对齐（industry / overview 的多键级联）整体移出到 03；
      不再输出 02_entity_pairs.csv；02_deal_master.csv 严格一行一 deal。
定位：通用并购研究数据库，不做任何论文口径筛选。
==================================================================
★ MOD: ENTITY_TIERS 升级，第一层T1_5key完整五元匹配 deal_num+tar_name+acq_name+tar_bvd_id_num+acq_bvd_id_num
★ v9 -> v10：彻底移除 vendor_name 合并单元（2026‑10‑17）★
==================================================================
[用户指出] 01a overview 里本来就有 vendor name，不需要再用
          01a_deal_vendor_name.csv 单独并一次。
          v9 中该模块并入后产生两列垃圾：
              ven_name_vendor_name
              ven_bvd_id_num_vendor_name
          （因与 industry/overview 已有列重名被加后缀，纯冗余）
★DEL★ MODULE_SPECS 中的 vendor_name 项
★DEL★ STEP 1b 覆盖对照循环中的 "vendor_name"
★DEL★ STEP 4 并入循环中的 "vendor_name"
★DEL★ try_numeric 触发条件中的 "vendor_name"
[vendor_type 状态] 保持 disabled（enabled=False），STEP 4 打印 [SKIP]。
  按用户决定：vendor type 挪到 07/08 之后再并。
  原因：any=["vendor_type"] 同时匹配两个文件，触发 ambiguous 报错——
      01b_deal_vendor_type.csv
      01b_deal_vendor_type_profile.csv
    歧义应暴露，不由代码猜选；放到 07/08 时显式指定
    01b_deal_vendor_type.csv 即可。
==================================================================
★ v8 -> v9：彻底移除 comment 合并单元（2026‑10‑17）★
==================================================================
[用户决定] comment 不在 02 合并，改到 06/07 之后再做。
  故本版删除全部 comment 相关代码：
    - MODULE_SPECS 中的 comments 项
    - COMM_SUM / COMM_MAX / COMM_MEAN / COMM_MIN_DT / COMM_MAX_DT / COMM_CONCAT
    - collapse_comments() 函数
    - STEP 4 循环中的 "comments"
    - STEP 1b 的 comment 覆盖对照、STEP 8 的 comment 覆盖率行
  不留开关、不留死代码。01c_comments_features.csv 仍在 cleaned 目录下，
  但不再被任何 role 匹配，无害。
  后续需要时由 06/07 之后的脚本按 deal_num 单独并入。
==================================================================
v5 -> v6：合并键修正（用户评审结论，2026‑10‑17）
==================================================================
[问题] v5 仅按 deal_num 合并：
  01_deal_overview.csv      真实 1,088,987 行 / 1,011,206 deal（多标的 +77,781 行）
  01_deal_sic_industry.csv  真实 1,096,864 行 / 1,018,369 deal（多标的 +78,495 行）
  v5 的 reduce_to_deal_level 在 merge 前先把每个模块各自折叠成 1 行/deal：
    - industry 丢弃 78,495 行 (7.16%)；overview 丢弃 77,781 行 (7.14%)
    - 折叠依据是 L1_sic > L2_ctry > L3_fill > L4_order 的"局内挑选"，
      各模块独立挑选 → 多标的交易会把不同 target 的属性拼进同一行
      （跨模块实体错配 / Frankenstein 行，沙盘实证 3/5 笔错配）
    - 列名冲突时 d.drop(columns=[...]) 静默丢弃后合并方（如 overview 的
      tar_key / n_targets / acq_orbis_id_num），无任何审计
[修法] v6 分两步：
  STEP‑A 实体模块在"deal × tar × acq"粒度先做多键级联合并对齐：
    T1_5key   (deal_num, tar_name, acq_name, tar_bvd_id_num, acq_bvd_id_num)  # MOD 完整五元最高优先级
    T2_2key_bvdid (deal_num, tar_bvd_id_num, acq_bvd_id_num)
    T3_2key_name  (deal_num, tar_name, acq_name)
    T4_tarkey     (deal_num, tar_key)
    T5_deal       (deal_num)                                    ← 最后兜底，显式报告
  每个模块记录 *_tier，并做"跨模块键一致性"审计（同键命中但其余键打架则报告）。
  STEP‑B 再折叠到 deal 层（主行 = 填充最全，平局取先），输出两份：
    - 02_deal_master.csv    一行一 deal（主行实体属性 + 全部 deal 级变量）
    - 02_entity_pairs.csv   deal×tar×acq 全量表（0 丢弃，真保真）
  列冲突不再静默 drop：重名列以 _{module} 后缀并存，并写冲突审计。
保留 v5 的正确部分：不删列 / 数值转换丢失即退回加 _num / 常量列排除
has_any_multiple / fillna(0) 前审计 / merge 接受 miss / 论文 moment 对照。
"""
import os
import sys
import glob
import time
import traceback
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
# =====================================================================
# Config
# =====================================================================
try:
    import config as CFG
except Exception:
    CFG = None
def cfg(name, default):
    return getattr(CFG, name, default) if CFG is not None else default
BASE_DIR    = cfg("BASE_DIR",    r"D:\MA")
CLEANED_DIR = cfg("CLEANED_DIR", os.path.join(BASE_DIR, "data", "cleaned"))
MERGED_DIR  = cfg("MERGED_DIR",  os.path.join(BASE_DIR, "data", "merged"))
os.makedirs(MERGED_DIR, exist_ok=True)
DEAL = cfg("MERGE_KEY_DEAL", "deal_num")
OUT_MASTER    = os.path.join(MERGED_DIR, "02_deal_master.csv")
OUT_PAIRS     = os.path.join(MERGED_DIR, "02_entity_pairs.csv")
OUT_DIAG      = os.path.join(MERGED_DIR, "02_deal_master_diagnostics.txt")
PAPER = {"p_A": 0.2563, "p_D": 0.6568, "p_B": 0.2081}
MODULE_SPECS = [
    {"role": "structure",   "required": True,  "any": ["structure_date"], "enabled": True},
    {"role": "industry",    "required": False, "any": ["_industry", "sic_industry"], "enabled": False},  # [V11] 实体对齐挪到 03，02 只做 deal 级
    {"role": "multiple",    "required": False, "any": ["_multiple"], "enabled": True},
    {"role": "value",       "required": False, "any": ["_value"], "enabled": True},
    {"role": "overview",    "required": False, "any": ["_overview"], "enabled": False},  # [V11] 实体对齐挪到 03，02 只做 deal 级
    # ★DEL★ v10: vendor_name 已删除 —— 01a overview 本身即含 ven_name / ven_bvd_id_num
    {"role": "vendor_type", "required": False, "any": ["vendor_type"], "enabled": False},
    {"role": "info_source", "required": False, "any": ["info_source"], "enabled": True},
    # comments 已移除（v9）：由 06/07 之后按 deal_num 单独并入
]
SIC_CANDIDATES  = ["tar_primary_sic_code", "tar_sic2", "tar_sic_code", "tar_primary_sic"]
CTRY_CANDIDATES = ["tar_country_code", "tar_ctry", "tar_country", "tar_nation"]
MULTIPLE_PREFIX   = "pre_"
MULTIPLE_FLAG_ABS = 1e4
MULTIPLE_HARD_CAP = 1e9
HEAVY_TEXT_DROP = ["tar_overview", "acq_overview", "deal_headline", "comments",
                   "raw_text", "tar_busi_descr", "acq_busi_descr",
                   "ven_busi_descr", "gr_acq_busi_descr", "gr_ven_busi_descr",
                   "reg_entity_list", "regulatory_bodies"]
DROP_INDEX_COLS = ["unnamed_0", "Unnamed: 0", "unnamed_0.1", "index", "level_0"]
BINARY_PREFIXES = ("has_", "reg_", "is_", "cross_", "common_law_")
COERCE_MAX_LOSS_PCT = 0.001
CONSTANT_COL_WARN   = True
SENTINEL = "__NA__"
# ── 实体对齐级联键【MODIFIED：第一层完整五元】──────────────────────────────────────────────
ENTITY_TIERS = [
    ("T1_5key",    [DEAL, "tar_name", "acq_name","tar_bvd_id_num", "acq_bvd_id_num"]),
    ("T2_2key_bvdid",  [DEAL, "tar_bvd_id_num", "acq_bvd_id_num"]),
    ("T3_2key_name",  [DEAL, "tar_name", "acq_name"]),
    ("T4_tarkey",   [DEAL, "tar_key"]),
    ("T5_deal",    [DEAL]),
]
ENTITY_ID_COLS = ["tar_bvd_id_num", "acq_bvd_id_num", "tar_orbis_id_num",
                  "acq_orbis_id_num", "tar_name", "acq_name", "tar_key"]
# =====================================================================
# Logging
# =====================================================================
_LINES = []
_T0 = time.time()
def log(msg=""):
    print(msg)
    _LINES.append(str(msg))
def flush_log():
    with open(OUT_DIAG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")
def banner(t):
    log("\n" + "=" * 80)
    log(t)
    log("=" * 80)
# =====================================================================
# 文件发现 / 载入
# =====================================================================
def scan_01_files():
    files = sorted(glob.glob(os.path.join(CLEANED_DIR, "01*.csv")))
    log(f"\n  [SCAN] {CLEANED_DIR}")
    log(f"    found {len(files)} file(s):")
    for f in files:
        log(f"      - {os.path.basename(f)}")
    return files
def resolve_role(spec, files):
    role = spec["role"]
    hits = [f for f in files
            if any(k.lower() in os.path.basename(f).lower() for k in spec["any"])]
    if len(hits) == 0:
        return None
    if len(hits) == 1:
        log(f"    [{role}] matched -> {os.path.basename(hits[0])}")
        return hits[0]
    # 多个候选：报错（歧义必须暴露，不猜、不选、不静默）
    log(f"\n    [!!][{role}] 多个候选文件，无法判定：")
    for h in hits:
        log(f"        - {os.path.basename(h)}")
    raise RuntimeError(f"ambiguous file for role={role}")
def load(spec, files):
    role = spec["role"]
    if not spec.get("enabled", True):
        log(f"  [DISABLED] {role}")
        return None
    p = resolve_role(spec, files)
    if p is None:
        log(f"  [MISSING] {role}" + ("  <- REQUIRED" if spec["required"] else "  (skip)"))
        if spec["required"]:
            raise FileNotFoundError(role)
        return None
    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False, on_bad_lines="warn")
    df.columns = [str(c).strip() for c in df.columns]
    drop_idx = [c for c in df.columns if c in DROP_INDEX_COLS]
    if drop_idx:
        df = df.drop(columns=drop_idx)
        log(f"      [DROP‑INDEX] {drop_idx}")
    if DEAL in df.columns:
        df[DEAL] = pd.to_numeric(df[DEAL], errors="coerce").astype("Int64")
        df = df.dropna(subset=[DEAL])
    log(f"  loaded {role}: {os.path.basename(p)} | {len(df):,} rows | "
        f"unique {DEAL}: {df[DEAL].nunique():,}")
    return df
# =====================================================================
# 通用工具
# =====================================================================
def norm_id_str(s):
    """ID/名称标准化：去首尾空白，去掉 float 尾巴 '.0'，空→NaN。"""
    def f(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip()
        if t.endswith(".0"):
            t = t[:-2]
        return t if t.lower() not in ("nan", "none", "na", "") else np.nan
    return s.map(f)
def prepare_entity_keys(df, keys):
    """把 keys 中除 deal 外的列填充 SENTINEL 并转字符串，保证可比较。"""
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            out[k] = SENTINEL
        elif k != DEAL:
            out[k] = norm_id_str(out[k]).fillna(SENTINEL).astype(str)
    return out
def dedup_most_filled(df, keys, label=""):
    """按 keys 去重，保留填充最全、顺序最靠前的行。"""
    n0 = len(df)
    if n0 == 0:
        return df
    tmp = df.copy()
    tmp["_fill"] = tmp.notna().sum(axis=1)
    tmp["_ord"] = np.arange(n0)
    tmp = tmp.sort_values(keys + ["_fill", "_ord"],
                          ascending=[True] * len(keys) + [False, True],
                          kind="mergesort")
    out = (tmp.drop_duplicates(subset=keys, keep="first")
              .drop(columns=["_fill", "_ord"]).reset_index(drop=True))
    if n0 != len(out):
        log(f"        dedup[{label}] {n0:,} -> {len(out):,} (dropped {n0 - len(out):,})")
    return out
def groupby_first(df, label=""):
    if df[DEAL].duplicated().sum() == 0:
        return df
    n0 = len(df)
    out = df.groupby(DEAL, sort=False).first().reset_index()
    out[DEAL] = pd.to_numeric(out[DEAL], errors="coerce").astype("Int64")
    log(f"    [{label}] groupby.first(): {n0:,} -> {len(out):,} "
        f"(折叠 {n0 - len(out):,} 行)")
    return out
# =====================================================================
# [FFILL-PATCH] 折叠前先填满 tar_/acq_ 稀疏实体列（只填 NaN，不覆盖）
# =====================================================================
# 只填实体标识/引用列，不填 sic/naics/overview/country 等属性列：
# 属性列每行自带，跨实体 ffill 会把上一 target 的属性复制给下一行
# （多标的时 tarB 行会拿到 tarA 的 sic / overview 文本），比 NaN 更危险，
# 且会均衡化 _fill 计数、破坏 02 折叠的"填充最全"区分度。
FFILL_ENTITY_COLS = ["tar_bvd_id_num", "tar_name", "tar_orbis_id_num",
                     "acq_bvd_id_num", "acq_name", "acq_orbis_id_num",
                     "tar_key"]


def ffill_block(df, prefix, exclude=None):
    """按 deal 组对实体键列（tar_/acq_ 前缀子集）做前向填充。
    - 只填 NaN：ffill 不会覆盖已有值。
    - 空串/占位符先归一为 NaN 再填（01a 后可能残留 ""/"nan" 等）。
    - 绝不碰 ven_*；不碰属性列（sic/naics/overview/country/财务）。
    - 返回完整 DataFrame（保留所有列）。"""
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
    """tar/acq 都填完后，按在场实体键一次联合去重：
    - 多标的（tarA/acq、tarB/acq）两行键不同 -> 保留为多行，不吞多标的；
    - 纯重复行（同一组合多行记录）-> 合并为一行；
    - tar_bvd 缺失的行按 acq 键归组（NaN 在 drop_duplicates 中视为同组）。
    注意：若先 tar 后 acq 顺序两次独立去重，同 acq 的多标的会被吞成
    一行，所以必须一次性按联合键去重。"""
    key_cols = [k for k in ("tar_bvd_id_num", "acq_bvd_id_num")
                if k in df.columns]
    if not key_cols:
        return df
    d = df.sort_values(["deal_num"] + key_cols, kind="mergesort")
    d = d.drop_duplicates(subset=["deal_num"] + key_cols, keep="first")
    return d


# =====================================================================
# 实体对齐：多键级联（v6 核心）
# =====================================================================
def cascade_align_entity(base, donor, module):
    """把 donor（实体级）按 ENTITY_TIERS 级联对齐到 base（实体级）。
    返回 (base, handled_mask)：
      - base      新增 donor 列 + {module}_tier；
      - handled_mask 长度 = 原始 donor 行数，True = 该行已被并入 base
        （含"与命中行同键而被去重吸收"的行）。
    未处理（handled=False）的 donor 行由调用方追加，保证 0 丢弃。"""
    if donor is None or len(donor) == 0:
        log(f"\n    [{module}] 无数据，跳过")
        return base, np.ones(0, dtype=bool)
    donor_cols = [c for c in donor.columns if c not in [DEAL] + ENTITY_ID_COLS]
    donor_cols = [c for c in donor_cols if c not in HEAVY_TEXT_DROP]
    if not donor_cols:
        log(f"\n    [{module}] 无可并入列，跳过")
        return base, np.ones(len(donor), dtype=bool)
    collide = [c for c in donor_cols if c in base.columns]
    if collide:
        ren = {c: f"{c}_{module}" for c in collide}
        donor = donor.rename(columns=ren)
        donor_cols = [ren.get(c, c) for c in donor_cols]
        log(f"\n    [{module}] 列冲突 {len(collide)} 列 -> 加 _{module} 后缀: {collide}")
    tier_col = f"{module}_tier"
    base[tier_col] = SENTINEL
    n0 = len(base)
    donor_proc = donor.copy()
    donor_proc["_src"] = np.arange(len(donor), dtype=np.int64)
    handled = np.zeros(len(donor), dtype=bool)
    filled = 0
    tier_counts = {}
    for tname, tkeys in ENTITY_TIERS:
        present = all(k in base.columns and k in donor.columns for k in tkeys)
        if not present:
            continue
        todo_mask = base[tier_col].eq(SENTINEL)
        if not todo_mask.any():
            break
        d = dedup_most_filled(donor_proc, tkeys, f"{module}@{tname}")
        d = prepare_entity_keys(d, tkeys)[tkeys + donor_cols + ["_src"]]
        idx_all = base.index[todo_mask.values].to_numpy()
        left = prepare_entity_keys(base.loc[todo_mask], tkeys)[tkeys].reset_index(drop=True)
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
            filled += int(ok.sum())
            tier_counts[tname] = int(ok.sum())
            srcs = mm.loc[ok, "_src"].to_numpy(dtype=np.int64)
            handled[srcs] = True
            # [ENTITY-AUDIT] Compare the entity keys of the base rows that
            # were just filled with the entity keys of the donor rows that
            # supplied the values. T1/T2 keys match by construction, so
            # conflicts here are data inconsistencies; T5 (deal_num only)
            # can glue a DIFFERENT target's columns into a base row, which
            # is exactly what this audit is meant to surface.
            try:
                b_rows = base.loc[sel]
                d_rows = donor_proc.loc[srcs]
                n_ek = 0
                for ek in ["tar_name", "tar_bvd_id_num",
                           "acq_name", "acq_bvd_id_num"]:
                    if ek not in b_rows.columns or ek not in d_rows.columns:
                        continue
                    bv = b_rows[ek].fillna(SENTINEL).astype(str).to_numpy()
                    dv = d_rows[ek].fillna(SENTINEL).astype(str).to_numpy()
                    n_ek += int(((bv != SENTINEL) & (dv != SENTINEL)
                                 & (bv != dv)).sum())
                if n_ek:
                    log(f"      [ENTITY-AUDIT][{tname}] 命中 {int(ok.sum()):,} 行，"
                        f"实体键不一致 {n_ek:,} 处（tar_name / tar_bvd / acq_name / "
                        f"acq_bvd 双方非空且不等）")
            except Exception as e:
                log(f"      [ENTITY-AUDIT][{tname}] 跳过（{e}）")
            d2 = prepare_entity_keys(donor_proc, tkeys)
            keyset = set(map(tuple, mm.loc[ok, tkeys].to_numpy()))
            if keyset:
                dup = np.array([tuple(r) in keyset for r in
                                d2[tkeys].to_numpy()], dtype=bool)
                dup_src = d2.loc[dup, "_src"].to_numpy(dtype=np.int64)
                handled[dup_src] = True
        log(f"      {tname:<14s} 命中 {int(ok.sum()):>6,} / 待补 {len(left):>6,}")
    unfilled = int(base[tier_col].eq(SENTINEL).sum())
    log(f"\n    [{module}] 对齐 {n0:,} 行：命中 {filled:,} ({filled/max(n0,1)*100:.2f}%) | "
        f"未命中 {unfilled:,}（接受 miss，列留空）| donor 未并入 "
        f"{int((~handled).sum()):,} 行（交由调用方追加）")
    if tier_counts:
        log(f"    级联分布: {tier_counts}")
    return base, handled
def check_entity_consistency(pairs):
    """跨模块键一致性审计：同 deal 内，各模块带来的 tar/acq 键应一致。"""
    log("\n  -- 跨模块实体键一致性审计 --")
    n_conflict = 0
    for c in ["tar_bvd_id_num", "acq_bvd_id_num", "tar_name", "acq_name"]:
        cols = [x for x in pairs.columns if x.startswith(c + "_") or x == c]
        if len(cols) < 2:
            continue
        base_col = c if c in pairs.columns else None
        ref = base_col if base_col else cols[0]
        for other in cols:
            if other == ref:
                continue
            a = pairs[ref].astype(str)
            b = pairs[other].astype(str)
            both = a.ne("nan") & b.ne("nan") & a.ne(SENTINEL) & b.ne(SENTINEL) & a.ne(b)
            n = int(both.sum())
            if n:
                n_conflict += n
                log(f"      [!!] {ref} vs {other}: {n:,} 行不一致（样例: "
                    f"{list(zip(pairs.loc[both, ref].head(3), pairs.loc[both, other].head(3)))})")
            else:
                log(f"      [OK] {ref} vs {other}: 一致")
    if n_conflict == 0:
        log("  [OK] 未发现跨模块键冲突")
    else:
        log(f"  [!!] 共 {n_conflict:,} 处键冲突 —— 建议人工核查上游匹配")
    return n_conflict
# =====================================================================
# 倍数审计 / 常量列 / 数值转换（沿用 v5 保真逻辑）
# =====================================================================
def audit_multiples(df, label=""):
    mcols = [c for c in df.columns
             if c.startswith(MULTIPLE_PREFIX) and c.endswith("_mul_ly")]
    if not mcols:
        return df, []
    log(f"\n  估值倍数审计（只标记，不改值，不删列）, {len(mcols)} 列")
    usable, flagged = [], []
    for c in mcols:
        s = pd.to_numeric(df[c], errors="coerce")
        nn = int(s.notna().sum())
        if nn == 0:
            continue
        s = s.where(s >= 0).where(s <= MULTIPLE_HARD_CAP)
        n_flag = int((s > MULTIPLE_FLAG_ABS).sum())
        if n_flag:
            df[c + "_implausible"] = np.where(s.notna(),
                                              (s > MULTIPLE_FLAG_ABS).astype(float), np.nan)
            flagged.append((c, n_flag))
        if s.nunique(dropna=True) > 1:
            usable.append(c)
        df[c] = s
    log(f"    可用列 {len(usable)} / {len(mcols)}；标记不可信极值 {len(flagged)} 列")
    return df, usable
def report_constant_columns(df, label=""):
    if not CONSTANT_COL_WARN:
        return
    const = []
    for c in df.columns:
        if c == DEAL:
            continue
        s = df[c]
        if s.notna().sum() == 0:
            const.append((c, "全空", 0))
            continue
        if s.nunique(dropna=True) <= 1:
            const.append((c, f"唯一值={list(s.dropna().unique()[:1])}",
                          int(s.notna().sum())))
    if const:
        log(f"\n  [常量列报告][{label}] {len(const)} 列（保留不删）：")
        for c, why, n in const[:15]:
            log(f"    {c:<32s} {why:<28s} non‑null={n:,}")
        if len(const) > 15:
            log(f"      ... 另 {len(const) - 15} 列")
def try_numeric(df, patterns=("pct", "_val_", "stake", "irr"), label=""):
    converted, rejected = [], []
    for c in df.columns:
        if c == DEAL:
            continue
        if not any(p in c.lower() for p in patterns):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        raw = df[c]
        orig_nn = int(raw.notna().sum())
        if orig_nn == 0:
            continue
        s = raw.astype(str).str.strip()
        conv = pd.to_numeric(s.str.replace(",", ""), errors="coerce")
        conv2 = pd.to_numeric(s.str.replace(",", "").str.replace("%", "")
                               .str.replace(" ", ""), errors="coerce")
        if int(conv2.notna().sum()) > int(conv.notna().sum()):
            conv = conv2
        new_nn = int(conv.notna().sum())
        loss = orig_nn - new_nn
        loss_pct = loss / max(orig_nn, 1)
        if loss == 0:
            df[c] = conv
            converted.append((c, orig_nn, 0))
        elif loss_pct <= COERCE_MAX_LOSS_PCT:
            df[c] = conv
            converted.append((c, orig_nn, loss))
            log(f"      [OK] {c:<26s} 转换，损失 {loss:,} ({loss_pct*100:.3f}%) 在容忍内")
        else:
            df[c + "_num"] = conv
            rejected.append((c, orig_nn, loss, loss_pct))
            bad = raw[conv.isna() & raw.notna()].astype(str)
            log(f"      [!] {c:<26s} 丢失 {loss:,} ({loss_pct*100:.2f}%) > "
                f"{COERCE_MAX_LOSS_PCT*100:.1f}% —— 保留原列，新增 {c}_num")
            log(f"          无法解析样例: {list(pd.unique(bad))[:8]}")
    if converted:
        log(f"\n  数值转换完成 {len(converted)} 列")
    if rejected:
        log(f"\n  [!] {len(rejected)} 列因丢失过大未转换（原列保留，另加 _num）")
    return df
# =====================================================================
# 报告
# =====================================================================
def module_report(df, label=""):
    log(f"\n  ── MODULE [{label}] ── {len(df):,} rows x {df.shape[1]} cols")
    log(f"    {'column':<36s} {'non‑null':>10s} {'pct':>7s} {'mean/uniq':>14s}")
    log("    " + "-" * 72)
    for c in df.columns:
        n = int(df[c].notna().sum())
        pct = n / max(len(df), 1) * 100
        extra = ""
        if c.startswith(BINARY_PREFIXES):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any() and s.nunique(dropna=True) > 1:
                extra = f"mean={s.mean():.4f}"
            elif df[c].dtype == object:
                extra = f"uniq={df[c].nunique():,}"
        elif not pd.api.types.is_numeric_dtype(df[c]):
            extra = f"uniq={df[c].nunique():,}"
        log(f"    {c:<36s} {n:>10,} {pct:>6.2f}% {extra:>14s}")
def light_report(df, label=""):
    log(f"\n  -- {label}: {len(df):,} rows x {df.shape[1]} cols --")
    miss = sorted([(int(df[c].isna().sum()), c) for c in df.columns], reverse=True)
    log("    top‑5 missing: " +
        ", ".join(f"{c} ({n/max(len(df),1)*100:.1f}%)" for n, c in miss[:5]))
# =====================================================================
# 变量对账：02 输出 vs 01 输入（回答"能不能一一对应"）
# =====================================================================
def reconcile_variables(loaded, master, pairs=None):
    """01 每个模块的变量 → 是否出现在 02 输出；缺失者逐条报告。"""
    log("\n  -- 变量对账：01 模块 → 02 输出（一一对应检查）--")
    out_cols = set(master.columns)
    if pairs is not None:
        out_cols |= set(pairs.columns)
    total_in = 0
    for role, df in loaded.items():
        if df is None:
            continue
        cols = [c for c in df.columns if c != DEAL]
        missing = [c for c in cols
                   if c not in out_cols and c not in HEAVY_TEXT_DROP
                   and c.lower() not in ("unnamed_0", "unnamed: 0", "index", "level_0")]
        if missing:
            log(f"    [{role}] 未出现在 02 输出: {missing}")
        else:
            log(f"    [{role}] 全部 {len(cols)} 个变量已出现在 02 输出")
        total_in += len(cols)
    n_pairs = pairs.shape[1] if pairs is not None else 0
    log(f"  合计: 01 输入变量 {total_in:,} | 02 输出列 "
        f"master={master.shape[1]:,} + pairs={n_pairs:,}")
# =====================================================================
# Main
# =====================================================================
def main():
    banner("02_merge_deal_master.py — v10‑mod 【T1五元最高优先级】")
    log(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"cleaned: {CLEANED_DIR}")
    log("  合并键: 实体模块 = 多键级联 (T1_5key → T2_2key_bvdid → T3_2key_name → T4_tarkey → T5_deal)，"
        "deal 级模块 = deal_num")
    log("  输出  : 02_deal_master.csv（一行一 deal）—— [V11] 02_entity_pairs.csv 不再输出，实体对齐在 03")
    log("  comment     : 不在本脚本合并（v9），由 06/07 之后按 deal_num 单独并入")
    log("  vendor_name : 不在本脚本合并（v10）—— 01a overview 本身即含 ven_name")
    log("  vendor_type : disabled，按用户决定挪到 07/08（文件匹配有歧义）")
    files = scan_01_files()
    if not files:
        raise RuntimeError("no 01*.csv found")
    banner("STEP 0 — Load modules")
    loaded = {}
    for spec in MODULE_SPECS:
        loaded[spec["role"]] = load(spec, files)
    banner("STEP 1 — Anchor = structure")
    df = loaded.get("structure")
    if df is None:
        raise RuntimeError("structure required")
    df = groupby_first(df, "structure")
    assert df[DEAL].duplicated().sum() == 0
    n_universe = len(df)
    log(f"  ANCHOR universe: {n_universe:,} deals")
    universe = set(df[DEAL].dropna().astype("int64").tolist())
    light_report(df, "anchor")
    banner("STEP 1b — Entity coverage vs universe (左连接自然缺失，不硬拗)")
    # ★DEL★ v10: 原为 ["industry", "overview", "vendor_name"]
    for k in ["industry", "overview"]:
        d = loaded.get(k)
        if d is None:
            continue
        have = set(d[DEAL].dropna().astype("int64").tolist())
        miss = len(universe - have)
        log(f"  {k:<12s}: {len(have):,} deals | 缺失 {miss:,} ({miss/n_universe*100:.2f}%)")
    # ── STEP 2 — 实体对齐（多键级联，0 丢弃）────────────────────────────
# [V11-DISABLED]     banner("STEP 2 — 实体模块对齐（deal×tar×acq 粒度，多键级联）")
# [V11-DISABLED]     # [FFILL-PATCH] 02：折叠前先填满 industry/overview 的 tar/acq 稀疏列，
# [V11-DISABLED]     # 再按 (deal, tar_bvd, acq_bvd) 联合去重，避免主行选到实体键全 NaN 的行。
# [V11-DISABLED]     for role in ["overview", "industry"]:
# [V11-DISABLED]         d = loaded.get(role)
# [V11-DISABLED]         if d is None:
# [V11-DISABLED]             continue
# [V11-DISABLED]         n0 = len(d)
# [V11-DISABLED]         d = ffill_block(d, "tar_")
# [V11-DISABLED]         d = ffill_block(d, "acq_")
# [V11-DISABLED]         d = dedup_entity_blocks(d)
# [V11-DISABLED]         loaded[role] = d
# [V11-DISABLED]         log(f"  [FFILL] {role}: {n0:,} -> {len(d):,} (tar/acq ffill + 联合去重)")
# [V11-DISABLED]     entity = None
# [V11-DISABLED]     for k in ["industry", "overview"]:
# [V11-DISABLED]         d = loaded.get(k)
# [V11-DISABLED]         if d is None:
# [V11-DISABLED]             log(f"\n  [SKIP] {k}")
# [V11-DISABLED]             continue
# [V11-DISABLED]         for c in ENTITY_ID_COLS:
# [V11-DISABLED]             if c in d.columns:
# [V11-DISABLED]                 d[c] = norm_id_str(d[c])
# [V11-DISABLED]         if entity is None:
# [V11-DISABLED]             entity = d.copy()
# [V11-DISABLED]             log(f"\n  [entity anchor] {k}: {len(entity):,} rows")
# [V11-DISABLED]             continue
# [V11-DISABLED]         entity, handled = cascade_align_entity(entity, d, k)
# [V11-DISABLED]         unmatched = d.loc[~handled].copy()
# [V11-DISABLED]         if len(unmatched):
# [V11-DISABLED]             unmatched[f"{k}_tier"] = f"{k}_only"
# [V11-DISABLED]             entity = pd.concat([entity, unmatched], ignore_index=True)
# [V11-DISABLED]             log(f"    [append] {k} 未并入 {len(unmatched):,} 行 -> 追加为独立实体行 "
# [V11-DISABLED]                 f"（{k}_only）")
# [V11-DISABLED]     if entity is None:
# [V11-DISABLED]         log("\n  [FATAL] 无任何实体模块（industry/overview），退化为 deal 级")
# [V11-DISABLED]         entity = df[[DEAL]].copy()
# [V11-DISABLED]     n_pairs_rows = len(entity)
# [V11-DISABLED]     log(f"\n  实体对齐完成: {n_pairs_rows:,} rows "
# [V11-DISABLED]         f"(0 丢弃) | unique {DEAL}: {entity[DEAL].nunique():,}")
# [V11-DISABLED]     n_multi = int(entity.groupby(DEAL).size().gt(1).sum())
# [V11-DISABLED]     log(f"  多标的 deal: {n_multi:,} 笔（多行保留在 02_entity_pairs.csv）")
# [V11-DISABLED]     check_entity_consistency(entity)
# [V11-DISABLED]     # ── STEP 3 — 折叠实体 → deal 主行（填充最全，平局取先）──────────────
# [V11-DISABLED]     banner("STEP 3 — 折叠实体 → deal 主行")
# [V11-DISABLED]     tmp = entity.copy()
# [V11-DISABLED]     tmp["_fill"] = tmp.notna().sum(axis=1)
# [V11-DISABLED]     tmp["_ord"] = np.arange(len(tmp))
# [V11-DISABLED]     tmp = tmp.sort_values([DEAL, "_fill", "_ord"],
# [V11-DISABLED]                           ascending=[True, False, True], kind="mergesort")
# [V11-DISABLED]     master_ent = tmp.drop_duplicates(subset=[DEAL], keep="first")
# [V11-DISABLED]     master_ent = master_ent.drop(columns=["_fill", "_ord"]).reset_index(drop=True)
# [V11-DISABLED]     n_targets = entity.groupby(DEAL).size().rename("n_targets")
# [V11-DISABLED]     master_ent = master_ent.merge(n_targets.reset_index(), on=DEAL, how="left")
# [V11-DISABLED]     master_ent[DEAL] = pd.to_numeric(master_ent[DEAL], errors="coerce").astype("Int64")
# [V11-DISABLED]     log(f"  entity {n_pairs_rows:,} rows -> deal 主行 {len(master_ent):,} "
# [V11-DISABLED]         f"(多标的 {n_multi:,} 笔取主行，全量在 entity_pairs)")
# [V11-DISABLED]     log(f"  n_targets 分布: {dict(n_targets.value_counts().sort_index().head(8))}")
# [V11-DISABLED]     # ── STEP 4 — deal 级模块并入 master ─────────────────────────────────
    banner("STEP 4 — Merge deal‑level modules (on deal_num)")
    # 以 structure 锚点为左表：不在实体模块的 deal 保留，实体列留空（接受 miss）
# [V11] 实体对齐已移出 02：master 主体 = structure anchor（不再 merge master_ent）
    # ★DEL★ v10: 原为 ["vendor_name", "vendor_type", "multiple", "value", "info_source"]
    for k in ["vendor_type", "multiple", "value", "info_source"]:
        d = loaded.get(k)
        if d is None:
            log(f"\n  [SKIP] {k}")
            continue
        banner(f"STEP 4 — Merge {k}")
        d = groupby_first(d, k)
        if k == "multiple":
            d, usable_muls = audit_multiples(d, k)
            report_constant_columns(d, k)
            globals()["_USABLE_MULS"] = usable_muls
        # ★DEL★ v10: 原为 k in ("value", "vendor_name", "info_source")
        elif k in ("value", "info_source"):
            d = try_numeric(d, label=k)
        collide = [c for c in d.columns if c != DEAL and c in df.columns]
        if collide:
            d = d.rename(columns={c: f"{c}_{k}" for c in collide})
            log(f"      [冲突] {len(collide)} 列加 _{k} 后缀: {collide}")
        module_report(d, k)
        before = len(df)
        df = df.merge(d, on=DEAL, how="left")
        assert len(df) == before, f"{k} 改变行数"
        log(f"  merged. rows={len(df):,} (unchanged)")
    # ── STEP 5 — entity_pairs 输出：实体全量 + deal 级变量 ──────────────
# [V11-DISABLED]     banner("STEP 5 — 组装并写出 02_entity_pairs.csv（全量）")
# [V11-DISABLED]     deal_cols = [c for c in df.columns if c not in entity.columns or c == DEAL]
# [V11-DISABLED]     pairs = entity.merge(df[deal_cols], on=DEAL, how="left", suffixes=("", "_deal"))
# [V11-DISABLED]     pairs[DEAL] = pd.to_numeric(pairs[DEAL], errors="coerce").astype("Int64")
# [V11-DISABLED]     pairs.to_csv(OUT_PAIRS, index=False, encoding="utf‑8‑sig")
# [V11-DISABLED]     sz = os.path.getsize(OUT_PAIRS) / 1024 / 1024
# [V11-DISABLED]     log(f"  [SAVED] {OUT_PAIRS}")
# [V11-DISABLED]     log(f"    rows={len(pairs):,} | cols={pairs.shape[1]} | {sz:.1f} MB")
# [V11-DISABLED]     # ── STEP 6 — 后处理（沿用 v5）───────────────────────────────────────
    banner("STEP 6 — Post‑processing")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].replace({"__MISSING__": np.nan, "__NA__": np.nan,
                                   "nan": np.nan})
    log("  has_* 列 fillna(0) 前审计：")
    susp = []
    for c in [c for c in df.columns if c.startswith("has_")]:
        s = pd.to_numeric(df[c], errors="coerce")
        m = s.mean() if s.notna().any() else float("nan")
        log(f"    {c:<28s} {str(df[c].dtype):<12s} {m:>12.4f}")
        if not pd.api.types.is_numeric_dtype(df[c]):
            susp.append((c, "非数值 dtype"))
        elif (s.notna().sum() > 0) and (m == 0):
            susp.append((c, "mean==0，与其他同类列不一致"))
    if susp:
        log("\n  [!!] 可疑列：")
        for c, why in susp:
            log(f"      - {c:<28s} {why}")
    for c in [c for c in df.columns if c.startswith("has_")]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    log("  fillna(0) 完成")
    usable_muls = globals().get("_USABLE_MULS", [])
    mul_cols = [c for c in df.columns
                if c.startswith(MULTIPLE_PREFIX) and c.endswith("_mul_ly")]
    if mul_cols:
        use = [c for c in usable_muls if c in df.columns] or [
            c for c in mul_cols
            if pd.to_numeric(df[c], errors="coerce").nunique(dropna=True) > 1]
        mat = df[use].notna()
        df["has_any_multiple"] = mat.any(axis=1).astype(int)
        df["n_multiples"] = mat.sum(axis=1)
        log(f"\n  has_any_multiple: {int(df['has_any_multiple'].sum()):,} "
            f"({df['has_any_multiple'].mean()*100:.2f}%)")
        log(f"    参与计算 {len(use)} 列；排除常量列 "
            f"{[c for c in mul_cols if c not in use]}")
    if "reg_body_name" in df.columns:
        df["reg_info_available"] = df["reg_body_name"].notna().astype(int)
        log(f"\n  reg_info_available (=1): {int(df['reg_info_available'].sum()):,} "
            f"({df['reg_info_available'].mean()*100:.2f}%)")
    n_rows, n_deals = len(df), df[DEAL].nunique()
    log(f"\n  rows={n_rows:,} | unique {DEAL}={n_deals:,}")
    assert n_rows == n_deals, f"不是一行一 deal ({n_rows} vs {n_deals})"
    log("  [OK] assertion passed: 一行一 deal")
    # ── STEP 7 — 论文 moment ────────────────────────────────────────────
    banner("STEP 7 — SAMPLE MOMENTS (分母 = 全集)")
    log(f"  N (deals)                : {n_rows:,}")
    for c, lab in [("rumor_jfe", "p_A (lead>=3d)"),
                   ("has_rumor_status", "p_A (仅 status)"),
                   ("completed_flag", "p_D (Completed*)"),
                   ("completed_strict", "p_D (仅 Completed)")]:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            log(f"  {lab:<26s}: {v.mean():.4f}  (n={int(v.notna().sum()):,})")
    if "pre_rev_mul_ly" in df.columns:
        log(f"\n  p_B (pre_rev_mul_ly)     : {df['pre_rev_mul_ly'].notna().mean():.4f} "
            f"(n={int(df['pre_rev_mul_ly'].notna().sum()):,})")
    if "has_any_multiple" in df.columns:
        log(f"  p_B (任意倍数, 宽口径)   : {df['has_any_multiple'].mean():.4f} "
            f"(n={int(df['has_any_multiple'].sum()):,})")
    log(f"\n  论文参照(仅供参考): p_A={PAPER['p_A']} p_D={PAPER['p_D']} p_B={PAPER['p_B']}")
    # ── STEP 8 — 变量对账 + 最终覆盖率 ──────────────────────────────────
    banner("STEP 8 — 变量对账与最终覆盖率")
    reconcile_variables(loaded, df)   # [V11] pairs 不再输出
    log("\n  FINAL COVERAGE (分母 = 全集):")
    keys = ["deal_status", "announced_d", "tar_primary_sic_code", "tar_sic2",
            "tar_country_code", "acq_country_code", "pre_rev_mul_ly",
            "has_any_multiple", "deal_val_usd", "stake_acq_pct", "n_targets",
            "reg_info_available"]
    # vendor name 来源核对（v10：只应来自 industry / overview，不再有 _vendor_name 后缀）
    keys += ["ven_name", "ven_name_overview"]
    for k in keys:
        if k in df.columns:
            n = int(df[k].notna().sum())
            log(f"  {k:<28s} {n:>10,}  {n/n_rows*100:>7.2f}%")
        else:
            log(f"  {k:<28s} {'-- NOT PRESENT --':>18s}")
    junk = [c for c in df.columns if c.endswith("_vendor_name")]
    if junk:
        log(f"\n  [!!] 仍存在 vendor_name 合并残留列: {junk}")
    else:
        log("\n  [OK] 无 _vendor_name 后缀残留列")
    df.to_csv(OUT_MASTER, index=False, encoding="utf‑8‑sig")
    size = os.path.getsize(OUT_MASTER) / 1024 / 1024
    log(f"\n  [SAVED] {OUT_MASTER}")
    log(f"    rows={len(df):,} | cols={df.shape[1]} | {size:.1f} MB")
    log(f"\n  done in {time.time()-_T0:.1f}s")
    flush_log()
if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[FATAL] " + traceback.format_exc())
        flush_log()
        raise
