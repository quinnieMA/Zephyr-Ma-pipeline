# -*- coding: utf-8 -*-
"""
06b_text_peer_multiple.py — v2
================================================================================
06b: Doc2Vec text-peer KNN benchmark
  1. 01_deal_sic_industry + 05_deal_firm_filtered 用 5 键一对一对齐
  2. 01 侧先删无 tar_overview 行；01/05 各自 5 键折叠，L1-L4 选一行
  3. validate="one_to_one"，报错不硬压；merge 后断言 deal_num 唯一
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import re
import sys
import logging
import argparse
import random
from pathlib import Path
from bisect import bisect_right
import contextlib
import io
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--K", type=int, default=10, help="Number of nearest neighbors")
_parser.add_argument("--min-peers", type=int, default=3, help="Minimum peers with valid multiple")
_parser.add_argument("--min-tokens", type=int, default=10)
_parser.add_argument("--epochs", type=int, default=40)
_parser.add_argument("--vector-size", type=int, default=300)
_args = _parser.parse_args()

K = _args.K
MIN_PEERS = _args.min_peers
MIN_TOKENS = _args.min_tokens
VECTOR_SIZE = _args.vector_size
EPOCHS = _args.epochs

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\MA")
CLEANED = ROOT / "data" / "cleaned"
MERGED = ROOT / "data" / "merged"
MODELS = ROOT / "data" / "models"
for d in [MERGED, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

FILE_INDUSTRY = CLEANED / "01_deal_sic_industry.csv"
FILE_MULTIPLES = MERGED / "05_deal_firm_filtered.csv"

OUTPUT_PATH = MERGED / "06b_text_peer.csv"
DETAIL_PATH = MERGED / "06b_text_peer_detail.csv"
MODEL_PATH = MODELS / "06b_doc2vec_overview.model"
LOG_PATH = MERGED / "06b_text_peer.txt"
LIT_SW = ROOT / "raw" / "stop_words.txt"

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("06_text_peer")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
if log.hasHandlers():
    log.handlers.clear()
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(fh)
log.addHandler(ch)

def _flush_and_exit(code=1):
    try:
        logging.shutdown()
    except Exception:
        pass
    sys.exit(code)

log.info("=" * 70)
log.info("=== Script 06b: Doc2Vec Text-Peer KNN Benchmark (v2) ===")
log.info("=" * 70)
log.info(f"Parameters: K={K}, MinPeers={MIN_PEERS}, MinTokens={MIN_TOKENS}, "
         f"VecSize={VECTOR_SIZE}, Epochs={EPOCHS}")

# ── Constants ─────────────────────────────────────────────────────────────────
MUL_COLS = ['pre_rev_mul_ly', 'pre_ebitda_mul_ly', 'pre_ebit_mul_ly',
            'pre_pbt_mul_ly', 'pre_ta_mul_ly', 'pre_eq_mul_ly']
BENCH_COLS = ['bench_rev', 'bench_ebitda', 'bench_ebit', 'bench_pbt', 'bench_ta', 'bench_eq']
N_PEERS_COLS = ['n_peers_rev', 'n_peers_ebitda', 'n_peers_ebit', 'n_peers_pbt', 'n_peers_ta', 'n_peers_eq']

# ── Stopwords ─────────────────────────────────────────────────────────────────
log.info("Loading stopwords...")
hp2025_stop = set()
if LIT_SW.exists():
    with open(LIT_SW, "r", encoding="utf-8") as f:
        hp2025_stop = {w.strip().lower() for w in f if w.strip()}
BUSINESS_BOILERPLATE = {
    "company", "companies", "corporation", "incorporated", "limited",
    "operations", "business", "businesses", "headquartered", "registered",
    "office", "offices", "subsidiary", "subsidiaries", "holding",
    "provides", "provide", "offering", "offers", "offer", "services",
    "service", "products", "product", "include", "includes", "including",
    "primarily", "engaged", "engage", "operates", "operate", "conduct",
    "conducts", "based", "located", "founded", "established", "known",
    "formerly", "also", "through", "well", "various", "range", "wide",
    "full", "number", "related", "activities", "activity", "deal", "said",
    "source", "million", "billion", "management", "international", "corporate",
    "head", "involved", "provision", "providing", "addition",
    "specialises", "specializes", "active", "leading", "principally",
    "serves", "strategically", "employs", "firm", "group",
    "keting", "kets", "solutions"
}
STOP_WORDS = hp2025_stop | BUSINESS_BOILERPLATE
log.info(f"  Total combined stopwords: {len(STOP_WORDS)}")

# ── Text Preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    try:
        text = re.sub(r"(?:(?:2(?:[0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9]|(\*?))\.){3}(?:(?:2([0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9]|(\*))?)", " ", text)
        text = re.sub(r"((\d\d?(st|nd|rd|th)?([\-\./])?)?\s?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)(\s?([\-\./])?(,)?\s?(\d\d\d?\d?))?)|(([\d]+)([\-])([\d]+)([\-])([\d]+))|(([\d]+)([/])([\d]+)([/])([\d]+))|(([\d]+)([.])([\d]+)([.])([\d]+))", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"(<script(\s|\S)*?<\/script>)|(<style(\s|\S)*?<\/style>)|(<!--(\s|\S)*?-->)|(<\/?(\s|\S)*?>)", " ", text)
        text = re.sub(r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?", " ", text)
        text = re.sub(r"[`~@$^&\\;\'<>.,!?()#\"|=\-:/_*\[\]+%]", " ", text)
        text = text.lower()
        return [w for w in text.split() if len(w) >= 4 and w not in STOP_WORDS]
    except Exception:
        return []

class ReIterableCorpus:
    def __init__(self, tokens_list, indices):
        self.tokens_list = tokens_list
        self.indices = indices
    def __iter__(self):
        for idx, tokens in zip(self.indices, self.tokens_list):
            yield TaggedDocument(words=tokens, tags=[idx])

# ── 1. Load & Merge Data ──────────────────────────────────────────────────────
try:
    log.info("\n--- 1. Loading Data ---")
    df_ind = pd.read_csv(FILE_INDUSTRY, low_memory=False)
    df_mul = pd.read_csv(FILE_MULTIPLES, low_memory=False)

    keep_ind = [c for c in ['deal_num', 'tar_overview', 'tar_bvd_id_num', 'tar_name',
                            'tar_primary_sic_code', 'acq_name', 'acq_bvd_id_num']
                if c in df_ind.columns]
    keep_from_05 = ["deal_num", "tar_incorp_d_year", "acq_incorp_d_year",
                    "announced_d", "completed_d", "rumour_d"]
    keep_from_05 = [c for c in keep_from_05 if c in df_mul.columns]
    df_ind = df_ind[keep_ind].copy()

    # =====================================================================
    # [5-KEY FOLD] KEYS = deal_num + tar_bvd_id_num + tar_name + acq_name + acq_bvd_id_num
    #   L1 tar_overview 非空 | L2 pre_ta_mul_ly | L3 pre_/post_ 非空数 | L4 原始行序
    #   缺失键填 "__NA__"，不删行；deal_num 缺失直接退出（不转字符串）
    # =====================================================================
    KEYS = ["deal_num", "tar_bvd_id_num", "tar_name", "acq_name", "acq_bvd_id_num"]

    def _fill_keys_na(_d, _tag):
        _na_rows = {}
        for _c in KEYS:
            if _c not in _d.columns:
                _d[_c] = "__NA__"
                _na_rows[_c] = len(_d)
                continue
            _miss = int(_d[_c].isna().sum())
            if _c == "deal_num":
                if _miss > 0:
                    log.error(f"  [{_tag}][FATAL] deal_num 缺失 {_miss:,} 行")
                    _flush_and_exit(1)
                continue  # 主键不缺，保持数值型
            _d[_c] = _d[_c].astype(str).str.strip()
            _d.loc[_d[_c].isin(["nan", "None", "", "NaT", "<NA>"]), _c] = "__NA__"
            if _miss > 0:
                _na_rows[_c] = _miss
        if _na_rows:
            log.info(f"  [{_tag}] 5 键缺失(填 __NA__，不删行): "
                     + " | ".join(f"{_c}={_n:,}" for _c, _n in _na_rows.items()))
        return _na_rows

    # ── 01 侧：先删无 overview 行 ──
    n_raw_ind = len(df_ind)
    _ovw_s = df_ind["tar_overview"].astype(str).str.strip()
    _empty_ovw = df_ind["tar_overview"].isna() | _ovw_s.isin(
        ["", "nan", "None", "NaT", "<NA>"])
    n_drop_pre = int(_empty_ovw.sum())
    df_ind = df_ind[~_empty_ovw].copy().reset_index(drop=True)
    n_after_pre = len(df_ind)
    log.info(f"  [PRE-FILTER-01] drop 无 tar_overview 行 {n_raw_ind:,} -> "
             f"{n_after_pre:,}（删除 {n_drop_pre:,}，占 "
             f"{100*n_drop_pre/max(n_raw_ind,1):.2f}%）")
    if n_after_pre == 0:
        log.error("  [PRE-FILTER-01][FATAL] 删除后 df_ind 为空")
        _flush_and_exit(1)

    _na01 = _fill_keys_na(df_ind, "FOLD-01")
    df_ind["_has_ovw"] = 1  # 已全有 overview，L1 恒 1
    df_ind["_ord"] = np.arange(n_after_pre)

    # L2: 按 5 键从 05 挂 pre_ta_mul_ly（仅排序用）
    if "pre_ta_mul_ly" in df_mul.columns:
        _m5 = df_mul[KEYS + ["pre_ta_mul_ly"]].copy()
        for _c in KEYS:
            if _c not in _m5.columns:
                _m5[_c] = "__NA__"
            elif _c == "deal_num":
                continue
            else:
                _m5[_c] = _m5[_c].astype(str).str.strip()
                _m5.loc[_m5[_c].isin(["nan", "None", "", "NaT", "<NA>"]), _c] = "__NA__"
        _m5["pre_ta_mul_ly"] = pd.to_numeric(_m5["pre_ta_mul_ly"], errors="coerce")
        _ta_map = _m5.groupby(KEYS, as_index=False)["pre_ta_mul_ly"].max()
        _ta_aligned = df_ind[KEYS].merge(_ta_map, on=KEYS, how="left")["pre_ta_mul_ly"]
        _ta_aligned.index = df_ind.index
        df_ind["_ta"] = _ta_aligned
    else:
        df_ind["_ta"] = np.nan

    _fin_cols_ind = [c for c in df_ind.columns if c.startswith("pre_") or c.startswith("post_")]
    df_ind["_fin"] = df_ind[_fin_cols_ind].notna().sum(axis=1)

    def _cov_nn(_d, _col):
        if _col not in _d.columns:
            return None
        _s = _d[_col]
        return int((_s.notna() & (_s.astype(str).str.strip() != "nan")).sum())

    _cov_cols = ["tar_overview", "tar_name", "tar_bvd_id_num", "tar_primary_sic_code",
                 "acq_name", "acq_bvd_id_num", "acq_primary_sic_code"]

    def _cov_str(_d, _tag):
        _parts = []
        for _c in _cov_cols:
            _nn = _cov_nn(_d, _c)
            if _nn is None:
                _parts.append(f"{_c} n/a")
            else:
                _parts.append(f"{_c} {_nn:,}/{len(_d)}={_nn/max(len(_d),1)*100:.2f}%")
        return f"  [{_tag}] 覆盖率: " + " | ".join(_parts)

    _pre_cov = _cov_str(df_ind, "FOLD-01 折叠前")
    df_ind = (df_ind.sort_values(
        KEYS + ["_has_ovw", "_ta", "_fin", "_ord"],
        ascending=[True] * 5 + [False, False, False, True], kind="mergesort")
        .drop_duplicates(subset=KEYS, keep="first")
        .drop(columns=["_has_ovw", "_ta", "_fin", "_ord"])
        .reset_index(drop=True))
    _dup01 = int(df_ind.duplicated(KEYS).sum())
    assert _dup01 == 0, f"[FOLD-01] KEYS duplicated={_dup01}"
    log.info(f"  [FOLD-01] {n_after_pre:,} rows -> {len(df_ind):,} rows "
             f"| duplicated(KEYS)={_dup01} | 丢弃 {n_after_pre - len(df_ind):,} 行")
    log.info(_pre_cov)
    log.info(_cov_str(df_ind, "FOLD-01 折叠后"))

    # ── 05 侧折叠 ──
    _keep05 = list(dict.fromkeys(KEYS + keep_from_05 +
                                 [c for c in MUL_COLS if c in df_mul.columns]))
    df_mul = df_mul[[c for c in _keep05 if c in df_mul.columns]].copy()
    n_raw_mul = len(df_mul)
    _na05 = _fill_keys_na(df_mul, "FOLD-05")
    # 05 的 _keep05 不含 tar_overview，L1 不适用，显式置 0
    df_mul["_has_ovw"] = 0
    df_mul["_ta"] = (pd.to_numeric(df_mul["pre_ta_mul_ly"], errors="coerce")
                     if "pre_ta_mul_ly" in df_mul.columns else np.nan)
    _fin_cols_mul = [c for c in df_mul.columns if c.startswith("pre_") or c.startswith("post_")]
    df_mul["_fin"] = df_mul[_fin_cols_mul].notna().sum(axis=1)
    df_mul["_ord"] = np.arange(n_raw_mul)
    df_mul = (df_mul.sort_values(
        KEYS + ["_has_ovw", "_ta", "_fin", "_ord"],
        ascending=[True] * 5 + [False, False, False, True], kind="mergesort")
        .drop_duplicates(subset=KEYS, keep="first")
        .drop(columns=["_has_ovw", "_ta", "_fin", "_ord"])
        .reset_index(drop=True))
    _dup05 = int(df_mul.duplicated(KEYS).sum())
    assert _dup05 == 0, f"[FOLD-05] KEYS duplicated={_dup05}"
    _dropped05 = n_raw_mul - len(df_mul)
    log.info(f"  [FOLD-05] {n_raw_mul:,} rows -> {len(df_mul):,} rows "
             f"| duplicated(KEYS)={_dup05} | 丢弃 {_dropped05:,} 行")

    # ── 5 键一对一 merge ──
    _nL, _nR = len(df_ind), len(df_mul)
    try:
        df = df_ind.merge(df_mul, on=KEYS, how="inner", validate="one_to_one",
                          suffixes=("", "_05"))
    except Exception as _me:
        log.error(f"  [MERGE-5KEY][FATAL] validate='one_to_one' 报错: {_me}")
        _dL = df_ind[df_ind.duplicated(KEYS)]
        _dR = df_mul[df_mul.duplicated(KEYS)]
        log.error(f"  [MERGE-5KEY][FATAL] 01 侧 KEYS 重复 {len(_dL):,} 行（前 20）:")
        for _, _r in _dL.head(20).iterrows():
            log.error("      " + str([_r[_c] for _c in KEYS]))
        log.error(f"  [MERGE-5KEY][FATAL] 05 侧 KEYS 重复 {len(_dR):,} 行（前 20）:")
        for _, _r in _dR.head(20).iterrows():
            log.error("      " + str([_r[_c] for _c in KEYS]))
        _flush_and_exit(1)
    _dup_df = int(df.duplicated(KEYS).sum())
    assert _dup_df == 0, f"[MERGE-5KEY] KEYS duplicated={_dup_df}"
    log.info(f"  [MERGE-5KEY] left={_nL:,} right={_nR:,} inner={len(df):,} "
             f"| duplicated(KEYS)={_dup_df}")

    log.info(f"Merged raw: {len(df):,} rows")
    log.info(f"After merge columns: {list(df.columns)}")
    try:
        _m_n = len(df)
        log.info(f"  [KEY-CHECK] merged rows={_m_n:,} | unique deal_num="
                 f"{df['deal_num'].nunique():,} | 重复行="
                 f"{int(df['deal_num'].duplicated().sum()):,}")
        log.info("  [COVERAGE] merged 关键列:")
        for _c in ["tar_overview", "tar_name", "tar_bvd_id_num",
                   "tar_primary_sic_code", "tar_incorp_d_year",
                   "acq_incorp_d_year", "announced_d", "completed_d", "rumour_d"]:
            if _c in df.columns:
                _nn = int(df[_c].notna().sum())
                log.info(f"      {_c:<24s} {_nn:>9,} ({_nn/max(_m_n,1)*100:6.2f}%)")
            else:
                log.info(f"      {_c:<24s} [列不存在]")
    except Exception as _e:
        log.warning(f"  [WARN] 合并后覆盖率诊断失败: {_e}")
except Exception as e:
    log.error(f"CRITICAL ERROR during data loading: {e}")
    _flush_and_exit(1)

# ── 2. Date Engineering ───────────────────────────────────────────────────────
log.info("\n--- 2. Date Engineering ---")
def parse_zephyr_date(series):
    s = series.astype(str).str.strip().replace(['nan', 'None', '', 'NaN'], pd.NaT)
    return pd.to_datetime(s, errors='coerce')

for col_in, col_out in [('announced_d', 'announced_dt'),
                        ('completed_d', 'completed_dt'),
                        ('rumour_d', 'rumour_dt')]:
    if col_in in df.columns:
        df[col_out] = parse_zephyr_date(df[col_in])
df['deal_date'] = df['announced_dt'].fillna(df['completed_dt']).fillna(df['rumour_dt'])
df['deal_year'] = df['deal_date'].dt.year

has_incorp_year = 'tar_incorp_d_year' in df.columns
if has_incorp_year:
    df['tar_incorp_d_year'] = pd.to_numeric(df['tar_incorp_d_year'], errors='coerce')
    log.info(f"Rows with valid tar_incorp_d_year: {df['tar_incorp_d_year'].notna().sum():,}")
else:
    log.warning("Column 'tar_incorp_d_year' not found. Skipping +/- 10 years filter.")

try:
    _n_dd = int(df['deal_date'].notna().sum())
    log.info(f"  deal_date 非空率: {_n_dd:,}/{len(df):,} = "
             f"{_n_dd/max(len(df),1)*100:.2f}%")
    _y_min = df['deal_year'].min(); _y_max = df['deal_year'].max()
    if pd.notna(_y_min):
        log.info(f"  deal_year 范围: {int(_y_min):,} ~ {int(_y_max):,}")
    for _c in ["announced_dt", "completed_dt", "rumour_dt"]:
        if _c in df.columns:
            _nn = int(df[_c].notna().sum())
            log.info(f"  {_c}: 有效 {_nn:,} ({_nn/max(len(df),1)*100:.2f}%)")
except Exception as _e:
    log.warning(f"  [WARN] 日期工程诊断失败: {_e}")

# ── 3. Text Cleaning ──────────────────────────────────────────────────────────
log.info("\n--- 3. Text Cleaning ---")
df['tar_overview'] = df['tar_overview'].fillna("")
tqdm.pandas(desc="Tokenizing (Stable)")
df['tokens'] = df['tar_overview'].progress_apply(preprocess_text)
df['token_count'] = df['tokens'].apply(len)

try:
    _tc = df['token_count']
    _zero = int((_tc == 0).sum())
    log.info(f"  Token stats: mean={_tc.mean():.1f}, median={_tc.median():.1f}, "
             f"max={int(_tc.max()):,} | 零词条行: {_zero:,} "
             f"({_zero/max(len(df),1)*100:.2f}%)")
except Exception as _e:
    log.warning(f"  [WARN] 分词统计失败: {_e}")

# ── 4. Multiples Winsorization ────────────────────────────────────────────────
log.info("\n--- 4. Multiples Winsorization ---")
for col in MUL_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
        log.info(f"  {col}: clipped [{lo:.3f},{hi:.3f}], valid={df[col].notna().sum():,}")

# ── 5. Final Filtering & Sorting ──────────────────────────────────────────────
valid_text_mask = df['token_count'] >= MIN_TOKENS
df_valid = df[valid_text_mask & df['deal_date'].notna()].copy()
log.info("\n--- 5. Final Corpus ---")
log.info(f"Final Corpus size (Text + Date): {len(df_valid):,} rows")
if len(df_valid) == 0:
    log.error("CRITICAL: No valid rows after filter!")
    _flush_and_exit(1)
df_valid = df_valid.sort_values('deal_date').reset_index(drop=True)
df_valid['doc_idx'] = df_valid.index

try:
    _tc2 = df_valid['token_count']
    _yc = df_valid['deal_year'].value_counts().sort_index()
    log.info(f"  Corpus token stats: mean={_tc2.mean():.1f}, "
             f"median={_tc2.median():.1f}, min={int(_tc2.min()):,}, "
             f"max={int(_tc2.max()):,}")
    log.info(f"  Corpus year range: {int(_yc.index.min()):,} ~ "
             f"{int(_yc.index.max()):,} | 覆盖年份数: {len(_yc)}")
    log.info(f"  Corpus 年均规模: 平均 {_yc.mean():.0f} 笔/年 | 峰值年份 "
             f"{int(_yc.index[_yc.argmax()])}: {int(_yc.max()):,} 笔")
except Exception as _e:
    log.warning(f"  [WARN] 最终语料诊断失败: {_e}")

# ── 6. Train Doc2Vec ──────────────────────────────────────────────────────────
log.info("\n--- 6. Train Doc2Vec ---")
tokens_list_train = df_valid['tokens'].tolist()
indices_train = df_valid['doc_idx'].tolist()
corpus = ReIterableCorpus(tokens_list_train, indices_train)
n_workers_train = 1
log.info(f"Doc2Vec workers={n_workers_train}, vector_size={VECTOR_SIZE}, epochs={EPOCHS}")
try:
    with contextlib.redirect_stderr(io.StringIO()):
        model = Doc2Vec(
            documents=corpus, dm=0, vector_size=VECTOR_SIZE, window=15,
            min_count=3, dbow_words=1, hs=1, negative=0, sample=1e-5,
            alpha=0.025, min_alpha=0.0001,
            epochs=EPOCHS, workers=n_workers_train, seed=SEED
        )
    model.save(str(MODEL_PATH))
    log.info(f"Doc2Vec saved, vocab size: {len(model.wv.key_to_index):,}")
except Exception as e:
    log.error(f"Doc2Vec train failed: {e}")
    _flush_and_exit(1)
del corpus, tokens_list_train, indices_train

# ── 7. Build Peer Pool & Matrices ─────────────────────────────────────────────
log.info("\n--- 7. Build matrix & peer pool ---")
pool_df = df_valid.copy()
pool_years_list = pool_df['deal_year'].tolist()

vec_matrix = np.zeros((len(pool_df), VECTOR_SIZE), dtype=np.float32)
for i, idx in enumerate(tqdm(pool_df['doc_idx'], desc="Extract doc vectors")):
    vec_matrix[i] = model.dv[idx]
norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
vec_matrix /= norms

mul_matrix = pool_df[MUL_COLS].values.astype(np.float32)
deal_nums = pool_df['deal_num'].values
bvd_ids = pool_df['tar_bvd_id_num'].values if 'tar_bvd_id_num' in pool_df.columns else None
inc_years = pool_df['tar_incorp_d_year'].values if has_incorp_year else None

# ── 8. Rolling Window KNN ─────────────────────────────────────────────────────
log.info("\n--- 8. Rolling-year KNN peer search ---")
results = {}
focal_years = sorted(pool_df['deal_year'].dropna().unique())
n_processed = 0

for year_t in tqdm(focal_years, desc="Process each focal year"):
    pool_cutoff = bisect_right(pool_years_list, year_t)
    if pool_cutoff < K:
        continue
    active_vecs = vec_matrix[:pool_cutoff]
    active_muls = mul_matrix[:pool_cutoff]
    active_dns = deal_nums[:pool_cutoff]
    active_bvds = bvd_ids[:pool_cutoff] if bvd_ids is not None else None
    active_incs = inc_years[:pool_cutoff] if inc_years is not None else None

    focal_in_year = pool_df[pool_df['deal_year'] == year_t]
    focal_indices = focal_in_year['doc_idx'].values
    focal_incs = focal_in_year['tar_incorp_d_year'].values if has_incorp_year else None
    focal_vecs = vec_matrix[focal_indices]

    sim_matrix = focal_vecs @ active_vecs.T
    for i, focal_idx in enumerate(focal_indices):
        sim_scores = sim_matrix[i].copy()
        focal_dn = deal_nums[focal_idx]
        mask = (active_dns == focal_dn)
        if active_bvds is not None and pd.notna(bvd_ids[focal_idx]):
            mask |= (active_bvds == bvd_ids[focal_idx])
        if active_incs is not None and pd.notna(focal_incs[i]):
            inc_diff = np.abs(active_incs - focal_incs[i])
            mask |= pd.isna(active_incs) | (inc_diff > 10)
        sim_scores[mask] = -2.0
        k_actual = min(K, int((sim_scores > -1.5).sum()))
        if k_actual == 0:
            continue
        top_k_idx = np.argpartition(sim_scores, -k_actual)[-k_actual:]
        top_k_idx = top_k_idx[sim_scores[top_k_idx] > -1.5]
        top_k_idx = top_k_idx[np.argsort(sim_scores[top_k_idx])[::-1]]
        peer_sims = sim_scores[top_k_idx]
        peer_muls = active_muls[top_k_idx]
        peer_dns_list = active_dns[top_k_idx]

        res_dict = {
            "text_peer_sim_mean": float(np.mean(peer_sims)),
            "text_peer_ids": "|".join(str(dn) for dn in peer_dns_list),
            "text_peer_sim_list": "|".join(f"{s:.6f}" for s in peer_sims),
        }
        for col_idx, (mc, bc, nc) in enumerate(zip(MUL_COLS, BENCH_COLS, N_PEERS_COLS)):
            col_data = peer_muls[:, col_idx]
            valid_data = col_data[(~np.isnan(col_data)) & (col_data > 0)]
            res_dict[nc] = int(len(valid_data))
            res_dict[bc] = float(np.median(valid_data)) if len(valid_data) >= MIN_PEERS else np.nan
        results[focal_dn] = res_dict
        n_processed += 1
log.info(f"KNN peer finished, processed {n_processed:,} focal deals")

# ── 9. Merge & Output Wide Table ──────────────────────────────────────────────
log.info("\n--- 9. Merge & Saving Wide Table ---")
assert df['deal_num'].is_unique, \
    f"[OUTPUT] df 的 deal_num 不唯一: {int(df['deal_num'].duplicated().sum()):,}"
res_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
res_df.rename(columns={'index': 'deal_num'}, inplace=True)
df_final = df.merge(res_df, on='deal_num', how='left')
assert df_final['deal_num'].is_unique, \
    f"[OUTPUT] merge 后 deal_num 不唯一: {int(df_final['deal_num'].duplicated().sum()):,}"

if 'deal_date_x' in df_final.columns:
    df_final['deal_date'] = df_final['deal_date_x']
elif 'deal_date_y' in df_final.columns:
    df_final['deal_date'] = df_final['deal_date_y']

output_cols = (['deal_num', 'deal_date', 'deal_year'] + MUL_COLS +
               ['text_peer_sim_mean'] + N_PEERS_COLS + BENCH_COLS)
existing_cols = [c for c in output_cols if c in df_final.columns]
df_out = df_final[existing_cols].copy()
df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
log.info(f"Saved wide table to: {OUTPUT_PATH} ({len(df_out):,} rows)")
log.info(f"  [KEY-CHECK] wide-table deal_num unique: "
         f"{bool(df_out['deal_num'].is_unique)} | rows={len(df_out):,} | "
         f"unique={df_out['deal_num'].nunique():,}")

try:
    _n_out = len(df_out)
    log.info(f"  [OUT-COVERAGE] 宽表 {_n_out:,} 行 × {df_out.shape[1]} 列:")
    for _c in df_out.columns:
        _nn = int(df_out[_c].notna().sum())
        _pct = _nn / _n_out * 100
        if pd.api.types.is_numeric_dtype(df_out[_c]):
            _vals = pd.to_numeric(df_out[_c], errors="coerce").dropna()
            if len(_vals):
                _stat = (f"min={_vals.min():.4g} | med={_vals.median():.4g} | "
                         f"mean={_vals.mean():.4g} | max={_vals.max():.4g}")
            else:
                _stat = "ALL MISSING"
        else:
            _stat = f"unique={df_out[_c].nunique():,}"
        log.info(f"      {_c:<24s} nonnull={_nn:>9,} ({_pct:6.2f}%) | {_stat}")
except Exception as _e:
    log.warning(f"  [WARN] 宽表覆盖率诊断失败: {_e}")

# ── 10. Long Detail Table ─────────────────────────────────────────────────────
log.info("\n--- 10. Generating Full Detail Table (Long Format) ---")
detail_records = []
detail_base_cols = ['deal_num', 'tar_name', 'tar_overview', 'tar_incorp_d_year',
                    'deal_year', 'tar_primary_sic_code']
detail_base_cols = [c for c in detail_base_cols if c in df.columns]
detail_all_cols = detail_base_cols + [c for c in MUL_COLS if c in df.columns]
peer_info = df[detail_all_cols].set_index('deal_num')

for focal_dn, res in tqdm(results.items(), desc="Building Detail"):
    peer_id_str = res.get("text_peer_ids", "")
    sim_str = res.get("text_peer_sim_list", "")
    if not peer_id_str:
        continue
    peer_dns = [int(pid) for pid in peer_id_str.split("|")]
    sim_values = [float(s) for s in sim_str.split("|")]
    combine = sorted(zip(sim_values, peer_dns), reverse=True)
    focal_row = df[df['deal_num'] == focal_dn].iloc[0]
    for rank, (sim, pid) in enumerate(combine, start=1):
        if pid in peer_info.index:
            p = peer_info.loc[pid]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
            detail_records.append({
                "focal_deal_num": focal_dn,
                "focal_company_name": focal_row.get('tar_name', ''),
                "focal_deal_year": focal_row.get('deal_year', np.nan),
                "focal_incorp_year": focal_row.get('tar_incorp_d_year', np.nan),
                "focal_tar_primary_sic": focal_row.get('tar_primary_sic_code', np.nan),
                "similarity_rank": rank,
                "cosine_similarity": sim,
                "peer_deal_num": pid,
                "peer_company_name": p.get('tar_name', ''),
                "peer_deal_year": p.get('deal_year', np.nan),
                "peer_incorp_year": p.get('tar_incorp_d_year', np.nan),
                "peer_tar_primary_sic": p.get('tar_primary_sic_code', np.nan),
                "peer_ev_rev": p.get('pre_rev_mul_ly', np.nan),
                "peer_ev_ebitda": p.get('pre_ebitda_mul_ly', np.nan),
                "peer_ev_ebit": p.get('pre_ebit_mul_ly', np.nan),
                "peer_ev_pbt": p.get('pre_pbt_mul_ly', np.nan),
                "peer_ev_ta": p.get('pre_ta_mul_ly', np.nan),
                "peer_ev_eq": p.get('pre_eq_mul_ly', np.nan),
            })
detail_df = pd.DataFrame(detail_records)
detail_df.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
log.info(f"Saved peer detail table to: {DETAIL_PATH} ({len(detail_df):,} rows)")

try:
    _n_det = len(detail_df)
    if _n_det > 0:
        _focal_unique = detail_df['focal_deal_num'].nunique()
        _per = detail_df.groupby('focal_deal_num').size()
        log.info(f"  [DETAIL] 长表 {_n_det:,} 行 | 唯一 focal deal: {_focal_unique:,} | "
                 f"每 deal 平均 peer 行: {_per.mean():.1f} | 中位数: {_per.median():.0f}")
        log.info("  [DETAIL][COVERAGE] 关键列:")
        for _c in ['focal_company_name', 'peer_company_name', 'cosine_similarity',
                   'peer_ev_rev', 'peer_ev_ebitda', 'peer_ev_ebit', 'peer_ev_pbt',
                   'peer_ev_ta', 'peer_ev_eq']:
            if _c in detail_df.columns:
                _nn = int(detail_df[_c].notna().sum())
                log.info(f"      {_c:<20s} nonnull={_nn:>9,} ({_nn/_n_det*100:6.2f}%)")
    else:
        log.info("  [DETAIL] 长表为空（无 peer 记录）")
except Exception as _e:
    log.warning(f"  [WARN] 明细表诊断失败: {_e}")

# ── 11. Diagnostics ───────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("=== COMPREHENSIVE DIAGNOSTICS REPORT ===")
log.info("=" * 70)
n_total = len(df_out)
for bc in BENCH_COLS:
    if bc in df_out.columns:
        n_valid = df_out[bc].notna().sum()
        pct = 100 * n_valid / n_total if n_total > 0 else 0
        log.info(f"  {bc}: {n_valid:,} / {n_total:,} ({pct:.2f}%)")
sim_vals = df_out['text_peer_sim_mean'].dropna()
if len(sim_vals) > 0:
    log.info(f"  Similarity: Mean={sim_vals.mean():.4f}, "
             f"Median={sim_vals.median():.4f}, P90={sim_vals.quantile(0.90):.4f}")

try:
    _pcols = [c for c in N_PEERS_COLS if c in df_out.columns]
    if _pcols:
        log.info("  Peer counts (n_peers_*):")
        for _c in _pcols:
            _vals = df_out[_c].dropna()
            if len(_vals):
                log.info(f"      {_c:<16s} 非零={int((_vals > 0).sum()):>9,} | "
                         f"mean={_vals.mean():.2f} | median={_vals.median():.0f} | "
                         f"p90={_vals.quantile(0.90):.0f}")
    _nn_sim = int(df_out['text_peer_sim_mean'].notna().sum())
    log.info(f"  FINAL: 宽表 {len(df_out):,} 行 | 唯一 deal "
             f"{df_out['deal_num'].nunique():,} | 有 peer 结果的 deal {_nn_sim:,} "
             f"({_nn_sim/max(len(df_out),1)*100:.2f}%)")
except Exception as _e:
    log.warning(f"  [WARN] 最终诊断失败: {_e}")

log.info("\n=== SCRIPT 06b COMPLETE ===")
logging.shutdown()