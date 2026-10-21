# -*- coding: utf-8 -*-
r"""
01a_clean_overview.py — Overview export (NO acquirer aggregation)
=============================================================
Input  : D:\zephyr\overview\cleaned_output\MA_overview.csv
Output : D:\MA\data\cleaned\01_deal_overview.csv        (overview + regulatory + legal + cross-border)
         D:\MA\data\cleaned\01a_deal_vendor_name.csv     (M3 vendor aggregation)
Log    : D:\MA\data\cleaned\01a_overview.txt

Observation unit: PRIMARY KEY = (deal_num, tar_name, acq_name, tar_bvd_id_num, acq_bvd_id_num)
DO NOT aggregate acquirer information! Keep raw per‑row acq_name / acq_bvd_id_num.

Updates:
  1. Added cross_border and cross_country variables.
  2. Added La Porta (1998) legal origin mapping (legal_origin_tar/acq, common_law_tar/acq).
  3. Added classification distance (classification_distance_tar/acq).
  4. Regulatory body / country dummy variables generated BEFORE deduplication.
  5. M3 vendor aggregation executed BEFORE the target‑acquirer dedup.
  6. Deduplicate key strictly: ["deal_num","tar_name","acq_name","tar_bvd_id_num","acq_bvd_id_num"].
"""
import os
import sys
import glob
import re
import pandas as pd
import numpy as np

# =====================================================================
# Module 1 — Config
# =====================================================================
BASE    = r"D:\MA"
ZEPHYR  = r"D:\zephyr"
CLEANED = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEANED, exist_ok=True)
RAW_OVW = os.path.join(ZEPHYR, "overview", "cleaned_output")
MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

# =====================================================================
# Logging
# =====================================================================
LOG_PATH = os.path.join(CLEANED, "01a_overview.txt")
_log_fh = open(LOG_PATH, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log_fh.write(str(msg) + "\n")
    _log_fh.flush()

# =====================================================================
# Helpers
# =====================================================================
def read_and_ffill(filepath, encoding="utf-8-sig"):
    if filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath, sheet_name=0, dtype=str)
    else:
        df = pd.read_csv(filepath, encoding=encoding, low_memory=False, dtype=str)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u200b", "", regex=False)
        .str.strip()
    )
    log(f"  [read] file={os.path.basename(filepath)} | "
        f"rows={len(df):,} | cols={len(df.columns)}")
    if "deal_num" in df.columns:
        df["deal_num"] = df["deal_num"].ffill()
    return df

def clean_missing_inplace(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(MISSING_VALS, np.nan)
        df[col] = df[col].str.strip()
    return df

def fix_orbis_val(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s in ("nan", "None", ""):
        return np.nan
    return s.zfill(9)

def normalize_orbis_id(ser):
    return ser.apply(fix_orbis_val)

def report_coverage(df, label, cols=None):
    cols = list(df.columns) if cols is None else [c for c in cols if c in df.columns]
    log(f"\n  [{label}] coverage — {len(df):,} rows")
    for c in cols:
        if c in df.columns:
            nn = int(df[c].notna().sum())
            pct = nn / len(df) * 100 if len(df) else 0.0
            log(f"    {c:<45s} non-null {nn:>9,} ({pct:6.2f}%)  "
                f"missing {len(df) - nn:>9,} ({100.0 - pct:6.2f}%)")

def report_expected(df, label, expected):
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log(f"  [expected] {label}: {len(missing)} expected column(s) missing -> "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}")
    else:
        log(f"  [expected] {label}: all {len(expected)} expected columns present")

# =====================================================================
# Regulatory classification constants
# =====================================================================
REG_CATEGORY_KEYWORDS = {
    "antitrust": [
        "competition", "antitrust", "anti-trust", "anti-monopoly",
        "antimonopoly", "monopoly", "kartell", "konkurr", "cade",
        "fair trade", "bundeskartellamt", "ftc", "department of justice",
        "doj", "dg competition", "wettbewerb", "concurrencia",
        "concurrence", "concorrenza", "mededingingsautoriteit",
        "competition commission", "competition and markets authority",
        "cma", "accc", "state administration for market regulation",
        "samr", "anti-monopoly bureau", "amb", "federal antimonopoly service",
        "federal trade commission", "autorité de la concurrence",
        "autorita garante della concorrenza", "office of competition",
        "competition bureau", "jersey competition regulatory authority",
        "monopolkommission", "direccion general de defensa de la competencia",
        "nederlandse mededingingsautoriteit", "nma",
        "direction de la concurrence", "autoriteit financiele markten",
    ],
    "securities": [
        "securities commission", "securities and exchange", "sec",
        "china securities regulatory commission", "csrc",
        "financial supervisory commission", "fsc", "securities and futures",
        "sfc", "autorité des marchés financiers", "amf",
        "financial services authority", "fsa", "financial conduct authority",
        "fca", "bafin", "bundesanstalt für finanzdienstleistungsaufsicht",
        "commission de surveillance du secteur financier",
        "financial services board", "securities and exchange board", "sebi",
        "monetary authority", "cbrc",
    ],
    "state_assets": [
        "state-owned assets supervision", "sasac",
        "assets supervision and administration commission", "state-owned",
    ],
    "foreign_invest": [
        "foreign investment", "foreign investments review",
        "investment commission", "firb", "foreign investment committee",
        "committee on foreign investment", "cfius", "investment review board",
    ],
    "financial": [
        "banking", "insurance", "financial services commission",
        "financial supervision", "financial regulatory", "china banking",
        "banking and insurance", "central bank", "reserve bank",
    ],
    "defense_tech": [
        "defence", "defense", "national security",
        "state administration of science",
        "science, technology and industry for national defence",
        "ministry of defence", "department of defense", "national defence",
    ],
}

COMMON_LAW_COUNTRIES = {
    "United States", "United Kingdom", "Australia", "Canada", "New Zealand",
    "Singapore", "Hong Kong", "India", "Malaysia", "South Africa", "Ireland",
    "Israel", "Pakistan", "Bangladesh", "Sri Lanka", "Kenya", "Nigeria",
    "Ghana", "Jamaica", "Trinidad and Tobago", "Bahamas", "Barbados",
    "Mauritius", "Cyprus", "Malta", "Guernsey", "Jersey", "Isle of Man",
    "Cayman Islands", "British Virgin Islands", "Bermuda",
}

CIVIL_LAW_COUNTRIES = {
    "China", "Germany", "France", "Italy", "Spain", "Netherlands", "Belgium",
    "Austria", "Switzerland", "Sweden", "Norway", "Denmark", "Finland",
    "Japan", "South Korea", "Taiwan", "Russia", "Brazil", "Mexico",
    "Argentina", "Chile", "Colombia", "Peru", "Venezuela", "Philippines",
    "Indonesia", "Thailand", "Vietnam", "Turkey", "Poland", "Czech Republic",
    "Hungary", "Romania", "Bulgaria", "Greece", "Portugal", "Luxembourg",
    "Iceland", "Estonia", "Latvia", "Lithuania", "Slovakia", "Slovenia",
    "Croatia", "Serbia", "Ukraine", "Egypt", "Morocco", "Algeria", "Tunisia",
}

TRANSNATIONAL_REGULATORS = {"European Union", "EU", "European Commission"}

REG_EXCLUDE = {
    "malaysia securities exchange", "securities industry council",
    "takeover regulation panel", "chamber of the amsterdam court of appeal",
    "the federal court of australia", "ontario superior court of justice",
    "supreme court of british columbia", "national company law tribunal",
    "federal court", "high court", "court of appeal", "supreme court",
}

REG_TRANSNATIONAL_BODIES = {
    "european commission", "european council", "european parliament",
    "council of the european union", "european central bank",
    "european banking authority", "european securities and markets authority",
    "european insurance and occupational pensions authority",
    "directorate general for competition",
}

REG_EXACT_MAP = {
    "china banking and insurance regulatory commission": {"financial"},
    "cbirc": {"financial"},
    "cbrc": {"financial"},
    "financial regulatory authority": {"financial"},
    "guernsey financial services commission": {"financial"},
    "jersey financial services commission": {"financial"},
    "ministry of commerce": {"foreign_invest"},
    "national development and reform commission": {"state_assets"},
    "state administration of foreign exchange": {"foreign_invest"},
    "industrial & commercial administration bureau": {"state_assets"},
    "ministry of finance": {"financial"},
    "national administration of financial regulation": {"financial"},
    "china's state council": {"state_assets"},
    "state council": {"state_assets"},
    "european commission": {"antitrust"},
    "anti-trust authority": {"antitrust"},
    "comisión nacional de la competencia": {"antitrust"},
    "comision nacional de los mercados y la competencia": {"antitrust"},
    "comisión federal de competencia": {"antitrust"},
    "autoridade de concorrencia": {"antitrust"},
    "autoriteit consument en markt": {"antitrust"},
    "office of fair trading": {"antitrust"},
    "gazdasagi versenyhivatal": {"antitrust"},
    "konkurentsiamet": {"antitrust"},
    "lietuvos respublikos konkurencijos taryba": {"antitrust"},
    "commerce commission": {"antitrust"},
    "illinois commerce commission": {"antitrust"},
    "consob": {"securities"},
    "comision nacional del mercado de valores": {"securities"},
    "komisija za hartije od vrjednosti": {"securities"},
    "federal service for financial markets": {"securities"},
    "finanstilsynet": {"securities"},
    "finansinspektionen": {"securities"},
    "autoriteit financiële markten": {"securities"},
    "croatian agency for the supervision of financial services": {"securities"},
    "financial supervisory authority": {"securities"},
    "egyptian financial supervisory authority": {"securities"},
    "capital market authority": {"securities"},
    "capital markets authority": {"securities"},
    "capital markets board": {"securities"},
    "superintendencia financiera de colombia": {"securities"},
    "australian securities and investments commission": {"securities"},
    "swiss financial market supervisory authority": {"securities"},
    "securities & commodities authority": {"securities"},
    "securities market agency": {"securities"},
    "comissão de valores mobiliários": {"securities"},
    "comissão do mercado de valores mobiliários": {"securities"},
    "autorité des services et marchés financiers": {"securities"},
    "bank negara malaysia": {"financial"},
    "federal reserve board": {"financial"},
    "bank of italy": {"financial"},
    "office of the comptroller of the currency": {"financial"},
    "national bank of cambodia": {"financial"},
    "central bank of mozambique": {"financial"},
    "nepal rastra bank": {"financial"},
    "federal energy regulatory commission": {"financial"},
    "prudential regulation authority": {"financial"},
    "overseas investment office": {"foreign_invest"},
    "ministry of international trade and industry": {"foreign_invest"},
    "ministry of economic affairs": {"foreign_invest"},
    "agência nacional de energia elétrica": {"financial"},
    "agência nacional de telecomunicações": {"defense_tech"},
    "federal communications commission": {"defense_tech"},
    "kilpailuvirasto": {"antitrust"},
    "australian prudential regulatory authority": {"financial"},
    "australian prudential regulation authority": {"financial"},
    "bureau of internal revenue": {"financial"},
    "ministry of domestic trade and consumer affairs": {"financial"},
    "agência nacional de saúde suplementar": {"financial"},
}

def _norm(s):
    return " ".join(str(s).lower().split())

def classify_regulatory_category(body_name):
    if pd.isna(body_name):
        return set()
    nm = _norm(str(body_name).strip())
    for frag, cats in REG_EXACT_MAP.items():
        if _norm(frag) in nm:
            return set(cats)
    for bad in REG_EXCLUDE:
        if _norm(bad) in nm:
            return set()
    cats = set()
    for cat, keywords in REG_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            kwl = kw.lower()
            if len(kwl) <= 4:
                if re.search(rf"\b{re.escape(kwl)}\b", nm):
                    cats.add(cat)
                    break
            elif kwl in nm:
                cats.add(cat)
                break
    return cats

def is_common_law_country(country):
    if pd.isna(country):
        return False
    return str(country).strip() in COMMON_LAW_COUNTRIES

def is_civil_law_country(country):
    if pd.isna(country):
        return False
    return str(country).strip() in CIVIL_LAW_COUNTRIES

def is_transnational(country):
    if pd.isna(country):
        return False
    return str(country).strip() in TRANSNATIONAL_REGULATORS

def build_regulatory_variables(df_full_raw):
    body_col = "reg_body_name" if "reg_body_name" in df_full_raw.columns else "regulatory_body_name"
    ctry_col = "reg_body_ctry" if "reg_body_ctry" in df_full_raw.columns else "regulatory_body_country"
    if body_col not in df_full_raw.columns:
        return pd.DataFrame()
    reg_df = df_full_raw[df_full_raw[body_col].notna()].copy()
    if len(reg_df) == 0:
        return pd.DataFrame()
    if ctry_col in reg_df.columns:
        country_series = reg_df[ctry_col]
    else:
        country_series = pd.Series([np.nan] * len(reg_df), index=reg_df.index)
    reg_df[ctry_col] = country_series
    reg_df["deal_num"] = pd.to_numeric(reg_df["deal_num"], errors="coerce")
    reg_df = reg_df[reg_df["deal_num"].notna()].copy()
    reg_df["deal_num"] = reg_df["deal_num"].astype("Int64")
    reg_agg = reg_df.groupby("deal_num").agg(
        regulatory_bodies=(body_col, lambda x: "|".join(sorted(set(x.dropna())))),
        regulatory_countries=(ctry_col, lambda x: "|".join(sorted(set(x.dropna())))),
        reg_body_count=(body_col, "nunique"),
        reg_country_count=(ctry_col, "nunique"),
    ).reset_index()
    for cat in REG_CATEGORY_KEYWORDS.keys():
        col_name = f"reg_{cat}"
        reg_agg[col_name] = reg_agg["regulatory_bodies"].apply(
            lambda bodies: any(
                cat in classify_regulatory_category(b)
                for b in bodies.split("|") if b
            )
        ).astype(int)
    def _is_transnational_body(bodies_str):
        parts = [p.strip() for p in str(bodies_str).split("|") if p.strip()]
        return any(
            any(_norm(t) in _norm(p) for t in REG_TRANSNATIONAL_BODIES)
            for p in parts
        )
    reg_agg["reg_cross_national"] = reg_agg.apply(
        lambda r: int(
            any(is_transnational(c)
                for c in str(r["regulatory_countries"]).split("|") if c)
            or _is_transnational_body(r["regulatory_bodies"])
        ), axis=1
    ).astype(int)
    reg_agg["reg_common_law"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(
            is_common_law_country(c) for c in countries.split("|") if c
        )
    ).astype(int)
    reg_agg["reg_civil_law"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(
            is_civil_law_country(c) for c in countries.split("|") if c
        )
    ).astype(int)
    reg_agg["reg_mixed_legal"] = (
        (reg_agg["reg_common_law"] == 1) & (reg_agg["reg_civil_law"] == 1)
    ).astype(int)
    return reg_agg

# =====================================================================
# Legal Origin & Classification Distance Constants (La Porta 1998)
# =====================================================================
LEGAL_ORIGIN = {
    "AU": "English", "CA": "English", "GH": "English", "HK": "English",
    "IN": "English", "IE": "English", "JM": "English", "KE": "English",
    "MY": "English", "NZ": "English", "NG": "English", "PK": "English",
    "SG": "English", "ZA": "English", "TZ": "English", "UG": "English",
    "GB": "English", "US": "English", "ZW": "English", "BD": "English",
    "LK": "English", "ZM": "English",
    "BE": "French", "BR": "French", "CL": "French", "CO": "French",
    "EG": "French", "FR": "French", "GR": "French", "ID": "French",
    "IT": "French", "JO": "French", "LB": "French", "LU": "French",
    "MX": "French", "MA": "French", "NL": "French", "PE": "French",
    "PT": "French", "RO": "French", "ES": "French", "TH": "French",
    "TN": "French", "TR": "French", "VE": "French", "CN": "French", "RU": "French",
    "UA": "French", "VN": "French", "KZ": "French", "AE": "French", "SA": "French",
    "QA": "French", "KW": "French", "BH": "French",
    "AT": "German", "CZ": "German", "DE": "German", "HU": "German",
    "JP": "German", "KR": "German", "SK": "German", "CH": "German",
    "TW": "German", "PL": "German", "HR": "German", "SI": "German",
    "RS": "German",
    "DK": "Scandinavian", "FI": "Scandinavian", "IS": "Scandinavian",
    "NO": "Scandinavian", "SE": "Scandinavian",
    "KY": "English", "IL": "English", "BM": "English", "MT": "English",
    "CY": "English", "MU": "English", "PG": "English", "VG": "English",
    "MH": "English", "GI": "English", "VC": "English",
    "BG": "French", "LT": "French", "BY": "French", "UY": "French",
    "UZ": "French", "AR": "French", "DZ": "French", "OM": "French",
    "PA": "French", "CD": "French", "MN": "French",
    "EE": "German", "LV": "German", "ME": "German", "BA": "German",
    "MK": "German",
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

# =====================================================================
# Main program
# =====================================================================
log("=" * 78)
log("Module 01a — Overview 诊断输出 (NO acquirer aggregation)")
log("PRIMARY KEY: deal_num + tar_name + acq_name + tar_bvd_id_num + acq_bvd_id_num")
log("=" * 78)

OVW_FILE = os.path.join(RAW_OVW, "MA_overview.csv")
log(f"Input file: {OVW_FILE}")
if not os.path.exists(OVW_FILE):
    log(f"[ERROR] File not found: {OVW_FILE}")
    _log_fh.close()
    sys.exit(1)

df_raw = read_and_ffill(OVW_FILE)
n_raw = len(df_raw)
log(f"Raw rows read: {n_raw:,}")

DROP_ALWAYS = ["_drop_index", "unnamed_0", "Unnamed: 0"]
drop_cols_always = [c for c in DROP_ALWAYS if c in df_raw.columns]
drop_cols_dup = [c for c in df_raw.columns if "__1" in c]
all_drop = drop_cols_always + drop_cols_dup
if all_drop:
    df_raw = df_raw.drop(columns=all_drop)
    log(f"Dropped {len(all_drop)} columns (index / __1 duplicates)")

df_raw = clean_missing_inplace(df_raw)

OVW_KEEP = [
    "deal_num",
    "deal_headline",
    "deal_type",
    "deal_status",
    "deal_val_usd",
    "tar_name",
    "tar_ctry",
    "tar_busi_descr",
    "tar_bvd_id_num",
    "tar_orbis_id_num",
    "acq_name",
    "acq_ctry",
    "acq_busi_descr",
    "acq_bvd_id_num",
    "acq_orbis_id_num",
    "gr_acq_name",
    "gr_acq_ctry",
    "gr_acq_busi_descr",
    "gr_acq_bvd_id_num",
    "gr_acq_orbis_id_num",
    "ven_name",
    "ven_ctry",
    "ven_busi_descr",
    "ven_bvd_id_num",
    "ven_orbis_id_num",
    "gr_ven_name",
    "gr_ven_ctry",
    "gr_ven_busi_descr",
    "gr_ven_bvd_id_num",
    "gr_ven_orbis_id_num",
    "reg_body_name",
    "reg_body_ctry",
    "deal_opportunity_type",
]

keep_present = [c for c in OVW_KEEP if c in df_raw.columns]
missing_cols = [c for c in OVW_KEEP if c not in df_raw.columns]
if missing_cols:
    log(f"[WARN] Missing expected columns: {missing_cols}")

df_full_raw = df_raw[keep_present].copy()
del df_raw
log(f"Columns kept: {len(keep_present)} / {len(OVW_KEEP)}")
log(f"Full raw rows (before dedup): {len(df_full_raw):,}")

# =====================================================================
# M3 — Vendor‑name aggregation（只vendor，不碰tar/acq行）
# =====================================================================
log("\n" + "=" * 60)
log("M3 — Vendor‑name aggregation -> 01a_deal_vendor_name.csv")
log("=" * 60)
VENDOR_AGG = ["ven_name", "ven_bvd_id_num"]
_ven_frames = []
for _vc in VENDOR_AGG:
    if _vc not in df_full_raw.columns:
        log(f"  {_vc:<24s} skipped (column not in source)")
        continue
    _s = df_full_raw.dropna(subset=[_vc])
    if _s.empty:
        log(f"  {_vc:<24s} all missing")
        continue
    _g = (_s.groupby("deal_num")[_vc]
          .apply(lambda x: "|".join(sorted(set(
              str(v).strip() for v in x
              if str(v).strip() and str(v).strip().lower() != "nan"))))
          .rename(_vc + "_all").to_frame())
    _g[_vc + "_n"] = _g[_vc + "_all"].apply(
        lambda s: len([x for x in str(s).split("|") if x]) if pd.notna(s) else 0)
    _g = _g.reset_index()
    _n_multi = int((_g[_vc + "_n"] > 1).sum())
    log(f"  {_vc:<24s} {len(_g):,} deals | multi {_n_multi:,} "
        f"| mean {_g[_vc + '_n'].mean():.2f}")
    _ven_frames.append(_g)

if _ven_frames:
    _ven_agg = _ven_frames[0]
    for _f in _ven_frames[1:]:
        _ven_agg = _ven_agg.merge(_f, on="deal_num", how="outer")
    _ven_agg["deal_num"] = pd.to_numeric(
        _ven_agg["deal_num"], errors="coerce").astype("Int64")
    _out_ven = os.path.join(CLEANED, "01a_deal_vendor_name.csv")
    _ven_agg.to_csv(_out_ven, index=False, encoding="utf-8-sig")
    log(f"  Saved -> {_out_ven} ({len(_ven_agg):,} deals)")
else:
    log("  no vendor columns found — 01a_deal_vendor_name.csv NOT produced")

# =====================================================================
# Step 2: 聚合 regulatory 变量 (BEFORE dedup)
# =====================================================================
log("\n" + "=" * 60)
log("Building regulatory variables from full raw data (all reg rows reserved)")
log("=" * 60)
reg_vars = build_regulatory_variables(df_full_raw)

# =====================================================================
# Step 3: 清洗主表，【NO acquirer groupby聚合！！！】
# =====================================================================
df_ovw = df_full_raw.copy()
del df_full_raw

for _oc in ["tar_orbis_id_num", "acq_orbis_id_num"]:
    if _oc in df_ovw.columns:
        df_ovw[_oc] = normalize_orbis_id(df_ovw[_oc])

COUNTRY_ALIAS = {
    "KV": "XK", "XK": "XK", "UK": "GB", "EN": "GB",
    "SL": "SI", "BU": "BG", "ROU": "RO", "ZR": "CD", "TP": "TL",
    "YU": "RS", "CS": "RS", "AN": "CW", "FX": "FR", "DD": "DE",
}
for _cc in ["tar_ctry", "acq_ctry"]:
    if _cc in df_ovw.columns:
        _v = df_ovw[_cc].astype("object").str.strip().str.upper()
        _v = _v.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
        _v = _v.replace(COUNTRY_ALIAS)
        df_ovw[_cc] = _v
log("  [FIX] tar_ctry / acq_ctry normalised (strip + upper + alias map)")

n_before_filter = len(df_ovw)
df_ovw = df_ovw[df_ovw["tar_name"].notna()].copy()
log(f"Drop blank tar_name rows: {n_before_filter - len(df_ovw):,} removed")

# =====================================================================
# [FIX-T1_5KEY] 严格去重逻辑
# 必须按照 deal_num, tar_name, acq_name, tar_bvd_id_num, acq_bvd_id_num 唯一！
# 为防止 bvd_id 缺失导致 dropna 丢行，先统一转字符串并填充 __NA__
# =====================================================================
n_before_dedup = len(df_ovw)

# 确保这5个字段存在且转为 string，避免混合类型引发 KeyError 或 drop_duplicates 异常
for _col in ["tar_name", "acq_name", "tar_bvd_id_num", "acq_bvd_id_num"]:
    if _col in df_ovw.columns:
        df_ovw[_col] = df_ovw[_col].fillna("__NA__").astype(str).str.strip()
    else:
        df_ovw[_col] = "__NA__"

# 严格定义主键
DEDUP_KEYS = ["deal_num", "tar_name", "acq_name", "tar_bvd_id_num", "acq_bvd_id_num"]

# 按五元组严格去重，保留第一条
df_ovw = df_ovw.drop_duplicates(subset=DEDUP_KEYS, keep="first").copy()

n_after_dedup = len(df_ovw)
n_multi = int(df_ovw.groupby("deal_num").size().gt(1).sum())

log(f"  [DEDUP-T1_5key] {n_before_dedup:,} -> {n_after_dedup:,} rows "
    f"(dropped {n_before_dedup - n_after_dedup:,} pure duplicates)")
log(f"  [DEDUP-T1_5key] unique deal_num: {df_ovw['deal_num'].nunique():,} | "
    f"多标的/多收购方 deal: {n_multi:,}（保留多行，未折叠）")
log(f"  [UNIT] rows={n_after_dedup:,} | unique composite PK="
    f"{len(df_ovw.drop_duplicates(subset=DEDUP_KEYS)):,}")

# 去重完成后，将占位符还原为 NaN 供下游使用
for _col in ["tar_name", "acq_name", "tar_bvd_id_num", "acq_bvd_id_num"]:
    df_ovw.loc[df_ovw[_col] == "__NA__", _col] = np.nan

df_ovw["deal_num"] = pd.to_numeric(df_ovw["deal_num"], errors="coerce")
df_ovw = df_ovw[df_ovw["deal_num"].notna()].copy()
df_ovw["deal_num"] = df_ovw["deal_num"].astype("Int64")

if "deal_val_usd" in df_ovw.columns:
    df_ovw["deal_value_ovw"] = pd.to_numeric(df_ovw["deal_val_usd"], errors="coerce")
    df_ovw = df_ovw.drop(columns=["deal_val_usd"])
    _neg = int((df_ovw["deal_value_ovw"] <= 0).sum())
    df_ovw.loc[df_ovw["deal_value_ovw"] <= 0, "deal_value_ovw"] = np.nan
    log(f"  [FIX] deal_value_ovw <= 0 set to NaN: {_neg:,} rows")

# =====================================================================
# Step 4: 合并 regulatory 变量
# =====================================================================
if len(reg_vars) > 0:
    log(f"\nReg agg stats: total deals with reg info = {len(reg_vars):,}")
    log(f"reg_body_count min/max/mean: "
        f"{reg_vars['reg_body_count'].min()} / "
        f"{reg_vars['reg_body_count'].max()} / "
        f"{reg_vars['reg_body_count'].mean():.2f}")
    log("\nCategory count summary:")
    for cat in REG_CATEGORY_KEYWORDS.keys():
        cnt = reg_vars[f"reg_{cat}"].sum()
        log(f"  reg_{cat} = {cnt:,}")
    df_ovw = df_ovw.merge(reg_vars, on="deal_num", how="left")
    del reg_vars
    count_cols = ["reg_body_count", "reg_country_count"]
    dummy_cols = (
        [f"reg_{cat}" for cat in REG_CATEGORY_KEYWORDS]
        + ["reg_cross_national", "reg_common_law", "reg_civil_law", "reg_mixed_legal"]
    )
    for c in count_cols:
        df_ovw[c] = df_ovw[c].fillna(0).astype(int)
    for c in dummy_cols:
        df_ovw[c] = df_ovw[c].fillna(0).astype(int)
    _n_reg = int(df_ovw["regulatory_bodies"].notna().sum())
    log(f"\nRegulatory variables merge complete, real multi‑count enabled")
    log(f"  [FIX] regulatory_bodies non‑null = {_n_reg:,} / {len(df_ovw):,} "
        f"({100.0 * _n_reg / max(len(df_ovw), 1):.2f}%) — kept as NaN when absent")
else:
    log("Warning: zero regulatory records found in raw data")
    df_ovw["reg_body_count"] = 0
    df_ovw["reg_country_count"] = 0
    for cat in REG_CATEGORY_KEYWORDS.keys():
        df_ovw[f"reg_{cat}"] = 0
    for c in ["reg_cross_national", "reg_common_law", "reg_civil_law", "reg_mixed_legal"]:
        df_ovw[c] = 0
    df_ovw["regulatory_bodies"] = np.nan
    df_ovw["regulatory_countries"] = np.nan

# =====================================================================
# Step 5: Cross‑border / Cross‑country variables
# =====================================================================
log("\n" + "=" * 60)
log("STEP 5 — Cross‑border / Cross‑country variables")
log("=" * 60)
if "tar_ctry" in df_ovw.columns and "acq_ctry" in df_ovw.columns:
    tar_c = df_ovw["tar_ctry"].astype(str).str.strip().str.upper()
    acq_c = df_ovw["acq_ctry"].astype(str).str.strip().str.upper()
    mask_valid = df_ovw["tar_ctry"].notna() & df_ovw["acq_ctry"].notna()
    df_ovw["cross_border"] = np.where(mask_valid, (tar_c != acq_c).astype(int), np.nan)
    df_ovw["cross_country"] = df_ovw["cross_border"]
    n_cb = int(df_ovw["cross_border"].sum()) if df_ovw["cross_border"].notna().any() else 0
    n_dom = int((df_ovw["cross_border"] == 0).sum()) if df_ovw["cross_border"].notna().any() else 0
    log(f"  cross_border = 1: {n_cb:,} deals")
    log(f"  cross_border = 0: {n_dom:,} deals")
    log(f"  cross_border missing: {df_ovw['cross_border'].isna().sum():,} deals")
else:
    log("  [WARN] tar_ctry or acq_ctry missing, cannot compute cross_border")
    df_ovw["cross_border"] = np.nan
    df_ovw["cross_country"] = np.nan

# =====================================================================
# Step 6: Legal Origin & Classification Distance (La Porta 1998)
# =====================================================================
log("\n" + "=" * 60)
log("STEP 6 — Country‑level Legal Origin & Classification Distance")
log("=" * 60)
if "tar_ctry" in df_ovw.columns:
    df_ovw["legal_origin_tar"] = df_ovw["tar_ctry"].map(LEGAL_ORIGIN)
    df_ovw["common_law_tar"] = (df_ovw["legal_origin_tar"] == "English").astype("Int64")
    df_ovw.loc[df_ovw["legal_origin_tar"].isna(), "common_law_tar"] = pd.NA
    df_ovw["classification_distance_tar"] = df_ovw["tar_ctry"].map(CLASSIFICATION_DIST)
else:
    df_ovw["legal_origin_tar"] = np.nan
    df_ovw["common_law_tar"] = pd.NA
    df_ovw["classification_distance_tar"] = np.nan

if "acq_ctry" in df_ovw.columns:
    df_ovw["legal_origin_acq"] = df_ovw["acq_ctry"].map(LEGAL_ORIGIN)
    df_ovw["common_law_acq"] = (df_ovw["legal_origin_acq"] == "English").astype("Int64")
    df_ovw.loc[df_ovw["legal_origin_acq"].isna(), "common_law_acq"] = pd.NA
    df_ovw["classification_distance_acq"] = df_ovw["acq_ctry"].map(CLASSIFICATION_DIST)
else:
    df_ovw["legal_origin_acq"] = np.nan
    df_ovw["common_law_acq"] = pd.NA
    df_ovw["classification_distance_acq"] = np.nan

df_ovw["classification_distance"] = df_ovw["classification_distance_tar"]

unmapped_tar = df_ovw.loc[df_ovw["legal_origin_tar"].isna(), "tar_ctry"].value_counts()
unmapped_acq = df_ovw.loc[df_ovw["legal_origin_acq"].isna(), "acq_ctry"].value_counts()
if len(unmapped_tar) > 0:
    log(f"  Unmapped target countries (top 10): {unmapped_tar.head(10).to_dict()}")
if len(unmapped_acq) > 0:
    log(f"  Unmapped acquirer countries (top 10): {unmapped_acq.head(10).to_dict()}")

log(f"  common_law_tar = 1: {df_ovw['common_law_tar'].sum():,} deals")
log(f"  common_law_acq = 1: {df_ovw['common_law_acq'].sum():,} deals")

if "cross_border" in df_ovw.columns:
    cross_df = df_ovw[df_ovw["cross_border"] == 1]
    if len(cross_df) > 0:
        both_common = ((cross_df["common_law_tar"] == 1) & (cross_df["common_law_acq"] == 1)).sum()
        tar_only = ((cross_df["common_law_tar"] == 1) & (cross_df["common_law_acq"] != 1)).sum()
        acq_only = ((cross_df["common_law_tar"] != 1) & (cross_df["common_law_acq"] == 1)).sum()
        neither = ((cross_df["common_law_tar"] != 1) & (cross_df["common_law_acq"] != 1)).sum()
        log(f"\n  Cross‑border deals legal origin breakdown:")
        log(f"    Both Common: {both_common:,}")
        log(f"    Target Common Only: {tar_only:,}")
        log(f"    Acquirer Common Only: {acq_only:,}")
        log(f"    Neither Common: {neither:,}")

n_tar_dist = df_ovw["classification_distance_tar"].notna().sum()
n_acq_dist = df_ovw["classification_distance_acq"].notna().sum()
log(f"\n  classification_distance_tar: {n_tar_dist:,}/{len(df_ovw):,} obs mapped ({100*n_tar_dist/len(df_ovw):.1f}%)")
log(f"  classification_distance_acq: {n_acq_dist:,}/{len(df_ovw):,} obs mapped ({100*n_acq_dist/len(df_ovw):.1f}%)")

# =====================================================================
# Step 7: 输出，保留acq_name/acq_bvd_id_num！！！
# =====================================================================
for _src, _dst in [("tar_ctry", "tar_country_code"), ("acq_ctry", "acq_country_code")]:
    if _src in df_ovw.columns:
        df_ovw[_dst] = df_ovw[_src]
        log(f"  [FIX] created {_dst} from {_src} "
            f"(non‑null {int(df_ovw[_dst].notna().sum()):,} / {len(df_ovw):,})")
    else:
        log(f"  [FATAL‑RISK] {_src} absent -> {_dst} cannot be created")

DROP_COLS_FINAL = [
    "deal_headline", "deal_type", "deal_status", "deal_val_usd",
    "tar_ctry", "acq_ctry", "tar_busi_descr", "tar_orbis_id_num",
    "acq_busi_descr", "acq_orbis_id_num",
    "gr_acq_name", "gr_acq_ctry", "gr_acq_busi_descr", "gr_acq_bvd_id_num", "gr_acq_orbis_id_num",
    "ven_name", "ven_ctry", "ven_busi_descr", "ven_bvd_id_num", "ven_orbis_id_num",
    "gr_ven_name", "gr_ven_ctry", "gr_ven_busi_descr", "gr_ven_bvd_id_num", "gr_ven_orbis_id_num",
]
cols_to_drop = [c for c in DROP_COLS_FINAL if c in df_ovw.columns]
if cols_to_drop:
    df_ovw = df_ovw.drop(columns=cols_to_drop)
    log(f"  Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
else:
    log("  No matching columns found to drop.")

# =====================================================================
# Step 8: 诊断输出
# =====================================================================
total_rows = len(df_ovw)
log("\n" + "-" * 60)
log("ROW COUNT SUMMARY")
log("-" * 60)
log(f"  raw rows (before dedup)         : {n_raw:,}")
log(f"  rows after dedup                : {total_rows:,}")
log(f"  unique deal_num                 : {df_ovw['deal_num'].nunique():,}")

# 使用最终有效的键进行统计，不再引用已删除的辅助列
FINAL_DEDUP_KEYS = [c for c in DEDUP_KEYS if c in df_ovw.columns]
log(f"  unique composite PK({','.join(FINAL_DEDUP_KEYS)}) : "
    f"{len(df_ovw.drop_duplicates(subset=FINAL_DEDUP_KEYS)):,}")

log("\n" + "-" * 60)
log("COLUMN COVERAGE (ALL OUTPUT VARIABLES)")
log("-" * 60)
for col in df_ovw.columns:
    n_nonnull = int(df_ovw[col].notna().sum())
    n_missing = total_rows - n_nonnull
    pct_nn = n_nonnull / total_rows * 100 if total_rows else 0.0
    pct_ms = n_missing / total_rows * 100 if total_rows else 0.0
    log(f"  {col:<35s} non-null {n_nonnull:>10,} ({pct_nn:6.2f}%)  "
        f"missing {n_missing:>10,} ({pct_ms:6.2f}%)")

# ---- NUMERIC SUMMARY ----
log("\n" + "-" * 60)
log("NUMERIC SUMMARY (ALL NUMERIC VARIABLES)")
log("-" * 60)
num_cols = df_ovw.select_dtypes(include=[np.number, "Int64", "Float64"]).columns
for col in num_cols:
    s = pd.to_numeric(df_ovw[col], errors="coerce")
    n_valid = int(s.notna().sum())
    if n_valid == 0:
        log(f"  [{col}] N=0 (all missing)")
        continue
    try:
        log(f"  [{col}] N={n_valid:,} | "
            f"min={s.min():.4g} | p25={s.quantile(0.25):.4g} | "
            f"median={s.median():.4g} | p75={s.quantile(0.75):.4g} | "
            f"max={s.max():.4g} | mean={s.mean():.4g}")
    except Exception as e:
        log(f"  [{col}] N={n_valid:,} | error computing stats: {e}")

# ---- CATEGORICAL SUMMARY ----
log("\n" + "-" * 60)
log("CATEGORICAL SUMMARY (NON-NUMERIC VARIABLES)")
log("-" * 60)
cat_cols = df_ovw.select_dtypes(exclude=[np.number, "Int64", "Float64"]).columns
for col in cat_cols:
    if col not in df_ovw.columns:
        continue
    s = df_ovw[col]
    n_valid = int(s.notna().sum())
    n_unique = int(s.nunique())
    log(f"  [{col}] {n_valid:,} valid | unique values: {n_unique:,}")

# ---- TOP 20 TARGET COUNTRY ----
if "tar_country_code" in df_ovw.columns:
    log("\n" + "-" * 60)
    log("TOP20 TARGET COUNTRY (RAW overview universe)")
    log("-" * 60)
    log(df_ovw["tar_country_code"].value_counts().head(20).to_string())

# ---- REGULATORY SUBSET APPENDIX ----
if "reg_body_count" in df_ovw.columns and "tar_country_code" in df_ovw.columns:
    _sub = df_ovw[df_ovw["reg_body_count"] > 0]
    log(f"\n[Appendix] Subset with regulatory records N={len(_sub):,} rows "
        f"(unique deal {_sub['deal_num'].nunique():,})")
    log("  reg* variables are non-zero only within this subset:")
    log(_sub["tar_country_code"].value_counts().head(10).to_string())


# =====================================================================
# Save
# =====================================================================
out_path = os.path.join(CLEANED, "01_deal_overview.csv")
df_ovw.to_csv(out_path, index=False, encoding="utf-8-sig")
size_kb = os.path.getsize(out_path) / 1024
log(f"\n  [01_deal_overview.csv] saved -> {out_path} ({size_kb:.0f} KB)")
log(f"  [01a_overview.txt] log saved -> {LOG_PATH}")

_log_fh.close()
print(f"\nLog saved -> {LOG_PATH}")