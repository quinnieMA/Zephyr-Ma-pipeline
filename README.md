# Zephyr-Ma-pipeline
Lossless, audit-first pipeline that turns raw BvD Zephyr M&amp;A exports into a deal-level research database (CSV + Stata .dta). Every merge key is explicit; every dropped row is logged.

## Repository layout

```text
.
├── config.py                     # shared paths (ZEPHYR_BASE, BASE_DIR, …)
├── data/
│   ├── cleaned/                  # 01-stage outputs
│   │   ├── 01_deal_structure_date.csv             # 02
│   │   ├── 01_deal_sic_industry.csv               # 03
│   │   ├── 01_deal_multiple.csv                   # 02
│   │   ├── 01_deal_value.csv                      # 02
│   │   ├── 01_deal_overview.csv                   # 03
│   │   ├── 01b_deal_info_source_count.csv         # 02
│   │   ├── 01b_deal_vendor_type.csv               # 06a
│   │   └── 01c_comments_features.csv              # 06a
│   └── merged/                                    # 02–07 outputs
│       ├── 02_deal_master.csv                     # 01_deal_structure_date.csv|01_deal_multiple.csv|01_deal_value.csv|01b_deal_info_source_count.csv
│       ├── 03_firm_deal_level.csv                 # 01_deal_sic_industry.csv| 01_deal_overview.csv
│       ├── 03_firm_entity_pairs.csv               # multiple targets 
│       ├── 04_deal_firm_full.csv                  # 02+03
│       ├── 05_deal_firm_filtered.csv              # 04 + filter
│       ├── 06a_merge_comment_vendor.csv           # 01c_comments_features.csv|01b_deal_vendor_type.csv+05_deal_firm_filtered.csv
│       ├── 06b_text_peer.csv                      # 01_deal_overview.csv+05_deal_firm_filtered.csv
│       ├── 07_full_sample.csv                     # 06a+06b
│       └── 07_full_sample.dta
├── scripts/
│   ├── 01a_clean_structure_date.py
│   ├── 01a_clean_industry.py
│   ├── 01a_clean_multiple.py
│   ├── 01a_clean_value.py
│   ├── 01a_clean_overview.py
│   ├── 01b_clean_vendor_type.py
│   ├── 01b_clean_info_source.py
│   ├── 01c_comments_features.py
│   ├── 02_merge_deal_master.py
│   ├── 03_clean_firm_modules.py
│   ├── 04_merge_firm_to_deal.py
│   ├── 05_sample_filter.py
│   ├── 06a_merge_comment_vendor.py
│   ├── 06b_text_peer_multiple.py
│   └── 07_full_sample.py
└── logs/                         # per-stage *.txt diagnostics
```

## Requirements

```text
pandas          >= 2.0
numpy           >= 1.24
pyreadstat      >= 1.2      # write Stata .dta (stage 07)
gensim          >= 4.3      # Doc2Vec text embeddings (stage 06b)
tqdm            >= 4.65     # progress bars (stage 06b)
stanfordcorenlp >= 3.9      # optional — vendor NER (stage 01b)
```
