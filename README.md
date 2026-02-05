# Project Structure

```
.
├── 1_pcf
│   ├── metadata
│   │   ├── full                                # Metadata of 11,496 Amazon Elec products (2nd experiment)
│   │   └── subset                              # Metadata of 63 electronic products (1st experiment)
│   ├── pcf_batch.py                            # Script to extract products' kg/CO2e via LLM batch inference
│   ├── pcf_real_time.py                        # Script to extract products' kg/CO2e via LLM real-time inference
│   └── results
│       ├── full                                # Metadata of 11,496 Amazon Elec products + kg/CO2e from LLMs (2nd experiment)
│       └── subset                              # PDF reports of products' kg/CO2e estimation by LLMs (1st experiment)
├── 2_recbole
│   ├── dataset                                 # Amazon Elec dataset and its splits (train-valid-test)
│   ├── models                                  # The best, trained RecSys
│   ├── process_data
│   │   ├── electronics_CORE_15.jsonl           # K-core filtering (with k=15) on Amazon Elec dataset
│   │   ├── maps
│   │   │   ├── item_map.tsv                    # Mapping of products: internal index ↔ product ID ↔ parent ASIN
│   │   │   └── user_map.tsv                    # Mapping of users: internal index ↔ user ID
│   │   └── process.py                          # Script to preprocess Amazon Elec dataset for RecBole
│   └── train_recsys.py                         # Script for HPO and RecSys training 
├── 3_reranking
│   ├── dataset                                 # Amazon Elec dataset and its splits (train-valid-test)
│   ├── rerank_recs.py                          # Script to generate standard and PCF-aware top-k (k=100) RecSys recommendations
│   └── results                                 # Saved top-k (k=100) RecSys recommendations for all models, LLMs, and alphas
├── 4_eval
│   ├── dataset                                 # Amazon Elec dataset and its splits (train-valid-test)
│   ├── eval.py                                 # Script to evaluate RecSys models and PCF-aware recommendations
│   └── results                                 # Evaluation results CSVs for all RecSys models, LLMs, alphas, and top-k settings
├── README.md                                   # Project structure overview
└── utils.py                                    # Utility functions
```