# `data/`

```text
.
├── interim
│   └── maps
│       ├── item_map.tsv  <- (item_id, item_index, parent_asin) mapping
│       └── user_map.tsv  <- (user_id, user_index) mapping
├── processed
│   ├── amazon_reviews_23
│   │   ├── item_metadata
│   │   │   ├── gemini_2_5_flash
│   │   │   │   └── electronics.jsonl  <- Item metadata enriched with PCF via Gemini 2.5 Flash
│   │   │   └── openai_o3_mini
│   │   │       └── electronics.jsonl  <- Item metadata enriched with PCF via OpenAI o3-mini
│   │   └── user_reviews
│   │       └── electronics
│   │           ├── electronics.inter  <- Binarized user-item interactions
│   │           ├── electronics.test.inter  <- Binarized user-item interactions (test set, 10%)
│   │           ├── electronics.train.inter  <- Binarized user-item interactions (train set, 80%)
│   │           └── electronics.valid.inter  <- Binarized user-item interactions (valid set, 10%)
│   └── ground_truths
│       ├── gemini_2_5_flash
│       │   └── electronics.jsonl  <- Ground truth items with PCF from Gemini 2.5 Flash
│       └── openai_o3_mini
│           └── electronics.jsonl  <- Ground truth items with PCF from OpenAI o3-mini
├── raw
│   ├── amazon_reviews_23
│   │   ├── item_metadata
│   │   │   └── electronics.jsonl  <- Item metadata (15-core filtered)
│   │   └── user_reviews
│   │       └── electronics.jsonl  <- User-item interactions (15-core filtered) 
│   └── ground_truths
│       └── electronics.jsonl  <- Ground truth for 'Electronics' items 
└── README.md  <- This file
```

**Note**: The repository doesn't include data files due to their size.