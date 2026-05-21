# `data/`

Directory for all the data organized by processing stage (raw → interim → processed).

**Note**: This project uses DVC to manage data files. Make sure `dvc` is installed, then pull all data artifacts from the remote bucket:
```bash
dvc pull
```

## `raw/`

| Name | Format | Source | Size | Rows | Schema | Description |
|------|--------|--------|------|------|--------|-------------|
| `amazon_reviews_23/item_metadata/electronics.jsonl` | JSONL | Amazon Reviews'23 | 48.2 MB | 9,710 | `main_category`, `title`, `average_rating`, `rating_number`, `features`, `description`, `price`, `images`, `videos`, `store`, `categories`, `details`, `parent_asin`, `bought_together` | Item metadata (15-core filtered) |
| `amazon_reviews_23/user_reviews/electronics.jsonl` | JSONL | Amazon Reviews'23 | 262.4 MB | 464,464 | `rating`, `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, `verified_purchase` | User reviews (15-core filtered) |
| `ground_truths/electronics.jsonl` | JSONL | Manually collected | 41 KB | 200 | `title`, `co2e_kg` | Items with known PCF values |

## `interim/`

| Name | Format | Source | Size | Rows | Schema | Description |
|------|--------|--------|------|------|--------|-------------|
| `maps/user_map.tsv` | TSV | Generated | 750 KB | 21,751 | `user_id`, `user_index` | Amazon user ID → numeric index mapping |
| `maps/item_map.tsv` | TSV | Generated | 311 KB | 11,495 | `item_id`, `item_index`, `parent_asin` | ASIN → numeric index + parent ASIN mapping |

## `processed/`

| Name | Format | Source | Size | Rows | Schema | Description |
|------|--------|--------|------|------|--------|-------------|
| `amazon_reviews_23/item_metadata/google_genai:gemini-2.5-flash/electronics.jsonl` | JSONL | Gemini 2.5 Flash enrichment | - | 9,710 | `main_category`, `title`, `average_rating`, `rating_number`, `features`, `description`, `price`, `images`, `videos`, `store`, `categories`, `details`, `parent_asin`, `bought_together`, `co2e_kg` | Item metadata enriched with PCF estimates via Gemini 2.5 Flash |
| `amazon_reviews_23/item_metadata/openai:o3-mini/electronics.jsonl` | JSONL | OpenAI o3-mini enrichment | - | 9,710 | `main_category`, `title`, `average_rating`, `rating_number`, `features`, `description`, `price`, `images`, `videos`, `store`, `categories`, `details`, `parent_asin`, `bought_together`, `co2e_kg` | Item metadata enriched with PCF estimates via OpenAI o3-mini |
| `amazon_reviews_23/user_reviews/electronics/electronics.inter` | TSV | Generated | 12.3 MB | 464,457 | `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float` | All user-item interactions binarized in RecBole format |
| `amazon_reviews_23/user_reviews/electronics/electronics.train.inter` | TSV | Generated | 9.6 MB | 363,022 | `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float` | Training split in RecBole format |
| `amazon_reviews_23/user_reviews/electronics/electronics.valid.inter` | TSV | Generated | 936 KB | 35,117 | `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float` | Validation split in RecBole format |
| `amazon_reviews_23/user_reviews/electronics/electronics.test.inter` | TSV | Generated | 1.8 MB | 65,862 | `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float` | Test split in RecBole format |
| `ground_truths/google_genai:gemini-2.5-flash/electronics.jsonl` | JSONL | Gemini 2.5 Flash enrichment | 305 KB | 200 | `title`, `co2e_kg` | Items with known PCF values enriched with Gemini 2.5 Flash estimates |
| `ground_truths/openai:o3-mini/electronics.jsonl` | JSONL | OpenAI o3-mini enrichment | 234 KB | 200 | `title`, `co2e_kg` | Items with known PCF values enriched with OpenAI o3-mini estimates |
