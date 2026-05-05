"""src/const.py

Project constants.

Centralize all constant values used across the project.
"""

# ============================================================================
# ABBREVIATIONS
# ============================================================================
# AR23 = Amazon Reviews'23
# ASIN = Amazon Standard Identification Number
# BAL = Balanced relevance-sustainability trade-off
# BPR = Bayesian Personalized Ranking
# COL = Column
# CONFIG = Configuration
# DEDUP = Deduplication
# DIR = Directory
# ELEC = Electronics
# FULL = Full dataset
# G25F = Gemini 2.5 Flash
# GT = Ground Truths
# ID = Identifier
# IM = Item Metadata
# IMAP = Item Map
# LLM = Large Language Model
# LIGHTGCN = Light Graph Convolutional Network
# MAX = Maximum
# MIN = Minimum
# O3M = OpenAI O3 Mini
# PREDS = Predictions
# PURE = Pure relevance, no sustainability
# REL = Relevance-first
# SRC = Source
# SUS = Sustainability-first
# TEST = Test set
# TOL = Tolerance
# TRAIN = Training set
# UMAP = User Map
# UR = User Reviews
# VALID = Validation set
# ============================================================================

# ============================================================================
# DATASETS
# ============================================================================

# Dataset names
DATASET_NAME_ELEC = "electronics"

# Supported datasets
SUPPORTED_DATASETS = [DATASET_NAME_ELEC]

# Dataset raw user reviews columns
DATASET_RAW_UR_USER_ID_COL = "user_id"
DATASET_RAW_UR_ASIN_COL = "asin"
DATASET_RAW_UR_PARENT_ASIN_COL = "parent_asin"
DATASET_RAW_UR_TIMESTAMP_COL = "timestamp"
DATASET_RAW_UR_RATING_COL = "rating"

# Dataset processed user reviews columns
DATASET_PROCESSED_UR_USER_ID_COL = "user_id:token"
DATASET_PROCESSED_UR_ITEM_ID_COL = "item_id:token"
DATASET_PROCESSED_UR_RATING_COL = "rating:float"
DATASET_PROCESSED_UR_TIMESTAMP_COL = "timestamp:float"

# ============================================================================
# MAPS
# ============================================================================

# User map columns
MAP_UMAP_USER_ID_COL = "user_id"
MAP_UMAP_USER_INDEX_COL = "user_index"

# Item map columns
MAP_IMAP_ITEM_ID_COL = "item_id"
MAP_IMAP_ITEM_INDEX_COL = "item_index"
MAP_IMAP_PARENT_ASIN_COL = "parent_asin"

# ============================================================================
# MODELS AND LLMS
# ============================================================================

# Model names
MODEL_NAME_BPR = "BPR"
MODEL_NAME_LIGHTGCN = "LightGCN"

# Supported models
SUPPORTED_MODELS = [MODEL_NAME_BPR, MODEL_NAME_LIGHTGCN]

# LLM names
LLM_NAME_G25F = "gemini_2_5_flash"
LLM_NAME_O3M = "openai_o3_mini"

# Supported LLMs
SUPPORTED_LLMS = [LLM_NAME_G25F, LLM_NAME_O3M]

# ============================================================================
# RATINGS
# ============================================================================

# Rating boundaries
RATING_MIN_VALUE = 1
RATING_MAX_VALUE = 5

# ============================================================================
# SPLIT RATIOS
# ============================================================================

# Split boundaries and tolerance
SPLIT_RATIO_MIN_VALUE = 0.0
SPLIT_RATIO_MAX_VALUE = 1.0
SPLIT_RATIO_SUM_TOL = 1e-6

# ============================================================================
# DEDUPLICATION KEEP STRATEGIES
# ============================================================================

# Deduplication configuration
DEDUP_KEEP_STRATEGY_FIRST = "first"
DEDUP_KEEP_STRATEGY_LAST = "last"
DEDUP_KEEP_STRATEGIES = (
    DEDUP_KEEP_STRATEGY_FIRST,
    DEDUP_KEEP_STRATEGY_LAST,
)

# ============================================================================
# RE-RANKING ALPHAS
# ============================================================================

# Re-ranking alpha values
RERANKING_ALPHA_SUS = 0.25
RERANKING_ALPHA_BAL = 0.5
RERANKING_ALPHA_REL = 0.75
RERANKING_ALPHA_PURE = 1.0

# Supported re-ranking alpha values
SUPPORTED_RERANKING_ALPHAS = [
    RERANKING_ALPHA_SUS,
    RERANKING_ALPHA_BAL,
    RERANKING_ALPHA_REL,
    RERANKING_ALPHA_PURE,
]

# ============================================================================
# DIRECTORIES AND PATHS
# ============================================================================

# Source directory
SRC_DIR = "src"

# Data directory
DATA_DIR = "data"

# Data raw directories
DATA_RAW_DIR = f"{DATA_DIR}/raw"
DATA_RAW_AR23_DIR = f"{DATA_RAW_DIR}/amazon_reviews_23"
DATA_RAW_AR23_IM_DIR = f"{DATA_RAW_AR23_DIR}/item_metadata"
DATA_RAW_AR23_UR_DIR = f"{DATA_RAW_AR23_DIR}/user_reviews"
DATA_RAW_GT_DIR = f"{DATA_RAW_DIR}/ground_truths"

# Data interim directories
DATA_INTERIM_DIR = f"{DATA_DIR}/interim"
DATA_INTERIM_MAPS_DIR = f"{DATA_INTERIM_DIR}/maps"

# Data processed directories
DATA_PROCESSED_DIR = f"{DATA_DIR}/processed"
DATA_PROCESSED_AR23_DIR = f"{DATA_PROCESSED_DIR}/amazon_reviews_23"
DATA_PROCESSED_AR23_IM_DIR = f"{DATA_PROCESSED_AR23_DIR}/item_metadata"
DATA_PROCESSED_AR23_IM_G25F_DIR = (
    f"{DATA_PROCESSED_AR23_IM_DIR}/{LLM_NAME_G25F}"
)
DATA_PROCESSED_AR23_IM_O3M_DIR = f"{DATA_PROCESSED_AR23_IM_DIR}/{LLM_NAME_O3M}"
DATA_PROCESSED_AR23_UR_DIR = f"{DATA_PROCESSED_AR23_DIR}/user_reviews"
DATA_PROCESSED_AR23_UR_ELEC_DIR = (
    f"{DATA_PROCESSED_AR23_UR_DIR}/{DATASET_NAME_ELEC}"
)
DATA_PROCESSED_GT_DIR = f"{DATA_PROCESSED_DIR}/ground_truths"
DATA_PROCESSED_GT_G25F_DIR = f"{DATA_PROCESSED_GT_DIR}/{LLM_NAME_G25F}"
DATA_PROCESSED_GT_O3M_DIR = f"{DATA_PROCESSED_GT_DIR}/{LLM_NAME_O3M}"

# Data raw paths
DATA_RAW_AR23_IM_ELEC_PATH = (
    f"{DATA_RAW_AR23_IM_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_RAW_AR23_UR_ELEC_PATH = (
    f"{DATA_RAW_AR23_UR_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_RAW_GT_ELEC_PATH = f"{DATA_RAW_GT_DIR}/{DATASET_NAME_ELEC}.jsonl"

# Data interim paths
DATA_INTERIM_MAPS_UMAP_PATH = f"{DATA_INTERIM_MAPS_DIR}/user_map.tsv"
DATA_INTERIM_MAPS_IMAP_PATH = f"{DATA_INTERIM_MAPS_DIR}/item_map.tsv"

# Data processed paths
DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH = (
    f"{DATA_PROCESSED_AR23_IM_G25F_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_PROCESSED_AR23_IM_O3M_ELEC_PATH = (
    f"{DATA_PROCESSED_AR23_IM_O3M_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.train.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.test.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.valid.inter"
)
DATA_PROCESSED_GT_G25F_ELEC_PATH = (
    f"{DATA_PROCESSED_GT_G25F_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_PROCESSED_GT_O3M_ELEC_PATH = (
    f"{DATA_PROCESSED_GT_O3M_DIR}/{DATASET_NAME_ELEC}.jsonl"
)

# Models directory
MODELS_DIR = "models"

# Models BPR directories
MODELS_BPR_DIR = f"{MODELS_DIR}/{MODEL_NAME_BPR}"
MODELS_BPR_PREDS_DIR = f"{MODELS_BPR_DIR}/predictions"
MODELS_BPR_PREDS_G25F_DIR = f"{MODELS_BPR_PREDS_DIR}/{LLM_NAME_G25F}"
MODELS_BPR_PREDS_O3M_DIR = f"{MODELS_BPR_PREDS_DIR}/{LLM_NAME_O3M}"

# Models LightGCN directories
MODELS_LIGHTGCN_DIR = f"{MODELS_DIR}/{MODEL_NAME_LIGHTGCN}"
MODELS_LIGHTGCN_PREDS_DIR = f"{MODELS_LIGHTGCN_DIR}/predictions"
MODELS_LIGHTGCN_PREDS_G25F_DIR = f"{MODELS_LIGHTGCN_PREDS_DIR}/{LLM_NAME_G25F}"
MODELS_LIGHTGCN_PREDS_O3M_DIR = f"{MODELS_LIGHTGCN_PREDS_DIR}/{LLM_NAME_O3M}"

# Models BPR paths
MODELS_BPR_PATH = f"{MODELS_BPR_DIR}/{MODEL_NAME_BPR}.pth"
MODELS_BPR_PREDS_G25F_SUS_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_SUS).replace('.', '_')}.pth"
MODELS_BPR_PREDS_G25F_BAL_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_BAL).replace('.', '_')}.pth"
MODELS_BPR_PREDS_G25F_REL_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_REL).replace('.', '_')}.pth"
MODELS_BPR_PREDS_G25F_PURE_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_PURE).replace('.', '_')}.pth"
MODELS_BPR_PREDS_O3M_SUS_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_SUS).replace('.', '_')}.pth"
MODELS_BPR_PREDS_O3M_BAL_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_BAL).replace('.', '_')}.pth"
MODELS_BPR_PREDS_O3M_REL_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_REL).replace('.', '_')}.pth"
MODELS_BPR_PREDS_O3M_PURE_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_PURE).replace('.', '_')}.pth"

# Models LightGCN paths
MODELS_LIGHTGCN_PATH = f"{MODELS_LIGHTGCN_DIR}/{MODEL_NAME_LIGHTGCN}.pth"
MODELS_LIGHTGCN_PREDS_G25F_SUS_PATH = f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_SUS).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_G25F_BAL_PATH = f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_BAL).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_G25F_REL_PATH = f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_REL).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_G25F_PURE_PATH = f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_PURE).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_O3M_SUS_PATH = f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_SUS).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_O3M_BAL_PATH = f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_BAL).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_O3M_REL_PATH = f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_REL).replace('.', '_')}.pth"
MODELS_LIGHTGCN_PREDS_O3M_PURE_PATH = f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_{str(RERANKING_ALPHA_PURE).replace('.', '_')}.pth"

# Config directory
CONFIG_DIR = f"{SRC_DIR}/config"

# Config paths
CONFIG_PATH = f"{CONFIG_DIR}/config.yaml"
