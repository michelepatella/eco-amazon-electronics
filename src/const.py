"""src/const.py

Project constants.

Centralize all constant values used across the project.
"""

# ============================================================================
# DIRECTORIES AND PATHS
# ============================================================================
# Abbreviations:
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# DIR = Directory
# AR23 = Amazon Reviews'23
# GT = Ground Truths
# UR = User Reviews
# IM = Item Metadata
# G25F = Gemini 2.5 Flash
# O3M = OpenAI O3 Mini
# ELEC = Electronics
# UMAP = User Map
# IMAP = Item Map
# PREDS = Predictions
# FULL = Full dataset
# TRAIN = Training set
# TEST = Test set
# VALID = Validation set
# SUS = Sustainability-first
# BAL = Balanced relevance-sustainability trade-off
# REL = Relevance-first
# PURE = Pure relevance, no sustainability
# CONFIG = Configuration
# ============================================================================

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
    f"{DATA_PROCESSED_AR23_IM_DIR}/gemini_2_5_flash"
)
DATA_PROCESSED_AR23_IM_O3M_DIR = f"{DATA_PROCESSED_AR23_IM_DIR}/openai_o3_mini"
DATA_PROCESSED_AR23_UR_DIR = f"{DATA_PROCESSED_AR23_DIR}/user_reviews"
DATA_PROCESSED_AR23_UR_ELEC_DIR = f"{DATA_PROCESSED_AR23_UR_DIR}/electronics"
DATA_PROCESSED_GT_DIR = f"{DATA_PROCESSED_DIR}/ground_truths"
DATA_PROCESSED_GT_G25F_DIR = f"{DATA_PROCESSED_GT_DIR}/gemini_2_5_flash"
DATA_PROCESSED_GT_O3M_DIR = f"{DATA_PROCESSED_GT_DIR}/openai_o3_mini"

# Data raw paths
DATA_RAW_AR23_IM_ELEC_PATH = f"{DATA_RAW_AR23_IM_DIR}/electronics.jsonl"
DATA_RAW_AR23_UR_ELEC_PATH = f"{DATA_RAW_AR23_UR_DIR}/electronics.jsonl"
DATA_RAW_GT_ELEC_PATH = f"{DATA_RAW_GT_DIR}/electronics.jsonl"

# Data interim paths
DATA_INTERIM_MAPS_UMAP_PATH = f"{DATA_INTERIM_MAPS_DIR}/user_map.tsv"
DATA_INTERIM_MAPS_IMAP_PATH = f"{DATA_INTERIM_MAPS_DIR}/item_map.tsv"

# Data processed paths
DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH = (
    f"{DATA_PROCESSED_AR23_IM_G25F_DIR}/electronics.jsonl"
)
DATA_PROCESSED_AR23_IM_O3M_ELEC_PATH = (
    f"{DATA_PROCESSED_AR23_IM_O3M_DIR}/electronics.jsonl"
)
DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/electronics.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/electronics.train.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/electronics.test.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/electronics.valid.inter"
)
DATA_PROCESSED_GT_G25F_ELEC_PATH = (
    f"{DATA_PROCESSED_GT_G25F_DIR}/electronics.jsonl"
)
DATA_PROCESSED_GT_O3M_ELEC_PATH = (
    f"{DATA_PROCESSED_GT_O3M_DIR}/electronics.jsonl"
)

# Models directory
MODELS_DIR = "models"

# Models BPR directories
MODELS_BPR_DIR = f"{MODELS_DIR}/BPR"
MODELS_BPR_PREDS_DIR = f"{MODELS_BPR_DIR}/predictions"
MODELS_BPR_PREDS_G25F_DIR = f"{MODELS_BPR_PREDS_DIR}/gemini_2_5_flash"
MODELS_BPR_PREDS_O3M_DIR = f"{MODELS_BPR_PREDS_DIR}/openai_o3_mini"

# Models LightGCN directories
MODELS_LIGHTGCN_DIR = f"{MODELS_DIR}/LightGCN"
MODELS_LIGHTGCN_PREDS_DIR = f"{MODELS_LIGHTGCN_DIR}/predictions"
MODELS_LIGHTGCN_PREDS_G25F_DIR = (
    f"{MODELS_LIGHTGCN_PREDS_DIR}/gemini_2_5_flash"
)
MODELS_LIGHTGCN_PREDS_O3M_DIR = f"{MODELS_LIGHTGCN_PREDS_DIR}/openai_o3_mini"

# Models BPR paths
MODELS_BPR_PATH = f"{MODELS_BPR_DIR}/BPR.pth"
MODELS_BPR_PREDS_G25F_SUS_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_0_25.pth"
MODELS_BPR_PREDS_G25F_BAL_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_0_5.pth"
MODELS_BPR_PREDS_G25F_REL_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_0_75.pth"
MODELS_BPR_PREDS_G25F_PURE_PATH = f"{MODELS_BPR_PREDS_G25F_DIR}/alpha_1_0.pth"
MODELS_BPR_PREDS_O3M_SUS_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_0_25.pth"
MODELS_BPR_PREDS_O3M_BAL_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_0_5.pth"
MODELS_BPR_PREDS_O3M_REL_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_0_75.pth"
MODELS_BPR_PREDS_O3M_PURE_PATH = f"{MODELS_BPR_PREDS_O3M_DIR}/alpha_1_0.pth"

# Models LightGCN paths
MODELS_LIGHTGCN_PATH = f"{MODELS_LIGHTGCN_DIR}/LightGCN.pth"
MODELS_LIGHTGCN_PREDS_G25F_SUS_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_0_25.pth"
)
MODELS_LIGHTGCN_PREDS_G25F_BAL_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_0_5.pth"
)
MODELS_LIGHTGCN_PREDS_G25F_REL_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_0_75.pth"
)
MODELS_LIGHTGCN_PREDS_G25F_PURE_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_G25F_DIR}/alpha_1_0.pth"
)
MODELS_LIGHTGCN_PREDS_O3M_SUS_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_0_25.pth"
)
MODELS_LIGHTGCN_PREDS_O3M_BAL_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_0_5.pth"
)
MODELS_LIGHTGCN_PREDS_O3M_REL_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_0_75.pth"
)
MODELS_LIGHTGCN_PREDS_O3M_PURE_PATH = (
    f"{MODELS_LIGHTGCN_PREDS_O3M_DIR}/alpha_1_0.pth"
)

# Config directory
CONFIG_DIR = "src/config"

# Config paths
CONFIG_PATH = f"{CONFIG_DIR}/config.yaml"

# ============================================================================
# DATASETS AND MAPS
# ============================================================================
# Abbreviations:
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ELEC = Electronics
# UR = User Reviews
# COL = Column
# UMAP = User Map
# IMAP = Item Map
# ============================================================================

# Dataset names
DATASET_NAME_ELEC = "elec"

# Supported datasets
SUPPORTED_DATASETS = [DATASET_NAME_ELEC]

# Raw user reviews
RAW_UR_USER_ID_COL = "user_id"
RAW_UR_ASIN_COL = "asin"
RAW_UR_PARENT_ASIN_COL = "parent_asin"
RAW_UR_TIMESTAMP_COL = "timestamp"
RAW_UR_RATING_COL = "rating"

# Processed user reviews
PROCESSED_UR_USER_ID_COL = "user_id:token"
PROCESSED_UR_ITEM_ID_COL = "item_id:token"
PROCESSED_UR_RATING_COL = "rating:float"

# Maps
MAPS_UMAP_USER_ID_COL = "user_id"
MAPS_UMAP_USER_INDEX_COL = "user_index"
MAPS_IMAP_ITEM_ID_COL = "item_id"
MAPS_IMAP_ITEM_INDEX_COL = "item_index"
MAPS_IMAP_PARENT_ASIN_COL = "parent_asin"
