import pandas as pd
from tqdm import tqdm

from src.const import (
    DATA_INTERIM_MAPS_IMAP_PATH,
    DATA_INTERIM_MAPS_UMAP_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH,
    DATA_RAW_AR23_UR_ELEC_PATH,
    CONFIG_PATH,
    DATASET_NAME_ELEC,
    MAPS_IMAP_ITEM_ID_COL,
    MAPS_IMAP_ITEM_INDEX_COL,
    MAPS_IMAP_PARENT_ASIN_COL,
    MAPS_UMAP_USER_ID_COL,
    MAPS_UMAP_USER_INDEX_COL,
    PROCESSED_UR_ITEM_ID_COL,
    PROCESSED_UR_RATING_COL,
    PROCESSED_UR_USER_ID_COL,
    RAW_UR_ASIN_COL,
    RAW_UR_PARENT_ASIN_COL,
    RAW_UR_RATING_COL,
    RAW_UR_TIMESTAMP_COL,
    RAW_UR_USER_ID_COL,
)
from src.utils import load_config


# Load configuration
config = load_config(CONFIG_PATH)

# Determine paths based on dataset name
if config.data.name == DATASET_NAME_ELEC:
    raw_user_reviews_path = DATA_RAW_AR23_UR_ELEC_PATH
    processed_user_reviews_path = DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH
    processed_user_reviews_train_path = DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH
    processed_user_reviews_valid_path = DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH
    processed_user_reviews_test_path = DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH


def _build_user_item_maps(
    user_reviews: pd.DataFrame,
) -> tuple[dict, list[tuple]]:
    """
    Build user and item maps from user reviews and save them.

    This function creates and saves two maps:
    - A map from user IDs to integer indices.
    - A map from item IDs (ASIN) to integer indices, enriched with parent
      ASIN information.
    Both maps are built from the provided user reviews dataset.

    Args:
        user_reviews (pd.DataFrame):
            User reviews dataset with at least the following columns:
            - "user_id": Unique identifier for users.
            - "asin": Item identifier.
            - "parent_asin": Parent item identifier.

    Returns:
        tuple[dict, list[tuple]]:
            A tuple containing:
            - map_users (dict): (User ID, User Index) map.
            - map_items (list[tuple]): List of (Item ID, Item Index, Parent ASIN) tuples.
    """
    # Create (User ID, User Index) map
    map_users = {
        user_id: i
        for i, user_id in enumerate(user_reviews[RAW_UR_USER_ID_COL].unique())
    }

    # Create (Item ID, Item Index) map
    map_items = {
        item_id: i
        for i, item_id in enumerate(user_reviews[RAW_UR_ASIN_COL].unique())
    }

    # Create (ASIN, Parent ASIN) map
    asin_to_parent = (
        user_reviews.drop_duplicates(RAW_UR_ASIN_COL)
        .set_index(RAW_UR_ASIN_COL)[RAW_UR_PARENT_ASIN_COL]
        .to_dict()
    )

    # Create (Item ID, Item Index, Parent ASIN) map,
    # overriding (Item ID, Item Index) map
    map_items = [
        (item_id, item_index, asin_to_parent.get(item_id, item_id))
        for item_id, item_index in map_items.items()
    ]

    # Save maps
    pd.DataFrame(
        list(map_users.items()),
        columns=[MAPS_UMAP_USER_ID_COL, MAPS_UMAP_USER_INDEX_COL],
    ).to_csv(DATA_INTERIM_MAPS_UMAP_PATH, sep="\t", index=False)

    pd.DataFrame(
        map_items,
        columns=[
            MAPS_IMAP_ITEM_ID_COL,
            MAPS_IMAP_ITEM_INDEX_COL,
            MAPS_IMAP_PARENT_ASIN_COL,
        ],
    ).to_csv(DATA_INTERIM_MAPS_IMAP_PATH, sep="\t", index=False)

    return map_users, map_items


def _binarize_user_reviews(
    user_reviews: pd.DataFrame, map_users: dict, map_items: list[tuple]
) -> pd.DataFrame:
    """Binarize ratings and save user-item interactions in RecBole format.

    This function transforms the raw user reviews into a binarized interaction
    matrix suitable for recommendation systems, applying three key transformations:
    1. Maps original user IDs and item ASINs to sequential indices.
    2. Binarizes ratings so that those equal to the positive threshold become 1
       (positive) all others become 0 (negative), enabling implicit feedback scenarios.
    3. Shuffles and saves in RecBole format.

    Args:
        user_reviews (pd.DataFrame):
            Raw user reviews dataset containing at least the following columns:
            - "user_id": Unique identifier for users.
            - "asin": Item identifier.
            - "rating": Numerical rating values to binarize.

        map_users (dict):
            (User ID, User Index) map.

        map_items (list[tuple]):
            List of (Item ID, Item Index, Parent ASIN) tuples.

    Returns:
        pd.DataFrame:
            The binarized user reviews dataset.
    """
    # Binarize ratings and save user-item interactions in RecBole format
    binarized_user_reviews = (
        pd.DataFrame(
            {
                PROCESSED_UR_USER_ID_COL: user_reviews[RAW_UR_USER_ID_COL].map(
                    map_users
                ),
                PROCESSED_UR_ITEM_ID_COL: user_reviews[RAW_UR_ASIN_COL].map(
                    {
                        item_id: item_index
                        for item_id, item_index, _ in map_items
                    }
                ),
                PROCESSED_UR_RATING_COL: (
                    user_reviews[RAW_UR_RATING_COL]
                    == config.data.preprocessing.binarization.rating.threshold
                ).astype(int),
            }
        )
        .sample(frac=1, random_state=config.seed)
        .reset_index(drop=True)
    )

    binarized_user_reviews.to_csv(
        processed_user_reviews_path, sep="\t", index=False
    )

    return binarized_user_reviews


def _split_user_reviews(user_reviews: pd.DataFrame) -> None:
    """Split user reviews into train, validation, and test sets and save them.

    This function splits the user reviews dataset into three subsets:
    - Training set: Used for model training.
    - Validation set: Used for hyperparameter tuning and model selection.
    - Test set: Used for final evaluation of the model's performance.
    The resulting subsets are saved in RecBole format.

    Args:
        user_reviews (pd.DataFrame):
            The user reviews dataset to split.

    Returns:
        None
    """
    # Calculate train and validation sizes
    train_size = int(config.data.preprocessing.split.train_ratio * len(user_reviews))
    valid_size = int(config.data.preprocessing.split.valid_ratio * len(user_reviews))

    # Split the dataset into train, validation, and test sets and
    # save them in RecBole format
    user_reviews.iloc[:train_size].to_csv(
        processed_user_reviews_train_path, sep="\t", index=False
    )
    user_reviews.iloc[train_size : train_size + valid_size].to_csv(
        processed_user_reviews_valid_path, sep="\t", index=False
    )
    user_reviews.iloc[train_size + valid_size :].to_csv(
        processed_user_reviews_test_path, sep="\t", index=False
    )


def preprocess_data() -> None:
    """Execute the complete preprocessing pipeline for data.

    This function orchestrates the entire data preprocessing workflow, which
    consists of the following stages:
    1. Data Loading: Load raw user reviews sorted by user and time.
    2. Mapping: Build user and item ID to index mappings
    3. Binarization: Convert ratings to implicit feedback (0 or 1) and shuffle
    4. Splitting: Partition interactions into train/validation/test sets
    All outputs are persisted to disk.

    Returns:
        None
    """
    steps = tqdm(total=3)

    # Load raw user reviews and sort chronologically by user and timestamp
    # to maintain temporal order of interactions for each user
    steps.set_description("Loading raw user reviews")
    user_reviews = pd.read_json(raw_user_reviews_path, lines=True).sort_values(
        by=[RAW_UR_USER_ID_COL, RAW_UR_TIMESTAMP_COL]
    )
    steps.update(1)

    # Create and persist user/item ID to index mappings
    steps.set_description("Building user/item maps")
    map_users, map_items = _build_user_item_maps(user_reviews)
    steps.update(1)

    # Binarize ratings using positive threshold, creating implicit feedback
    # matrix suitable for recommendation models, then shuffle for robustness
    steps.set_description("Binarizing user-item interactions")
    binarized_user_reviews = _binarize_user_reviews(
        user_reviews, map_users, map_items
    )
    steps.update(1)

    # Split binarized interactions into train/valid/test sets and persist
    # each split separately for downstream model training and evaluation
    steps.set_description("Splitting user reviews into train/valid/test")
    _split_user_reviews(binarized_user_reviews)
    steps.set_description("Data preprocessing completed")
    steps.close()


if __name__ == "__main__":
    preprocess_data()
