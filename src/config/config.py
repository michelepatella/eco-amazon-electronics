"""src/config/config.py

Configuration module.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

from src.const import (
    DEDUP_KEEP_STRATEGIES,
    RATING_MAX_VALUE,
    RATING_MIN_VALUE,
    SPLIT_RATIO_MAX_VALUE,
    SPLIT_RATIO_MIN_VALUE,
    SPLIT_RATIO_SUM_TOL,
    SUPPORTED_DATASETS,
)


class SplitConfig(BaseModel):
    """Configuration for dataset split ratios.

    Defines the proportions for splitting a dataset into training, validation,
    and test sets.

    Attributes:
        train_ratio (float):
            Proportion of data for training.
        valid_ratio (float):
            Proportion of data for validation.
        test_ratio (float):
            Proportion of data for testing.
    """

    train_ratio: float = Field(
        ge=SPLIT_RATIO_MIN_VALUE,
        le=SPLIT_RATIO_MAX_VALUE,
    )
    valid_ratio: float = Field(
        ge=SPLIT_RATIO_MIN_VALUE,
        le=SPLIT_RATIO_MAX_VALUE,
    )
    test_ratio: float = Field(
        ge=SPLIT_RATIO_MIN_VALUE,
        le=SPLIT_RATIO_MAX_VALUE,
    )

    @model_validator(mode="after")
    def check_sum(self) -> SplitConfig:
        """Validate the split ratios sum.

        Returns:
            SplitConfig:
                Self, if validation passes.

        Raises:
            ValueError:
                If the sum of ratios does not equal the predefined value
                within tolerance.
        """
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if (
            abs(total - (SPLIT_RATIO_MAX_VALUE - SPLIT_RATIO_MIN_VALUE))
            > SPLIT_RATIO_SUM_TOL
        ):
            raise ValueError(
                f"train_ratio + valid_ratio + test_ratio must sum to {SPLIT_RATIO_MAX_VALUE - SPLIT_RATIO_MIN_VALUE}",
            )
        return self


class RatingConfig(BaseModel):
    """Configuration for rating-based binarization.

    Defines the threshold used to convert continuous ratings into binary
    labels.

    Attributes:
        threshold (int):
            Rating threshold for binarization.
    """

    threshold: int = Field(
        ge=RATING_MIN_VALUE,
        le=RATING_MAX_VALUE,
    )


class DeduplicationConfig(BaseModel):
    """Configuration for deduplication preprocessing step.

    Defines how duplicate user-item interactions are handled.

    Attributes:
        keep (str):
            Which duplicate to keep.
    """

    keep: str = Field()

    @field_validator("keep")
    @classmethod
    def validate_keep_strategy(cls, v: str) -> str:
        """Validate that keep strategy is valid.

        Args:
            v (str):
                The keep strategy to validate.

        Returns:
            str:
                The validated keep strategy.

        Raises:
            ValueError:
                If keep strategy is not valid.
        """
        if v not in DEDUP_KEEP_STRATEGIES:
            raise ValueError(
                f"keep strategy must be one of {DEDUP_KEEP_STRATEGIES}, "
                f"got '{v}'",
            )
        return v


class BinarizationConfig(BaseModel):
    """Configuration for binarization preprocessing step.

    Encapsulates all settings related to converting features
    into binary representations for collaborative filtering models.

    Attributes:
        rating (RatingConfig):
            Configuration for rating binarization.
    """

    rating: RatingConfig


class PreprocessingConfig(BaseModel):
    """Configuration for all preprocessing steps.

    Combines configurations for different preprocessing stages.

    Attributes:
        deduplication (DeduplicationConfig):
            Configuration for deduplication.
        binarization (BinarizationConfig):
            Configuration for binarization.
        split (SplitConfig):
            Configuration for train/valid/test split.
    """

    deduplication: DeduplicationConfig
    binarization: BinarizationConfig
    split: SplitConfig


class DataConfig(BaseModel):
    """Configuration for data-related settings.

    Aggregates configuration for the data.

    Attributes:
        name (str):
            Name of the dataset.
        preprocessing (PreprocessingConfig):
            Preprocessing configuration for the dataset.
    """

    name: str = Field()
    preprocessing: PreprocessingConfig

    @field_validator("name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate that dataset name is supported.

        Args:
            v (str):
                The dataset name to validate.

        Returns:
            str:
                The validated dataset name.

        Raises:
            ValueError:
                If dataset name is not a supported dataset.
        """
        if v not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{v}' is not supported. "
                f"Supported datasets: {SUPPORTED_DATASETS}",
            )
        return v


class Config(BaseModel):
    """Root configuration.

    Top-level configuration that aggregates all other configuration sections.

    Attributes:
        data (DataConfig):
            Data configuration.
        seed (int):
            Random seed for reproducibility.
    """

    data: DataConfig
    seed: int = Field(ge=0)
