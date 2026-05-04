"""src/config/config.py

Configuration module.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

from src.const import SUPPORTED_DATASETS


class SplitConfig(BaseModel):
    """Configuration for dataset split ratios.

    Defines the proportions for splitting a dataset into training, validation,
    and test sets.

    Attributes:
        train_ratio (float):
            Proportion of data for training. Must be in [0.0, 1.0].
        valid_ratio (float):
            Proportion of data for validation. Must be in [0.0, 1.0].
        test_ratio (float):
            Proportion of data for testing. Must be in [0.0, 1.0].
    """

    train_ratio: float = Field(ge=0.0, le=1.0)
    valid_ratio: float = Field(ge=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> SplitConfig:
        """Validate that split ratios sum to 1.0.

        Returns:
            SplitConfig:
                Self, if validation passes.

        Raises:
            ValueError:
                If the sum of ratios does not equal 1.0 within tolerance.
        """
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + valid_ratio + test_ratio must sum to 1.0",
            )
        return self


class RatingConfig(BaseModel):
    """Configuration for rating-based binarization.

    Defines the threshold used to convert continuous ratings into binary
    labels.

    Attributes:
        threshold (int):
            Rating threshold for binarization. Must be in range [1, 5]
            to match standard Amazon review scale.
    """

    threshold: int = Field(ge=1, le=5)


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
        binarization (BinarizationConfig):
            Configuration for binarization.
        split (SplitConfig):
            Configuration for train/valid/test split.
    """

    binarization: BinarizationConfig
    split: SplitConfig


class DataConfig(BaseModel):
    """Configuration for data-related settings.

    Aggregates configuration for the data.

    Attributes:
        name (str):
            Name of the dataset. Must be one of the supported datasets.
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
            Random seed for reproducibility. Must be non-negative.
    """

    data: DataConfig
    seed: int = Field(ge=0)
