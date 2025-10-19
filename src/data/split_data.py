"""Split cleaned data into train/validation/test Parquet files with stratification."""

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import (
    INTERIM_DATA_DIR,
    INTERIM_DATA_NAME,
    PROCESSED_DATA_DIR,
    SEED,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)


def create_split_dataframes(df: pd.DataFrame):
    """Split a DataFrame into train, validation, and test DataFrames."""
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_SPLIT),
        stratify=df["labels"],  # keep label proportions
        random_state=SEED,
    )

    validation_size = VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - validation_size),
        stratify=temp_df["labels"],
        random_state=SEED,
    )

    return train_df, validation_df, test_df


def split_data():
    """
    Split the cleaned data into train, validation, and test sets and save them
    to the 'data/processed' directory.

    The split uses 70% train, 20% validation, and 10% test with a fixed seed.
    """
    logger.info("Splitting data into train, validation, and test sets")
    try:
        df = pd.read_parquet(INTERIM_DATA_DIR / INTERIM_DATA_NAME)
    except FileNotFoundError as err:
        logger.error("Input file not found: %s", err)
        return
    except (OSError, ValueError) as err:
        logger.error("Error reading input file: %s", err)
        return

    try:
        train_df, validation_df, test_df = create_split_dataframes(df)
    except ValueError as err:
        # Raised by train_test_split for invalid sizes / stratify issues
        logger.error("Error during stratified splitting: %s", err)
        return

    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet")
        validation_df.to_parquet(PROCESSED_DATA_DIR / "validation.parquet")
        test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet")
    except OSError as err:
        logger.error("Failed to save split datasets (I/O): %s", err)
        return
    except ValueError as err:
        logger.error("Failed to save split datasets (data issue): %s", err)
        return

    logger.info("Data split and saved successfully.")


if __name__ == "__main__":
    split_data()
