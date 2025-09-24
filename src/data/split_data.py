import pandas as pd
from loguru import logger
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_NAME, PROCESSED_DATA_DIR, SEED, TRAIN_SPLIT, VALIDATION_SPLIT


def split_data():
    """
    Split the cleaned data into train, validation, and test sets and save them to the 'data/processed' directory.

    The split is done in the ratio of 70% train, 20% validation, and 10% test.
    The random seed is set to ensure reproducibility.
    """
    logger.info("Splitting data into train, validation, and test sets")
    try:
        df = pd.read_parquet(INTERIM_DATA_DIR / INTERIM_DATA_NAME)
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return

    try:
        n_samples = df.shape[0]
        train_size = int(n_samples * TRAIN_SPLIT)
        validation_size = int(n_samples * VALIDATION_SPLIT)

        if train_size + validation_size >= n_samples:
            logger.error("Split sizes are too large for the dataset.")
            return

        train_df = df.sample(n=train_size, replace=False, random_state=SEED)
        remaining_df = df.drop(train_df.index)
        validation_df = remaining_df.sample(n=validation_size, replace=False, random_state=SEED)
        test_df = remaining_df.drop(validation_df.index)
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return

    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet")
        validation_df.to_parquet(PROCESSED_DATA_DIR / "validation.parquet")
        test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet")
    except Exception as e:
        logger.error(f"Failed to save split datasets: {e}")
        return

    logger.info("Data split and saved successfully.")

if __name__ == "__main__":
    split_data()
