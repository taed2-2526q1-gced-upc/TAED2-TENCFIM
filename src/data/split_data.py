import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_NAME, PROCESSED_DATA_DIR, SEED, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

def create_split_dataframes(df: pd.DataFrame):
    """ Given a dataframe, split it into train, validation, and test dataframes"""
    
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_SPLIT),
        stratify=df["labels"],                  # ensures that the split contains the same proportion
        random_state=SEED                      # of labels that were in the original df
    )

    validation_size = VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - validation_size),
        stratify=temp_df["labels"],
        random_state=SEED
    )
    
    return train_df, validation_df, test_df

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
        train_df, validation_df, test_df = create_split_dataframes(df)
    except Exception as e:
        logger.error(f"Error during stratified splitting: {e}")
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