import pandas as pd
from loguru import logger
from src.config import INTERIM_DATA_DIR, INTERIM_DATA_NAME, RAW_DATA_DIR, RAW_DATA_NAME


def sanitize_text(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize the text text by removing special characters and normalizing.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the text text.

    Returns
    -------
    pd.DataFrame
        The sanitized DataFrame with cleaned text text.
    """
    dataframe["text"] = (
        dataframe["text"]
        .astype(str)
        .str.replace(r"(\n|\r|\s+)", " ", regex=True)
        .str.strip()
        .str.normalize("NFKD")
    )

    return dataframe


def remove_empty_or_duplicate(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty or duplicate rows from the DataFrame.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input DataFrame containing the text text and labels.

    Returns
    -------
    pl.DataFrame
        The cleaned DataFrame with no duplicate or empty rows.
    """
    dataframe = dataframe[dataframe["text"].notnull() & dataframe["labels"].notnull()]
    dataframe = dataframe[dataframe["text"] != ""]

    return dataframe.drop_duplicates(subset=["text", "labels"])


def preprocess_data():
    """
    Preprocess the raw data and save it to the 'data/interim' directory.
    """
    logger.info("Preprocessing raw data")

    try:
        df = pd.read_parquet(RAW_DATA_DIR / RAW_DATA_NAME)

        df = df.rename(columns={"Label": "labels", "Sentence": "text"})

        df = sanitize_text(df)

        df = remove_empty_or_duplicate(df)

        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

        df.to_parquet(INTERIM_DATA_DIR / INTERIM_DATA_NAME)
        
        logger.info(f"Preprocessed data saved to {INTERIM_DATA_DIR / INTERIM_DATA_NAME}")

    except Exception as e:
        logger.error(f"Failed to preprocess or save the dataset: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
