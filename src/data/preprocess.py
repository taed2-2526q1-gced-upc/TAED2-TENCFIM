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

    # drop rows where text is not a Python str
    mask = dataframe["text"].apply(lambda v: isinstance(v, str))
    dataframe = dataframe[mask].reset_index(drop=True)
    dataframe["text"] = dataframe["text"].astype(str)
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

    return dataframe.drop_duplicates(subset=["text"])


def preprocess_data():
    """
    Preprocess the raw data and save it to the 'data/interim' directory.
    """
    logger.info("Preprocessing raw data")

    try:
        df = pd.read_parquet(RAW_DATA_DIR / RAW_DATA_NAME)

        # Robustly rename common column variants to the canonical names used downstream
        rename_map = {}
        if "Sentence" in df.columns:
            rename_map["Sentence"] = "text"
        if "sentence" in df.columns:
            rename_map["sentence"] = "text"
        if "Sentence_text" in df.columns:  # defensive: some sources use different names
            rename_map["Sentence_text"] = "text"

        if "Label" in df.columns:
            rename_map["Label"] = "labels"
        if "label" in df.columns:
            rename_map["label"] = "labels"

        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure required columns exist after renaming
        if "text" not in df.columns or "labels" not in df.columns:
            raise KeyError(
                "Expected columns 'text' and 'labels' in raw dataset after renaming. "
                f"Found columns: {list(df.columns)}"
            )

        df = sanitize_text(df)

        df = remove_empty_or_duplicate(df)

        # Keep only the expected columns and a consistent order
        df = df.reset_index(drop=True)
        df = df[["text", "labels"]]

        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

        df.to_parquet(INTERIM_DATA_DIR / INTERIM_DATA_NAME, index=False)

        logger.info(f"Preprocessed data saved to {INTERIM_DATA_DIR / INTERIM_DATA_NAME}")

    except Exception as e:
        logger.error(f"Failed to preprocess or save the dataset: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
