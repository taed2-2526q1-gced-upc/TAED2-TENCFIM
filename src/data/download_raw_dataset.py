"""Download the raw emotions dataset and save it under data/raw."""

import os

import pandas as pd
from loguru import logger

from src.config import RAW_DATA_DIR, RAW_DATA_NAME


def download_raw_dataset() -> None:
    """Fetch the dataset from Hugging Face and store it in data/raw."""
    try:
        df = pd.read_parquet(
            "hf://datasets/boltuix/emotions-dataset/emotions_dataset.parquet"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to download the dataset: %s", exc)
        raise

    try:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_parquet(RAW_DATA_DIR / RAW_DATA_NAME, index=False)
        logger.info("Dataset saved to %s", RAW_DATA_DIR / RAW_DATA_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save the dataset: %s", exc)
        raise


if __name__ == "__main__":
    download_raw_dataset()
