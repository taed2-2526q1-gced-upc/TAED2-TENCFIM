import pandas as pd
import os
from loguru import logger

from src.config import RAW_DATA_DIR, RAW_DATA_NAME

def download_dataset():
    """
    Download the dataset from Hugging Face and save it to the 'data/raw' directory.
    """
    
    logger.info("Downloading dataset from Hugging Face")
    try:
        df = pd.read_parquet("hf://datasets/boltuix/emotions-dataset/emotions_dataset.parquet")
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_parquet(RAW_DATA_DIR / RAW_DATA_NAME, index=False)
        logger.info(f"Dataset saved to {RAW_DATA_DIR / RAW_DATA_NAME}")
    except Exception as e:
        logger.error(f"Failed to download or save the dataset: {e}")
        raise

if __name__ == "__main__":
    download_dataset()
