import pandas as pd
from pathlib import Path
import types
from pytest import fixture
from src.config import ROOT_DIR
from src.data.gx_context_configuration import CLEAN_DATA_VALIDATOR
from src.data import preprocess, download_raw_dataset, split_data, gx_context_configuration, validate_data
import great_expectations as gx
from great_expectations import ValidationDefinition

#FIxtures. Mirar que comleixi totes les de gx.
#Mirar que les columnes estiguin totes


# Test GX expectations
@fixture
def clean_data_validator() -> ValidationDefinition:
    """ Fixture to provide the Great Expectations ValidationDefinition for clean data validation."""
    
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)
    return context.validation_definitions.get(CLEAN_DATA_VALIDATOR)

@fixture
def make_sample_df():
    """ Makes a sample dataframe with some dirty data"""
    return pd.DataFrame(
        {
            "text": ["Hello\nworld", "  space  ", "", None, "dup", "dup"],
            "labels": ["Happiness", "Sadness", "Neutral", "Anger", "Love", "Love"],
        }
    )
    
@fixture
def make_clean_sample_df():
    """ Makes a clean dataframe with 10 rows"""
    
    return pd.DataFrame(
        {
            "text": [
                "Hello world",
                "Good morning",
                "I love this",
                "So sad",
                "Neutral statement",
                "Angry reply",
                "Another happy",
                "Calm and neutral",
                "Repeated",
                "Unique thought",
            ],
            "labels": [
                "Happiness",
                "Happiness",
                "Love",
                "Love",
                "Happiness",
                "Love",
                "Love",
                "Happiness",
                "Love",
                "Happiness",
            ],
        }
    )

def test_clean_data(clean_data_validator: ValidationDefinition):
    """ Test that the clean data meets all Great Expectations defined in the validator."""
    
    validation_result = clean_data_validator.run(result_format="BOOLEAN_ONLY")

    expectations_failed = validation_result["statistics"]["unsuccessful_expectations"]

    assert expectations_failed == 0, f"There were {expectations_failed} failing expectations."

def test_download_raw_dataset():
    """ Test the download of the raw dataset."""
    
    download_raw_dataset.download_dataset()
    out = Path(download_raw_dataset.RAW_DATA_DIR) / download_raw_dataset.RAW_DATA_NAME
    assert out.exists()
    df = pd.read_parquet(out)
    assert "Sentence" in df.columns and "Label" in df.columns

def test_preprocess_data():
    """ Test the preprocessing of the raw dataset."""
    
    preprocess.preprocess_data()

    out = Path(preprocess.INTERIM_DATA_DIR) / preprocess.INTERIM_DATA_NAME
    assert out.exists()
    df_out = pd.read_parquet(out)
    assert "text" in df_out.columns and "labels" in df_out.columns
    assert df_out["text"].str.len().min() > 0
    assert df_out["text"].duplicated().sum() == 0

def test_sanitize_text(make_sample_df):
    """ Test the sanitization of text in the dataframe."""
    
    df = make_sample_df
    sanitized = preprocess.sanitize_text(df.copy())
    assert type(sanitized) is pd.DataFrame
    assert "text" in sanitized.columns and "labels" in sanitized.columns
    assert all("\n" not in t for t in sanitized["text"])  # newlines removed
    assert sanitized["text"].str.len().min() >= 0

def test_remove_empty_or_duplicate(make_sample_df):
    """ Test the removal of empty or duplicate entries in the dataframe."""
    
    df = make_sample_df
    cleaned = preprocess.remove_empty_or_duplicate(df.copy())
    assert type(cleaned) is pd.DataFrame
    assert "text" in cleaned.columns and "labels" in cleaned.columns
    assert "" not in cleaned["text"].values
    assert cleaned["text"].duplicated().sum() == 0
    
def test_create_split_dataframes(make_clean_sample_df):
    """ Test the creation of train, validation, and test splits from the dataframe."""
    
    df = make_clean_sample_df
    train_df, val_df, test_df = split_data.create_split_dataframes(df)
    
    # Check that the splits are correct
    total_rows = len(df)
    assert len(train_df) + len(val_df) + len(test_df) == total_rows
    assert abs(len(train_df) - 0.7 * total_rows) <= 1
    assert abs(len(val_df) - 0.2 * total_rows) <= 1
    assert abs(len(test_df) - 0.1 * total_rows) <= 1
 
def test_split_data():
    """ Test the full data splitting process, ensuring output files are created and row counts match."""
    
    split_data.split_data()

    processed_dir = Path(split_data.PROCESSED_DATA_DIR)
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "validation.parquet"
    test_path = processed_dir / "test.parquet"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    original_df = pd.read_parquet(Path(split_data.INTERIM_DATA_DIR) / split_data.INTERIM_DATA_NAME)
    assert total_rows == len(original_df)