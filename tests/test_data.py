import pandas as pd
from pathlib import Path
import types

import pytest

from src.data import preprocess, download_raw_dataset, split_data, gx_context_configuration, validate_data
from src import config


def make_sample_df():
    return pd.DataFrame(
        {
            "Sentence": ["Hello\nworld", "  space  ", "", None, "dup", "dup"],
            "Label": ["Happiness", "Sadness", "Neutral", "Anger", "Love", "Love"],
        }
    )


def test_sanitize_and_remove_empty_or_duplicate(tmp_path):
    df = make_sample_df().rename(columns={"Sentence": "text", "Label": "labels"})

    sanitized = preprocess.sanitize_text(df.copy())
    assert all("\n" not in t for t in sanitized["text"])  # newlines removed
    assert sanitized["text"].str.len().min() >= 0

    cleaned = preprocess.remove_empty_or_duplicate(sanitized.copy())
    assert "" not in cleaned["text"].values
    assert cleaned["text"].duplicated().sum() == 0


def test_preprocess_data_writes_interim(tmp_path, monkeypatch):
    df = make_sample_df()
    # make preprocess module use our tmp paths and stub pd.read_parquet
    monkeypatch.setattr(preprocess, "pd", preprocess.pd)
    monkeypatch.setattr("src.data.preprocess.pd.read_parquet", lambda path: df)

    monkeypatch.setattr(preprocess, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "RAW_DATA_NAME", "raw_emotions.parquet")
    monkeypatch.setattr(preprocess, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(preprocess, "INTERIM_DATA_NAME", "emotions_cleaned.parquet")

    preprocess.preprocess_data()

    out = Path(preprocess.INTERIM_DATA_DIR) / preprocess.INTERIM_DATA_NAME
    assert out.exists()
    df_out = pd.read_parquet(out)
    assert "Sentence" in df_out.columns and "Label" in df_out.columns


def test_download_dataset_writes_parquet(tmp_path, monkeypatch):
    sample = pd.DataFrame({"Sentence": ["a"], "Label": ["Happiness"]})
    monkeypatch.setattr("src.data.download_raw_dataset.pd.read_parquet", lambda path: sample)

    # Make the download_raw_dataset module write into tmp_path
    monkeypatch.setattr(download_raw_dataset, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(download_raw_dataset, "RAW_DATA_NAME", "raw_emotions.parquet")

    download_raw_dataset.download_dataset()

    out = Path(download_raw_dataset.RAW_DATA_DIR) / download_raw_dataset.RAW_DATA_NAME
    assert out.exists()
    df = pd.read_parquet(out)
    assert "Sentence" in df.columns and "Label" in df.columns


def test_split_data_creates_processed_files(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "text": [f"t{i}" for i in range(20)],
        "labels": ["A", "B"] * 10,
    })

    # Ensure split_data module reads/writes from our tmp folders
    monkeypatch.setattr(split_data, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(split_data, "INTERIM_DATA_NAME", "emotions_cleaned.parquet")
    monkeypatch.setattr(split_data, "PROCESSED_DATA_DIR", tmp_path / "processed")

    path = Path(split_data.INTERIM_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / split_data.INTERIM_DATA_NAME, index=False)

    split_data.split_data()

    train = Path(split_data.PROCESSED_DATA_DIR / "train.parquet")
    val = Path(split_data.PROCESSED_DATA_DIR / "validation.parquet")
    test = Path(split_data.PROCESSED_DATA_DIR / "test.parquet")

    assert train.exists() and val.exists() and test.exists()

    df_train = pd.read_parquet(train)
    df_val = pd.read_parquet(val)
    df_test = pd.read_parquet(test)

    assert len(df_train) + len(df_val) + len(df_test) == len(df)


def test_gx_context_configuration_constants():
    assert hasattr(gx_context_configuration, "DATASOURCE_NAME")
    assert hasattr(gx_context_configuration, "CLEAN_DATA_ASSET")
    assert hasattr(gx_context_configuration, "CHECKPOINT")


class DummyCheckpoint:
    def run(self):
        fake = types.SimpleNamespace()
        fake.run_results = {"run_1": {"statistics": {"evaluated_expectations": 5, "unsuccessful_expectations": 1}}}
        return fake


class DummyContext:
    def __init__(self, checkpoint):
        self._ck = checkpoint

    @property
    def checkpoints(self):
        class C:
            def __init__(self, ck):
                self._ck = ck

            def get(self, name):
                return self._ck

        return C(self._ck)


def test_validate_data_monkeypatched(monkeypatch):
    # monkeypatch gx.get_context used in validate_data module import
    monkeypatch.setattr("src.data.validate_data.gx.get_context", lambda mode, project_root_dir: DummyContext(DummyCheckpoint()))

    # reload module to execute top-level code under monkeypatched get_context
    import importlib
    import src.data.validate_data as vd
    importlib.reload(vd)

    # the module runs top-level code and defines expectations_run etc as module variables
    assert hasattr(vd, "expectations_run")
    assert hasattr(vd, "expectations_failed")
    assert vd.expectations_run == 5
    assert vd.expectations_failed == 1
