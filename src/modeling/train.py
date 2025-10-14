from pathlib import Path
from datasets import load_dataset, Dataset
from evaluate import load
from loguru import logger
import mlflow
import dagshub
import numpy as np
import torch
import os
import sys

# Use UTF-8 encoding for stdout/stderr to avoid UnicodeEncodeError on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # some environments don't support reconfigure (py39 minimal)
    pass

# Take out colors and emojis from rich logs (loguru uses rich under the hood)
os.environ["RICH_NO_COLOR"] = "1"
os.environ["RICH_NO_EMOJI"] = "1"

from sklearn.metrics import f1_score
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from typing import Optional, Dict, Any

# Optional integrations
try:
    from codecarbon import EmissionsTracker
except Exception: # pragma: no cover - optional dependency
    EmissionsTracker = None


try:
    import great_expectations as gx # type: ignore
except Exception: # pragma: no cover - optional dependency
    gx = None

from src.config import MODELS_DIR, SEED, PROCESSED_DATA_DIR
from dotenv import load_dotenv
from codecarbon import EmissionsTracker
load_dotenv()

# ------------------------------------------------------------------
# DagsHub + MLflow configuration via environment variables
# Set these in your .env or environment:
#   DAGSHUB_USER, DAGSHUB_REPO, DAGSHUB_TOKEN (preferred)
# or standard MLflow variables:
#   MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
# ------------------------------------------------------------------
DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "TAED2-TENCFIM")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
dagshub.init(repo_owner='BielBota8', repo_name='TAED2-TENCFIM', mlflow=True)


DEFAULT_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

# If a DagsHub token is provided and basic auth env vars aren't set, set them.
if DAGSHUB_TOKEN and not os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Configure MLflow endpoints (tracking and registry)
mlflow.set_tracking_uri(MLFLOW_URI)
try:
    mlflow.set_registry_uri(MLFLOW_URI)
except Exception:
    pass

SPEEDUP_TRAINING = True  # Set to True to speed up training by using a smaller dataset
SAMPLE_SIZE = 5000  

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"


def _find_local_hf_model() -> Optional[str]:
    """
    Look for a local HF model snapshot under MODELS_DIR / 'base-model'.
    Returns a path string to the snapshot directory if found, otherwise None.
    """
    local_root = Path(MODELS_DIR) / "base-model"
    if not local_root.exists():
        return None

    # look for repo dir like 'models--Owner--repo'
    repo_candidates = list(local_root.glob("models--*"))
    if not repo_candidates:
        return None

    repo_dir = repo_candidates[0]
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snaps = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if snaps:
            return str(snaps[-1])

    return str(repo_dir)


def freeze_base_model(model, unfreeze_last_n: int = 2):
    """
    Freeze base model except the classification head.
    Optionally unfreeze last N encoder layers (useful for RoBERTa/BERT).
    """
    base_attrs = ["base_model", "roberta", "bert", "distilbert", "albert",
                  "xlm_roberta", "electra", "camembert", "deberta", "mpnet", "model"]
    base = None
    for attr in base_attrs:
        base = getattr(model, attr, None)
        if base is not None:
            break

    if base is None:
        # fallback: freeze everything except names that look like classifier/pooler/head/score
        for name, p in model.named_parameters():
            if any(k in name.lower() for k in ("classifier", "pooler", "head", "score")):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return

    # Freeze all base params
    for p in base.parameters():
        p.requires_grad = False

    # Try to locate encoder layers and unfreeze last N
    encoder = getattr(base, "encoder", None) or getattr(base, "transformer", None)
    if encoder is not None and hasattr(encoder, "layer") and unfreeze_last_n > 0:
        layers = encoder.layer
        n = min(unfreeze_last_n, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Ensure classifier/head params are trainable
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ("classifier", "pooler", "head", "score")):
            p.requires_grad = True


def _train_model(hf_model: str, model_name: str):
    # ------------------------------------------------------------------
    # Prepare tokenizer & metrics
    # ------------------------------------------------------------------
    # -----------------------------
    # Hyperparameters (local) --- edit here
    # -----------------------------
    SPEEDUP_TRAINING = True  # set to False to use the full dataset
    SAMPLE_SIZE = 60000
    BATCH_SIZE = 32
    EPOCHS = 4  # only fine-tune head by default
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER = "linear"

    # If hf_model points to a local snapshot folder, pass that to from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")

    # Emotion label set (13 classes) of boltuix/emotions-dataset
    label_list = [
        "Happiness",
        "Sadness",
        "Neutral",
        "Anger",
        "Love",
        "Fear",
        "Disgust",
        "Confusion",
        "Surprise",
        "Shame",
        "Guilt",
        "Sarcasm",
        "Desire",
    ]
    label2id = {lbl.lower(): i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for i, lbl in enumerate(label_list)}

    def preprocess(batch):
        # Accept multiple possible column names for text and label
        text_keys = [k for k in ("Sentence", "sentence", "text") if k in batch]
        label_keys = [k for k in ("Label", "label", "labels") if k in batch]

        if not text_keys or not label_keys:
            raise KeyError("Expected text column (Sentence/text) and label column (Label/label) in the dataset batch")

        text_key = text_keys[0]
        label_key = label_keys[0]

        # batch[text_key] will be a list of strings when batched=True
        tokenized = tokenizer(batch[text_key], truncation=True, max_length=512)

        # Map label strings to ids; tolerate ints already present
        labels_out = []
        for lbl in batch[label_key]:
            if isinstance(lbl, str):
                labels_out.append(label2id[lbl.lower()])
            else:
                # assume already numeric
                labels_out.append(int(lbl))

        tokenized["labels"] = labels_out
        return tokenized

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        
        # Calculate accuracy
        accuracy = accuracy_metric.compute(predictions=preds, references=labels)
        
        # Calculate F1 score (macro average for multi-class)
        f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")
        f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        
        return {
            "accuracy": accuracy["accuracy"],
            "f1_macro": f1_macro["f1"],
            "f1_weighted": f1_weighted["f1"]
        }

    # Load processed parquet files (pandas) and convert to HF Dataset objects
    logger.info("Loading processed parquet files and converting to HF Datasets (train, val, test)")
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    validation_df = pd.read_parquet(PROCESSED_DATA_DIR / "validation.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")

    # Convert pandas DataFrames to Hugging Face Datasets
    train_raw = Dataset.from_pandas(train_df.reset_index(drop=True))
    validation_raw = Dataset.from_pandas(validation_df.reset_index(drop=True))
    test_raw = Dataset.from_pandas(test_df.reset_index(drop=True))

    if SPEEDUP_TRAINING:
        logger.info("Speeding up training by sub-sampling (%d examples).", SAMPLE_SIZE)
        train_raw = train_raw.shuffle(SEED).select(range(min(SAMPLE_SIZE, len(train_raw))))
        validation_raw = validation_raw.shuffle(SEED).select(range(min(SAMPLE_SIZE, len(validation_raw))))

    train_ds = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
    validation_ds = validation_raw.map(preprocess, batched=True, remove_columns=validation_raw.column_names)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    validation_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / f"{model_name}-checkpoint"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        push_to_hub=False,
        metric_for_best_model="accuracy",
        load_best_model_at_end=False,
        report_to=[],  # avoid auto-integrations (wandb/comet/dagshub)
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU memory before model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # Clear cache to avoid memory issues
        torch.cuda.empty_cache()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id={k: v for v, k in id2label.items()},
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    # Freeze base model so only the classification head is trained (fine-tuning)
    freeze_base_model(model, 4)  # unfreeze last 4 layers
    logger.info("Base model frozen; classification head will be fine-tuned.")
    
    if device == "cuda":
        logger.info(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=compute_metrics,
    )

    # Workaround: Some versions of transformers+dagshub raise
    # AttributeError: 'DagsHubCallback' object has no attribute '_auto_end_run'.
    # Remove the callback if it was auto-registered.
    try:
        from transformers.integrations import DagsHubCallback  # type: ignore

        trainer.remove_callback(DagsHubCallback)
        logger.info("DagsHubCallback removed to avoid '_auto_end_run' error.")
    except Exception:
        pass

    logger.info(f"Starting training of model {model_name}...")
    mlflow.set_experiment("Emotion detection (13-class)")
    # mlflow.set_system_metrics_sampling_interval(5)
    
    try:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model": hf_model,
                    "num_labels": len(label_list),
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "scheduler": LR_SCHEDULER,
                    "speedup_training": SPEEDUP_TRAINING,
                    "sample_size": SAMPLE_SIZE if SPEEDUP_TRAINING else None,
                }
            )
            logger.info("MLflow parameters logged successfully.")
            
            logger.info("Starting trainer.train()...")
            train_result = trainer.train()
            logger.info(f"Training completed successfully! Final loss: {train_result.training_loss}")
            
            # Evaluate on validation set to get final metrics
            logger.info("Evaluating model on validation set...")
            eval_results = trainer.evaluate()
            logger.info(f"Validation results: {eval_results}")
            
            # Log training and validation metrics
            mlflow.log_metrics({
                "final_train_loss": train_result.training_loss,
                "total_steps": train_result.global_step,
                "eval_loss": eval_results["eval_loss"],
                "eval_accuracy": eval_results["eval_accuracy"],
                "eval_f1_macro": eval_results["eval_f1_macro"],
                "eval_f1_weighted": eval_results["eval_f1_weighted"]
            })
            
            logger.info("MLflow run completed successfully.")
    
    except Exception as e:
        logger.error(f"Error during training or MLflow logging: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        raise  # Re-raise the exception to see the full traceback
    
    logger.info(f"Saving model to {MODELS_DIR / model_name}...")
    try:
        trainer.save_model(str(MODELS_DIR / model_name))
        logger.info("Model saved successfully!")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
    
    # Clean up resources
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared.")
    
    del model, trainer  # Explicitly delete large objects
    
    logger.success(f"🎉 Training pipeline completed successfully for model '{model_name}'!")
    logger.info(f"Model saved at: {MODELS_DIR / model_name}")
    logger.info("Script execution finished. Staying in WSL environment...")


def main():
    # Resolve local model if available, otherwise fall back to HF hub id
    hf_model = _find_local_hf_model() or "SamLowe/roberta-base-go_emotions"
    # Model names: roberta-emotions-v1.x --> First version: testing
    #              roberta-emotions-v2.x --> Second version: fine tuning
    
    model_name = "roberta-emotions-v2.8"
    
    logger.info(f"Starting training pipeline for model: {model_name}")
    logger.info(f"Base HF model (local or hub id): {hf_model}")
    with EmissionsTracker() as tracker:
        _train_model(hf_model, model_name)


if __name__ == "__main__":
    try:
        logger.info("Script starting...")
        main()
        logger.success("Script completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
c    finally:
        logger.info("Cleaning up and exiting...")

#python src/modeling/train.py SamLowe/roberta-base-go_emotions roberta-emotions-13