import click
from datasets import load_dataset
from evaluate import load
from loguru import logger
import mlflow
import sys
import os

# Use UTF-8 encoding for stdout/stderr to avoid UnicodeEncodeError on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Take out colors and emojis from rich logs (loguru uses rich under the hood)
os.environ["RICH_NO_COLOR"] = "1"
os.environ["RICH_NO_EMOJI"] = "1"

import dagshub
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score
)

from src.config import MODELS_DIR, SEED
from dotenv import load_dotenv
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
SAMPLE_SIZE = 10000  # Number of samples to use if SPEEDUP_TRAINING is True

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"


@click.command()
@click.argument("hf_model", type=str)
@click.argument("model_name", type=str)
def main(hf_model: str, model_name: str):
    # ------------------------------------------------------------------
    # Prepare tokenizer & metric
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    metric = load("accuracy")

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
        tokenized = tokenizer(batch["Sentence"], truncation=True, max_length=512)
        tokenized["labels"] = [label2id[label.lower()] for label in batch["Label"]]
        return tokenized

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average="macro")
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        acc = metric.compute(predictions=preds, references=labels)["accuracy"]
        return {"accuracy": acc, "f1_macro": f1, "precision_macro": precision, "recall_macro": recall}

    # Load Hugging Face dataset (single 'train' split) and create validation split
    logger.info("Loading boltuix/emotions-dataset and creating validation split...")
    raw = load_dataset("boltuix/emotions-dataset")
    split = raw["train"].train_test_split(test_size=0.1, seed=SEED)
    train_raw, validation_raw = split["train"], split["test"]

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
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        push_to_hub=False,
        metric_for_best_model="f1_macro",
        load_best_model_at_end=True,
        greater_is_better=True,
        seed=SEED,
        report_to=[],  # avoid auto-integrations (wandb/comet/dagshub)
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id={k: v for v, k in id2label.items()},
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,  # different head than original 28-label multi-label
    ).to(device)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Stop training if no improvement after 2 evals
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
        trainer.train()

    # Final evaluation
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    trainer.save_model(str(MODELS_DIR / model_name))


if __name__ == "__main__":
    main()

#python src/modeling/train.py SamLowe/roberta-base-go_emotions roberta-emotions-13-v1