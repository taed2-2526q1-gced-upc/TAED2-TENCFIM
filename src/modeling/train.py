import click
from datasets import load_dataset
from evaluate import load
from loguru import logger
import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED

SPEEDUP_TRAINING = True  # Set to True to speed up training by using a smaller dataset
SAMPLE_SIZE = 500

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"


@click.command()
@click.argument("hf_model", type=str)
@click.argument("model_name", type=str)
def main(hf_model: str, model_name: str):
    def tokenize(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512)
        return outputs

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = AutoTokenizer.from_pretrained(hf_model, num_labels=2)
    metric = load("accuracy")

    logger.info("Loading the dataset...")
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(PROCESSED_DATA_DIR / "train.parquet"),
            "validation": str(PROCESSED_DATA_DIR / "validation.parquet"),
        },
    )

    if SPEEDUP_TRAINING:
        logger.info("Speeding up training by using a smaller dataset.")
        train_ds = ds["train"].shuffle(SEED).select(range(SAMPLE_SIZE)).map(tokenize, batched=True)
        validation_ds = ds["validation"].shuffle(SEED).select(range(SAMPLE_SIZE)).map(tokenize, batched=True)
    else:
        train_ds = ds["train"].map(tokenize, batched=True)
        validation_ds = ds["validation"].map(tokenize, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    validation_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=MODELS_DIR / f"{model_name}-checkpoint",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        push_to_hub=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model, num_labels=2, label2id=label2id, id2label=id2label, problem_type="single_label_classification"
    ).to(device)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Starting training of model {model_name}...")
    mlflow.set_experiment("IMDB sentiment analysis")
    mlflow.set_system_metrics_sampling_interval(5)
    with mlflow.start_run(log_system_metrics=True):
        trainer.train()

    trainer.save_model(str(MODELS_DIR / model_name))


if __name__ == "__main__":
    main()
