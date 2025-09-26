import click
from datasets import load_dataset
from evaluate import load
from loguru import logger
import mlflow
import dagshub
import numpy as np
import torch
import os
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
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
SAMPLE_SIZE = 1000  

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"


@click.command()
@click.argument("hf_model", type=str)
@click.argument("model_name", type=str)
def main(hf_model: str, model_name: str):
    try:
        logger.info(f"Starting training pipeline for model: {model_name}")
        logger.info(f"Base HF model: {hf_model}")
        _train_model(hf_model, model_name)
    except Exception as e:
        logger.error(f"Fatal error in main(): {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise


def _train_model(hf_model: str, model_name: str):
    # ------------------------------------------------------------------
    # Prepare tokenizer & metrics
    # ------------------------------------------------------------------
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
        tokenized = tokenizer(batch["Sentence"], truncation=True, max_length=512)
        tokenized["labels"] = [label2id[label.lower()] for label in batch["Label"]]
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
        ignore_mismatched_sizes=True,  # different head than original 28-label multi-label
    ).to(device)
    
    if device == "cuda":
        logger.info(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
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


if __name__ == "__main__":
    try:
        logger.info("Script starting...")
        main()
        logger.success("Script completed successfully. Press any key to continue...")
        input()  # Wait for user input to prevent immediate exit
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.error("Press any key to exit...")
        input()  # Wait before exiting so user can see the error
    finally:
        logger.info("Cleaning up and exiting...")

#python src/modeling/train.py SamLowe/roberta-base-go_emotions roberta-emotions-13