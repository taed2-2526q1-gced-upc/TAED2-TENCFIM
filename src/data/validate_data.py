"""Run Great Expectations validation using a configured checkpoint and log results."""

import great_expectations as gx
from loguru import logger

from src.config import ROOT_DIR
from src.data.gx_context_configuration import CHECKPOINT


def main() -> None:
    """Load DataContext, run the checkpoint, and log validation summary."""
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

    checkpoint = context.checkpoints.get(CHECKPOINT)
    checkpoint_result = checkpoint.run()

    # Extract the Validation Result object from the Checkpoint results.
    first_key = next(iter(checkpoint_result.run_results))
    validation_result = checkpoint_result.run_results[first_key]

    stats = validation_result["statistics"]
    expectations_run = stats["evaluated_expectations"]
    expectations_failed = stats["unsuccessful_expectations"]

    logger.info(
        f"Validation results: {expectations_run} expectations evaluated, {expectations_failed} failed.",
    )


if __name__ == "__main__":
    main()
