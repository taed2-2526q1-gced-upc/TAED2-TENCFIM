"""Download a Hugging Face model snapshot into the local `models/base-model` folder."""

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise RuntimeError(
        "Missing required package 'huggingface_hub'. Install with: "
        "pip install huggingface-hub python-dotenv"
    ) from exc


def main():
    """Load env vars, validate HF token, and download the model snapshot."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not found in environment. Please set HF_TOKEN in your "
            ".env or environment."
        )

    repo_id = "SamLowe/roberta-base-go_emotions"
    dest = Path(__file__).resolve().parents[2] / "models" / "base-model"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} into {dest}...")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=str(dest),
        repo_type="model",
        use_auth_token=hf_token,
        local_dir_use_symlinks=False,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
