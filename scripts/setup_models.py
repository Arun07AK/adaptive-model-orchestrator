#!/usr/bin/env python3
"""Download and verify MLX models for local inference.

Usage:
    python scripts/setup_models.py          # Download all local models
    python scripts/setup_models.py --check  # Verify models are available
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LOCAL_MODELS = [
    {
        "name": "Qwen 2.5 1.5B (Router)",
        "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "ram_gb": 1.0,
    },
    {
        "name": "DeepSeek R1 Distill 7B (Math)",
        "model_id": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "ram_gb": 5.0,
    },
    {
        "name": "Qwen 2.5 7B (General)",
        "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "ram_gb": 5.0,
    },
    {
        "name": "DeepSeek Coder V2 16B (Code)",
        "model_id": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
        "ram_gb": 10.0,
    },
]


def download_model(model_id: str) -> bool:
    print(f"  Downloading {model_id}...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"from mlx_lm import load; load('{model_id}')"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            print(f"  OK: {model_id}")
            return True
        else:
            print(f"  FAILED: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {model_id}")
        return False


def check_models() -> None:
    print("Checking local models...\n")
    total_ram = 0.0
    for model in LOCAL_MODELS:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir_name = "models--" + model["model_id"].replace("/", "--")
        exists = (cache_dir / model_dir_name).exists()
        status = "READY" if exists else "NOT DOWNLOADED"
        print(f"  [{status}] {model['name']} ({model['model_id']})")
        if exists:
            total_ram += model["ram_gb"]

    print(f"\nEstimated RAM if all loaded: {sum(m['ram_gb'] for m in LOCAL_MODELS):.1f} GB")
    print(f"Currently downloaded models need: {total_ram:.1f} GB")


def download_all() -> None:
    print("Downloading MLX models...\n")
    total_ram = sum(m["ram_gb"] for m in LOCAL_MODELS)
    print(f"Total estimated RAM when loaded: {total_ram:.1f} GB")
    print(f"Models to download: {len(LOCAL_MODELS)}\n")

    success = 0
    for model in LOCAL_MODELS:
        print(f"\n[{model['name']}] (~{model['ram_gb']} GB RAM)")
        if download_model(model["model_id"]):
            success += 1

    print(f"\n{'='*40}")
    print(f"Downloaded: {success}/{len(LOCAL_MODELS)} models")


def main():
    parser = argparse.ArgumentParser(description="Setup MLX models for local inference")
    parser.add_argument("--check", action="store_true", help="Check model availability")
    args = parser.parse_args()

    if args.check:
        check_models()
    else:
        download_all()


if __name__ == "__main__":
    main()
