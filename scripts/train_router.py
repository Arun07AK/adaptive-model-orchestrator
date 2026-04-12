#!/usr/bin/env python3
"""Fine-tune Qwen 2.5 1.5B as a domain classifier for the router.

Usage:
    python scripts/train_router.py --epochs 3 --output models/router-classifier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_training_data() -> list[dict]:
    from src.orchestrator.analyzer import _MATH_SUBJECTS, _CODE_SUBJECTS, _REASONING_SUBJECTS

    training_data = []

    for subject in _MATH_SUBJECTS:
        training_data.append({"text": f"Subject: {subject.replace('_', ' ')}", "label": "math"})

    for subject in _CODE_SUBJECTS:
        training_data.append({"text": f"Subject: {subject.replace('_', ' ')}", "label": "code"})

    for subject in _REASONING_SUBJECTS:
        training_data.append({"text": f"Subject: {subject.replace('_', ' ')}", "label": "reasoning"})

    all_known = _MATH_SUBJECTS | _CODE_SUBJECTS | _REASONING_SUBJECTS
    mmlu_subjects = [
        "anatomy", "business_ethics", "clinical_knowledge", "college_biology",
        "college_medicine", "conceptual_physics", "global_facts",
        "high_school_biology", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "high_school_us_history", "high_school_world_history",
        "human_aging", "human_sexuality", "international_law", "jurisprudence",
        "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "prehistory", "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology",
        "world_religions",
    ]
    for subject in mmlu_subjects:
        if subject not in all_known:
            training_data.append({"text": f"Subject: {subject.replace('_', ' ')}", "label": "general"})

    return training_data


def main():
    parser = argparse.ArgumentParser(description="Train router domain classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=str, default="models/router-classifier")
    args = parser.parse_args()

    data = build_training_data()
    print(f"Training data: {len(data)} examples")
    print(f"Distribution: { {l: sum(1 for d in data if d['label'] == l) for l in set(d['label'] for d in data)} }")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nTraining data saved to {output_dir / 'training_data.json'}")
    print("NOTE: Full fine-tuning requires HuggingFace Transformers + MLX.")
    print("For now, the keyword-based analyzer handles benchmark routing well.")


if __name__ == "__main__":
    main()
