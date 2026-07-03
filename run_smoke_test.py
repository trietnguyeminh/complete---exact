#!/usr/bin/env python3
"""Lightweight smoke test for the curated EXACT 2026 archive.

This script does not run Qwen/vLLM/Z3 inference. It validates that the
portfolio-critical audit artifacts are present and readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "06b_combined_summary.json",
    "06b_smoke_gate.json",
    "SUMMARY.json",
    "MANIFEST.json",
    "build_v6_finetune.py",
    "build_v6_inference.py",
    "z3_entailment_poc.py",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Path to complete---exact repository")
    args = parser.parse_args()

    root = Path(args.root)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]

    print("EXACT 2026 archive smoke test")
    print(f"Root: {root.resolve()}")

    if missing:
        print("\nMissing required files:")
        for name in missing:
            print(f"  - {name}")
        return 1

    combined = load_json(root / "06b_combined_summary.json")
    summary = load_json(root / "SUMMARY.json")

    print("\nKey files: OK")

    # These keys may vary slightly across report versions, so print defensively.
    print("\n06b_combined_summary.json")
    for key in ["dataset", "smoke_size", "old_correct", "new_correct", "accuracy", "micro_f1", "macro7_f1", "weighted_f1", "prompt_identity_match", "gate"]:
        if key in combined:
            print(f"  {key}: {combined[key]}")

    print("\nSUMMARY.json")
    for key in ["version", "replay_cases", "fired", "correct_when_fired", "precision_when_fired", "coverage", "gate"]:
        if key in summary:
            print(f"  {key}: {summary[key]}")

    print("\nSmoke test passed: archive artifacts are readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
