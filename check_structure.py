#!/usr/bin/env python3
"""
Quick structural checks for train/test CSVs and an optional NPZ bundle.

What it does (lightweight):
- Compares column lists (order + names) between train and test CSV (header only).
- Optionally samples a few rows to compare inferred dtypes.
- If an NPZ is provided, checks that:
  * keys exist (X, y, numeric_cols, classes)
  * X/y shapes are consistent with numeric_cols
  * numeric_cols match the CSV numeric columns inferred from a sample (excluding the label column)
  * y indices are within range of classes

Usage:
    python check_structure.py --train fusion_train_smart6.csv --test fusion_test_smart6.csv [--npz preprocessed_dataset.npz]

This script only reads headers and a small sample (default 2k rows) to avoid loading huge files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_header(path: Path) -> list[str]:
    df0 = pd.read_csv(path, nrows=0, low_memory=False)
    return [str(c) for c in df0.columns]


def read_sample_dtypes(path: Path, *, nrows: int) -> dict[str, str]:
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    return {str(c): str(df[c].dtype) for c in df.columns}


def read_numeric_columns(path: Path, *, nrows: int, label_col: str = "Label") -> list[str]:
    """Infer numeric columns from a small sample, dropping the label if present."""
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    return [str(c) for c in numeric_cols]


def compare_csv(train: Path, test: Path, sample_rows: int) -> tuple[bool, list[str]]:
    messages: list[str] = []
    train_cols = read_header(train)
    test_cols = read_header(test)

    if train_cols == test_cols:
        messages.append("OK Columns match (order + names).")
        cols_match = True
    else:
        messages.append("ERROR Columns differ between train and test.")
        missing_in_test = [c for c in train_cols if c not in test_cols]
        missing_in_train = [c for c in test_cols if c not in train_cols]
        if missing_in_test:
            messages.append(f"   In train only: {missing_in_test}")
        if missing_in_train:
            messages.append(f"   In test only:  {missing_in_train}")
        cols_match = False

    # Optional dtype check on a small sample
    try:
        train_dt = read_sample_dtypes(train, nrows=sample_rows)
        test_dt = read_sample_dtypes(test, nrows=sample_rows)
        dtype_mismatches = []
        for c in set(train_dt) | set(test_dt):
            dt_tr = train_dt.get(c)
            dt_te = test_dt.get(c)
            if dt_tr != dt_te:
                dtype_mismatches.append((c, dt_tr, dt_te))
        if dtype_mismatches:
            messages.append("ERROR Dtype differences (sample-based):")
            for c, dt_tr, dt_te in dtype_mismatches:
                messages.append(f"   {c}: train={dt_tr}, test={dt_te}")
        else:
            messages.append("OK Dtypes look consistent on sampled rows.")
    except Exception as exc:
        messages.append(f"WARN Skipped dtype sampling: {exc}")

    return cols_match, messages


def check_npz(npz_path: Path, csv_numeric_cols: list[str]) -> list[str]:
    messages: list[str] = []
    data = np.load(npz_path, allow_pickle=True)
    required_keys = {"X", "y", "numeric_cols", "classes"}
    missing = required_keys - set(data.files)
    if missing:
        messages.append(f"ERROR NPZ missing keys: {sorted(missing)}")
        return messages

    X = data["X"]
    y = data["y"]
    numeric_cols = [str(c) for c in data["numeric_cols"]]
    classes = data["classes"]

    # Basic shape checks
    if X.shape[0] != y.shape[0]:
        messages.append(f"ERROR X rows ({X.shape[0]}) != y rows ({y.shape[0]}).")
    else:
        messages.append(f"OK X/y row counts match: {X.shape[0]}.")

    if X.shape[1] != len(numeric_cols):
        messages.append(
            f"ERROR X columns ({X.shape[1]}) != numeric_cols length ({len(numeric_cols)})."
        )
    else:
        messages.append(f"OK X column count matches numeric_cols ({X.shape[1]}).")

    if y.size > 0 and (y.min() < 0 or y.max() >= len(classes)):
        messages.append(
            f"ERROR y indices out of range for classes (min={y.min()}, max={y.max()}, classes={len(classes)})."
        )
    else:
        messages.append("OK y indices fit within classes.")

    # Compare numeric_cols vs CSV numeric columns
    csv_numeric = csv_numeric_cols
    mismatch_npz = [c for c in numeric_cols if c not in csv_numeric]
    missing_npz = [c for c in csv_numeric if c not in numeric_cols]
    if not mismatch_npz and not missing_npz:
        messages.append("OK numeric_cols align with CSV numeric columns (excluding label).")
    else:
        if mismatch_npz:
            messages.append(f"ERROR In NPZ numeric_cols but not CSV: {mismatch_npz}")
        if missing_npz:
            messages.append(f"ERROR In CSV numeric columns but missing in NPZ: {missing_npz}")

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Check structural consistency of train/test CSVs and optional NPZ.")
    parser.add_argument("--train", required=True, type=Path, help="Train CSV path")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV path")
    parser.add_argument("--npz", type=Path, help="Optional NPZ path to validate against train")
    parser.add_argument("--sample-rows", type=int, default=2000, help="Rows to sample for dtype/numeric inference")
    args = parser.parse_args()

    if not args.train.exists() or not args.test.exists():
        print("Train or test CSV does not exist.", file=sys.stderr)
        return 1

    cols_match, msgs = compare_csv(args.train, args.test, args.sample_rows)
    print("\n".join(msgs))

    if args.npz:
        if not args.npz.exists():
            print(f"NPZ not found: {args.npz}", file=sys.stderr)
            return 1
        csv_numeric_cols = read_numeric_columns(args.train, nrows=args.sample_rows)
        npz_msgs = check_npz(args.npz, csv_numeric_cols)
        print("\n[NPZ]")
        print("\n".join(npz_msgs))

    return 0 if cols_match else 2


if __name__ == "__main__":
    sys.exit(main())
