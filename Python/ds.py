#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidateddata_CORRECTED.py
=============================
Data Consolidation GUI - Production build (score-aware chunking + napkin-math graphs)

Key points:
- Reads only the first N rows of each file by default (N=1000) for TON_IoT + every CIC CSV.
  Set FULL_RUN=1 in env to process full files.
- Label column is fixed (case-insensitive detection -> renamed to "Label") for compatibility
  with downstream scripts expecting df["Label"].
- Worker cap is adaptive and explains WHY it cannot add workers (RAM / score / file_count / caps).
- Chunk size is adaptive based on RAM + overall score + worker cap, with explanation shown in UI.

Outputs (same names for pipeline compatibility):
- fusion_train_smart4.csv
- fusion_test_smart4.csv
- preprocessed_dataset.npz
"""

from __future__ import annotations

import gc
import os
import time
import traceback
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
import psutil
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# CONFIG
# ============================================================
MAX_RAM_PERCENT = 90
MIN_CHUNK_SIZE = 50_000
MAX_CHUNK_SIZE = 750_000
MAX_THREADS = 12
TARGET_FLOAT_DTYPE = np.float32
PROGRESS_TITLE = "Overall progress"

# Default behavior requested: only read first 1000 rows (TON_IoT + each CIC file).
DEFAULT_SAMPLE_ROWS = 1000

# Override: FULL_RUN=1 processes full files.
FULL_RUN = os.environ.get("FULL_RUN", "0").strip() in {"1", "true", "True", "YES", "yes"}


# ============================================================
# UTIL STRUCTS
# ============================================================
@dataclass
class CapDecision:
    cap: int
    desired: int
    reasons: list[str]
    score: float
    ram: float
    cpu: float
    file_count: int


# ============================================================
# CACHE + MONITOR
# ============================================================
class SmartCache:
    """Tiny stats-only cache to avoid hoarding data in memory."""

    def __init__(self, max_items: int = 0) -> None:
        self.max_items = max_items
        self.cache: dict[Any, Any] = {}
        self.order = deque()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.hits += 1
                try:
                    self.order.remove(key)
                except ValueError:
                    pass
                self.order.append(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        if self.max_items <= 0:
            self.misses += 1
            return

        with self.lock:
            self.cache[key] = value
            self.order.append(key)
            while len(self.order) > self.max_items:
                old = self.order.popleft()
                self.cache.pop(old, None)

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.order.clear()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0.0
        return {"items": len(self.cache), "hit_rate": hit_rate}


class AdvancedMonitor:
    """Minimal monitor capturing peaks and counters."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.ram_peak = 0.0
        self.cpu_peak = 0.0
        self.total_data_loaded = 0
        self.total_files = 0
        self.lock = threading.Lock()

    def record_metric(self) -> tuple[float, float]:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.0)
        with self.lock:
            self.ram_peak = max(self.ram_peak, vm.percent)
            self.cpu_peak = max(self.cpu_peak, cpu)
        return vm.percent, cpu

    def track_data(self, rows: int) -> None:
        with self.lock:
            self.total_data_loaded += int(rows)

    def track_file(self) -> None:
        with self.lock:
            self.total_files += 1

    def get_stats(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "elapsed": elapsed,
                "ram_peak": self.ram_peak,
                "cpu_peak": self.cpu_peak,
                "data_loaded": self.total_data_loaded,
                "files": self.total_files,
            }


# ============================================================
# PROCESSOR
# ============================================================
class OptimizedDataProcessor:
    """Loader/processor tuned for low RAM pressure (score-aware)."""

    def __init__(
        self,
        monitor: AdvancedMonitor,
        cache: SmartCache,
        max_ram_percent: int = MAX_RAM_PERCENT,
        logger: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.monitor = monitor
        self.cache = cache
        self.max_ram_percent = max_ram_percent
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.processed_rows = 0
        self.start_time = time.time()
        self.logger = logger
        self.chunk_history: deque[int] = deque(maxlen=10)

        # Score + concurrency signals (set by GUI)
        self.current_score: float = 100.0
        self.active_worker_cap: int = 1

        # Warm start: start aggressive (max threads + max chunk) then stabilize down with score/RAM.
        self.warm_start_seconds: float = float(os.environ.get("WARM_START_SECONDS", "12"))
        self.warm_start_until: float = time.time() + self.warm_start_seconds

        # UI hooks
        self.gui_hook_bw_sample: Optional[Callable[[int, float], None]] = None
        self.gui_hook_cap_decision: Optional[Callable[[CapDecision], None]] = None
        self.gui_hook_chunk_decision: Optional[Callable[[int, int, str], None]] = None  # old,new,reason

        vm = psutil.virtual_memory()
        ram_gb = vm.total / (1024**3)
        if ram_gb < 8:
            self.max_chunk_size = min(MAX_CHUNK_SIZE, 300_000)
            self.max_threads = min(MAX_THREADS, 4)
        elif ram_gb < 16:
            self.max_chunk_size = min(MAX_CHUNK_SIZE, 500_000)
            self.max_threads = min(MAX_THREADS, 8)
        else:
            self.max_chunk_size = MAX_CHUNK_SIZE
            self.max_threads = MAX_THREADS

        self.start_threads = 4
        self._log(
            f"Adaptive caps -> RAM {ram_gb:.1f} GB | max_chunk_size {self.max_chunk_size:,} | "
            f"max_threads {self.max_threads} | start_threads {self.start_threads}"
        )

    # --- logging helper -------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger(msg, "DEBUG")
            except Exception:
                pass

    # --- helpers ---------------------------------------------------------
    def _wait_for_ram(self, target: float | None = None) -> None:
        target = target or self.max_ram_percent
        while psutil.virtual_memory().percent >= target:
            gc.collect()
            time.sleep(0.2)

    def _estimate_chunk_size(self, filepath: str, workers_hint: int = 1) -> int:
        vm = psutil.virtual_memory()

        # Warm start: go BIG early if the machine is clearly safe.
        if time.time() < getattr(self, "warm_start_until", 0.0) and vm.percent < 50:
            chunk = int(self.max_chunk_size)
            self._log(f"WARMSTART chunk -> {chunk:,} rows (RAM {vm.percent:.1f}%, workers_hint {workers_hint})")
            return chunk

        # Headroom budget: also account for parallelism (more workers => smaller safe chunk).
        parallel_penalty = 1.0 + 0.12 * max(workers_hint - 1, 0)
        headroom = max((self.max_ram_percent / 100 * vm.total) - vm.used, vm.available * 0.5)
        budget = max((headroom * 0.6) / parallel_penalty, 64 * 1024 * 1024)

        try:
            sample = pd.read_csv(filepath, nrows=2000, low_memory=False, memory_map=True)
            sample = self._optimize_dtypes(sample)
            per_row = sample.memory_usage(deep=True).sum() / max(len(sample), 1)
            est = int(budget / max(per_row, 1))
        except Exception:
            est = self.max_chunk_size // 2

        # Dynamic safety based on current RAM
        if vm.percent > 60:
            est = int(est * 0.6)
        if vm.percent > 70:
            est = int(est * 0.5)

        chunk = max(self.min_chunk_size, min(self.max_chunk_size, est))
        self._log(
            f"Chunk size estimate for {os.path.basename(filepath)} -> {chunk:,} rows "
            f"(budget {budget/1e6:.1f} MB, RAM {vm.percent:.1f}%, workers_hint {workers_hint})"
        )
        return chunk

    def _tune_chunk_size(self, current_size: int, *, workers_cap: int) -> tuple[int, str]:
        """
        Score-aware chunk sizing:
        - Low score or high RAM => shrink
        - High score + low RAM => grow
        - More workers => shrink a bit (parallel memory pressure)
        """
        vm = psutil.virtual_memory()
        score = float(getattr(self, "current_score", 100.0))
        workers_cap = max(1, int(workers_cap))

        warm = (time.time() < getattr(self, "warm_start_until", 0.0))

        # Base multiplier from RAM
        mult = 1.0
        # Warm start: grow aggressively early, then let score/RAM shrink it.
        if warm and vm.percent < 55 and score > 80:
            mult *= 1.30
        if vm.percent > 75:
            mult *= 0.55
        elif vm.percent > 65:
            mult *= 0.75
        elif vm.percent < 45:
            mult *= 1.15

        # Score influence
        if score >= 90 and vm.percent < 50:
            mult *= 1.25
        elif score >= 80 and vm.percent < 60:
            mult *= 1.10
        elif score <= 60:
            mult *= 0.80
        elif score <= 40:
            mult *= 0.65

        # Parallelism penalty
        mult /= (1.0 + 0.10 * max(workers_cap - 1, 0))

        tuned = int(current_size * mult)
        tuned = max(self.min_chunk_size, min(self.max_chunk_size, tuned))

        reason = (
            f"chunk_tune: {current_size:,}→{tuned:,} | RAM {vm.percent:.1f}% | score {score:.1f} | workers_cap {workers_cap} | mult {mult:.2f}"
        )

        if tuned != current_size:
            self._log(reason)
            self.chunk_history.append(tuned)
            if self.gui_hook_chunk_decision:
                try:
                    self.gui_hook_chunk_decision(int(current_size), int(tuned), reason)
                except Exception:
                    pass

        else:
            # Still push the reason so the UI footer stays fresh.
            if self.gui_hook_chunk_decision:
                try:
                    self.gui_hook_chunk_decision(int(current_size), int(tuned), reason)
                except Exception:
                    pass

        return tuned, reason

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        int_cols = df.select_dtypes(include=["int64", "int32"]).columns
        float_cols = df.select_dtypes(include=["float64"]).columns
        obj_cols = df.select_dtypes(include=["object"]).columns

        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float").astype(TARGET_FLOAT_DTYPE)

        for col in obj_cols:
            # Keep Label as object; we normalize later
            if col.lower() == "label":
                continue
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                numeric_ratio = numeric.notna().mean()
            except Exception:
                numeric = None
                numeric_ratio = 0

            if numeric is not None and numeric_ratio > 0.5:
                df[col] = pd.to_numeric(numeric, downcast="float").astype(TARGET_FLOAT_DTYPE)
                continue

            nunique = df[col].nunique(dropna=False)
            if nunique / max(len(df[col]), 1) < 0.5 or nunique < 50:
                df[col] = df[col].astype("category")

        return df

    # -------- Label normalization (compat downstream) --------
    def normalize_label_column(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
        """Find label column case-insensitive, rename to 'Label', cast to str."""
        label_col = None
        for col in df.columns:
            if str(col).lower() == "label":
                label_col = col
                break

        if label_col is None:
            return df, None

        if label_col != "Label":
            df = df.rename(columns={label_col: "Label"})

        # Ensure string labels, keep NaNs for now (clean() decides)
        try:
            df["Label"] = df["Label"].astype(str)
        except Exception:
            df["Label"] = df["Label"].apply(lambda x: str(x))

        return df, "Label"

    # -------- Concurrency decisions + reasons --------
    def _choose_threads(self, file_count: int, avg_mb: float = 0.0) -> int:
        cpu_threads = psutil.cpu_count(logical=True) or 4
        max_threads = min(cpu_threads, self.max_threads, file_count if file_count else cpu_threads)
        vm = psutil.virtual_memory()

        if vm.total < 8 * 1024**3:
            max_threads = min(max_threads, 4)
        if vm.percent > 70:
            max_threads = min(max_threads, 3)
        elif vm.percent > 60:
            max_threads = min(max_threads, 4)

        if avg_mb and avg_mb > 500:
            max_threads = min(max_threads, 4)
        if avg_mb and avg_mb > 1000:
            max_threads = min(max_threads, 2)

        threads = max(1, min(max_threads, self.start_threads))
        self._log(
            f"Thread choice -> {threads} workers for {file_count} file(s) "
            f"(RAM {vm.percent:.1f}%, CPU threads {cpu_threads}, avg size ~{avg_mb:.1f} MB)"
        )
        return threads

    def _dynamic_thread_cap(self) -> int:
        vm = psutil.virtual_memory()
        score = float(getattr(self, "current_score", 100.0))
        if score < 30 or vm.percent > 80:
            return 2
        if score < 50 or vm.percent > 70:
            return 3
        if score < 70 or vm.percent > 60:
            return 4
        return self.max_threads

    def _decide_worker_cap(self, *, base_threads: int, file_count: int) -> CapDecision:
        score = float(getattr(self, "current_score", 100.0))
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.0)

        desired = int(base_threads)
        reasons: list[str] = []

        # Warm start: start with maximum parallelism if RAM is low (then stabilize down with score).
        if time.time() < getattr(self, "warm_start_until", 0.0) and vm.percent < 50:
            desired = int(self.max_threads)
            reasons.append(f"WARMSTART: RAM<{vm.percent:.1f}% -> desired=max_threads={desired}")

        # Aggressive boost ladder (requested)
        if score > 90 and vm.percent < 40:
            desired = self.max_threads
            reasons.append(f"boost: score>90 & RAM<40 -> desired={desired}")
        elif score > 85 and vm.percent < 50:
            desired = max(desired, int(self.max_threads * 0.8))
            reasons.append("boost: score>85 & RAM<50 -> +80% max_threads")
        elif score > 75 and vm.percent < 60:
            desired = min(self.max_threads, desired + 3)
            reasons.append("boost: score>75 & RAM<60 -> +3")
        elif score > 70:
            desired = min(self.max_threads, desired + 2)
            reasons.append("boost: score>70 -> +2")
        else:
            reasons.append("no boost: score low or RAM high")

        # Hard constraints
        hard = []
        hard_cap = self._dynamic_thread_cap()
        if hard_cap < desired:
            hard.append(f"dyn_cap={hard_cap} (score/RAM)")
        if file_count and desired > file_count:
            hard.append(f"file_count={file_count}")
        if desired > self.max_threads:
            hard.append(f"max_threads={self.max_threads}")

        cap = min(desired, hard_cap, self.max_threads, file_count if file_count else desired)

        if cap < desired:
            reasons.append("blocked: " + ", ".join(hard) if hard else "blocked by cap")

        # Update processor signal for chunking
        self.active_worker_cap = int(cap)

        return CapDecision(
            cap=int(cap),
            desired=int(desired),
            reasons=reasons[:],
            score=float(score),
            ram=float(vm.percent),
            cpu=float(cpu),
            file_count=int(file_count),
        )

    # -------- File loading --------
    def _load_single_file(
        self, filepath: str, sample_rows: int | None = None
    ) -> tuple[pd.DataFrame | None, str, float, float]:
        """returns: (df_or_none, filepath, elapsed_seconds, bytes_used)"""
        self._wait_for_ram(self.max_ram_percent - 2)
        try:
            start = time.time()
            if sample_rows:
                df = pd.read_csv(filepath, nrows=sample_rows, low_memory=False, memory_map=True)
            else:
                df = pd.read_csv(filepath, low_memory=False, memory_map=True)
            df = self._optimize_dtypes(df)
            df, _ = self.normalize_label_column(df)

            self.monitor.record_metric()

            bytes_used = float(df.memory_usage(deep=True).sum())
            elapsed = time.time() - start

            mem_mb = bytes_used / (1024 * 1024)
            self._log(f"Loaded: {os.path.basename(filepath)} ({len(df):,} rows, ~{mem_mb:.1f} MB, {elapsed:.2f}s)")
            return df, filepath, elapsed, bytes_used
        except Exception:
            return None, filepath, 0.0, 0.0

    # --- public API ------------------------------------------------------
    def load_toniot_optimized(
        self,
        filepath: str,
        callback=None,
        sample_rows: int | None = None,
        workers_hint: int = 1,
    ) -> pd.DataFrame:
        """
        TON_IoT loader:
        - sample_rows != None => read only sample_rows.
        - else => chunked read with score-aware chunk tuning.
        """
        chunk_size = self._estimate_chunk_size(filepath, workers_hint=workers_hint)

        if sample_rows:
            df = pd.read_csv(filepath, nrows=sample_rows, low_memory=False, memory_map=True)
            df = self._optimize_dtypes(df)
            df, _ = self.normalize_label_column(df)
            self._log(
                f"TON_IoT sample load -> {len(df):,} rows (~{df.memory_usage(deep=True).sum()/(1024*1024):.1f} MB)"
            )
            return df

        reader = pd.read_csv(filepath, chunksize=chunk_size, low_memory=False, memory_map=True)
        chunks: list[pd.DataFrame] = []
        processed = 0
        self._log(f"Starting TON_IoT load with chunk_size={chunk_size:,}")

        for idx, chunk in enumerate(reader, 1):
            self._wait_for_ram(self.max_ram_percent - 2)
            chunk = self._optimize_dtypes(chunk)
            chunk, _ = self.normalize_label_column(chunk)
            chunks.append(chunk)

            processed += len(chunk)
            self.processed_rows += len(chunk)
            self.monitor.record_metric()
            self.monitor.track_data(len(chunk))
            chunk_mem = chunk.memory_usage(deep=True).sum() / (1024 * 1024)
            self._log(f"TON_IoT chunk {idx}: {len(chunk):,} rows (~{chunk_mem:.1f} MB)")

            if callback:
                # crude but stable progress estimation
                progress = min(100.0, (idx * chunk_size) / max(processed, chunk_size) * 100)
                callback(idx, len(chunk), progress, (idx - 1) % 2)

            if len(chunks) >= 4 and psutil.virtual_memory().percent > self.max_ram_percent - 5:
                self._log(f"Concatenating buffered TON_IoT chunks at idx {idx} to free memory")
                chunks = [pd.concat(chunks, ignore_index=True, copy=False)]
                gc.collect()

            # score-aware adapt chunk size
            chunk_size, _ = self._tune_chunk_size(chunk_size, workers_cap=max(1, workers_hint))
            reader.chunksize = chunk_size

        result = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame()
        result, _ = self.normalize_label_column(result)
        gc.collect()
        res_mem = result.memory_usage(deep=True).sum() / (1024 * 1024) if not result.empty else 0
        self._log(f"TON_IoT loaded -> {len(result):,} rows (~{res_mem:.1f} MB)")
        return result

    def load_cic_optimized(
        self,
        folder: str,
        callback=None,
        threads_hook=None,
        sample_rows: int | None = None,
    ) -> tuple[list[pd.DataFrame], int, int]:
        cic_files = []
        sizes_mb = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv"):
                    path = os.path.join(root, f)
                    cic_files.append(path)
                    try:
                        sizes_mb.append(os.path.getsize(path) / (1024 * 1024))
                    except OSError:
                        sizes_mb.append(0)
        cic_files.sort()
        self._log(f"Discovered {len(cic_files)} CIC file(s) in {folder}")

        avg_mb = sum(sizes_mb) / len(sizes_mb) if sizes_mb else 0.0
        self._log(f"Avg CIC file size ~{avg_mb:.1f} MB (max ~{max(sizes_mb) if sizes_mb else 0:.1f} MB)")
        base_threads = self._choose_threads(len(cic_files), avg_mb=avg_mb)
        if threads_hook:
            threads_hook(base_threads)

        dfs_cic: list[pd.DataFrame] = []
        failures: list[str] = []
        active = 0
        cap_prev = base_threads

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {}

            for path in cic_files:
                while True:
                    decision = self._decide_worker_cap(base_threads=base_threads, file_count=len(cic_files))
                    cap_dynamic = int(decision.cap)

                    if self.gui_hook_cap_decision:
                        try:
                            self.gui_hook_cap_decision(decision)
                        except Exception:
                            pass

                    if cap_dynamic != cap_prev:
                        self._log(f"[CIC] worker cap changed: {cap_prev} -> {cap_dynamic} | " + " ; ".join(decision.reasons))
                        cap_prev = cap_dynamic

                    if active < cap_dynamic:
                        break
                    time.sleep(0.08)

                future = executor.submit(self._load_single_file, path, sample_rows)
                futures[future] = path
                active += 1

            for done_idx, future in enumerate(as_completed(futures), 1):
                df, path, elapsed, bytes_used = future.result()
                active = max(active - 1, 0)

                if df is not None:
                    dfs_cic.append(df)
                    self.monitor.track_file()
                    self.monitor.track_data(len(df))
                    self._log(f"[CIC] done {done_idx}/{len(cic_files)} -> {os.path.basename(path)} ({len(df):,} rows)")

                    if bytes_used > 0 and elapsed > 0 and self.gui_hook_bw_sample:
                        try:
                            gbps = (bytes_used / max(elapsed, 1e-6)) / 1e9
                            self.gui_hook_bw_sample(int(cap_prev), float(gbps))
                        except Exception:
                            pass

                    if done_idx % 3 == 0:
                        mem_mb = sum(x.memory_usage(deep=True).sum() for x in dfs_cic) / (1024 * 1024)
                        self._log(f"[CIC] accumulated {done_idx} files, ~{mem_mb:.1f} MB in memory")
                else:
                    failures.append(path)
                    self._log(f"[CIC] failed to load {os.path.basename(path)} (returned None)")

                if callback:
                    progress = (done_idx / max(len(cic_files), 1)) * 100
                    callback(
                        done_idx,
                        len(cic_files),
                        progress,
                        (done_idx - 1) % max(int(cap_prev), 1),
                        f"Loaded {os.path.basename(path)}",
                    )

                if psutil.virtual_memory().percent > self.max_ram_percent:
                    gc.collect()

        gc.collect()
        if failures:
            self._log(f"[CIC] missing/failed files: {len(failures)} -> {[os.path.basename(f) for f in failures]}")
        return dfs_cic, len(cic_files), base_threads

    def merge_optimized(self, dfs_list: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs_list:
            return pd.DataFrame()

        mem_before = sum(df.memory_usage(deep=True).sum() for df in dfs_list) / (1024 * 1024)
        rows_before = sum(len(df) for df in dfs_list)
        cols_before = max(len(df.columns) for df in dfs_list)
        self._log(
            f"Merging {len(dfs_list)} dataframe(s) "
            f"(rows {rows_before:,}, cols ~{cols_before}, ~{mem_before:.1f} MB pre-merge)"
        )
        result = pd.concat(dfs_list, ignore_index=True, copy=False)
        result = self._optimize_dtypes(result)
        result, _ = self.normalize_label_column(result)

        mem_after = result.memory_usage(deep=True).sum() / (1024 * 1024)
        dtypes_summary = result.dtypes.value_counts().to_dict()
        self._log(
            f"Merged dataframe -> {len(result):,} rows, {len(result.columns)} cols "
            f"(~{mem_after:.1f} MB, dtypes {dtypes_summary})"
        )
        gc.collect()
        return result

    def clean_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df, label_col = self.normalize_label_column(df)

        if label_col == "Label":
            before = len(df)
            df = df.dropna(subset=["Label"])
            after = len(df)
            self._log(f"Label fixed: dropped NaN labels -> {before:,}→{after:,}")
        else:
            self._log("No Label column found during clean().")

        gc.collect()
        return df

    def split_optimized(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split with robust Label handling:
        - If Label exists: stratified split
        - If not: random split
        """
        df, label_col = self.normalize_label_column(df)

        self._log(f"[SPLIT] Dataframe shape: {df.shape}")
        self._log(f"[SPLIT] Available columns: {list(df.columns)}")

        if label_col != "Label":
            self._log("[SPLIT] No Label column found - using random 60/40 split (no stratification)")
            n = len(df)
            train_size = int(n * 0.6)
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

        # Stratified split
        try:
            label_counts = df["Label"].value_counts(dropna=False)
            self._log(f"[SPLIT] Label distribution: {label_counts.to_dict()}")
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
            for train_idx, test_idx in sss.split(df, df["Label"]):
                self._log(f"[SPLIT] Stratified split OK: {len(train_idx)} train, {len(test_idx)} test")
                return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        except Exception as e:
            self._log(f"[SPLIT] Stratified split failed: {e}. Falling back to random split.")
            n = len(df)
            train_size = int(n * 0.6)
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

        return df.copy(), df.copy()

    def get_eta(self, processed: int, total: int) -> timedelta:
        if processed <= 0 or total <= 0:
            return timedelta(0)
        elapsed = time.time() - self.start_time
        rate = processed / max(elapsed, 1e-6)
        remaining = (total - processed) / max(rate, 1e-6)
        return timedelta(seconds=int(remaining))


# ============================================================
# GUI
# ============================================================
class ConsolidationGUIEnhanced:
    """Tkinter UI with live dashboards (scores + napkin graphs + explanations)."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Data Consolidation - CORRECTED (score-aware chunking)")
        self.root.geometry("1400x980")

        self.toniot_file: str | None = None
        self.cic_dir: str | None = None
        self.is_running = False
        self.start_time: float | None = None

        self.monitor = AdvancedMonitor()
        self.cache = SmartCache(max_items=0)
        self.processor = OptimizedDataProcessor(self.monitor, self.cache, logger=self.log)
        self.processor.gui_hook_bw_sample = self._on_bw_sample
        self.processor.gui_hook_cap_decision = self._on_cap_decision
        self.processor.gui_hook_chunk_decision = self._on_chunk_decision

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Idle")
        self.ram_var = tk.StringVar(value="-- %")
        self.cpu_var = tk.StringVar(value="-- %")
        self.rows_var = tk.StringVar(value="Rows: 0")
        self.files_var = tk.StringVar(value="Files: 0")

        self.thread_bars: dict[int, dict[str, tk.Variable | ttk.Label]] = {}
        self.progress_tick = 0
        self.progress_last_detail = -1
        self.last_progress_refresh = time.time()
        self.last_thread_refresh = time.time()

        self.current_score = 100.0
        self._score_state = {"ram": None, "cpu": None, "overall": None}

        # decision memories for UI
        self._last_cap_decision: Optional[CapDecision] = None
        self._last_chunk_reason: str = "chunk_tune: --"
        self._last_chunk_size: int = self.processor.min_chunk_size

        # napkin-math data
        self._bw_lock = threading.Lock()
        self._bw_by_threads: deque[tuple[float, int, float]] = deque(maxlen=250)
        self._plateau: deque[tuple[float, float]] = deque(maxlen=250)
        self._bw_by_size_seq: list[tuple[float, float]] = []
        self._bw_by_size_rand: list[tuple[float, float]] = []
        self._seqrand_last: dict[str, float] = {}
        self._last_bench = 0.0
        self._bench_running = False

        self.setup_ui()
        self.start_monitoring_loop()

    # ========================================================
    # UI SETUP
    # ========================================================
    def setup_ui(self) -> None:
        file_frame = ttk.LabelFrame(self.root, text="Input")
        file_frame.pack(fill="x", padx=10, pady=8)

        ttk.Label(file_frame, text="TON_IoT CSV:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.toniot_path_label = ttk.Label(file_frame, text="No file selected")
        self.toniot_path_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.select_toniot).grid(row=0, column=2, sticky="e", padx=5, pady=5)

        ttk.Label(file_frame, text="CIC Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.cic_path_label = ttk.Label(file_frame, text="No folder selected")
        self.cic_path_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.select_cic).grid(row=1, column=2, sticky="e", padx=5, pady=5)

        mode = "FULL_RUN=1 (full files)" if FULL_RUN else f"SAMPLE (first {DEFAULT_SAMPLE_ROWS} rows/file)"
        ttk.Label(file_frame, text=f"Mode: {mode}", foreground="#2c3e50").grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky="w")

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.start_button = ttk.Button(control_frame, text="▶ Start", command=self.start_consolidation)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(control_frame, text="⏹ Stop", command=self.stop_consolidation, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5)

        alerts_frame = ttk.LabelFrame(self.root, text="Alerts")
        alerts_frame.pack(fill="both", padx=10, pady=5)
        self.alert_canvas = tk.Canvas(alerts_frame, height=100)
        alerts_scroll = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.alert_canvas.yview)
        self.alert_canvas.configure(yscrollcommand=alerts_scroll.set)
        self.alert_canvas.pack(side="left", fill="both", expand=True)
        alerts_scroll.pack(side="right", fill="y")
        self.alert_inner = ttk.Frame(self.alert_canvas)
        self.alert_canvas.create_window((0, 0), window=self.alert_inner, anchor="nw")
        self.alert_inner.bind("<Configure>", lambda e: self.alert_canvas.configure(scrollregion=self.alert_canvas.bbox("all")))

        monitor_frame = ttk.LabelFrame(self.root, text="Monitoring")
        monitor_frame.pack(fill="x", padx=10, pady=8)

        ttk.Label(monitor_frame, text="RAM:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Label(monitor_frame, textvariable=self.ram_var).grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ttk.Label(monitor_frame, text="CPU:").grid(row=0, column=2, sticky="w", padx=5, pady=3)
        ttk.Label(monitor_frame, textvariable=self.cpu_var).grid(row=0, column=3, sticky="w", padx=5, pady=3)

        ttk.Label(monitor_frame, textvariable=self.rows_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=3)
        ttk.Label(monitor_frame, textvariable=self.files_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=5, pady=3)

        ttk.Label(monitor_frame, text="Status:").grid(row=2, column=0, sticky="w", padx=5, pady=3)
        ttk.Label(monitor_frame, textvariable=self.status_var).grid(row=2, column=1, columnspan=3, sticky="w", padx=5, pady=3)

        ttk.Label(monitor_frame, text=PROGRESS_TITLE + ":").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        self.progress_bar = ttk.Progressbar(monitor_frame, maximum=100, variable=self.progress_var)
        self.progress_bar.grid(row=3, column=1, columnspan=3, sticky="ew", padx=5, pady=6)
        self.progress_detail = ttk.Label(monitor_frame, text="", anchor="w")
        self.progress_detail.grid(row=4, column=0, columnspan=4, sticky="w", padx=5, pady=2)
        monitor_frame.columnconfigure(1, weight=1)
        monitor_frame.columnconfigure(3, weight=1)

        threads_frame = ttk.LabelFrame(self.root, text="Thread progress")
        threads_frame.pack(fill="x", padx=10, pady=5)
        self.thread_container = ttk.Frame(threads_frame)
        self.thread_container.pack(fill="x", padx=5, pady=5)

        # score gauges
        gauges = ttk.LabelFrame(self.root, text="Scores + Controls (with reasons)")
        gauges.pack(fill="x", padx=10, pady=5)

        self.ram_canvas = tk.Canvas(gauges, width=260, height=75, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        self.ram_canvas.grid(row=0, column=0, padx=5, pady=5)
        self.cpu_canvas = tk.Canvas(gauges, width=260, height=75, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        self.cpu_canvas.grid(row=0, column=1, padx=5, pady=5)
        self.workers_canvas = tk.Canvas(gauges, width=360, height=75, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        self.workers_canvas.grid(row=0, column=2, padx=5, pady=5)
        self.chunk_canvas = tk.Canvas(gauges, width=520, height=75, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        self.chunk_canvas.grid(row=0, column=3, padx=5, pady=5)

        gauges.columnconfigure(0, weight=1)
        gauges.columnconfigure(1, weight=1)
        gauges.columnconfigure(2, weight=1)
        gauges.columnconfigure(3, weight=2)

        # dashboard
        dash_frame = ttk.Frame(self.root)
        dash_frame.pack(fill="both", padx=10, pady=5, expand=True)

        napkin_frame = ttk.LabelFrame(dash_frame, text="Napkin-math style throughput (GB/s) + explanations")
        napkin_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.napkin_canvas_threads = tk.Canvas(napkin_frame, width=520, height=140, bg="#f5f5f5",
                                               highlightthickness=1, highlightbackground="#ccc")
        self.napkin_canvas_threads.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.napkin_canvas_sizes = tk.Canvas(napkin_frame, width=520, height=140, bg="#f5f5f5",
                                             highlightthickness=1, highlightbackground="#ccc")
        self.napkin_canvas_sizes.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.napkin_canvas_seqrand = tk.Canvas(napkin_frame, width=520, height=140, bg="#f5f5f5",
                                               highlightthickness=1, highlightbackground="#ccc")
        self.napkin_canvas_seqrand.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.napkin_canvas_plateau = tk.Canvas(napkin_frame, width=520, height=140, bg="#f5f5f5",
                                               highlightthickness=1, highlightbackground="#ccc")
        self.napkin_canvas_plateau.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        napkin_frame.columnconfigure(0, weight=1)
        napkin_frame.columnconfigure(1, weight=1)

        score_frame = ttk.LabelFrame(dash_frame, text="Live parameters")
        score_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.score_canvas = tk.Canvas(score_frame, width=340, height=320, bg="#f5f5f5",
                                      highlightthickness=1, highlightbackground="#ccc")
        self.score_canvas.pack(fill="both", expand=True, padx=5, pady=5)

        dash_frame.columnconfigure(0, weight=3)
        dash_frame.columnconfigure(1, weight=1)
        dash_frame.rowconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(self.root, text="Logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, font=("Courier", 8))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ========================================================
    # INPUT + LOGS
    # ========================================================
    def select_toniot(self) -> None:
        filepath = filedialog.askopenfilename(title="Select TON_IoT CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filepath:
            self.toniot_file = filepath
            self.toniot_path_label.config(text=filepath)

    def select_cic(self) -> None:
        folder = filedialog.askdirectory(title="Select CIC folder")
        if folder:
            self.cic_dir = folder
            self.cic_path_label.config(text=folder)

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")

        def _append():
            self.log_text.insert(tk.END, f"[{timestamp}] [{level}] {message}\n")
            self.log_text.see(tk.END)

        self.root.after(0, _append)

    def add_alert(self, message: str, level: str = "INFO") -> None:
        colors = {"ERROR": "#c0392b", "WARN": "#d35400", "INFO": "#2980b9", "OK": "#27ae60"}
        color = colors.get(level.upper(), "#2c3e50")

        def _append():
            row = ttk.Frame(self.alert_inner)
            tk.Label(row, text=level.upper(), fg=color, width=8, font=("Arial", 8, "bold")).pack(side="left", padx=4)
            tk.Label(row, text=message, fg=color, wraplength=1100, anchor="w", justify="left").pack(
                side="left", fill="x", expand=True
            )
            row.pack(fill="x", padx=4, pady=1)
            self.alert_canvas.yview_moveto(1.0)
            if len(self.alert_inner.winfo_children()) > 60:
                self.alert_inner.winfo_children()[0].destroy()

        self.root.after(0, _append)

    # ========================================================
    # THREAD UI
    # ========================================================
    def ensure_thread_bars(self, count: int) -> None:
        for tid in range(count):
            if tid in self.thread_bars:
                continue
            var = tk.DoubleVar(value=0)
            row = ttk.Frame(self.thread_container)
            ttk.Label(row, text=f"T{tid}").pack(side="left", padx=4)
            bar = ttk.Progressbar(row, maximum=100, variable=var)
            bar.pack(side="left", fill="x", expand=True, padx=4, pady=2)
            label = ttk.Label(row, text="Idle", width=40)
            label.pack(side="left", padx=4)
            row.pack(fill="x", padx=2, pady=1)
            self.thread_bars[tid] = {"var": var, "label": label}

    def reset_thread_bars(self) -> None:
        for entry in self.thread_bars.values():
            entry["var"].set(0)
            entry["label"].config(text="Idle")

    def update_thread_progress(self, thread_id: int, progress: float, text: str | None = None) -> None:
        entry = self.thread_bars.get(thread_id)
        if not entry:
            return

        def _update():
            entry["var"].set(progress)
            if text:
                entry["label"].config(text=text)

        self.root.after(0, _update)

    # ========================================================
    # SCORE (3 zones) + GAUGES
    # ========================================================
    def _clamp(self, x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def _ewma(self, value: float, prev: float | None, alpha: float = 0.25) -> float:
        return value if prev is None else (prev + alpha * (value - prev))

    def _usage_to_score(
        self,
        usage_percent: float,
        *,
        ok: float,
        danger: float,
        power: float = 2.6,
        floor: float = 0.0,
        ceil: float = 100.0,
    ) -> float:
        u = self._clamp(usage_percent / 100.0)
        ok_n = self._clamp(ok / 100.0)
        danger_n = self._clamp(danger / 100.0)
        if danger_n <= ok_n:
            danger_n = min(0.999, ok_n + 0.05)

        if u <= ok_n:
            safe_drop = 5.0 * (u / max(ok_n, 1e-6))
            return self._clamp(ceil - safe_drop, floor, ceil)

        if u >= danger_n:
            return floor

        t = (u - ok_n) / (danger_n - ok_n)
        curve = (1.0 - t) ** power
        score = 95.0 * curve + floor * (1.0 - curve)
        return self._clamp(score, floor, ceil)

    def _calc_scores(self, ram_percent: float, cpu_percent: float) -> tuple[float, float, float]:
        ram_raw = self._usage_to_score(ram_percent, ok=50.0, danger=float(self.processor.max_ram_percent), power=3.0)
        cpu_raw = self._usage_to_score(cpu_percent, ok=55.0, danger=95.0, power=2.0)

        overall_raw = 0.7 * ram_raw + 0.3 * cpu_raw

        if ram_percent >= (self.processor.max_ram_percent - 3):
            overall_raw *= 0.6
        elif ram_percent >= (self.processor.max_ram_percent - 6):
            overall_raw *= 0.8

        ram_s = self._ewma(ram_raw, self._score_state["ram"], alpha=0.25)
        cpu_s = self._ewma(cpu_raw, self._score_state["cpu"], alpha=0.25)
        overall_s = self._ewma(overall_raw, self._score_state["overall"], alpha=0.25)

        self._score_state["ram"] = ram_s
        self._score_state["cpu"] = cpu_s
        self._score_state["overall"] = overall_s
        return ram_s, cpu_s, overall_s

    def _draw_gauge(self, canvas: tk.Canvas, percent: float, score: float, label: str, footer: str = "") -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        bar_w = int((percent / 100) * (w - 20))
        score_w = int((score / 100) * (w - 20))
        canvas.create_text(10, 10, anchor="w", text=f"{label}: {percent:.1f}% | score {score:.1f}", font=("Arial", 10))
        canvas.create_rectangle(10, 25, 10 + bar_w, 35, fill="#ff6666", outline="")
        canvas.create_rectangle(10, 45, 10 + score_w, 55, fill="#66cc66", outline="")
        if footer:
            canvas.create_text(10, 68, anchor="w", text=footer, font=("Arial", 11), fill="#2c3e50")

    def _draw_workers_panel(self) -> None:
        c = self.workers_canvas
        c.delete("all")
        w = int(c["width"])
        decision = self._last_cap_decision

        if not decision:
            c.create_text(10, 10, anchor="w", text="Workers: --", font=("Arial", 10))
            return

        cap = decision.cap
        desired = decision.desired
        max_t = self.processor.max_threads
        pct = (cap / max(max_t, 1)) * 100.0
        bar_w = int((pct / 100.0) * (w - 20))

        title = f"Workers cap: {cap}/{max_t} (desired {desired})"
        c.create_text(10, 10, anchor="w", text=title, font=("Arial", 10))
        c.create_rectangle(10, 25, 10 + bar_w, 35, fill="#3498db", outline="")

        # Reason: show WHY we can't add a worker (if cap < desired)
        if cap < desired:
            reason = "why not +1: " + (decision.reasons[-1] if decision.reasons else "cap")
        else:
            reason = "reason: " + (" ; ".join(decision.reasons[:2]) if decision.reasons else "ok")
        c.create_text(10, 52, anchor="w", text=reason[:60] + ("..." if len(reason) > 60 else ""), font=("Arial", 11), fill="#2c3e50")

    def _draw_chunk_panel(self) -> None:
        c = self.chunk_canvas
        c.delete("all")
        w = int(c["width"])
        cur = int(self._last_chunk_size or self.processor.min_chunk_size)
        maxc = int(self.processor.max_chunk_size)
        pct = (cur / max(maxc, 1)) * 100.0
        bar_w = int((pct / 100.0) * (w - 20))

        c.create_text(10, 10, anchor="w", text=f"Chunk size: {cur:,} / {maxc:,}", font=("Arial", 10))
        c.create_rectangle(10, 25, 10 + bar_w, 35, fill="#9b59b6", outline="")
        # Footer: explain tuning
        footer = self._last_chunk_reason
        footer = footer.replace("chunk_tune:", "").strip()
        c.create_text(10, 52, anchor="w", text=footer[:90] + ("..." if len(footer) > 90 else ""), font=("Arial", 11), fill="#2c3e50")

    def update_score_graphs(self, ram_percent: float, cpu_percent: float) -> None:
        ram_score, cpu_score, combined = self._calc_scores(ram_percent, cpu_percent)
        self._draw_gauge(self.ram_canvas, ram_percent, ram_score, "RAM")
        self._draw_gauge(self.cpu_canvas, cpu_percent, cpu_score, "CPU")
        self.current_score = combined
        self.processor.current_score = combined

        self._draw_workers_panel()
        self._draw_chunk_panel()

    # ========================================================
    # NAPKIN MATH GRAPHS
    # ========================================================
    def _gbps(self, bytes_count: float, seconds: float) -> float:
        return (bytes_count / max(seconds, 1e-9)) / 1e9

    def _now(self) -> float:
        return time.time()

    def _draw_line_chart(
        self,
        canvas: tk.Canvas,
        points: list[tuple[float, float]],
        title: str,
        footer: str = "",
    ) -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        pad = 16
        footer_h = 18
        canvas.create_text(6, 4, anchor="nw", text=title, font=("Arial", 8, "bold"))

        if len(points) < 2:
            canvas.create_text(w // 2, h // 2, text="pas assez de data", anchor="center", font=("Arial", 10))
            if footer:
                canvas.create_text(6, h - 6, anchor="sw", text=footer, font=("Arial", 11), fill="#2c3e50")
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = 0.0, (max(ys) * 1.05) if ys else 1.0

        def tx(x):
            if x_max == x_min:
                return pad
            return pad + (x - x_min) / (x_max - x_min) * (w - 2 * pad)

        def ty(y):
            usable_h = h - 2 * pad - footer_h
            if y_max == y_min:
                return h - pad - footer_h
            return (h - pad - footer_h) - (y - y_min) / (y_max - y_min) * usable_h

        # axes
        canvas.create_line(pad, h - pad - footer_h, w - pad, h - pad - footer_h, width=1)
        canvas.create_line(pad, h - pad - footer_h, pad, pad, width=1)

        coords = []
        for x, y in points:
            coords += [tx(x), ty(y)]
        if len(coords) >= 4:
            canvas.create_line(*coords, width=2, fill="#2c3e50")

        if footer:
            canvas.create_text(6, h - 6, anchor="sw", text=footer[:110] + ("..." if len(footer) > 110 else ""), font=("Arial", 11), fill="#2c3e50")

    def _draw_two_line_chart(
        self,
        canvas: tk.Canvas,
        a: list[tuple[float, float]],
        b: list[tuple[float, float]],
        title: str,
    ) -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        pad = 16
        canvas.create_text(6, 4, anchor="nw", text=title, font=("Arial", 8, "bold"))

        all_pts = (a or []) + (b or [])
        if len(all_pts) < 2:
            canvas.create_text(w // 2, h // 2, text="pas assez de data", anchor="center", font=("Arial", 10))
            return

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = 0.0, (max(ys) * 1.05) if ys else 1.0

        def tx(x):
            if x_max == x_min:
                return pad
            return pad + (x - x_min) / (x_max - x_min) * (w - 2 * pad)

        def ty(y):
            if y_max == y_min:
                return h - pad
            return (h - pad) - (y - y_min) / (y_max - y_min) * (h - 2 * pad)

        canvas.create_line(pad, h - pad, w - pad, h - pad, width=1)
        canvas.create_line(pad, h - pad, pad, pad, width=1)

        if len(a) >= 2:
            coords = []
            for x, y in a:
                coords += [tx(x), ty(y)]
            canvas.create_line(*coords, width=2, fill="#2c3e50")
        if len(b) >= 2:
            coords = []
            for x, y in b:
                coords += [tx(x), ty(y)]
            canvas.create_line(*coords, width=2, fill="#7f8c8d", dash=(3, 2))

        canvas.create_text(w - pad, pad + 6, anchor="ne", text="solid=seq | dashed=rand", font=("Arial", 11), fill="#2c3e50")

    def _draw_bars(self, canvas: tk.Canvas, bars: list[tuple[str, float]], title: str) -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        pad = 16
        canvas.create_text(6, 4, anchor="nw", text=title, font=("Arial", 8, "bold"))

        if not bars:
            canvas.create_text(w // 2, h // 2, text="pas de data", anchor="center", font=("Arial", 10))
            return

        maxv = max(v for _, v in bars) or 1.0
        n = len(bars)
        bw = (w - 2 * pad) / max(n, 1)

        for i, (name, v) in enumerate(bars):
            x0 = pad + i * bw + 3
            x1 = pad + (i + 1) * bw - 3
            y1 = h - pad
            y0 = y1 - (v / maxv) * (h - 2 * pad)
            canvas.create_rectangle(x0, y0, x1, y1, fill="#3498db")
            canvas.create_text((x0 + x1) / 2, h - pad + 6, text=name, anchor="n", font=("Arial", 11))
            canvas.create_text((x0 + x1) / 2, y0 - 3, text=f"{v:.1f}", anchor="s", font=("Arial", 11))

    def _on_bw_sample(self, threads: int, gbps: float) -> None:
        with self._bw_lock:
            t = self._now()
            self._bw_by_threads.append((t, int(threads), float(gbps)))
            self._plateau.append((t, float(gbps)))

    def _on_cap_decision(self, decision: CapDecision) -> None:
        self._last_cap_decision = decision
        # Auto-add progress bars when worker cap grows (T0,T1,...)
        try:
            self.ensure_thread_bars(int(max(decision.cap, decision.desired)))
        except Exception:
            pass

    def _on_chunk_decision(self, old: int, new: int, reason: str) -> None:
        self._last_chunk_size = int(new)
        self._last_chunk_reason = reason

    def _run_memory_microbench(self) -> None:
        if self._bench_running:
            return
        self._bench_running = True

        def worker():
            try:
                sizes_mb = [0.016, 0.256, 2.0, 16.0, 128.0]
                dtype = np.int32
                seq_points, rand_points = [], []

                for mb in sizes_mb:
                    nbytes = int(mb * 1024 * 1024)
                    n = max(1, nbytes // np.dtype(dtype).itemsize)
                    arr = np.arange(n, dtype=dtype)

                    t0 = time.perf_counter()
                    _ = int(arr.sum())
                    t1 = time.perf_counter()
                    seq_gbps = self._gbps(arr.nbytes, t1 - t0)

                    idx = np.arange(n, dtype=np.int32)
                    np.random.shuffle(idx)
                    t0 = time.perf_counter()
                    _ = int(np.take(arr, idx).sum())
                    t1 = time.perf_counter()
                    rand_gbps = self._gbps(arr.nbytes, t1 - t0)

                    seq_points.append((mb, seq_gbps))
                    rand_points.append((mb, rand_gbps))

                with self._bw_lock:
                    self._bw_by_size_seq = seq_points
                    self._bw_by_size_rand = rand_points
                    if seq_points and rand_points:
                        self._seqrand_last = {"seq": seq_points[-1][1], "rand": rand_points[-1][1]}
            except Exception as e:
                self.log(f"Microbench error: {e}", "WARN")
            finally:
                self._bench_running = False

        threading.Thread(target=worker, daemon=True).start()

    def _draw_napkin_graphs(self) -> None:
        with self._bw_lock:
            # Graph 1: GB/s vs worker cap (with reason below)
            by_t: dict[int, list[float]] = {}
            for _, th, gbps in self._bw_by_threads:
                by_t.setdefault(th, []).append(gbps)
            thread_points = sorted((float(th), sum(v) / len(v)) for th, v in by_t.items()) if by_t else []

            footer = ""
            if self._last_cap_decision:
                d = self._last_cap_decision
                if d.cap < d.desired:
                    footer = f"why not +1 worker? {d.reasons[-1] if d.reasons else 'cap'}"
                else:
                    footer = "cap ok: " + (" ; ".join(d.reasons[:2]) if d.reasons else "ok")
            self._draw_line_chart(self.napkin_canvas_threads, thread_points, "GB/s vs workers", footer=footer)

            # Graph 2: Seq vs Rand microbench
            self._draw_two_line_chart(self.napkin_canvas_sizes, self._bw_by_size_seq[:], self._bw_by_size_rand[:], "Seq vs Rand (microbench)")

            # Graph 3: bars
            if self._seqrand_last:
                bars = [("seq", self._seqrand_last.get("seq", 0.0)), ("rand", self._seqrand_last.get("rand", 0.0))]
                self._draw_bars(self.napkin_canvas_seqrand, bars, "128MB compare (microbench)")
            else:
                self._draw_bars(self.napkin_canvas_seqrand, [], "128MB compare (microbench)")

            # Graph 4: plateau
            if len(self._plateau) >= 2:
                t0 = self._plateau[0][0]
                plateau_pts = [(t - t0, g) for t, g in self._plateau]
            else:
                plateau_pts = []
            self._draw_line_chart(self.napkin_canvas_plateau, plateau_pts, "Observed throughput over time", footer="from file loads (bytes/time)")

    def _draw_score_panel(self) -> None:
        c = self.score_canvas
        c.delete("all")
        w = int(c["width"])
        y = 10

        score = float(getattr(self, "current_score", 0.0))
        c.create_text(10, y, anchor="nw", text="LIVE", font=("Arial", 10, "bold"), fill="#2c3e50")
        y += 18
        c.create_text(10, y, anchor="nw", text=f"Overall score: {score:.1f}", font=("Arial", 9, "bold"), fill="#3498db")
        y += 16
        c.create_text(10, y, anchor="nw", text=f"RAM: {self.ram_var.get()}  |  CPU: {self.cpu_var.get()}", font=("Arial", 11))
        y += 14
        c.create_text(10, y, anchor="nw", text=f"Max threads: {self.processor.max_threads}", font=("Arial", 11))
        y += 14
        c.create_text(10, y, anchor="nw", text=f"Max chunk: {self.processor.max_chunk_size:,}", font=("Arial", 11))
        y += 18

        # score bar
        bar_w = int((score / 100) * (w - 20))
        c.create_rectangle(10, y, 10 + bar_w, y + 12, fill="#27ae60", outline="")
        c.create_rectangle(10, y, w - 10, y + 12, outline="#bdc3c7", width=1)
        y += 22

        # last cap reason
        if self._last_cap_decision:
            d = self._last_cap_decision
            c.create_text(10, y, anchor="nw", text=f"Workers cap: {d.cap}/{self.processor.max_threads} (desired {d.desired})", font=("Arial", 9, "bold"))
            y += 14
            for line in d.reasons[-3:]:
                c.create_text(14, y, anchor="nw", text=f"• {line}", font=("Arial", 10), fill="#2c3e50")
                y += 12
            y += 6

        # chunk reason
        c.create_text(10, y, anchor="nw", text="Chunk decision:", font=("Arial", 9, "bold"))
        y += 14
        c.create_text(14, y, anchor="nw", text=self._last_chunk_reason[:110] + ("..." if len(self._last_chunk_reason) > 110 else ""), font=("Arial", 10), fill="#2c3e50")
        y += 22

        # mode
        mode = "FULL_RUN=1" if FULL_RUN else f"SAMPLE {DEFAULT_SAMPLE_ROWS} rows"
        c.create_text(10, y, anchor="nw", text=f"Mode: {mode}", font=("Arial", 11), fill="#8e44ad")

    # ========================================================
    # MONITOR LOOP
    # ========================================================
    def start_monitoring_loop(self) -> None:
        ram, cpu = self.monitor.record_metric()
        stats = self.monitor.get_stats()
        self.ram_var.set(f"{ram:.1f}%")
        self.cpu_var.set(f"{cpu:.1f}%")
        self.rows_var.set(f"Rows: {stats['data_loaded']:,}")
        self.files_var.set(f"Files: {stats['files']:,}")

        self.update_score_graphs(ram, cpu)
        self._draw_score_panel()

        now = time.time()
        if now - self.last_progress_refresh >= 10:
            detail = f"Status: {self.status_var.get()} | {self.progress_var.get():.1f}% | RAM {ram:.1f}% | CPU {cpu:.1f}%"
            self.progress_detail.config(text=detail)
            self.last_progress_refresh = now

        if (now - self._last_bench) > 30.0 and not self._bench_running:
            self._last_bench = now
            self._run_memory_microbench()

        self._draw_napkin_graphs()

        self.root.after(700, self.start_monitoring_loop)

    # ========================================================
    # PIPELINE
    # ========================================================
    def start_consolidation(self) -> None:
        if not self.toniot_file or not self.cic_dir:
            messagebox.showerror("Missing input", "Select both TON_IoT CSV and CIC folder first.")
            self.add_alert("Select both TON_IoT CSV and CIC folder first.", "ERROR")
            return

        self.is_running = True
        self.start_time = time.time()
        self.progress_var.set(0)
        self.status_var.set("Starting...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.add_alert("Pipeline starting... ▶", "INFO")
        self.reset_thread_bars()
        self.progress_tick = 0
        self.progress_last_detail = -1

        threading.Thread(target=self.consolidate_worker, daemon=True).start()

    def stop_consolidation(self) -> None:
        self.is_running = False
        self.status_var.set("Stopping...")
        self.log("Stop requested. The current step will finish before exiting.", "WARN")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def consolidate_worker(self) -> None:
        try:
            self._run_pipeline()
        except Exception as exc:
            self.log(f"Critical error: {exc}", "ERROR")
            traceback.print_exc()
        finally:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _run_pipeline(self) -> None:
        steps = 6
        step = 0

        # Sample rows: as requested
        sample_rows = None if FULL_RUN else DEFAULT_SAMPLE_ROWS

        # Load TON_IoT
        self.status_var.set("Loading TON_IoT")
        self.ensure_thread_bars(2)
        df_toniot = self.processor.load_toniot_optimized(
            self.toniot_file,
            callback=self._progress_callback("TON_IoT"),
            sample_rows=sample_rows,
            workers_hint=max(1, self.processor.active_worker_cap),
        )
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"TON_IoT loaded: {len(df_toniot):,} rows", "OK")
        if not self.is_running:
            return

        # Load CIC
        self.status_var.set("Loading CIC")
        dfs_cic, total_files, cic_threads = self.processor.load_cic_optimized(
            self.cic_dir,
            callback=self._progress_callback("CIC"),
            threads_hook=self.ensure_thread_bars,
            sample_rows=sample_rows,
        )
        self.log(f"CIC loader base threads: {cic_threads}", "INFO")
        if len(dfs_cic) != total_files:
            missing = total_files - len(dfs_cic)
            self.add_alert(f"CIC: {missing} failed (loaded {len(dfs_cic)}/{total_files})", "WARN")
            self.log(f"CIC: {missing} failed (loaded {len(dfs_cic)}/{total_files})", "WARN")
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"CIC loaded: {len(dfs_cic)} file(s)", "OK")
        if not self.is_running:
            return

        # Merge + clean
        self.status_var.set("Merging and cleaning")
        combined = self.processor.merge_optimized([df_toniot] + dfs_cic)
        combined = self.processor.clean_optimized(combined)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"Merged: {len(combined):,} rows, {len(combined.columns)} cols", "INFO")
        if not self.is_running:
            return

        # Split
        self.status_var.set("Splitting train/test")
        df_train, df_test = self.processor.split_optimized(combined)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"Split: train {len(df_train):,} / test {len(df_test):,}", "INFO")
        if not self.is_running:
            return

        # Enforce Label name for downstream scripts
        df_train, _ = self.processor.normalize_label_column(df_train)
        df_test, _ = self.processor.normalize_label_column(df_test)

        # Export CSV
        self.status_var.set("Writing CSV")
        df_train.to_csv("fusion_train_smart4.csv", index=False, encoding="utf-8")
        df_test.to_csv("fusion_test_smart4.csv", index=False, encoding="utf-8")
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log("CSV files written", "OK")
        if not self.is_running:
            return

        # Export NPZ
        self.status_var.set("Writing NPZ")
        if "Label" not in df_train.columns:
            self.log("No 'Label' column -> cannot build NPZ labels safely", "ERROR")
        else:
            numeric_cols = [c for c in df_train.columns if c != "Label" and np.issubdtype(df_train[c].dtype, np.number)]
            if numeric_cols:
                X_train = df_train[numeric_cols].astype(np.float32).fillna(df_train[numeric_cols].mean())
                y_train = df_train["Label"].astype(str)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train).astype(np.float32)

                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y_train)

                np.savez_compressed(
                    "preprocessed_dataset.npz",
                    X=X_scaled,
                    y=y_encoded,
                    classes=encoder.classes_,
                    numeric_cols=np.array(numeric_cols, dtype=object),
                )
                self.log(f"NPZ created: {len(numeric_cols)} features, scaled & encoded", "OK")
            else:
                self.log("No numeric columns for NPZ", "WARN")

        step += 1
        self.progress_var.set(100)
        self.status_var.set("✓ Completed")

        elapsed = time.time() - (self.start_time or time.time())
        self.log(f"Consolidation finished in {self._format_duration(elapsed)}", "INFO")
        self.add_alert(f"✓ Completed in {self._format_duration(elapsed)}", "OK")

    def _format_duration(self, seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _progress_callback(self, name: str):
        def _cb(idx, size, progress, thread_id, action=None):
            if not self.is_running:
                return
            msg = action or f"{name} chunk {idx} ({size:,} rows)"
            self.update_thread_progress(thread_id, progress, msg)
            self.progress_tick += 1
            if progress - self.progress_last_detail >= 1:
                vm = psutil.virtual_memory()
                detail = f"{name}: {progress:.1f}% | RAM {vm.percent:.1f}%"
                self.progress_detail.config(text=detail)
                self.progress_var.set(progress)
                self.progress_last_detail = progress

        return _cb


def main() -> None:
    root = tk.Tk()
    app = ConsolidationGUIEnhanced(root)
    root.mainloop()


if __name__ == "__main__":
    main()
