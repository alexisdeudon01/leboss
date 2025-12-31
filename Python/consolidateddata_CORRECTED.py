#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Consolidation GUI - DYNAMIC UTILIZATION EDITION v4
======================================================
Goals (per your spec):
  - Start aggressive: MAX chunk + MAX workers, then stabilize down using score.
  - Chunk size dynamic AND depends on score + RAM + worker cap.
  - CIC files are also read in chunks in FULL_RUN mode.
  - Scoring is dynamic and aims to maximize RAM and CPU utilization
    while respecting RAM threshold (default 90%).
  - Alerts/Errors shown in a canvas.
  - Logs shown in a canvas (lots of logs).
  - 4 napkin math graphs removed.

Defaults:
  - SAMPLE mode ON by default: read first 1000 rows from TON_IoT and from each CIC file.
  - FULL mode by env: FULL_RUN=1 (then dynamic chunking applies).

Outputs:
  - fusion_train_smart4.csv
  - fusion_test_smart4.csv
  - preprocessed_dataset.npz

Python: 3.10+
"""

from __future__ import annotations

import gc
import os
import time
import traceback
import threading
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# ============================================================
# CONFIG
# ============================================================
MAX_RAM_PERCENT = int(os.getenv("MAX_RAM_PERCENT", "90"))          # hard ceiling
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50000"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "750000"))
MAX_THREADS = int(os.getenv("MAX_THREADS", "12"))

# keep sample runs heavier to stress CPU/RAM
DEFAULT_SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "50000"))
FULL_RUN = os.getenv("FULL_RUN", "1").strip() == "1"              # full processing if 1
WARM_START_SECONDS = float(os.getenv("WARM_START_SECONDS", "12")) # aggressive period at beginning

TARGET_FLOAT_DTYPE = np.float32

PROGRESS_TITLE = "Overall progress"

UI_FONT = ("Arial", 10)
UI_FONT_BOLD = ("Arial", 10, "bold")
SMALL_FONT = ("Arial", 9)
SMALL_FONT_BOLD = ("Arial", 9, "bold")

# ============================================================
# Utility: safe wrappers
# ============================================================

def _safe(func):
    """Decorator: catch exceptions, return None if fails (caller should handle)."""
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            # attempt to log via self._log if available
            try:
                self = args[0]
                if hasattr(self, "_log"):
                    self._log(f"[EXC] {func.__name__}:\n{traceback.format_exc()}", level="ERROR")
            except Exception:
                pass
            return None
    return wrap

# ============================================================
# UI helpers: Canvas-based alerts & logs
# ============================================================

class CanvasFeed:
    """Scrollable canvas feed (alerts/logs)."""

    def __init__(self, parent, *, height=140, max_items=400, bg="#ffffff", fg="#2c3e50"):
        self.max_items = max_items
        self.bg = bg
        self.fg = fg
        self.canvas = tk.Canvas(parent, height=height, bg=bg, highlightthickness=1, highlightbackground="#ccc")
        self.scroll = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def grid(self, **kwargs):
        self.canvas.grid(**kwargs)
        self.scroll.grid(**{**kwargs, "column": kwargs.get("column", 0) + 1, "sticky": "ns"})

    def pack(self, **kwargs):
        kwargs_canvas = dict(kwargs)
        kwargs_canvas.setdefault("side", "left")
        kwargs_canvas.setdefault("fill", "both")
        kwargs_canvas.setdefault("expand", True)
        self.canvas.pack(**kwargs_canvas)

        kwargs_scroll = {"side": "right", "fill": "y"}
        self.scroll.pack(**kwargs_scroll)

    def add(self, label: str, message: str, color: str | None = None):
        color = color or self.fg
        row = tk.Frame(self.inner, bg=self.bg)
        tk.Label(row, text=label, fg=color, width=10, font=SMALL_FONT_BOLD, bg=self.bg).pack(side="left", padx=4)
        tk.Label(
            row,
            text=message,
            fg=color,
            wraplength=1200,
            anchor="w",
            justify="left",
            font=SMALL_FONT,
            bg=self.bg,
        ).pack(side="left", fill="x", expand=True)
        row.pack(fill="x", padx=4, pady=1)

        children = self.inner.winfo_children()
        if len(children) > self.max_items:
            for w in children[: len(children) - self.max_items]:
                try:
                    w.destroy()
                except Exception:
                    pass
        try:
            self.canvas.yview_moveto(1.0)
        except Exception:
            pass

# ============================================================
# Monitor
# ============================================================

class AdvancedMonitor:
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
# Processor (chunking + workers)
# ============================================================

class OptimizedDataProcessor:
    def __init__(self, monitor: AdvancedMonitor, max_ram_percent: int = MAX_RAM_PERCENT, logger=None) -> None:
        self.monitor = monitor
        self.max_ram_percent = int(max_ram_percent)
        self.min_chunk_size = int(MIN_CHUNK_SIZE)

        self.processed_rows = 0
        self.start_time = time.time()
        self.logger = logger

        # live knobs from GUI
        self.current_score = 100.0
        self.current_worker_cap = 1
        self.last_worker_reason = "init"
        self.last_chunk_reason = "init"

        # label column (set during merge/clean)
        self.label_col: str | None = None

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

        self._log(
            f"Caps: RAM={ram_gb:.1f}GB | max_chunk={self.max_chunk_size:,} | max_threads={self.max_threads} | FULL_RUN={FULL_RUN} | SAMPLE_ROWS={DEFAULT_SAMPLE_ROWS}"
        )

    def _log(self, msg: str, level: str = "DEBUG") -> None:
        if self.logger:
            try:
                self.logger(msg, level)
            except Exception:
                pass

    def _wait_for_ram(self, target: float) -> None:
        try:
            while psutil.virtual_memory().percent >= target:
                gc.collect()
                time.sleep(0.15)
        except Exception:
            pass

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            int_cols = df.select_dtypes(include=["int64", "int32"]).columns
            float_cols = df.select_dtypes(include=["float64"]).columns
            obj_cols = df.select_dtypes(include=["object"]).columns

            if len(int_cols) > 0:
                df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
            if len(float_cols) > 0:
                df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float").astype(TARGET_FLOAT_DTYPE)

            for col in obj_cols:
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    numeric_ratio = float(numeric.notna().mean())
                except Exception:
                    numeric = None
                    numeric_ratio = 0.0

                if numeric is not None and numeric_ratio > 0.5:
                    df[col] = pd.to_numeric(numeric, downcast="float").astype(TARGET_FLOAT_DTYPE)
                    continue

                # categorical wins for repeated strings
                try:
                    nunq = int(df[col].nunique(dropna=False))
                    if nunq / max(len(df[col]), 1) < 0.5 or nunq < 50:
                        df[col] = df[col].astype("category")
                except Exception:
                    pass

            return df
        except Exception:
            self._log(f"optimize_dtypes failed:\n{traceback.format_exc()}", "WARN")
            return df

    # --------------------------
    # Label normalization
    # --------------------------
    def detect_label_column(self, df: pd.DataFrame) -> str | None:
        try:
            for c in df.columns:
                if str(c).strip().lower() == "label":
                    return c
            return None
        except Exception:
            return None

    def ensure_label_is_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames any label-like column to exact 'Label' for downstream compatibility."""
        try:
            c = self.detect_label_column(df)
            if c is None:
                self.label_col = None
                return df
            if c != "Label":
                df = df.rename(columns={c: "Label"})
                self._log(f"[LABEL] Renamed '{c}' -> 'Label'", "INFO")
            self.label_col = "Label"
            return df
        except Exception:
            self._log(f"ensure_label_is_standard failed:\n{traceback.format_exc()}", "WARN")
            return df

    # --------------------------
    # Dynamic chunking helpers
    # --------------------------
    def _estimate_bytes_per_row(self, filepath: str) -> float:
        try:
            sample = pd.read_csv(filepath, nrows=2000, low_memory=False, memory_map=True)
            sample = self._optimize_dtypes(sample)
            b = float(sample.memory_usage(deep=True).sum())
            return b / max(len(sample), 1)
        except Exception:
            return 512.0  # fallback guess

    def _estimate_chunk_size(self, filepath: str, workers_hint: int) -> int:
        vm = psutil.virtual_memory()
        headroom = max((self.max_ram_percent / 100 * vm.total) - vm.used, vm.available * 0.5)
        budget = max(headroom * 0.6, 64 * 1024 * 1024)

        per_row = self._estimate_bytes_per_row(filepath)
        parallel_penalty = 1.0 + 0.12 * max(workers_hint - 1, 0)  # more workers => smaller chunks
        est = int((budget / max(per_row, 1.0)) / parallel_penalty)

        # safety by current RAM
        if vm.percent > 60:
            est = int(est * 0.6)
        if vm.percent > 70:
            est = int(est * 0.5)

        chunk = max(self.min_chunk_size, min(self.max_chunk_size, est))
        self._log(
            f"[CHUNK_INIT] {os.path.basename(filepath)} -> {chunk:,} rows | per_rowâ‰ˆ{per_row:.0f}B | workers_hint={workers_hint} | RAM={vm.percent:.1f}%",
            "DEBUG",
        )
        return chunk

    def _tune_chunk_size(self, current_size: int, *, ram: float, cpu: float, score: float, workers_cap: int) -> int:
        mult = 1.0
        reason_parts = []

        target_ram = float(self.max_ram_percent) - 1.0

        # RAM-driven adjustments (aim near target, avoid ceiling)
        if ram >= self.max_ram_percent:
            mult *= 0.45
            reason_parts.append("RAM>=ceiling => hard shrink")
        elif ram > target_ram + 2:
            mult *= 0.65
            reason_parts.append("RAM>target => shrink")
        elif ram < target_ram - 10:
            mult *= 1.25
            reason_parts.append("RAM<<target => grow")
        elif ram < target_ram - 4:
            mult *= 1.12
            reason_parts.append("RAM<target => grow")

        # score nudges stability (no CPU ceiling)
        if score >= 90 and ram < target_ram:
            mult *= 1.08
            reason_parts.append("score>=90 => mild grow")
        elif score <= 55:
            mult *= 0.82
            reason_parts.append("score<=55 => shrink")

        # more workers => reduce chunk to avoid peak RAM
        mult /= (1.0 + 0.10 * max(workers_cap - 1, 0))
        if workers_cap > 1:
            reason_parts.append(f"workers_cap={workers_cap} => penalty")

        tuned = int(current_size * mult)
        tuned = max(self.min_chunk_size, min(self.max_chunk_size, tuned))
        if tuned == self.max_chunk_size:
            reason_parts.append("hit MAX_CHUNK_SIZE")
        if tuned == self.min_chunk_size:
            reason_parts.append("hit MIN_CHUNK_SIZE")
        self.last_chunk_reason = (
            f"chunk_tune {current_size:,}->{tuned:,} | RAM {ram:.1f}% score {score:.1f} | cap {workers_cap} | mult {mult:.2f} | "
            + "; ".join(reason_parts)
        )
        return tuned

    # --------------------------
    # Dynamic worker cap
    # --------------------------
    def compute_worker_cap(self, *, base_threads: int, file_count: int, ram: float, cpu: float, score: float, warm: bool) -> tuple[int, str]:
        # start aggressive
        if warm and ram < (self.max_ram_percent - 8):
            cap = self.max_threads
            return cap, f"warmstart<{WARM_START_SECONDS:.0f}s & RAM safe => cap={cap}"

        # hard RAM ceiling
        if ram >= self.max_ram_percent:
            return 1, f"RAM {ram:.1f}% >= ceiling {self.max_ram_percent}%"

        # near ceiling, clamp hard
        if ram >= (self.max_ram_percent - 3):
            cap = max(1, min(2, base_threads))
            return cap, f"near RAM ceiling (>= {self.max_ram_percent-3}%) => cap={cap}"

        ram_headroom = max(0.0, (self.max_ram_percent - ram))
        cap = base_threads

        # grow when we have room; CPU encourages growth
        if ram_headroom >= 10 and score >= 70:
            cap = min(self.max_threads, base_threads + 4)
            return cap, f"headroom {ram_headroom:.1f}% & score {score:.1f} => +4"
        if ram_headroom >= 6:
            cap = min(self.max_threads, base_threads + 2)
            return cap, f"headroom {ram_headroom:.1f}% => +2"
        if ram_headroom >= 3 and cpu < 90:
            cap = min(self.max_threads, base_threads + 1)
            return cap, f"CPU {cpu:.1f}% low & headroom {ram_headroom:.1f}% => +1"

        # if score poor, back off slightly
        if score < 50:
            cap = max(1, min(base_threads, 2))
            return cap, f"score {score:.1f} low => reduce cap={cap}"

        cap = min(self.max_threads, max(1, base_threads))
        return cap, f"stable: cap={cap}"

    # --------------------------
    # Loaders
    # --------------------------
    def _read_csv_chunked(self, filepath: str, *, workers_hint: int, callback=None, thread_id: int = 0) -> pd.DataFrame:
        # initial
        chunk_size = self._estimate_chunk_size(filepath, workers_hint=workers_hint)
        chunks: list[pd.DataFrame] = []
        processed = 0

        # estimate total rows for progress from file size / bytes per row
        try:
            per_row = self._estimate_bytes_per_row(filepath)
            fsize = float(os.path.getsize(filepath))
            est_total_rows = max(int(fsize / max(per_row, 1.0)), 1)
        except Exception:
            est_total_rows = 1

        reader = pd.read_csv(filepath, chunksize=chunk_size, low_memory=False, memory_map=True)
        for idx, chunk in enumerate(reader, 1):
            self._wait_for_ram(self.max_ram_percent - 1)
            chunk = self._optimize_dtypes(chunk)
            chunks.append(chunk)

            processed += len(chunk)
            self.processed_rows += len(chunk)
            self.monitor.track_data(len(chunk))
            ram, cpu = self.monitor.record_metric()

            # callback/progress
            if callback:
                prog = min(100.0, (processed / est_total_rows) * 100.0)
                callback(idx, len(chunk), prog, thread_id, f"{os.path.basename(filepath)} chunk {idx}")

            # occasionally concat to reduce fragmentation
            if len(chunks) >= 4 and psutil.virtual_memory().percent > (self.max_ram_percent - 6):
                chunks = [pd.concat(chunks, ignore_index=True, copy=False)]
                gc.collect()

            # dynamic tune
            workers_cap = max(1, int(getattr(self, "current_worker_cap", workers_hint)))
            score = float(getattr(self, "current_score", 100.0))
            chunk_size = self._tune_chunk_size(chunk_size, ram=ram, cpu=cpu, score=score, workers_cap=workers_cap)
            try:
                reader.chunksize = chunk_size
            except Exception:
                pass

        out = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame()
        gc.collect()
        return out

    def load_toniot(self, filepath: str, callback=None, sample_rows: int | None = None) -> pd.DataFrame:
        try:
            if sample_rows:
                self._log(f"[TON] sample nrows={sample_rows}", "INFO")
                df = pd.read_csv(filepath, nrows=sample_rows, low_memory=False, memory_map=True)
                df = self._optimize_dtypes(df)
                return df

            self._log("[TON] full chunked read", "INFO")
            return self._read_csv_chunked(filepath, workers_hint=max(1, self.current_worker_cap), callback=callback, thread_id=0)
        except Exception:
            self._log(f"load_toniot failed:\n{traceback.format_exc()}", "ERROR")
            return pd.DataFrame()

    def _load_cic_file(self, filepath: str, sample_rows: int | None, callback=None, thread_id: int = 0) -> tuple[pd.DataFrame | None, str]:
        try:
            if sample_rows:
                df = pd.read_csv(filepath, nrows=sample_rows, low_memory=False, memory_map=True)
                df = self._optimize_dtypes(df)
                return df, filepath

            # FULL_RUN => chunked read too
            df = self._read_csv_chunked(filepath, workers_hint=max(1, self.current_worker_cap), callback=callback, thread_id=thread_id)
            return df, filepath
        except Exception:
            self._log(f"_load_cic_file failed for {os.path.basename(filepath)}:\n{traceback.format_exc()}", "WARN")
            return None, filepath

    def load_cic_folder(self, folder: str, callback=None, threads_hook=None, sample_rows: int | None = None, worker_policy=None) -> tuple[list[pd.DataFrame], int, int]:
        # discover
        cic_files = []
        sizes_mb = []
        try:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".csv"):
                        path = os.path.join(root, f)
                        cic_files.append(path)
                        try:
                            sizes_mb.append(os.path.getsize(path) / (1024 * 1024))
                        except OSError:
                            sizes_mb.append(0.0)
        except Exception:
            self._log(f"[CIC] os.walk failed:\n{traceback.format_exc()}", "ERROR")

        cic_files.sort()
        avg_mb = (sum(sizes_mb) / len(sizes_mb)) if sizes_mb else 0.0
        self._log(f"[CIC] found {len(cic_files)} file(s) | avgâ‰ˆ{avg_mb:.1f}MB | sample_rows={sample_rows}", "INFO")

        base_threads = min(max(1, psutil.cpu_count(logical=True) or 4), self.max_threads, len(cic_files) or self.max_threads)
        base_threads = max(1, min(base_threads, 4))  # baseline; dynamic can go higher
        chosen = base_threads

        if threads_hook:
            try:
                threads_hook(chosen)
            except Exception:
                pass

        dfs: list[pd.DataFrame] = []
        failures: list[str] = []

        active = 0
        cap_prev = 0
        start = time.time()

        # executor always uses max_threads; we gate submissions ourselves
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {}

            for idx_file, path in enumerate(cic_files, 1):
                while True:
                    ram, cpu = self.monitor.record_metric()
                    warm = (time.time() - start) < WARM_START_SECONDS
                    score = float(getattr(self, "current_score", 100.0))

                    cap, reason = self.compute_worker_cap(
                        base_threads=chosen,
                        file_count=len(cic_files),
                        ram=ram,
                        cpu=cpu,
                        score=score,
                        warm=warm,
                    )
                    cap = min(cap, len(cic_files))
                    self.current_worker_cap = cap
                    self.last_worker_reason = reason

                    if cap != cap_prev:
                        cap_prev = cap
                        self._log(f"[CIC] cap -> {cap} ({reason})", "INFO")
                        if threads_hook:
                            try:
                                threads_hook(cap)
                            except Exception:
                                pass

                    if active < cap:
                        break

                    time.sleep(0.08)

                # thread id is assigned round-robin just for UI
                tid = (idx_file - 1) % max(1, cap_prev)
                fut = executor.submit(self._load_cic_file, path, sample_rows, callback, tid)
                futures[fut] = path
                active += 1

            for done_idx, fut in enumerate(as_completed(futures), 1):
                try:
                    df, path = fut.result()
                except Exception:
                    df, path = None, futures.get(fut, "unknown")
                    self._log(f"[CIC] future crashed for {path}:\n{traceback.format_exc()}", "ERROR")

                active = max(active - 1, 0)

                if df is not None:
                    dfs.append(df)
                    self.monitor.track_file()
                    self.monitor.track_data(len(df))
                    self._log(f"[CIC] done {done_idx}/{len(cic_files)} -> {os.path.basename(path)} rows={len(df):,}", "DEBUG")
                else:
                    failures.append(path)
                    self._log(f"[CIC] failed {os.path.basename(path)}", "WARN")

                if callback:
                    try:
                        progress = (done_idx / max(len(cic_files), 1)) * 100
                        callback(done_idx, len(cic_files), progress, done_idx % max(1, self.current_worker_cap), f"Loaded {os.path.basename(path)}")
                    except Exception:
                        pass

                if psutil.virtual_memory().percent > self.max_ram_percent:
                    gc.collect()

        if failures:
            self._log(f"[CIC] failures={len(failures)} -> {[os.path.basename(x) for x in failures][:10]}", "WARN")
        gc.collect()
        return dfs, len(cic_files), chosen

    # --------------------------
    # Merge / clean / split
    # --------------------------
    def merge(self, dfs_list: list[pd.DataFrame]) -> pd.DataFrame:
        try:
            if not dfs_list:
                return pd.DataFrame()
            mem_before = sum(float(df.memory_usage(deep=True).sum()) for df in dfs_list) / (1024 * 1024)
            self._log(f"[MERGE] inputs={len(dfs_list)} memâ‰ˆ{mem_before:.1f}MB", "INFO")
            df = pd.concat(dfs_list, ignore_index=True, copy=False)
            df = self._optimize_dtypes(df)
            df = self.ensure_label_is_standard(df)
            mem_after = float(df.memory_usage(deep=True).sum()) / (1024 * 1024) if not df.empty else 0.0
            self._log(f"[MERGE] out rows={len(df):,} cols={len(df.columns)} memâ‰ˆ{mem_after:.1f}MB label_col={self.label_col}", "INFO")
            gc.collect()
            return df
        except Exception:
            self._log(f"merge failed:\n{traceback.format_exc()}", "ERROR")
            return pd.DataFrame()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop_duplicates()
            df = self.ensure_label_is_standard(df)
            if "Label" in df.columns:
                before = len(df)
                df = df.dropna(subset=["Label"])
                self._log(f"[CLEAN] dropna Label: {before:,}->{len(df):,}", "INFO")
            else:
                self._log("[CLEAN] no Label column found", "WARN")
            gc.collect()
            return df
        except Exception:
            self._log(f"clean failed:\n{traceback.format_exc()}", "ERROR")
            return df

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df = self.ensure_label_is_standard(df)
            if "Label" not in df.columns:
                # fallback random split
                self._log("[SPLIT] no Label => random 60/40", "WARN")
                n = len(df)
                train_size = int(n * 0.6)
                idx = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(idx)
                train_idx = idx[:train_size]
                test_idx = idx[train_size:]
                return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

            # stratified
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
            for train_idx, test_idx in sss.split(df, df["Label"]):
                self._log(f"[SPLIT] stratified ok: train={len(train_idx):,} test={len(test_idx):,}", "INFO")
                return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

            return df.copy(), df.copy()
        except Exception:
            self._log(f"split failed:\n{traceback.format_exc()}", "ERROR")
            return df.copy(), df.copy()

# ============================================================
# GUI
# ============================================================

class ConsolidationGUIEnhanced:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Data Consolidation - Dynamic v4")
        self.root.geometry("1250x900")

        self.toniot_file: str | None = None
        self.cic_dir: str | None = None
        self.is_running = False
        self.start_time: float | None = None
        self.start_ts = time.time()

        self.monitor = AdvancedMonitor()
        self.processor = OptimizedDataProcessor(self.monitor, logger=self.log)

        # live ui vars
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Idle")
        self.ram_var = tk.StringVar(value="-- %")
        self.cpu_var = tk.StringVar(value="-- %")
        self.rows_var = tk.StringVar(value="Rows: 0")
        self.files_var = tk.StringVar(value="Files: 0")

        self.thread_bars: dict[int, dict[str, tk.Variable | ttk.Label]] = {}
        self.current_score = 100.0
        self.score_state: dict[str, float | None] = {"ram": None, "cpu": None, "overall": None}

        self.setup_ui()
        self.start_monitoring_loop()

    # ---------------- UI ----------------
    def setup_ui(self) -> None:
        # Input
        file_frame = ttk.LabelFrame(self.root, text="Input", padding=8)
        file_frame.pack(fill="x", padx=10, pady=8)

        ttk.Label(file_frame, text="TON_IoT CSV:", font=UI_FONT_BOLD).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.toniot_path_label = ttk.Label(file_frame, text="No file selected", font=UI_FONT)
        self.toniot_path_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.select_toniot).grid(row=0, column=2, sticky="e", padx=5, pady=5)

        ttk.Label(file_frame, text="CIC Folder:", font=UI_FONT_BOLD).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.cic_path_label = ttk.Label(file_frame, text="No folder selected", font=UI_FONT)
        self.cic_path_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.select_cic).grid(row=1, column=2, sticky="e", padx=5, pady=5)

        # Controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)
        self.start_button = ttk.Button(control_frame, text="â–¶ Start", command=self.start_consolidation)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(control_frame, text="â¹ Stop", command=self.stop_consolidation, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5)

        # Alerts
        alerts_frame = ttk.LabelFrame(self.root, text="Alerts / Errors", padding=6)
        alerts_frame.pack(fill="both", padx=10, pady=6)
        self.alert_feed = CanvasFeed(alerts_frame, height=110, max_items=120)
        self.alert_feed.pack(fill="both", expand=True)

        # Monitoring
        monitor_frame = ttk.LabelFrame(self.root, text="Monitoring", padding=8)
        monitor_frame.pack(fill="x", padx=10, pady=8)
        ttk.Label(monitor_frame, text="RAM:", font=UI_FONT_BOLD).grid(row=0, column=0, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.ram_var, font=UI_FONT).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(monitor_frame, text="CPU:", font=UI_FONT_BOLD).grid(row=0, column=2, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.cpu_var, font=UI_FONT).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Label(monitor_frame, textvariable=self.rows_var, font=UI_FONT).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.files_var, font=UI_FONT).grid(row=1, column=2, columnspan=2, sticky="w")

        ttk.Label(monitor_frame, text="Status:", font=UI_FONT_BOLD).grid(row=2, column=0, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.status_var, font=UI_FONT).grid(row=2, column=1, columnspan=3, sticky="w")

        ttk.Label(monitor_frame, text=PROGRESS_TITLE + ":", font=UI_FONT_BOLD).grid(row=3, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(monitor_frame, maximum=100, variable=self.progress_var)
        self.progress_bar.grid(row=3, column=1, columnspan=3, sticky="ew", padx=5, pady=6)

        self.progress_detail = ttk.Label(monitor_frame, text="", anchor="w", font=SMALL_FONT)
        self.progress_detail.grid(row=4, column=0, columnspan=4, sticky="w", padx=5, pady=2)
        monitor_frame.columnconfigure(1, weight=1)
        monitor_frame.columnconfigure(3, weight=1)

        # Threads + tuning panel
        mid = ttk.Frame(self.root)
        mid.pack(fill="x", padx=10, pady=6)

        threads_frame = ttk.LabelFrame(mid, text="Thread progress", padding=6)
        threads_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.thread_container = ttk.Frame(threads_frame)
        self.thread_container.pack(fill="both", expand=True)

        tuning_frame = ttk.LabelFrame(mid, text="Dynamic decisions", padding=6)
        tuning_frame.pack(side="left", fill="both", expand=True)

        # canvases with scrollbars
        score_wrap = ttk.Frame(tuning_frame)
        score_wrap.pack(fill="both", expand=True, padx=4, pady=4)
        self.score_canvas = tk.Canvas(score_wrap, width=450, height=170, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        score_scroll = ttk.Scrollbar(score_wrap, orient="vertical", command=self.score_canvas.yview)
        self.score_canvas.configure(yscrollcommand=score_scroll.set)
        self.score_canvas.pack(side="left", fill="both", expand=True)
        score_scroll.pack(side="right", fill="y")

        chunk_wrap = ttk.Frame(tuning_frame)
        chunk_wrap.pack(fill="both", expand=True, padx=4, pady=4)
        self.chunk_canvas = tk.Canvas(chunk_wrap, width=450, height=120, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        chunk_scroll = ttk.Scrollbar(chunk_wrap, orient="vertical", command=self.chunk_canvas.yview)
        self.chunk_canvas.configure(yscrollcommand=chunk_scroll.set)
        self.chunk_canvas.pack(side="left", fill="both", expand=True)
        chunk_scroll.pack(side="right", fill="y")

        worker_wrap = ttk.Frame(tuning_frame)
        worker_wrap.pack(fill="both", expand=True, padx=4, pady=4)
        self.worker_canvas = tk.Canvas(worker_wrap, width=450, height=110, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ccc")
        worker_scroll = ttk.Scrollbar(worker_wrap, orient="vertical", command=self.worker_canvas.yview)
        self.worker_canvas.configure(yscrollcommand=worker_scroll.set)
        self.worker_canvas.pack(side="left", fill="both", expand=True)
        worker_scroll.pack(side="right", fill="y")

        # Logs (canvas)
        logs_frame = ttk.LabelFrame(self.root, text="Logs (canvas)", padding=6)
        logs_frame.pack(fill="both", padx=10, pady=8, expand=True)
        self.log_feed = CanvasFeed(logs_frame, height=260, max_items=800, bg="#0f172a", fg="#e2e8f0")
        self.log_feed.pack(fill="both", expand=True)
        # ensure log area starts visible
        self.log("Log canvas ready", "INFO")

    # ---------------- UI events ----------------
    def select_toniot(self) -> None:
        try:
            filepath = filedialog.askopenfilename(title="Select TON_IoT CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if filepath:
                self.toniot_file = filepath
                self.toniot_path_label.config(text=filepath)
                self.add_alert("TON_IoT selected", "OK")
        except Exception as e:
            self.add_alert(f"Select TON failed: {e}", "ERROR")

    def select_cic(self) -> None:
        try:
            folder = filedialog.askdirectory(title="Select CIC folder")
            if folder:
                self.cic_dir = folder
                self.cic_path_label.config(text=folder)
                self.add_alert("CIC folder selected", "OK")
        except Exception as e:
            self.add_alert(f"Select CIC failed: {e}", "ERROR")

    def add_alert(self, message: str, level: str = "INFO") -> None:
        colors = {"ERROR": "#c0392b", "WARN": "#d35400", "INFO": "#2980b9", "OK": "#27ae60"}
        color = colors.get(level.upper(), "#2c3e50")
        try:
            self.root.after(0, lambda: self.alert_feed.add(level.upper(), message, color))
        except Exception:
            pass

    def log(self, message: str, level: str = "INFO") -> None:
        colors = {"ERROR": "#c0392b", "WARN": "#d35400", "INFO": "#2c3e50", "DEBUG": "#7f8c8d", "OK": "#27ae60"}
        color = colors.get(level.upper(), "#2c3e50")
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {message}"
        try:
            self.root.after(0, lambda: self.log_feed.add(level.upper(), msg, color))
        except Exception:
            pass

    # ---------------- Thread bars ----------------
    def ensure_thread_bars(self, count: int) -> None:
        # remove bars that are above the current cap
        for tid in sorted([k for k in self.thread_bars.keys() if k >= count], reverse=True):
            entry = self.thread_bars.pop(tid, None)
            try:
                row = entry.get("row") or entry["label"].master  # type: ignore[arg-type]
                row.destroy()
            except Exception:
                pass
            self.log(f"[UI] Removed thread bar T{tid}", "DEBUG")

        # create missing bars up to count-1
        for tid in range(count):
            if tid in self.thread_bars:
                continue
            var = tk.DoubleVar(value=0)
            row = ttk.Frame(self.thread_container)
            ttk.Label(row, text=f"T{tid}", font=UI_FONT_BOLD, width=4).pack(side="left", padx=4)
            bar = ttk.Progressbar(row, maximum=100, variable=var)
            bar.pack(side="left", fill="x", expand=True, padx=4, pady=2)
            label = ttk.Label(row, text="Idle", width=52, font=SMALL_FONT)
            label.pack(side="left", padx=4)
            row.pack(fill="x", padx=2, pady=2)
            self.thread_bars[tid] = {"var": var, "label": label, "row": row}
            self.log(f"[UI] Added thread bar T{tid}", "DEBUG")

    def reset_thread_bars(self) -> None:
        for entry in self.thread_bars.values():
            try:
                entry["var"].set(0)
                entry["label"].config(text="Idle")
            except Exception:
                pass

    def update_thread_progress(self, thread_id: int, progress: float, text: str | None = None) -> None:
        entry = self.thread_bars.get(thread_id)
        if not entry:
            return

        def _update():
            try:
                entry["var"].set(progress)
                if text:
                    entry["label"].config(text=text[:90])
            except Exception:
                pass

        try:
            self.root.after(0, _update)
        except Exception:
            pass

    # ---------------- Dynamic scoring ----------------
    def _clamp(self, x: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, x))

    def _ewma(self, value: float, prev: float | None, alpha: float = 0.25) -> float:
        return value if prev is None else (prev + alpha * (value - prev))

    def _target_score(self, value: float, target: float, tol: float, hard_max: float | None = None) -> float:
        """
        Score is highest when value is near target.
        If hard_max is set and value > hard_max => score=0.
        """
        if hard_max is not None and value >= hard_max:
            return 0.0
        # normalized distance
        d = abs(value - target)
        # inside tolerance => near-100
        if d <= tol:
            return 100.0 - (d / max(tol, 1e-6)) * 10.0  # 100..90
        # outside tol => drop faster
        return self._clamp(90.0 - (d - tol) * 2.2, 0.0, 100.0)

    def _calc_scores(self, ram: float, cpu: float) -> tuple[float, float, float]:
        target_ram = float(self.processor.max_ram_percent) - 1.0  # aim just below the ceiling
        delta = abs(ram - target_ram)

        if ram >= self.processor.max_ram_percent:
            ram_score_raw = 0.0
        else:
            ram_score_raw = max(0.0, 100.0 - delta * 5.0)
            if ram < target_ram and delta <= 12:
                ram_score_raw = min(100.0, ram_score_raw + (target_ram - ram) * 1.2)

        cpu_bonus = min(max(cpu, 0.0), 100.0) * 0.35  # target 100% CPU, bonus only
        overall_raw = 0.65 * ram_score_raw + 0.35 * cpu_bonus

        ram_s = self._ewma(ram_score_raw, self.score_state["ram"], alpha=0.20)
        cpu_s = self._ewma(cpu_bonus, self.score_state["cpu"], alpha=0.20)
        overall_s = self._ewma(overall_raw, self.score_state["overall"], alpha=0.20)

        self.score_state["ram"] = ram_s
        self.score_state["cpu"] = cpu_s
        self.score_state["overall"] = overall_s
        return ram_s, cpu_s, overall_s

    def _draw_score_panel(self, ram: float, cpu: float, score: float) -> None:
        c = self.score_canvas
        c.delete("all")
        w = int(c["width"])
        y = 10

        c.create_text(10, y, anchor="nw", text="SCORE (target RAM near ceiling, CPU bonus to 100%)", font=SMALL_FONT_BOLD)
        y += 18
        c.create_text(10, y, anchor="nw", text=f"RAM: {ram:.1f}%   CPU: {cpu:.1f}%   Score: {score:.1f}", font=UI_FONT_BOLD)
        y += 20

        bar_w = int((score / 100.0) * (w - 20))
        c.create_rectangle(10, y, w - 10, y + 16, outline="#bdc3c7")
        c.create_rectangle(10, y, 10 + bar_w, y + 16, fill="#27ae60", outline="")
        y += 24

        c.create_text(10, y, anchor="nw", text=f"Targets: RAM near {self.processor.max_ram_percent-1:.0f}% (CPU bonus only)", font=SMALL_FONT)
        y += 14
        c.create_text(10, y, anchor="nw", text="CPU is never capped; RAM drives chunk/worker adjustments.", font=SMALL_FONT)

    def _draw_chunk_panel(self) -> None:
        c = self.chunk_canvas
        c.delete("all")
        txt = self.processor.last_chunk_reason or "chunk: n/a"
        c.create_text(10, 10, anchor="nw", text="CHUNK DECISION", font=SMALL_FONT_BOLD)
        c.create_text(10, 30, anchor="nw", text=txt, font=SMALL_FONT, width=int(c["width"]) - 20)

    def _draw_worker_panel(self) -> None:
        c = self.worker_canvas
        c.delete("all")
        cap = getattr(self.processor, "current_worker_cap", 1)
        reason = getattr(self.processor, "last_worker_reason", "")
        c.create_text(10, 10, anchor="nw", text="WORKER CAP DECISION", font=SMALL_FONT_BOLD)
        c.create_text(10, 30, anchor="nw", text=f"cap={cap} / max={self.processor.max_threads}", font=UI_FONT_BOLD)
        c.create_text(10, 55, anchor="nw", text=f"Reason: {reason}", font=SMALL_FONT, width=int(c["width"]) - 20)

    # ---------------- Monitoring loop ----------------
    def start_monitoring_loop(self) -> None:
        try:
            ram, cpu = self.monitor.record_metric()
            stats = self.monitor.get_stats()
            self.ram_var.set(f"{ram:.1f}%")
            self.cpu_var.set(f"{cpu:.1f}%")
            self.rows_var.set(f"Rows: {stats['data_loaded']:,}")
            self.files_var.set(f"Files: {stats['files']:,}")

            ram_s, cpu_s, overall = self._calc_scores(ram, cpu)
            self.current_score = overall
            self.processor.current_score = overall

            # feed panels
            self._draw_score_panel(ram, cpu, overall)
            self._draw_chunk_panel()
            self._draw_worker_panel()

            # soft alerts
            if ram >= self.processor.max_ram_percent:
                self.add_alert(f"RAM ceiling hit: {ram:.1f}% >= {self.processor.max_ram_percent}%", "ERROR")
            elif ram >= (self.processor.max_ram_percent - 3):
                self.add_alert(f"RAM near ceiling: {ram:.1f}%", "WARN")
        except Exception:
            # do not crash loop
            self.log(f"monitoring loop exception:\n{traceback.format_exc()}", "ERROR")

        self.root.after(650, self.start_monitoring_loop)

    # ---------------- Pipeline ----------------
    def start_consolidation(self) -> None:
        if not self.toniot_file or not self.cic_dir:
            messagebox.showerror("Missing input", "Select both TON_IoT CSV and CIC folder first.")
            self.add_alert("Select both TON_IoT CSV and CIC folder first.", "ERROR")
            return

        self.is_running = True
        self.start_time = time.time()
        self.start_ts = time.time()
        self.progress_var.set(0)
        self.status_var.set("Starting...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.reset_thread_bars()
        self.ensure_thread_bars(2)  # start with 2 bars shown
        self.add_alert("Pipeline started", "INFO")
        self.log("Pipeline started", "INFO")

        threading.Thread(target=self.consolidate_worker, daemon=True).start()

    def stop_consolidation(self) -> None:
        self.is_running = False
        self.status_var.set("Stopping...")
        self.log("Stop requested (will stop after current step).", "WARN")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def consolidate_worker(self) -> None:
        try:
            self._run_pipeline()
        except Exception as exc:
            self.log(f"Critical error: {exc}\n{traceback.format_exc()}", "ERROR")
            self.add_alert(f"Critical error: {exc}", "ERROR")
        finally:
            self.is_running = False
            try:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
            except Exception:
                pass

    def _run_pipeline(self) -> None:
        steps = 6
        step = 0

        # sample by default, full only if env says so
        sample_rows = None if FULL_RUN else max(DEFAULT_SAMPLE_ROWS, 100_000)
        self.log(f"Mode: {'FULL_RUN' if FULL_RUN else 'SAMPLE'} | sample_rows={sample_rows}", "INFO")
        self.add_alert(f"Mode: {'FULL' if FULL_RUN else 'SAMPLE'} (rows={sample_rows or 'ALL'})", "INFO")

        # TON
        self.status_var.set("Loading TON_IoT")
        df_ton = self.processor.load_toniot(self.toniot_file, callback=self._progress_callback("TON"), sample_rows=sample_rows)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"TON loaded rows={len(df_ton):,}", "OK")
        if not self.is_running:
            return

        # CIC
        self.status_var.set("Loading CIC")
        dfs_cic, total_files, base_threads = self.processor.load_cic_folder(
            self.cic_dir,
            callback=self._progress_callback("CIC"),
            threads_hook=self.ensure_thread_bars,
            sample_rows=sample_rows,
        )
        self.log(f"CIC loaded {len(dfs_cic)}/{total_files} file(s)", "OK")
        if len(dfs_cic) != total_files:
            self.add_alert(f"CIC missing/failed: {total_files - len(dfs_cic)}", "WARN")
        step += 1
        self.progress_var.set(step / steps * 100)
        if not self.is_running:
            return

        # Merge + clean
        self.status_var.set("Merging & cleaning")
        combined = self.processor.merge([df_ton] + dfs_cic)
        combined = self.processor.clean(combined)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"Merged rows={len(combined):,} cols={len(combined.columns)}", "INFO")
        if not self.is_running:
            return

        # Split
        self.status_var.set("Splitting train/test")
        df_train, df_test = self.processor.split(combined)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"Split train={len(df_train):,} test={len(df_test):,}", "INFO")
        if not self.is_running:
            return

        # Export CSV
        self.status_var.set("Writing CSV")
        try:
            df_train.to_csv("fusion_train_smart4.csv", index=False, encoding="utf-8")
            df_test.to_csv("fusion_test_smart4.csv", index=False, encoding="utf-8")
            self.log("CSV written: fusion_train_smart4.csv / fusion_test_smart4.csv", "OK")
        except Exception:
            self.log(f"CSV export failed:\n{traceback.format_exc()}", "ERROR")
            self.add_alert("CSV export failed", "ERROR")
        step += 1
        self.progress_var.set(step / steps * 100)
        if not self.is_running:
            return

        # Export NPZ
        self.status_var.set("Writing NPZ")
        try:
            if "Label" not in df_train.columns:
                self.log("NPZ skipped: no Label column", "WARN")
            else:
                numeric_cols = [c for c in df_train.columns if c != "Label" and np.issubdtype(df_train[c].dtype, np.number)]
                if numeric_cols:
                    X_train = df_train[numeric_cols].astype(np.float32)
                    X_train = X_train.fillna(X_train.mean(numeric_only=True))
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
                    self.log(f"NPZ written: {len(numeric_cols)} features", "OK")
                else:
                    self.log("NPZ skipped: no numeric columns", "WARN")
        except Exception:
            self.log(f"NPZ export failed:\n{traceback.format_exc()}", "ERROR")
            self.add_alert("NPZ export failed", "ERROR")

        step += 1
        self.progress_var.set(100)
        self.status_var.set("Completed")

        elapsed = time.time() - (self.start_time or time.time())
        self.log(f"Finished in {self._format_duration(elapsed)}", "INFO")
        self.add_alert(f"Completed in {self._format_duration(elapsed)}", "OK")

    def _format_duration(self, seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _progress_callback(self, name: str):
        def _cb(idx, size, progress, thread_id, action=None):
            if not self.is_running:
                return
            msg = action or f"{name} part {idx} ({size:,} rows)"
            self.update_thread_progress(int(thread_id), float(progress), msg)
            # reduce spam: log every ~10 updates or major % step
            if int(progress) % 10 == 0:
                self.log(f"{name} {progress:.1f}% -> {msg}", "DEBUG")
            try:
                self.progress_detail.config(text=f"{name}: {msg} | {progress:.1f}%")
                self.progress_var.set(progress)
            except Exception:
                pass
        return _cb

# ============================================================
# main
# ============================================================

def main() -> None:
    root = tk.Tk()
    app = ConsolidationGUIEnhanced(root)
    root.mainloop()

if __name__ == "__main__":
    main()











