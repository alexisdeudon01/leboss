#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Consolidation GUI - STREAMING + CHECKPOINT + AI OPTIMIZATION v6
====================================================================

What this version fixes vs v5:
- ✅ Removed old embedded LinUCB algorithm (was redundant)
- ✅ Now uses ONLY AIOptimizationServer (separate thread, clean architecture)
- ✅ Removed duplicate code in start_monitoring_loop()
- ✅ Cleaner metrics handling every 5 seconds
- ✅ Simplified UI - no longer needs to redraw old algorithm graphs
- ✅ All optimization logic in AIOptimizationServer with separate monitoring GUI
- ✅ Main GUI focused on pipeline progress

Outputs (final):
- fusion_train_smart6.csv
- fusion_test_smart6.csv
- preprocessed_dataset.npz (best effort; falls back to memmap folder if too big)

Python: 3.10+

Notes:
- FULL_RUN=1 (default) means full streaming processing.
- FULL_RUN=0 means sample mode; SAMPLE_ROWS controls nrows per file.
- Checkpoint + staging dir: ".run_state/" next to this script.
- AIOptimizationServer runs in separate thread with its own GUI.

"""

from __future__ import annotations

import csv
import gc
import json
import os
import time
import queue
import shutil
import hashlib
import traceback
import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sklearn.preprocessing import StandardScaler, LabelEncoder
from ai_server_with_gui import AIOptimizationServer, Metrics as AIMetrics

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Data Consolidation - Streaming v6 (checkpoint + AI)"
PROGRESS_TITLE = "Overall progress"

MAX_RAM_PERCENT = int(os.getenv("MAX_RAM_PERCENT", "90"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50_000"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "750_000"))
MAX_THREADS = int(os.getenv("MAX_THREADS", "12"))

FULL_RUN = os.getenv("FULL_RUN", "1").strip() == "1"
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "10_000"))  # used only when FULL_RUN=0

TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.60"))
UI_REFRESH_MS = 1000  # exactly 1 second

RUN_STATE_DIR = Path(os.getenv("RUN_STATE_DIR", ".run_state")).resolve()
CHECKPOINT_FILE = RUN_STATE_DIR / "checkpoint.json"
UNION_COLS_FILE = RUN_STATE_DIR / "union_cols.json"
PARTS_DIR = RUN_STATE_DIR / "parts"

FINAL_TRAIN = Path("fusion_train_smart6.csv").resolve()
FINAL_TEST = Path("fusion_test_smart6.csv").resolve()

NPZ_OUT = Path("preprocessed_dataset.npz").resolve()
NPZ_FALLBACK_DIR = Path("preprocessed_dataset_memmap").resolve()

UI_FONT = ("Arial", 10)
UI_FONT_BOLD = ("Arial", 10, "bold")
SMALL_FONT = ("Arial", 9)
SMALL_FONT_BOLD = ("Arial", 9, "bold")

TARGET_FLOAT_DTYPE = np.float32
NPZ_FLOAT_DTYPE = np.float64  # safer for very large magnitudes during NPZ build

# ============================================================
# Helpers
# ============================================================

def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _human_bytes(n: float) -> str:
    unit = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(unit) - 1:
        n /= 1024
        i += 1
    return f"{n:.1f}{unit[i]}"

# ============================================================
# UI Canvas Feed
# ============================================================

class CanvasFeed:
    """Scrollable feed based on Canvas -> Frame -> Labels (fast enough for logs)."""

    def __init__(self, parent, *, height=140, max_items=600, bg="#ffffff", fg="#2c3e50"):
        self.max_items = int(max_items)
        self.bg = bg
        self.fg = fg
        self.canvas = tk.Canvas(parent, height=height, bg=bg, highlightthickness=1, highlightbackground="#ccc")
        self.scroll = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def pack(self, **kwargs):
        kwargs_canvas = dict(kwargs)
        kwargs_canvas.setdefault("side", "left")
        kwargs_canvas.setdefault("fill", "both")
        kwargs_canvas.setdefault("expand", True)
        self.canvas.pack(**kwargs_canvas)
        self.scroll.pack(side="right", fill="y")

    def add(self, label: str, message: str, color: str | None = None):
        color = color or self.fg
        row = tk.Frame(self.inner, bg=self.bg)
        tk.Label(row, text=label, fg=color, width=8, font=SMALL_FONT_BOLD, bg=self.bg).pack(side="left", padx=4)
        tk.Label(
            row,
            text=message,
            fg=color,
            wraplength=1300,
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


class DualColorProgressBar:
    """Canvas-based progress bar: existing progress stays base_color, new delta is red."""

    def __init__(
        self,
        parent,
        variable: tk.DoubleVar,
        *,
        height: int = 18,
        base_color: str = "#22c55e",
        delta_color: str = "#ef4444",
        trough_color: str = "#e5e7eb",
        border_color: str = "#cbd5e1",
    ) -> None:
        self.variable = variable
        self.base_color = base_color
        self.delta_color = delta_color
        self.trough_color = trough_color
        self.border_color = border_color
        self.height = int(height)

        self.prev_value = float(self.variable.get() or 0.0)
        self.curr_value = float(self.variable.get() or 0.0)

        bg = "#ffffff"
        self.canvas = tk.Canvas(parent, height=self.height + 4, highlightthickness=0, bg=bg)
        self._trace = self.variable.trace_add("write", self._on_change)
        self.canvas.bind("<Configure>", lambda _e: self._draw())
        self._draw()

    def _on_change(self, *_args) -> None:
        try:
            new_val = float(self.variable.get())
        except Exception:
            new_val = 0.0
        new_val = max(0.0, min(100.0, new_val))

        if new_val >= self.curr_value:
            self.prev_value = self.curr_value
        else:
            self.prev_value = new_val
        self.curr_value = new_val
        self._draw()

    def _draw(self) -> None:
        try:
            c = self.canvas
            c.delete("all")
            w = max(int(c.winfo_width()), 1)
            h = self.height
            pad = 2
            x0, y0, x1, y1 = pad, pad, w - pad, h + pad
            c.create_rectangle(x0, y0, x1, y1, fill=self.trough_color, outline=self.border_color)

            total_w = max((x1 - x0), 1)
            green_w = int(total_w * max(min(self.prev_value, self.curr_value), 0.0) / 100.0)
            red_w = int(total_w * max(self.curr_value - self.prev_value, 0.0) / 100.0)

            if green_w > 0:
                c.create_rectangle(x0, y0, x0 + green_w, y1, fill=self.base_color, outline="")
            if red_w > 0:
                c.create_rectangle(x0 + green_w, y0, x0 + green_w + red_w, y1, fill=self.delta_color, outline="")
        except Exception:
            pass

    def grid(self, *args, **kwargs):
        return self.canvas.grid(*args, **kwargs)

    def pack(self, *args, **kwargs):
        return self.canvas.pack(*args, **kwargs)

    def destroy(self) -> None:
        try:
            if self._trace:
                self.variable.trace_remove("write", self._trace)
        except Exception:
            pass
        try:
            self.canvas.destroy()
        except Exception:
            pass

# ============================================================
# Checkpointing
# ============================================================

@dataclass
class Checkpoint:
    run_id: str
    toniot_file: str | None
    cic_dir: str | None
    full_run: bool
    sample_rows: int
    union_cols_ready: bool
    union_cols_hash: str | None
    completed_files: list[str]
    stage: str
    rows_written_train: int
    rows_written_test: int
    last_update_ts: str

    @classmethod
    def fresh(cls) -> "Checkpoint":
        return cls(
            run_id=_sha1(str(time.time())),
            toniot_file=None,
            cic_dir=None,
            full_run=FULL_RUN,
            sample_rows=SAMPLE_ROWS,
            union_cols_ready=False,
            union_cols_hash=None,
            completed_files=[],
            stage="idle",
            rows_written_train=0,
            rows_written_test=0,
            last_update_ts=_now_ts(),
        )

class CheckpointManager:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def load(self) -> Checkpoint:
        if not self.path.exists():
            return Checkpoint.fresh()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Checkpoint(**data)
        except Exception:
            return Checkpoint.fresh()

    def save(self, ckpt: Checkpoint) -> None:
        with self.lock:
            ckpt.last_update_ts = _now_ts()
            _safe_mkdir(self.path.parent)
            self.path.write_text(json.dumps(ckpt.__dict__, indent=2), encoding="utf-8")

# ============================================================
# Monitor (metrics + throughput)
# ============================================================

class AdvancedMonitor:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.ram_peak = 0.0
        self.cpu_peak = 0.0
        self.total_rows_seen = 0
        self.total_files_done = 0
        self.lock = threading.Lock()
        self._prev_rows = 0
        self._prev_t = time.time()
        self.rows_per_sec = 0.0

    def record_metric(self) -> tuple[float, float]:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.0)
        with self.lock:
            self.ram_peak = max(self.ram_peak, vm.percent)
            self.cpu_peak = max(self.cpu_peak, cpu)
        return vm.percent, cpu

    def track_rows(self, rows: int) -> None:
        with self.lock:
            self.total_rows_seen += int(rows)

    def track_file_done(self) -> None:
        with self.lock:
            self.total_files_done += 1

    def update_throughput(self) -> float:
        with self.lock:
            now = time.time()
            dt = max(now - self._prev_t, 1e-6)
            dr = self.total_rows_seen - self._prev_rows
            self.rows_per_sec = dr / dt
            self._prev_rows = self.total_rows_seen
            self._prev_t = now
            return self.rows_per_sec

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "elapsed": elapsed,
                "ram_peak": self.ram_peak,
                "cpu_peak": self.cpu_peak,
                "rows_seen": self.total_rows_seen,
                "files_done": self.total_files_done,
                "rows_per_sec": self.rows_per_sec,
            }

# ============================================================
# Processor (streaming writers + dynamic chunking)
# ============================================================

class OptimizedDataProcessor:
    def __init__(self, monitor: AdvancedMonitor, logger=None) -> None:
        self.monitor = monitor
        self.logger = logger

        self.max_ram_percent = MAX_RAM_PERCENT
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.max_chunk_size = MAX_CHUNK_SIZE
        self.max_threads = MAX_THREADS
        self.current_worker_cap = 1
        self.current_chunk_size = MIN_CHUNK_SIZE

        vm = psutil.virtual_memory()
        ram_gb = vm.total / (1024**3)
        if ram_gb < 8:
            self.max_chunk_size = min(self.max_chunk_size, 300_000)
            self.max_threads = min(self.max_threads, 4)
        elif ram_gb < 16:
            self.max_chunk_size = min(self.max_chunk_size, 500_000)
            self.max_threads = min(self.max_threads, 8)

        self.chunk_multiplier = 1.0
        self.last_worker_reason = "init"
        self.last_chunk_reason = "init"
        self.last_ai_action = "init"
        self.last_chunk_delta = 0
        self.last_worker_delta = 0

        self._log(
            f"Caps: RAM={ram_gb:.1f}GB | max_chunk={self.max_chunk_size:,} | max_threads={self.max_threads} | FULL_RUN={FULL_RUN} | SAMPLE_ROWS={SAMPLE_ROWS}",
            "INFO",
        )

    def _log(self, msg: str, level: str = "DEBUG") -> None:
        if self.logger:
            try:
                self.logger(msg, level)
            except Exception:
                pass

    # --------------------------
    # dtype optimization + label
    # --------------------------

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
                    nunq = int(df[col].nunique(dropna=False))
                    if nunq / max(len(df[col]), 1) < 0.5 or nunq < 50:
                        df[col] = df[col].astype("category")
                except Exception:
                    pass
            return df
        except Exception:
            self._log(f"optimize_dtypes failed:\n{traceback.format_exc()}", "WARN")
            return df

    def ensure_label_is_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for c in df.columns:
                if str(c).strip().lower() == "label":
                    if c != "Label":
                        df = df.rename(columns={c: "Label"})
                    return df
            return df
        except Exception:
            return df

    def _sanitize_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf by NaN and clip numeric values to avoid float32 overflows downstream."""
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                return df
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            df[num_cols] = df[num_cols].clip(-1e6, 1e6)
            return df
        except Exception:
            self._log(f"sanitize_chunk failed:\n{traceback.format_exc()}", "WARN")
            return df

    @staticmethod
    def _sanitize_numeric_block(
        df: pd.DataFrame,
        numeric_cols: list[str],
        *,
        clip_val: float | None = 1e6,
        dtype: Any = np.float32,
    ) -> pd.DataFrame:
        """Sanitize numeric columns: replace inf, optional clip, fill NaN with column mean."""
        try:
            if not numeric_cols:
                return df
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            if clip_val is not None:
                df[numeric_cols] = df[numeric_cols].clip(-clip_val, clip_val)
            means = df[numeric_cols].mean(numeric_only=True)
            df[numeric_cols] = df[numeric_cols].fillna(means).astype(dtype, copy=False)
            return df
        except Exception:
            return df

    # --------------------------
    # union columns (schema)
    # --------------------------

    def build_union_columns(self, toniot: str, cic_files: list[str], sample_rows: int | None) -> list[str]:
        cols: set[str] = set()

        def _add_cols(path: str) -> None:
            try:
                df0 = pd.read_csv(path, nrows=0, low_memory=False)
                cols.update([str(c) for c in df0.columns])
            except Exception:
                self._log(f"[SCHEMA] failed header read: {Path(path).name}", "WARN")

        _add_cols(toniot)
        for p in cic_files:
            _add_cols(p)

        # normalize label name in schema
        cols_norm = []
        has_label = False
        for c in cols:
            if c.strip().lower() == "label":
                has_label = True
            else:
                cols_norm.append(c)
        if has_label:
            cols_norm.append("Label")

        # stable order: keep Label at end if present
        cols_norm = sorted(set(cols_norm), key=lambda x: (x == "Label", x.lower()))
        return cols_norm

    # --------------------------
    # chunk sizing
    # --------------------------

    def _estimate_bytes_per_row(self, filepath: str) -> float:
        try:
            sample = pd.read_csv(filepath, nrows=2000, low_memory=False, memory_map=True)
            sample = self._optimize_dtypes(sample)
            b = float(sample.memory_usage(deep=True).sum())
            return b / max(len(sample), 1)
        except Exception:
            return 512.0

    def _estimate_chunk_size(self, filepath: str, workers_hint: int) -> int:
        vm = psutil.virtual_memory()
        headroom = max((self.max_ram_percent / 100 * vm.total) - vm.used, vm.available * 0.5)
        budget = max(headroom * 0.55, 64 * 1024 * 1024)

        per_row = self._estimate_bytes_per_row(filepath)
        parallel_penalty = 1.0 + 0.12 * max(workers_hint - 1, 0)
        est = int((budget / max(per_row, 1.0)) / parallel_penalty)

        # AI chunk multiplier
        est = int(est * float(self.chunk_multiplier))

        # safety by current RAM
        if vm.percent > 70:
            est = int(est * 0.50)
        elif vm.percent > 60:
            est = int(est * 0.65)

        chunk = max(self.min_chunk_size, min(self.max_chunk_size, est))
        return chunk

    def _tune_chunk_size(self, current_size: int, *, ram: float, workers_cap: int) -> int:
        mult = 1.0
        reason = []

        target_ram = float(self.max_ram_percent) - 1.0
        if ram >= self.max_ram_percent:
            mult *= 0.40
            reason.append("RAM>=ceiling => hard shrink")
        elif ram > target_ram + 2:
            mult *= 0.65
            reason.append("RAM>target => shrink")
        elif ram < target_ram - 10:
            mult *= 1.25
            reason.append("RAM<<target => grow")
        elif ram < target_ram - 4:
            mult *= 1.12
            reason.append("RAM<target => grow")

        mult /= (1.0 + 0.10 * max(workers_cap - 1, 0))
        if workers_cap > 1:
            reason.append(f"workers_cap={workers_cap} => penalty")

        tuned = int(current_size * mult)
        tuned = max(self.min_chunk_size, min(self.max_chunk_size, tuned))
        self.last_chunk_reason = f"{current_size:,}->{tuned:,} | RAM {ram:.1f}% | mult {mult:.2f} | " + "; ".join(reason)
        self.last_chunk_delta = tuned - current_size
        return tuned

    def _rebalance_chunk_size(self, current_size: int, *, workers_cap: int, per_row: float) -> int:
        """Fair-share headroom across active workers to let chunks regrow when RAM is available."""
        try:
            vm = psutil.virtual_memory()
            target_ram = float(self.max_ram_percent) - 1.0
            if vm.percent >= target_ram - 5:
                return current_size

            headroom = max((self.max_ram_percent / 100.0 * vm.total) - vm.used, vm.available * 0.5)
            fair_budget = max(headroom * 0.9 / max(workers_cap, 1), 0.0)
            est_rows = int(fair_budget / max(per_row, 1.0))
            est_rows = max(self.min_chunk_size, min(self.max_chunk_size, est_rows))

            if est_rows > current_size:
                grown = int(min(self.max_chunk_size, current_size * 0.6 + est_rows * 0.4))
                return grown
            return current_size
        except Exception:
            return current_size

    # --------------------------
    # splitting (streaming stratified-ish)
    # --------------------------

    @staticmethod
    def _split_train_test_streaming(df: pd.DataFrame, train_ratio: float, counters: dict[str, tuple[int, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Per-label streaming split:
          counters[label] = (train_count, test_count)
        Rule: push to the side that is behind target ratio.
        """
        if "Label" not in df.columns:
            mask = np.random.RandomState(42).rand(len(df)) < train_ratio
            return df[mask].copy(), df[~mask].copy()

        train_idx = []
        test_idx = []
        for i, lab in enumerate(df["Label"].astype(str).tolist()):
            tr, te = counters.get(lab, (0, 0))
            total = tr + te
            desired_tr = int(round((total + 1) * train_ratio))
            if tr < desired_tr:
                train_idx.append(i)
                counters[lab] = (tr + 1, te)
            else:
                test_idx.append(i)
                counters[lab] = (tr, te + 1)

        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    # --------------------------
    # parts writing helpers
    # --------------------------

    @staticmethod
    def _write_csv_append(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")

    # --------------------------
    # main per-file worker
    # --------------------------

    def process_file_to_parts(
        self,
        *,
        filepath: str,
        union_cols: list[str],
        out_dir: Path,
        sample_rows: int | None,
        callback=None,
        thread_id: int = 0,
        stop_flag: threading.Event | None = None,
    ) -> tuple[str, int, int]:
        """
        Streams one CSV into 2 part files (train/test). Returns (filepath, train_rows, test_rows).
        Safe for parallel execution because output is per-file.
        """
        p = Path(filepath)
        file_key = _sha1(str(p.resolve()))
        train_part = out_dir / f"train__{file_key}__{p.name}.csv"
        test_part = out_dir / f"test__{file_key}__{p.name}.csv"

        # If parts already exist, we consider it done (resume).
        if train_part.exists() and test_part.exists() and train_part.stat().st_size > 0:
            # count rows quickly (best effort)
            tr = self._count_csv_rows_fast(train_part)
            te = self._count_csv_rows_fast(test_part)
            self._log(f"[RESUME] skip {p.name} (parts exist) train~{tr:,} test~{te:,}", "INFO")
            return filepath, tr, te

        counters: dict[str, tuple[int, int]] = {}
        train_written = 0
        test_written = 0

        per_row = self._estimate_bytes_per_row(filepath)
        try:
            fsize = float(p.stat().st_size)
        except Exception:
            fsize = 1.0

        workers_hint = max(1, int(self.current_worker_cap))
        chunk_size = self._estimate_chunk_size(filepath, workers_hint=workers_hint)

        self._log(f"[FILE] start {p.name} | chunk={chunk_size:,} | est_row≈{per_row:.0f}B | size={_human_bytes(fsize)} | T{thread_id}", "INFO")

        t0 = time.time()
        bytes_done = 0.0
        last_tick = time.time()
        last_rows = 0

        def _progress(chunk_i: int, rows_i: int) -> None:
            nonlocal bytes_done, last_tick, last_rows
            bytes_done = min(fsize, bytes_done + rows_i * per_row)
            pct = 100.0 * (bytes_done / max(fsize, 1.0))
            now = time.time()
            dt = max(now - last_tick, 1e-6)
            rps = (rows_i + last_rows) / dt
            last_rows = 0
            last_tick = now

            if callback:
                callback(chunk_i, rows_i, min(100.0, pct), thread_id, f"{p.name} chunk {chunk_i} ({rows_i:,} rows) r/s≈{rps:,.0f}")

        # sample mode
        if sample_rows is not None:
            df = pd.read_csv(filepath, nrows=sample_rows, low_memory=False, memory_map=True)
            df = self._optimize_dtypes(self.ensure_label_is_standard(df))
            df = self._sanitize_chunk(df)
            df = df.reindex(columns=union_cols)
            if "Label" in df.columns:
                df = df.dropna(subset=["Label"])
            tr_df, te_df = self._split_train_test_streaming(df, TRAIN_RATIO, counters)
            if len(tr_df) > 0:
                self._write_csv_append(train_part, tr_df)
            if len(te_df) > 0:
                self._write_csv_append(test_part, te_df)
            train_written += len(tr_df)
            test_written += len(te_df)
            self.monitor.track_rows(len(df))
            ram, cpu = self.monitor.record_metric()
            self._log(f"[FILE] sample done {p.name} rows={len(df):,} -> train={train_written:,} test={test_written:,} | RAM {ram:.1f}% CPU {cpu:.1f}%", "OK")
            _progress(1, len(df))
            return filepath, train_written, test_written

        # full streaming
        chunk_i = 0
        try:
            reader = pd.read_csv(filepath, chunksize=chunk_size, low_memory=False, memory_map=True)
            for chunk in reader:
                if stop_flag and stop_flag.is_set():
                    self._log(f"[STOP] requested, stop reading {p.name}", "WARN")
                    break

                chunk_i += 1
                chunk = self._optimize_dtypes(self.ensure_label_is_standard(chunk))
                chunk = self._sanitize_chunk(chunk)
                chunk = chunk.reindex(columns=union_cols)
                if "Label" in chunk.columns:
                    chunk = chunk.dropna(subset=["Label"])

                tr_df, te_df = self._split_train_test_streaming(chunk, TRAIN_RATIO, counters)
                if len(tr_df) > 0:
                    self._write_csv_append(train_part, tr_df)
                if len(te_df) > 0:
                    self._write_csv_append(test_part, te_df)

                train_written += len(tr_df)
                test_written += len(te_df)

                self.monitor.track_rows(len(chunk))
                ram, cpu = self.monitor.record_metric()
                workers_cap = max(1, int(getattr(self, "current_worker_cap", workers_hint)))
                chunk_size = self._tune_chunk_size(chunk_size, ram=ram, workers_cap=workers_cap)
                rebalance = self._rebalance_chunk_size(chunk_size, workers_cap=workers_cap, per_row=per_row)
                if rebalance != chunk_size:
                    self.last_chunk_reason += f" | rebalance->{rebalance:,}"
                chunk_size = rebalance
                try:
                    reader.chunksize = chunk_size
                except Exception:
                    pass

                _progress(chunk_i, len(chunk))

                # extra-verbose decision log (every chunk)
                if chunk_i % 1 == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    self._log(
                        f"[CHUNK] {p.name}#{chunk_i} rows={len(chunk):,} train+={len(tr_df):,} test+={len(te_df):,} "
                        f"| tot_train={train_written:,} tot_test={test_written:,} | RAM {ram:.1f}% CPU {cpu:.1f}% "
                        f"| chunk_next={chunk_size:,} | cap={workers_cap}",
                        "DEBUG",
                    )

                if ram >= self.max_ram_percent:
                    gc.collect()
                    time.sleep(0.15)

        except Exception:
            self._log(f"[FILE] crashed {p.name}:\n{traceback.format_exc()}", "ERROR")

        elapsed = time.time() - t0
        self._log(f"[FILE] done {p.name} in {elapsed:.1f}s -> train={train_written:,} test={test_written:,}", "OK")
        return filepath, train_written, test_written

    @staticmethod
    def _count_csv_rows_fast(path: Path, max_lines: int = 2_000_000) -> int:
        # Best-effort: count lines minus header (capped for speed)
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                n = 0
                for n, _ in enumerate(f, 1):
                    if n >= max_lines:
                        return max_lines
            return max(0, n - 1)
        except Exception:
            return 0

# ============================================================
# GUI
# ============================================================

class ConsolidationGUIEnhanced:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x1050")
        self.root.minsize(1350, 950)

        self.monitor = AdvancedMonitor()
        self.processor = OptimizedDataProcessor(self.monitor, logger=self.log)

        # ===== AI OPTIMIZATION SERVER =====
        # NOW WORKS! Uses root.update() in server thread instead of daemon GUI
        self.ai_server = AIOptimizationServer(
            max_workers=MAX_THREADS,
            max_chunk_size=MAX_CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_ram_percent=MAX_RAM_PERCENT,
            show_gui=True,  # ✅ NOW WORKS!
        )

        self.ai_server_thread = threading.Thread(
            target=self.ai_server.run,
            daemon=True,
            name="AIOptimizationServer",
        )
        self.ai_server_thread.start()
        self.log("AI Optimization Server started", "INFO")

        self._ai_metrics_counter = 0
        self._ai_last_rows_seen = 0
        # ===== END AI =====

        self.ckpt_mgr = CheckpointManager(CHECKPOINT_FILE)
        self.ckpt = self.ckpt_mgr.load()

        self.toniot_file: str | None = self.ckpt.toniot_file
        self.cic_dir: str | None = self.ckpt.cic_dir

        self.is_running = False
        self.stop_event = threading.Event()

        # UI vars
        self.status_var = tk.StringVar(value="Idle")
        self.ram_var = tk.StringVar(value="-- %")
        self.cpu_var = tk.StringVar(value="-- %")
        self.rps_var = tk.StringVar(value="Rows/s: --")
        self.rows_var = tk.StringVar(value="Rows seen: 0")
        self.files_var = tk.StringVar(value="Files done: 0")

        self.progress_overall = tk.DoubleVar(value=0.0)
        self.progress_ton = tk.DoubleVar(value=0.0)
        self.progress_cic = tk.DoubleVar(value=0.0)
        self.progress_finalize = tk.DoubleVar(value=0.0)
        self.progress_npz = tk.DoubleVar(value=0.0)
        self._progress_cache = {"overall": 0.0, "ton": 0.0, "cic": 0.0, "finalize": 0.0, "npz": 0.0}
        self._progress_var_to_key = {
            id(self.progress_overall): "overall",
            id(self.progress_ton): "ton",
            id(self.progress_cic): "cic",
            id(self.progress_finalize): "finalize",
            id(self.progress_npz): "npz",
        }

        self.thread_bars: dict[int, dict[str, Any]] = {}

        self._style = ttk.Style()
        self._init_styles()

        self.setup_ui()
        self._restore_ui_from_checkpoint()
        self.start_monitoring_loop()

    # ---------------- Styles ----------------

    def _init_styles(self) -> None:
        self._style.configure("StageTON.Horizontal.TProgressbar", troughcolor="#e5e7eb", background="#2563eb")
        self._style.configure("StageCIC.Horizontal.TProgressbar", troughcolor="#e5e7eb", background="#16a34a")
        self._style.configure("StageFIN.Horizontal.TProgressbar", troughcolor="#e5e7eb", background="#f59e0b")
        self._style.configure("StageNPZ.Horizontal.TProgressbar", troughcolor="#e5e7eb", background="#7c3aed")
        self._style.configure("StageALL.Horizontal.TProgressbar", troughcolor="#e5e7eb", background="#0ea5e9")

    # ---------------- UI ----------------

    def setup_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        # Input
        input_frame = ttk.LabelFrame(top, text="Input", padding=8)
        input_frame.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ttk.Label(input_frame, text="TON_IoT CSV:", font=UI_FONT_BOLD).grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.toniot_path_label = ttk.Label(input_frame, text=self.toniot_file or "No file selected", font=UI_FONT)
        self.toniot_path_label.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(input_frame, text="Browse", command=self.select_toniot).grid(row=0, column=2, sticky="e", padx=5, pady=3)

        ttk.Label(input_frame, text="CIC Folder:", font=UI_FONT_BOLD).grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.cic_path_label = ttk.Label(input_frame, text=self.cic_dir or "No folder selected", font=UI_FONT)
        self.cic_path_label.grid(row=1, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(input_frame, text="Browse", command=self.select_cic).grid(row=1, column=2, sticky="e", padx=5, pady=3)

        input_frame.columnconfigure(1, weight=1)

        # Controls
        control_frame = ttk.LabelFrame(top, text="Controls", padding=8)
        control_frame.pack(side="left", fill="x")

        self.start_button = ttk.Button(control_frame, text="▶ Start / Resume", command=self.start_consolidation)
        self.start_button.grid(row=0, column=0, padx=6, pady=4, sticky="ew")

        self.stop_button = ttk.Button(control_frame, text="⏹ Stop", command=self.stop_consolidation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=6, pady=4, sticky="ew")

        self.reset_button = ttk.Button(control_frame, text="Reset state", command=self.reset_state)
        self.reset_button.grid(row=1, column=0, columnspan=2, padx=6, pady=4, sticky="ew")

        # Alerts
        alerts_frame = ttk.LabelFrame(self.root, text="Alerts / Errors", padding=6)
        alerts_frame.pack(fill="x", padx=10, pady=6)
        self.alert_feed = CanvasFeed(alerts_frame, height=110, max_items=200)
        self.alert_feed.pack(fill="both", expand=True)

        # Main Paned layout
        paned = ttk.PanedWindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        upper = ttk.Frame(paned)
        lower = ttk.Frame(paned)
        paned.add(upper, weight=5)
        paned.add(lower, weight=1)

        # Monitoring + progress bars
        monitor_frame = ttk.LabelFrame(upper, text="Monitoring + Progress", padding=8)
        monitor_frame.pack(fill="x", pady=(0, 6))

        ttk.Label(monitor_frame, text="RAM:", font=UI_FONT_BOLD).grid(row=0, column=0, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.ram_var, font=UI_FONT).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(monitor_frame, text="CPU:", font=UI_FONT_BOLD).grid(row=0, column=2, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.cpu_var, font=UI_FONT).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(monitor_frame, textvariable=self.rps_var, font=UI_FONT).grid(row=0, column=4, sticky="w", padx=12)

        ttk.Label(monitor_frame, textvariable=self.rows_var, font=UI_FONT).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.files_var, font=UI_FONT).grid(row=1, column=3, columnspan=2, sticky="w")

        ttk.Label(monitor_frame, text="Status:", font=UI_FONT_BOLD).grid(row=2, column=0, sticky="w")
        ttk.Label(monitor_frame, textvariable=self.status_var, font=UI_FONT).grid(row=2, column=1, columnspan=4, sticky="w")

        # Overall + stage bars (color-coded)
        ttk.Label(monitor_frame, text=PROGRESS_TITLE + ":", font=UI_FONT_BOLD).grid(row=3, column=0, sticky="w")
        self.pb_overall = DualColorProgressBar(
            monitor_frame, self.progress_overall, base_color="#16a34a", delta_color="#ef4444"
        )
        self.pb_overall.grid(row=3, column=1, columnspan=4, sticky="ew", padx=5, pady=3)

        ttk.Label(monitor_frame, text="TON:", font=SMALL_FONT_BOLD).grid(row=4, column=0, sticky="w")
        self.pb_ton = DualColorProgressBar(
            monitor_frame, self.progress_ton, base_color="#2563eb", delta_color="#ef4444"
        )
        self.pb_ton.grid(row=4, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(monitor_frame, text="CIC:", font=SMALL_FONT_BOLD).grid(row=4, column=2, sticky="w")
        self.pb_cic = DualColorProgressBar(
            monitor_frame, self.progress_cic, base_color="#16a34a", delta_color="#ef4444"
        )
        self.pb_cic.grid(row=4, column=3, sticky="ew", padx=5, pady=2)

        ttk.Label(monitor_frame, text="Finalize:", font=SMALL_FONT_BOLD).grid(row=5, column=0, sticky="w")
        self.pb_fin = DualColorProgressBar(
            monitor_frame, self.progress_finalize, base_color="#f59e0b", delta_color="#ef4444"
        )
        self.pb_fin.grid(row=5, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(monitor_frame, text="NPZ:", font=SMALL_FONT_BOLD).grid(row=5, column=2, sticky="w")
        self.pb_npz = DualColorProgressBar(
            monitor_frame, self.progress_npz, base_color="#7c3aed", delta_color="#ef4444"
        )
        self.pb_npz.grid(row=5, column=3, sticky="ew", padx=5, pady=2)

        monitor_frame.columnconfigure(1, weight=1)
        monitor_frame.columnconfigure(3, weight=1)

        # Thread bars
        threads_frame = ttk.LabelFrame(upper, text="Thread progress (never empty)", padding=6)
        threads_frame.pack(fill="both", expand=True)
        thread_canvas_holder = ttk.Frame(threads_frame)
        thread_canvas_holder.pack(fill="both", expand=True)
        self.thread_canvas = tk.Canvas(thread_canvas_holder, highlightthickness=0)
        self.thread_scroll = ttk.Scrollbar(thread_canvas_holder, orient="vertical", command=self.thread_canvas.yview)
        self.thread_canvas.configure(yscrollcommand=self.thread_scroll.set)
        self.thread_inner = ttk.Frame(self.thread_canvas)
        self.thread_inner.bind(
            "<Configure>", lambda e: self.thread_canvas.configure(scrollregion=self.thread_canvas.bbox("all"))
        )
        self.thread_inner_window = self.thread_canvas.create_window((0, 0), window=self.thread_inner, anchor="nw")
        self.thread_canvas.bind(
            "<Configure>", lambda e: self.thread_canvas.itemconfigure(self.thread_inner_window, width=e.width)
        )
        self.thread_canvas.pack(side="left", fill="both", expand=True)
        self.thread_scroll.pack(side="right", fill="y")

        # Logs
        logs_frame = ttk.LabelFrame(lower, text="Logs (VERY verbose)", padding=6)
        log_container = tk.Frame(logs_frame, bg="#0f172a")
        log_container.pack(fill="both", expand=True)
        self.log_feed = CanvasFeed(log_container, height=220, max_items=1600, bg="#0f172a", fg="#e2e8f0")
        self.log_feed.pack(fill="both", expand=True)
        logs_frame.pack(fill="both", expand=True)

        self.log("Log canvas ready", "INFO")

        self.ensure_thread_bars(max(1, int(self.processor.current_worker_cap)))

    # ---------------- UI events ----------------

    def _restore_ui_from_checkpoint(self) -> None:
        if self.toniot_file:
            self.toniot_path_label.config(text=self.toniot_file)
        if self.cic_dir:
            self.cic_path_label.config(text=self.cic_dir)

        if CHECKPOINT_FILE.exists():
            self.add_alert(f"Checkpoint found: stage={self.ckpt.stage} | completed_files={len(self.ckpt.completed_files)} | last={self.ckpt.last_update_ts}", "INFO")

    def select_toniot(self) -> None:
        filepath = filedialog.askopenfilename(title="Select TON_IoT CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filepath:
            self.toniot_file = filepath
            self.toniot_path_label.config(text=filepath)
            self.add_alert("TON_IoT selected", "OK")
            self._update_ckpt_inputs()

    def select_cic(self) -> None:
        folder = filedialog.askdirectory(title="Select CIC folder")
        if folder:
            self.cic_dir = folder
            self.cic_path_label.config(text=folder)
            self.add_alert("CIC folder selected", "OK")
            self._update_ckpt_inputs()

    def reset_state(self) -> None:
        if self.is_running:
            messagebox.showwarning("Running", "Stop first.")
            return

        try:
            if RUN_STATE_DIR.exists():
                shutil.rmtree(RUN_STATE_DIR, ignore_errors=True)
            for p in [FINAL_TRAIN, FINAL_TEST, NPZ_OUT]:
                if p.exists():
                    p.unlink()
            if NPZ_FALLBACK_DIR.exists():
                shutil.rmtree(NPZ_FALLBACK_DIR, ignore_errors=True)
        except Exception:
            pass

        self.ckpt = Checkpoint.fresh()
        self.ckpt_mgr.save(self.ckpt)
        self._set_progress_async(self.progress_overall, 0)
        self._set_progress_async(self.progress_ton, 0)
        self._set_progress_async(self.progress_cic, 0)
        self._set_progress_async(self.progress_finalize, 0)
        self._set_progress_async(self.progress_npz, 0)
        self.add_alert("State reset. Ready.", "OK")
        self.log("State reset. Ready.", "OK")

    def _update_ckpt_inputs(self) -> None:
        self.ckpt.toniot_file = self.toniot_file
        self.ckpt.cic_dir = self.cic_dir
        self.ckpt.full_run = FULL_RUN
        self.ckpt.sample_rows = SAMPLE_ROWS
        self.ckpt_mgr.save(self.ckpt)

    # ---------------- Logs/alerts ----------------

    def add_alert(self, message: str, level: str = "INFO") -> None:
        colors = {"ERROR": "#ef4444", "WARN": "#f59e0b", "INFO": "#38bdf8", "OK": "#22c55e"}
        color = colors.get(level.upper(), "#e2e8f0")
        self.root.after(0, lambda: self.alert_feed.add(level.upper(), message, color))

    def log(self, message: str, level: str = "INFO") -> None:
        colors = {"ERROR": "#ef4444", "WARN": "#f59e0b", "INFO": "#e2e8f0", "DEBUG": "#94a3b8", "OK": "#22c55e"}
        color = colors.get(level.upper(), "#e2e8f0")
        ts = datetime.now().strftime("%H:%M:%S")
        th = threading.current_thread().name
        if th == "MainThread":
            tag = "MAIN"
        elif th.startswith("T"):
            tag = th
        else:
            tag = th.split("-")[-1]
        msg = f"[{tag}] [{ts}] {message}"
        self.root.after(0, lambda: self.log_feed.add(level.upper(), msg, color))

    # ---------------- Thread bars ----------------

    def ensure_thread_bars(self, count: int) -> None:
        count = max(1, int(count))  # NEVER empty
        # remove extra
        for tid in sorted([k for k in self.thread_bars.keys() if k >= count], reverse=True):
            entry = self.thread_bars.pop(tid, None)
            if entry:
                try:
                    if entry.get("bar"):
                        entry["bar"].destroy()
                except Exception:
                    pass
                try:
                    entry["row"].destroy()
                except Exception:
                    pass
                self.log(f"[UI] Removed thread bar T{tid}", "DEBUG")

        # add missing
        for tid in range(count):
            if tid in self.thread_bars:
                continue
            var = tk.DoubleVar(value=0)
            row = ttk.Frame(self.thread_inner)
            ttk.Label(row, text=f"T{tid}", font=UI_FONT_BOLD, width=4).pack(side="left", padx=4)
            bar = DualColorProgressBar(row, var, base_color="#16a34a", delta_color="#ef4444")
            bar.pack(side="left", fill="x", expand=True, padx=4, pady=2)
            label = ttk.Label(row, text="Idle", width=62, font=SMALL_FONT)
            label.pack(side="left", padx=4)
            row.pack(fill="x", padx=2, pady=2)
            self.thread_bars[tid] = {"var": var, "label": label, "row": row, "bar": bar}
            self.log(f"[UI] Added thread bar T{tid}", "DEBUG")

        try:
            self.thread_inner.update_idletasks()
            self.thread_canvas.configure(scrollregion=self.thread_canvas.bbox("all"))
        except Exception:
            pass

    def update_thread_progress(self, thread_id: int, progress: float, text: str | None = None) -> None:
        entry = self.thread_bars.get(int(thread_id))
        if not entry:
            return

        def _update():
            try:
                entry["var"].set(float(progress))
                if text:
                    entry["label"].config(text=text[:115])
            except Exception:
                pass

        self.root.after(0, _update)

    def _set_progress_async(self, var: tk.DoubleVar, value: float) -> None:
        try:
            key = self._progress_var_to_key.get(id(var))
            if key:
                self._progress_cache[key] = float(value)
        except Exception:
            pass
        try:
            self.root.after(0, lambda v=var, x=float(value): v.set(x))
        except Exception:
            pass

    # ===== SIMPLIFIED MONITORING (AI handled by AIOptimizationServer) =====

    def start_monitoring_loop(self) -> None:
        try:
            ram, cpu = self.monitor.record_metric()
            _ = self.monitor.update_throughput()
            stats = self.monitor.get_stats()

            self.ram_var.set(f"{ram:.1f}% (peak {stats['ram_peak']:.1f}%)")
            self.cpu_var.set(f"{cpu:.1f}% (peak {stats['cpu_peak']:.1f}%)")
            throughput = stats.get("rows_per_sec", 0.0)
            self.rps_var.set(f"Rows/s (5s): {throughput:,.0f}")
            self.rows_var.set(f"Rows seen: {stats['rows_seen']:,}")
            self.files_var.set(f"Files done: {stats['files_done']:,}")

            # Send metrics to AI server EVERY second (not every 5!)
            # This keeps the AI window responsive
            try:
                ai_metrics = AIMetrics(
                    timestamp=time.time(),
                    num_workers=max(1, int(getattr(self.processor, "current_worker_cap", 1))),
                    chunk_size=int(getattr(self.processor, "current_chunk_size", MIN_CHUNK_SIZE)),
                    rows_processed=int(stats.get("rows_seen", 0) - self._ai_last_rows_seen),
                    ram_percent=float(ram),
                    cpu_percent=float(cpu),
                    throughput=float(throughput),
                )
                
                self.ai_server.send_metrics(ai_metrics)
                self._ai_last_rows_seen = stats.get("rows_seen", 0)
                
                # Get recommendation (non-blocking)
                rec = self.ai_server.get_recommendation(timeout=0.05)
                if rec:
                    self.log(
                        f"[AI] Δworkers={rec.d_workers:+d} chunk×={rec.chunk_mult:.2f} | {rec.reason}",
                        "INFO",
                    )
                    
                    try:
                        new_workers = max(1, min(MAX_THREADS, int(self.processor.current_worker_cap) + rec.d_workers))
                        self.processor.current_worker_cap = new_workers
                        self.ensure_thread_bars(new_workers)
                        
                        new_chunk = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, int(self.processor.current_chunk_size * rec.chunk_mult)))
                        self.processor.current_chunk_size = new_chunk
                    except Exception as e:
                        self.log(f"[AI] Failed to apply recommendation: {e}", "WARN")
                
            except Exception as e:
                self.log(f"[AI] Error in metrics: {e}", "WARN")

        except Exception:
            self.log(f"monitoring loop exception:\n{traceback.format_exc()}", "ERROR")

        self.root.after(UI_REFRESH_MS, self.start_monitoring_loop)

    # ===== END MONITORING =====

    # ---------------- Pipeline ----------------

    def start_consolidation(self) -> None:
        if not self.toniot_file or not self.cic_dir:
            messagebox.showerror("Missing input", "Select both TON_IoT CSV and CIC folder first.")
            self.add_alert("Select both TON_IoT CSV and CIC folder first.", "ERROR")
            return

        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.add_alert("Pipeline started (streaming + checkpoint + AI)", "INFO")
        self.log(f"Mode: {'FULL_RUN' if FULL_RUN else 'SAMPLE'} | SAMPLE_ROWS={SAMPLE_ROWS}", "INFO")

        self._update_ckpt_inputs()

        t = threading.Thread(target=self._pipeline_worker, daemon=True)
        t.start()

    def stop_consolidation(self) -> None:
        if not self.is_running:
            return
        self.stop_event.set()
        self.status_var.set("Stopping...")
        self.log("Stop requested: will stop after current chunk/file.", "WARN")

    def _pipeline_worker(self) -> None:
        try:
            self._run_pipeline()
        except Exception as exc:
            self.log(f"Critical error: {exc}\n{traceback.format_exc()}", "ERROR")
            self.add_alert(f"Critical error: {exc}", "ERROR")
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            try:
                if hasattr(self, "ai_server"):
                    self.ai_server.stop()
                    if hasattr(self.ai_server, "root") and self.ai_server.root:
                        try:
                            self.ai_server.root.quit()
                            self.ai_server.root.destroy()
                        except Exception:
                            pass
            except Exception:
                pass

    def _discover_cic_files(self, folder: str) -> list[str]:
        files: list[str] = []
        for root, _, names in os.walk(folder):
            for n in names:
                if n.lower().endswith(".csv"):
                    files.append(str(Path(root) / n))
        files.sort()
        return files

    def _run_pipeline(self) -> None:
        _safe_mkdir(RUN_STATE_DIR)
        _safe_mkdir(PARTS_DIR)

        # stage 0: discover files
        cic_files = self._discover_cic_files(self.cic_dir or "")
        if not cic_files:
            self.add_alert("No CIC CSV files found.", "ERROR")
            self.log("No CIC CSV files found.", "ERROR")
            return

        # sample control
        sample_rows = None if FULL_RUN else SAMPLE_ROWS

        # stage 1: union columns (schema)
        self.status_var.set("Schema: union columns")
        if not self.ckpt.union_cols_ready or not UNION_COLS_FILE.exists():
            self.ckpt.stage = "schema"
            self.ckpt_mgr.save(self.ckpt)

            union_cols = self.processor.build_union_columns(self.toniot_file, cic_files, sample_rows)
            UNION_COLS_FILE.write_text(json.dumps(union_cols, indent=2), encoding="utf-8")
            self.ckpt.union_cols_ready = True
            self.ckpt.union_cols_hash = _sha1(json.dumps(union_cols))
            self.ckpt_mgr.save(self.ckpt)

            self.log(f"[SCHEMA] union cols={len(union_cols)} (Label={'yes' if 'Label' in union_cols else 'no'})", "OK")
        else:
            union_cols = json.loads(UNION_COLS_FILE.read_text(encoding="utf-8"))
            self.log(f"[SCHEMA] loaded union cols={len(union_cols)}", "INFO")

        # stage 2: stream TON -> parts
        self.status_var.set("TON: streaming to parts")
        self.ckpt.stage = "ton"
        self.ckpt_mgr.save(self.ckpt)

        ton_key = str(Path(self.toniot_file).resolve())
        if ton_key not in self.ckpt.completed_files:
            self._process_one_file(
                filepath=self.toniot_file,
                union_cols=union_cols,
                part_dir=PARTS_DIR,
                sample_rows=sample_rows,
                stage_var=self.progress_ton,
                stage_name="TON",
            )
            if not self.stop_event.is_set():
                self.ckpt.completed_files.append(ton_key)
                self.ckpt_mgr.save(self.ckpt)
        else:
            self.log("[RESUME] TON already completed (checkpoint).", "INFO")
            self._set_progress_async(self.progress_ton, 100.0)

        if self.stop_event.is_set():
            self.status_var.set("Stopped")
            self.add_alert("Stopped by user.", "WARN")
            return

        # stage 3: stream CIC in parallel -> parts (dynamic worker cap)
        self.status_var.set("CIC: streaming to parts")
        self.ckpt.stage = "cic"
        self.ckpt_mgr.save(self.ckpt)

        remaining = [p for p in cic_files if str(Path(p).resolve()) not in set(self.ckpt.completed_files)]
        done_already = len(cic_files) - len(remaining)
        if done_already:
            self.log(f"[RESUME] CIC already done: {done_already}/{len(cic_files)}", "INFO")

        total = len(cic_files)
        completed = done_already

        from concurrent.futures import ThreadPoolExecutor, as_completed

        active = 0
        futures: dict[Any, tuple[str, int]] = {}

        start = time.time()
        idx_iter = 0

        with ThreadPoolExecutor(max_workers=self.processor.max_threads) as ex:
            # submit loop
            while idx_iter < len(cic_files):
                if self.stop_event.is_set():
                    break

                # skip completed
                p = cic_files[idx_iter]
                idx_iter += 1
                p_key = str(Path(p).resolve())
                if p_key in set(self.ckpt.completed_files):
                    continue

                # gate by cap
                while True:
                    cap = max(1, int(self.processor.current_worker_cap))
                    if active < cap:
                        break
                    time.sleep(0.05)
                    if self.stop_event.is_set():
                        break

                if self.stop_event.is_set():
                    break

                tid = (len(futures) % max(1, int(self.processor.current_worker_cap)))
                fut = ex.submit(
                    self.processor.process_file_to_parts,
                    filepath=p,
                    union_cols=union_cols,
                    out_dir=PARTS_DIR,
                    sample_rows=sample_rows,
                    callback=self._progress_callback(stage="CIC", stage_var=self.progress_cic, total_files=total),
                    thread_id=tid,
                    stop_flag=self.stop_event,
                )
                futures[fut] = (p, tid)
                active += 1

            # collect
            for fut in as_completed(list(futures.keys())):
                p, tid = futures.get(fut, ("unknown", 0))
                active = max(0, active - 1)

                try:
                    file_path, tr_rows, te_rows = fut.result()
                except Exception:
                    self.log(f"[CIC] crash future {Path(p).name}:\n{traceback.format_exc()}", "ERROR")
                    continue

                completed += 1
                self.monitor.track_file_done()

                p_key = str(Path(file_path).resolve())
                if p_key not in self.ckpt.completed_files:
                    self.ckpt.completed_files.append(p_key)
                self.ckpt_mgr.save(self.ckpt)

                self.update_thread_progress(tid, 100.0, f"{Path(file_path).name} done (train={tr_rows:,} test={te_rows:,})")
                self.log(f"[CIC] done {completed}/{total}: {Path(file_path).name} train={tr_rows:,} test={te_rows:,}", "OK")

                self._set_progress_async(self.progress_cic, 100.0 * (completed / max(total, 1)))

                # periodic checkpoint ping
                if (time.time() - start) > 4.0:
                    start = time.time()
                    self.ckpt_mgr.save(self.ckpt)

        if self.stop_event.is_set():
            self.status_var.set("Stopped")
            self.add_alert("Stopped by user.", "WARN")
            return

        self._set_progress_async(self.progress_cic, 100.0)

        # stage 4: finalize - concat parts into final train/test
        self.status_var.set("Finalize: assembling final CSVs")
        self.ckpt.stage = "finalize"
        self.ckpt_mgr.save(self.ckpt)
        self._finalize_from_parts(union_cols)

        # stage 5: NPZ (best effort)
        self.status_var.set("NPZ: building (best effort)")
        self.ckpt.stage = "npz"
        self.ckpt_mgr.save(self.ckpt)
        self._build_npz_best_effort(union_cols)

        # done
        self._set_progress_async(self.progress_overall, 100.0)
        self.status_var.set("Completed")
        self.ckpt.stage = "done"
        self.ckpt_mgr.save(self.ckpt)

        self.add_alert(f"Completed. Train={FINAL_TRAIN.name} Test={FINAL_TEST.name}", "OK")
        self.log(f"Completed. Train={FINAL_TRAIN} Test={FINAL_TEST}", "OK")
        if NPZ_OUT.exists():
            self.log(f"NPZ created: {NPZ_OUT} ({_human_bytes(NPZ_OUT.stat().st_size)})", "OK")
        elif NPZ_FALLBACK_DIR.exists():
            self.log(f"NPZ fallback created: {NPZ_FALLBACK_DIR}", "WARN")

    def _process_one_file(
        self,
        *,
        filepath: str,
        union_cols: list[str],
        part_dir: Path,
        sample_rows: int | None,
        stage_var: tk.DoubleVar,
        stage_name: str,
    ) -> None:
        self._set_progress_async(stage_var, 0.0)
        cb = self._progress_callback(stage=stage_name, stage_var=stage_var, total_files=1)
        fp, tr, te = self.processor.process_file_to_parts(
            filepath=filepath,
            union_cols=union_cols,
            out_dir=part_dir,
            sample_rows=sample_rows,
            callback=cb,
            thread_id=0,
            stop_flag=self.stop_event,
        )
        self.monitor.track_file_done()
        self._set_progress_async(stage_var, 100.0)
        self.log(f"[{stage_name}] done {Path(fp).name} -> train={tr:,} test={te:,}", "OK")

    def _progress_callback(self, *, stage: str, stage_var: tk.DoubleVar, total_files: int):
        def _cb(chunk_idx, rows, progress, thread_id, action=None):
            if self.stop_event.is_set():
                return
            msg = action or f"{stage} chunk {chunk_idx} ({rows:,} rows)"
            self.update_thread_progress(int(thread_id), float(progress), msg)
            self._set_progress_async(stage_var, float(progress))
            # overall weighted
            overall = (
                0.15 * self._progress_cache["ton"]
                + 0.65 * self._progress_cache["cic"]
                + 0.15 * self._progress_cache["finalize"]
                + 0.05 * self._progress_cache["npz"]
            )
            self._set_progress_async(self.progress_overall, overall)

            # intentionally verbose
            if int(progress) % 5 == 0:
                self.log(f"[{stage}] {progress:5.1f}% | {msg}", "DEBUG")
        return _cb

    def _finalize_from_parts(self, union_cols: list[str]) -> None:
        self._set_progress_async(self.progress_finalize, 0.0)

        train_parts = sorted(PARTS_DIR.glob("train__*.csv"))
        test_parts = sorted(PARTS_DIR.glob("test__*.csv"))
        if not train_parts or not test_parts:
            self.add_alert("No parts found to finalize. Did processing crash?", "ERROR")
            self.log("No parts found to finalize. Did processing crash?", "ERROR")
            return

        self.log(f"[FINALIZE] parts: train={len(train_parts)} test={len(test_parts)}", "INFO")

        def _concat(parts: list[Path], out_path: Path, label: str) -> None:
            if out_path.exists():
                out_path.unlink()
            _safe_mkdir(out_path.parent)

            total = len(parts)
            written = 0
            with out_path.open("w", encoding="utf-8", newline="") as out_f:
                writer = None
                for i, part in enumerate(parts, 1):
                    if self.stop_event.is_set():
                        break
                    with part.open("r", encoding="utf-8", errors="ignore", newline="") as in_f:
                        reader = csv.reader(in_f)
                        header = next(reader, None)
                        if header is None:
                            continue
                        if writer is None:
                            writer = csv.writer(out_f)
                            writer.writerow(header)
                        for row in reader:
                            writer.writerow(row)
                            written += 1
                    self._set_progress_async(self.progress_finalize, 100.0 * (i / max(total, 1)))
                    self._set_progress_async(
                        self.progress_overall,
                        0.15 * self._progress_cache["ton"]
                        + 0.65 * self._progress_cache["cic"]
                        + 0.15 * self._progress_cache["finalize"]
                        + 0.05 * self._progress_cache["npz"],
                    )
                    self.log(f"[FINALIZE] {label} {i}/{total} part={part.name} (rows~{written:,})", "DEBUG")

            self.log(f"[FINALIZE] wrote {label}: {out_path.name} rows~{written:,}", "OK")

        _concat(train_parts, FINAL_TRAIN, "train")
        _concat(test_parts, FINAL_TEST, "test")
        self._set_progress_async(self.progress_finalize, 100.0)

        if FINAL_TRAIN.exists():
            self.log(f"[FINALIZE] train size={_human_bytes(FINAL_TRAIN.stat().st_size)}", "INFO")
        if FINAL_TEST.exists():
            self.log(f"[FINALIZE] test size={_human_bytes(FINAL_TEST.stat().st_size)}", "INFO")

    def _build_npz_best_effort(self, union_cols: list[str]) -> None:
        self._set_progress_async(self.progress_npz, 0.0)
        self._set_progress_async(
            self.progress_overall,
            0.15 * self._progress_cache["ton"]
            + 0.65 * self._progress_cache["cic"]
            + 0.15 * self._progress_cache["finalize"]
            + 0.05 * self._progress_cache["npz"],
        )

        if not FINAL_TRAIN.exists():
            self.log("[NPZ] skipped: train CSV missing", "ERROR")
            self.add_alert("NPZ skipped: train CSV missing", "ERROR")
            return

        # determine numeric columns by sampling
        try:
            sample = pd.read_csv(FINAL_TRAIN, nrows=5000, low_memory=False)
        except Exception:
            self.log(f"[NPZ] failed reading train sample:\n{traceback.format_exc()}", "ERROR")
            return

        if "Label" not in sample.columns:
            self.log("[NPZ] skipped: no Label column", "WARN")
            self.add_alert("NPZ skipped: no Label column", "WARN")
            return

        numeric_cols = [c for c in sample.columns if c != "Label" and np.issubdtype(sample[c].dtype, np.number)]
        if not numeric_cols:
            self.log("[NPZ] skipped: no numeric columns", "WARN")
            self.add_alert("NPZ skipped: no numeric columns", "WARN")
            return

        self.log(f"[NPZ] numeric features={len(numeric_cols)} (2-pass streaming)", "INFO")

        # Pass 1: partial_fit scaler and collect labels
        scaler = StandardScaler(with_mean=True, with_std=True)
        le = LabelEncoder()
        label_set: set[str] = set()

        chunk_size = 150_000
        total_rows = 0

        try:
            for i, chunk in enumerate(pd.read_csv(FINAL_TRAIN, chunksize=chunk_size, low_memory=False), 1):
                if self.stop_event.is_set():
                    return
                y = chunk["Label"].astype(str)
                label_set.update(y.unique().tolist())
                chunk = OptimizedDataProcessor._sanitize_numeric_block(
                    chunk, numeric_cols, clip_val=None, dtype=NPZ_FLOAT_DTYPE
                )
                X = chunk[numeric_cols].astype(NPZ_FLOAT_DTYPE, copy=False)
                scaler.partial_fit(X)
                total_rows += len(chunk)
                if i % 1 == 0:
                    self._set_progress_async(self.progress_npz, min(45.0, 45.0 * (i / max(i + 1, 2))))
                    self._set_progress_async(
                        self.progress_overall,
                        0.15 * self._progress_cache["ton"]
                        + 0.65 * self._progress_cache["cic"]
                        + 0.15 * self._progress_cache["finalize"]
                        + 0.05 * self._progress_cache["npz"],
                    )
                    self.log(f"[NPZ] pass1 chunk#{i} rows={len(chunk):,} total={total_rows:,}", "DEBUG")
        except Exception:
            self.log(f"[NPZ] pass1 failed:\n{traceback.format_exc()}", "ERROR")
            return

        # fit label encoder
        le.fit(sorted(label_set))
        n = total_rows
        d = len(numeric_cols)

        # Decide memory strategy
        est_bytes = n * d * 4
        use_memmap = est_bytes > (1.2 * psutil.virtual_memory().available)
        if use_memmap:
            _safe_mkdir(NPZ_FALLBACK_DIR)
            X_path = NPZ_FALLBACK_DIR / "X_train.npy"
            y_path = NPZ_FALLBACK_DIR / "y_train.npy"
            meta_path = NPZ_FALLBACK_DIR / "meta.json"
            self.log(f"[NPZ] dataset too big for RAM -> memmap fallback ({_human_bytes(est_bytes)})", "WARN")
        else:
            X_arr = np.zeros((n, d), dtype=NPZ_FLOAT_DTYPE)
            y_arr = np.zeros((n,), dtype=np.int64)

        # Pass 2: transform and write
        row_cursor = 0
        try:
            if use_memmap:
                X_mm = np.memmap(X_path, mode="w+", dtype=NPZ_FLOAT_DTYPE, shape=(n, d))
                y_mm = np.memmap(y_path, mode="w+", dtype=np.int64, shape=(n,))
            for i, chunk in enumerate(pd.read_csv(FINAL_TRAIN, chunksize=chunk_size, low_memory=False), 1):
                if self.stop_event.is_set():
                    return
                y = le.transform(chunk["Label"].astype(str))
                chunk = OptimizedDataProcessor._sanitize_numeric_block(
                    chunk, numeric_cols, clip_val=None, dtype=NPZ_FLOAT_DTYPE
                )
                X = chunk[numeric_cols].astype(NPZ_FLOAT_DTYPE, copy=False)
                Xs = scaler.transform(X).astype(NPZ_FLOAT_DTYPE, copy=False)

                r = len(chunk)
                if use_memmap:
                    X_mm[row_cursor: row_cursor + r, :] = Xs
                    y_mm[row_cursor: row_cursor + r] = y
                else:
                    X_arr[row_cursor: row_cursor + r, :] = Xs
                    y_arr[row_cursor: row_cursor + r] = y
                row_cursor += r

                self._set_progress_async(self.progress_npz, 45.0 + 55.0 * (row_cursor / max(n, 1)))
                self._set_progress_async(
                    self.progress_overall,
                    0.15 * self._progress_cache["ton"]
                    + 0.65 * self._progress_cache["cic"]
                    + 0.15 * self._progress_cache["finalize"]
                    + 0.05 * self._progress_cache["npz"],
                )
                self.log(f"[NPZ] pass2 chunk#{i} rows={r:,} cursor={row_cursor:,}/{n:,}", "DEBUG")

            if use_memmap:
                X_mm.flush()
                y_mm.flush()
                meta = {"classes": le.classes_.tolist(), "numeric_cols": numeric_cols}
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                self.add_alert("NPZ fallback done: memmap folder created.", "WARN")
            else:
                # NPZ creation (in-RAM)
                np.savez_compressed(
                    NPZ_OUT,
                    X=X_arr,
                    y=y_arr,
                    classes=le.classes_,
                    numeric_cols=np.array(numeric_cols, dtype=object),
                )
                self.add_alert("NPZ created.", "OK")
        except Exception:
            self.log(f"[NPZ] pass2 failed:\n{traceback.format_exc()}", "ERROR")
            self.add_alert("NPZ build failed (see logs).", "ERROR")
        finally:
            self._set_progress_async(self.progress_npz, 100.0)

# ============================================================
# main
# ============================================================

def main() -> None:
    root = tk.Tk()
    app = ConsolidationGUIEnhanced(root)
    root.mainloop()

if __name__ == "__main__":
    main()