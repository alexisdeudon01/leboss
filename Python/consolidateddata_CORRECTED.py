#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data consolidation GUI with adaptive chunking and threading.
The loader targets a RAM ceiling (90% by default) by sizing chunks dynamically,
throttling threads when memory gets tight, and downcasting dtypes aggressively.
"""

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
from tkinter import ttk, filedialog, scrolledtext, messagebox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

MAX_RAM_PERCENT = 90
MIN_CHUNK_SIZE = 50_000
MAX_CHUNK_SIZE = 750_000
MAX_THREADS = 12
TARGET_FLOAT_DTYPE = np.float32
PROGRESS_TITLE = "Overall progress"


class SmartCache:
    """Tiny stats-only cache to avoid hoarding data in memory."""

    def __init__(self, max_items: int = 0) -> None:
        self.max_items = max_items
        self.cache = {}
        self.order = deque()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.hits += 1
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        if self.max_items <= 0:
            # Cache disabled; still count the access so stats stay meaningful.
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
            self.total_data_loaded += rows

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


class OptimizedDataProcessor:
    """Loader/processor tuned for low RAM pressure."""

    def __init__(
        self,
        monitor: AdvancedMonitor,
        cache: SmartCache,
        max_ram_percent: int = MAX_RAM_PERCENT,
        logger=None,
    ) -> None:
        self.monitor = monitor
        self.cache = cache
        self.max_ram_percent = max_ram_percent
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.max_chunk_size = MAX_CHUNK_SIZE
        self.processed_rows = 0
        self.start_time = time.time()
        self.logger = logger

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

    def _estimate_chunk_size(self, filepath: str) -> int:
        vm = psutil.virtual_memory()
        headroom = max((self.max_ram_percent / 100 * vm.total) - vm.used, vm.available * 0.5)
        budget = max(headroom * 0.6, 64 * 1024 * 1024)  # keep plenty of headroom

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
            f"(budget {budget/1e6:.1f} MB, RAM {vm.percent:.1f}%)"
        )
        return chunk

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
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
                numeric_ratio = numeric.notna().mean()
            except Exception:
                numeric = None
                numeric_ratio = 0

            if numeric is not None and numeric_ratio > 0.5:
                df[col] = pd.to_numeric(numeric, downcast="float").astype(TARGET_FLOAT_DTYPE)
                continue

            if df[col].nunique(dropna=False) / max(len(df[col]), 1) < 0.5:
                df[col] = df[col].astype("category")
            elif df[col].nunique(dropna=False) < 50:
                df[col] = df[col].astype("category")

        return df

    def _choose_threads(self, file_count: int, avg_mb: float = 0.0) -> int:
        cpu_threads = psutil.cpu_count(logical=True) or 4
        max_threads = min(cpu_threads, MAX_THREADS, file_count if file_count else cpu_threads)
        vm = psutil.virtual_memory()

        if vm.total < 8 * 1024**3:  # small machines: keep concurrency low
            max_threads = min(max_threads, 4)
        if vm.percent > 70:
            max_threads = min(max_threads, 3)
        elif vm.percent > 60:
            max_threads = min(max_threads, 4)

        if avg_mb and avg_mb > 500:
            max_threads = min(max_threads, 4)
        if avg_mb and avg_mb > 1000:
            max_threads = min(max_threads, 2)

        threads = max(1, max_threads)
        self._log(
            f"Thread choice -> {threads} workers for {file_count} file(s) "
            f"(RAM {vm.percent:.1f}%, CPU threads {cpu_threads}, avg size ~{avg_mb:.1f} MB)"
        )
        return threads

    def _load_single_file(self, filepath: str) -> tuple[pd.DataFrame | None, str]:
        self._wait_for_ram(self.max_ram_percent - 2)
        try:
            df = pd.read_csv(filepath, low_memory=False, memory_map=True)
            df = self._optimize_dtypes(df)
            self.monitor.record_metric()
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            self._log(f"Loaded file on thread: {os.path.basename(filepath)} ({len(df):,} rows, ~{mem_mb:.1f} MB)")
            return df, filepath
        except Exception:
            return None, filepath

    # --- public API ------------------------------------------------------
    def load_toniot_optimized(self, filepath: str, callback=None) -> pd.DataFrame:
        chunk_size = self._estimate_chunk_size(filepath)
        reader = pd.read_csv(filepath, chunksize=chunk_size, low_memory=False, memory_map=True)
        chunks: list[pd.DataFrame] = []
        processed = 0
        self._log(f"Starting TON_IoT load with chunk_size={chunk_size:,}")

        for idx, chunk in enumerate(reader, 1):
            self._wait_for_ram(self.max_ram_percent - 2)
            chunk = self._optimize_dtypes(chunk)
            chunks.append(chunk)

            processed += len(chunk)
            self.processed_rows += len(chunk)
            self.monitor.record_metric()
            self.monitor.track_data(len(chunk))
            chunk_mem = chunk.memory_usage(deep=True).sum() / (1024 * 1024)
            self._log(f"TON_IoT chunk {idx}: {len(chunk):,} rows (~{chunk_mem:.1f} MB)")

            if callback:
                progress = min(100.0, (idx * chunk_size) / max(processed, chunk_size) * 100)
                callback(idx, len(chunk), progress, (idx - 1) % 2)

            if len(chunks) >= 4 and psutil.virtual_memory().percent > self.max_ram_percent - 5:
                self._log(f"Concatenating buffered TON_IoT chunks at idx {idx} to free memory")
                chunks = [pd.concat(chunks, ignore_index=True, copy=False)]
                gc.collect()

        result = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame()
        gc.collect()
        res_mem = result.memory_usage(deep=True).sum() / (1024 * 1024) if not result.empty else 0
        self._log(f"TON_IoT loaded -> {len(result):,} rows (~{res_mem:.1f} MB)")
        return result

    def load_cic_optimized(self, folder: str, callback=None, threads_hook=None) -> tuple[list[pd.DataFrame], int, int]:
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
        threads = self._choose_threads(len(cic_files), avg_mb=avg_mb)
        if threads_hook:
            threads_hook(threads)
        dfs_cic: list[pd.DataFrame] = []
        failures: list[str] = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(self._load_single_file, path): path for path in cic_files}

            for done_idx, future in enumerate(as_completed(futures), 1):
                df, path = future.result()
                if df is not None:
                    dfs_cic.append(df)
                    self.monitor.track_file()
                    self.monitor.track_data(len(df))
                    self._log(f"[CIC] done {done_idx}/{len(cic_files)} -> {os.path.basename(path)} ({len(df):,} rows)")
                    if done_idx % 3 == 0:
                        mem_mb = sum(x.memory_usage(deep=True).sum() for x in dfs_cic) / (1024 * 1024)
                        self._log(f"[CIC] accumulated {done_idx} files, ~{mem_mb:.1f} MB in memory")
                else:
                    failures.append(path)
                    self._log(f"[CIC] failed to load {os.path.basename(path)} (returned None)")

                if callback:
                    progress = (done_idx / max(len(cic_files), 1)) * 100
                    callback(done_idx, len(cic_files), progress, done_idx % max(threads, 1), f"Loaded {os.path.basename(path)}")

                if psutil.virtual_memory().percent > self.max_ram_percent:
                    gc.collect()

        gc.collect()
        if failures:
            self._log(f"[CIC] missing/failed files: {len(failures)} -> {[os.path.basename(f) for f in failures]}")
        return dfs_cic, len(cic_files), threads

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
        mem_after = result.memory_usage(deep=True).sum() / (1024 * 1024)
        dtypes_summary = result.dtypes.value_counts().to_dict()
        self._log(f"Merged dataframe -> {len(result):,} rows, {len(result.columns)} cols (~{mem_after:.1f} MB, dtypes {dtypes_summary})")
        gc.collect()
        return result

    def clean_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        if "Label" in df.columns:
            df = df.dropna(subset=["Label"])
        gc.collect()
        return df

    def split_optimized(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
        for train_idx, test_idx in sss.split(df, df["Label"]):
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        return df.copy(), df.copy()

    def get_eta(self, processed: int, total: int) -> timedelta:
        if processed <= 0 or total <= 0:
            return timedelta(0)
        elapsed = time.time() - self.start_time
        rate = processed / max(elapsed, 1e-6)
        remaining = (total - processed) / max(rate, 1e-6)
        return timedelta(seconds=int(remaining))


class ConsolidationGUIEnhanced:
    """Tkinter UI that wires up the optimized loader."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Data Consolidation")
        self.root.geometry("1100x800")

        self.toniot_file: str | None = None
        self.cic_dir: str | None = None
        self.is_running = False
        self.start_time: float | None = None

        self.monitor = AdvancedMonitor()
        self.cache = SmartCache(max_items=0)
        self.processor = OptimizedDataProcessor(self.monitor, self.cache, logger=self.log)

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Idle")
        self.ram_var = tk.StringVar(value="-- %")
        self.cpu_var = tk.StringVar(value="-- %")
        self.rows_var = tk.StringVar(value="Rows: 0")
        self.files_var = tk.StringVar(value="Files: 0")
        self.thread_bars: dict[int, dict[str, tk.Variable | ttk.Label]] = {}

        self.setup_ui()
        self.start_monitoring_loop()

    # --- UI setup -------------------------------------------------------
    def setup_ui(self) -> None:
        file_frame = ttk.LabelFrame(self.root, text="Input")
        file_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(file_frame, text="TON_IoT CSV:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.toniot_path_label = ttk.Label(file_frame, text="No file selected")
        self.toniot_path_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.select_toniot).grid(row=0, column=2, sticky="e", padx=5, pady=5)

        ttk.Label(file_frame, text="CIC Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.cic_path_label = ttk.Label(file_frame, text="No folder selected")
        self.cic_path_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.select_cic).grid(row=1, column=2, sticky="e", padx=5, pady=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_consolidation)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_consolidation, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5)

        alerts_frame = ttk.LabelFrame(self.root, text="Alerts")
        alerts_frame.pack(fill="both", padx=10, pady=5)
        self.alert_canvas = tk.Canvas(alerts_frame, height=120)
        alerts_scroll = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.alert_canvas.yview)
        self.alert_canvas.configure(yscrollcommand=alerts_scroll.set)
        self.alert_canvas.pack(side="left", fill="both", expand=True)
        alerts_scroll.pack(side="right", fill="y")
        self.alert_inner = ttk.Frame(self.alert_canvas)
        self.alert_canvas.create_window((0, 0), window=self.alert_inner, anchor="nw")
        self.alert_inner.bind(
            "<Configure>", lambda e: self.alert_canvas.configure(scrollregion=self.alert_canvas.bbox("all"))
        )

        monitor_frame = ttk.LabelFrame(self.root, text="Monitoring")
        monitor_frame.pack(fill="x", padx=10, pady=10)

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
        monitor_frame.columnconfigure(1, weight=1)
        monitor_frame.columnconfigure(3, weight=1)

        threads_frame = ttk.LabelFrame(self.root, text="Thread progress")
        threads_frame.pack(fill="x", padx=10, pady=5)
        self.thread_container = ttk.Frame(threads_frame)
        self.thread_container.pack(fill="x", padx=5, pady=5)

        log_frame = ttk.LabelFrame(self.root, text="Logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD, font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    # --- UI events ------------------------------------------------------
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
            tk.Label(row, text=level.upper(), fg=color, width=8).pack(side="left", padx=4)
            tk.Label(row, text=message, fg=color, wraplength=800, anchor="w", justify="left").pack(
                side="left", fill="x", expand=True
            )
            row.pack(fill="x", padx=4, pady=1)
            self.alert_canvas.yview_moveto(1.0)
            if len(self.alert_inner.winfo_children()) > 100:
                self.alert_inner.winfo_children()[0].destroy()

        self.root.after(0, _append)

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

    def start_monitoring_loop(self) -> None:
        ram, cpu = self.monitor.record_metric()
        stats = self.monitor.get_stats()
        self.ram_var.set(f"{ram:.1f}%")
        self.cpu_var.set(f"{cpu:.1f}%")
        self.rows_var.set(f"Rows: {stats['data_loaded']:,}")
        self.files_var.set(f"Files: {stats['files']:,}")

        self.root.after(700, self.start_monitoring_loop)

    # --- pipeline -------------------------------------------------------
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
        self.log("Consolidation started", "INFO")
        self.add_alert("Consolidation started", "INFO")
        self.reset_thread_bars()

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

        # Load TON_IoT
        self.status_var.set("Loading TON_IoT")
        self.ensure_thread_bars(2)
        df_toniot = self.processor.load_toniot_optimized(self.toniot_file, callback=self._progress_callback("TON_IoT"))
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"TON_IoT loaded: {len(df_toniot):,} rows", "OK")
        if not self.is_running:
            return

        # Load CIC
        self.status_var.set("Loading CIC")
        dfs_cic, total_files, cic_threads = self.processor.load_cic_optimized(
            self.cic_dir, callback=self._progress_callback("CIC"), threads_hook=self.ensure_thread_bars
        )
        self.log(f"CIC loader using {cic_threads} thread(s)", "INFO")
        if len(dfs_cic) != total_files:
            missing = total_files - len(dfs_cic)
            self.add_alert(f"CIC missing/failed files: {missing} (loaded {len(dfs_cic)}/{total_files})", "WARN")
            self.log(f"CIC missing/failed files: {missing} (loaded {len(dfs_cic)}/{total_files})", "WARN")
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"CIC loaded: {len(dfs_cic)} file(s) (expected {total_files})", "OK")
        if not self.is_running:
            return

        # Merge and clean
        self.status_var.set("Merging and cleaning")
        combined = self.processor.merge_optimized([df_toniot] + dfs_cic)
        combined = self.processor.clean_optimized(combined)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"Merged dataset: {len(combined):,} rows, {len(combined.columns)} columns", "INFO")
        if not self.is_running:
            return

        # Split
        self.status_var.set("Splitting train/test")
        df_train, df_test = self.processor.split_optimized(combined)
        step += 1
        self.progress_var.set(step / steps * 100)
        self.log(f"Split complete: train {len(df_train):,} / test {len(df_test):,}", "INFO")
        if not self.is_running:
            return

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
            self.log(f"NPZ created with {len(numeric_cols)} numeric features", "OK")
        else:
            self.log("No numeric columns found for NPZ export", "WARN")

        step += 1
        self.progress_var.set(100)
        self.status_var.set("Completed")

        elapsed = time.time() - (self.start_time or time.time())
        self.log(f"Consolidation finished in {self._format_duration(elapsed)}", "INFO")
        self.add_alert(f"Consolidation completed in {self._format_duration(elapsed)}", "OK")

    def _format_duration(self, seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _progress_callback(self, name: str):
        def _cb(idx, size, progress, thread_id, action=None):
            if not self.is_running:
                return
            msg = action or f"{name} chunk {idx} ({size:,} rows)"
            self.log(f"{name}: {msg} [{progress:.1f}% - thread {thread_id}]", "INFO")
            self.update_thread_progress(thread_id, progress, msg)

        return _cb


def main() -> None:
    root = tk.Tk()
    app = ConsolidationGUIEnhanced(root)
    root.mainloop()


if __name__ == "__main__":
    main()
