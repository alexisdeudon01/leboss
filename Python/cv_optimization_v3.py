#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV OPTIMIZATION V3 - AMÉLIORÉ
======================================
✅ Grid Search: Hyperparamètres variables
✅ Graphiques scrollables (paramètres vs scores)
✅ Gestion RAM dynamique (<90%)
✅ Tkinter GUI avancée
✅ Visualisation complète résultats
======================================
"""

import os
import sys
import time
import gc
import json
import traceback
import psutil
import threading
import multiprocessing
import warnings
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from consolidation_style_shell import ConsolidationStyleShell
from ai_optimization_server_with_sessions_v4 import AIOptimizationServer, Metrics as AIMetrics
from cv_graphics_window_v2_revised import CVGraphicsWindowV2
from cv_graphics_window_v2_revised import CVGraphicsWindowV2
NPZ_FLOAT_DTYPE = np.float64

def _normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure label column is named exactly 'Label' (case-insensitive match)."""
    if df is None or df.empty:
        return df
    if 'Label' in df.columns:
        return df
    for c in df.columns:
        if str(c).lower() == 'label':
            return df.rename(columns={c: 'Label'})
    return df


try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score, recall_score, precision_score
except ImportError:
    print("Erreur: sklearn non installé")
    sys.exit(1)
# Silence sklearn deprecations for logistic regression penalty/n_jobs
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs.*has no effect.*", category=FutureWarning)

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    print("Erreur: tkinter ou matplotlib non installé")
    sys.exit(1)

os.environ['JOBLIB_PARALLEL_BACKEND'] = 'loky'

NUM_CORES = multiprocessing.cpu_count()
K_FOLD = 5
STRATIFIED_SAMPLE_RATIO = 0.5
RAM_THRESHOLD = 90.0

# GRID SEARCH CONFIGURATION
PARAM_GRIDS = {
    'Logistic Regression': {
        # penalty deprecated in >=1.8; keep default penalty and tune via C only
        'C': [0.1, 1, 10],
        'max_iter': [1000, 2000],
    },
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7],
    },
    'Decision Tree': {
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5],
    }
}


class MemoryManager:
    """Gère la mémoire dynamiquement"""
    
    @staticmethod
    def get_ram_usage():
        try:
            return psutil.virtual_memory().percent
        except:
            return 50

    @staticmethod
    def get_available_ram_gb():
        try:
            return psutil.virtual_memory().available / (1024**3)
        except:
            return 8

    @staticmethod
    def get_optimal_chunk_size(total_size=None, min_chunk=100000, max_chunk=1000000):
        """Calcule chunk size optimal basé sur RAM libre"""
        ram_free = MemoryManager.get_available_ram_gb()
        ram_usage = MemoryManager.get_ram_usage()
        
        if ram_usage > 80:
            chunk_size = int(min_chunk * (100 - ram_usage) / 20)
        else:
            chunk_size = int(max_chunk * (ram_free / 16))
        
        return max(min_chunk, min(chunk_size, max_chunk))

    @staticmethod
    def check_memory():
        """Vérifie et nettoie mémoire si nécessaire"""
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > RAM_THRESHOLD:
            gc.collect()
            return False
        return True


class CVOptimizationGUI:
    """Interface Tkinter avancée avec graphiques scrollables"""
    
    def __init__(self, root):
        self.root = root
        self.root.title('CV Optimization V3 - Grid Search')
        self.root.geometry('1600x1000')
        self.root.configure(bg='#f0f0f0')

        # Shared consolidation-style shell
        self.ui_shell = ConsolidationStyleShell(
            title="CV Optimization V3 - Grid Search",
            stages=[
                ("overall", "Overall"),
                ("load", "Load"),
                ("prep", "Prep"),
                ("grid", "Grid"),
                ("graphs", "Graphs"),
            ],
            thread_slots=4,
            parent=self.root,
            use_parent_main=True,
        )
        
        self.running = False
        self.results = {}
        self.optimal_configs = {}
        self.start_time = None
        self.completed_operations = 0
        self.total_operations = 0
        self.current_workers = max(1, min(NUM_CORES, 4))
        self.checkpoint_path = Path(".run_state/cv_checkpoint.json")
        self.ckpt = {"model_idx": 0, "combo_idx": 0, "best_score": 0.0}
        self.rows_seen = 0
        self.files_done = 0
        self.stage_start = {}
        self.stage_eta = {}
        self.current_task = "Init"
        self.thread_slots = len(self.ui_shell.thread_vars)
        self.worker_thread: threading.Thread | None = None
        self._last_ai_ts = time.time()
        self._last_ai_rows = 0

        # Core data containers
        self.df = None
        self.X_scaled = None
        self.y = None
        self.label_encoder = None

        # Bind shell controls and placeholders for legacy UI handles
        self.ui_shell.start_btn.config(command=self.start_optimization)
        self.ui_shell.stop_btn.config(command=self.stop_optimization)
        self.start_btn = self.ui_shell.start_btn
        self.stop_btn = self.ui_shell.stop_btn
        self.live_text = self.ui_shell.log_text
        self.alerts_text = self.ui_shell.alerts_text

        class _StatusProxy:
            def __init__(self, setter):
                self._setter = setter

            def config(self, text=None, fg=None):
                if text is not None:
                    self._setter(text)

        self.status_label = _StatusProxy(lambda t: self.ui_shell.set_status(t))
        self.progress_bar = tk.DoubleVar(value=0.0)
        self.progress_label = _StatusProxy(lambda t: self.ui_shell.set_tasks(t))
        self.ram_label = _StatusProxy(lambda t: None)
        self.cpu_label = _StatusProxy(lambda t: None)
        self.ram_progress = {"value": 0}
        self.cpu_progress = {"value": 0}
        self.graphs_btn = self.ui_shell.reset_btn  # placeholder, not used
        self.thread_slots = len(self.ui_shell.thread_vars)

        # graph window will be created on demand with CVGraphicsWindowV2
        self.graph_window = None
        self.graph_ui = None

        # AI optimization server (headless)
        self.ai_server = AIOptimizationServer(
            max_workers=1,
            max_chunk_size=1_000_000,
            min_chunk_size=50_000,
            max_ram_percent=90.0,
            with_gui=False,
        )
        self.ai_server_thread = threading.Thread(target=self.ai_server.run, daemon=True, name="AIOptimizationServer")
        self.ai_server_thread.start()
        self.rows_seen = 0
        self.current_chunk_size = MemoryManager.get_optimal_chunk_size()

    # -------------------- UI safe setters (main thread) --------------------
    def _ui_stage(self, key: str, val: float):
        self.root.after(0, lambda: self.ui_shell.set_stage_progress(key, val))

    def _ui_overall(self, val: float):
        self.root.after(0, lambda: self.ui_shell.set_overall_progress(val))

    def _ui_eta(self, key: str, text: str):
        self.root.after(0, lambda: self.ui_shell.set_stage_eta(key, text))

    def _ui_model(self, key: str, val: float):
        self.root.after(0, lambda: self.ui_shell.set_model_progress(key, val, key))

    def _ui_thread(self, tid: int, val: float, msg: str, done: int | None = None, total: int | None = None):
        try:
            self.root.after(0, lambda: self.ui_shell.set_thread_progress(tid, val, msg, done, total))
        except Exception:
            self.root.after(0, lambda: self.ui_shell.update_thread(tid, val, msg))

    def _ui_best_score(self, text: str):
        self.root.after(0, lambda: self.ui_shell.set_best_score(text))

    def _ui_ai_rec(self, text: str):
        self.root.after(0, lambda: self.ui_shell.set_ai_recommendation(text))

    def _ui_tasks(self, text: str):
        self.root.after(0, lambda: self.ui_shell.set_tasks(text))

    def _send_ai_metric(self, rows: int, chunk_size: int):
        try:
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.0)
            now = time.time()
            dt = max(now - self._last_ai_ts, 1e-6)
            tp = rows / dt
            self._last_ai_ts = now
            self._last_ai_rows += rows
            self.ai_server.send_metrics(
                AIMetrics(
                    timestamp=time.time(),
                    num_workers=int(self.current_workers),
                    chunk_size=int(chunk_size),
                    rows_processed=int(rows),
                    ram_percent=float(ram),
                    cpu_percent=float(cpu),
                    throughput=float(tp),
                )
            )
            rec = self.ai_server.get_recommendation(timeout=0.02)
            if rec:
                new_workers = max(1, min(NUM_CORES, self.current_workers + rec.d_workers))
                if new_workers != self.current_workers:
                    self.current_workers = new_workers
                    self.log_live(f"[AI] workers->{new_workers} (reason: {rec.reason})", "info")
                new_chunk = int(self.current_chunk_size * rec.chunk_mult)
                self.current_chunk_size = max(20_000, min(1_000_000, new_chunk))
                self.log_live(f"[AI] chunk->{self.current_chunk_size:,} (reason: {rec.reason})", "info")
                self.ui_shell.set_ai_recommendation(rec.reason)
                # surface throughput + server instruction
                self.log_live(f"[AI] instr: {rec.reason} | rows/s≈{tp:,.1f}", "info")
        except Exception:
            pass

    def _reset_thread_bars(self):
        for tid in range(self.thread_slots):
            self._ui_thread(tid, 0.0, "Idle", 0, 0)

    def _update_stage_eta(self, key: str, done: int, total: int):
        if total <= 0:
            self._ui_eta(key, "ETA: --")
            return
        now = time.time()
        if key not in self.stage_start:
            self.stage_start[key] = now
        elapsed = now - self.stage_start[key]
        pace = elapsed / max(done, 1)
        remaining = max(total - done, 0) * pace
        eta_dt = timedelta(seconds=int(remaining))
        self._ui_eta(key, f"ETA: {eta_dt}")

    def _write_checkpoint(self, force: bool = False):
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.ckpt["rows_seen"] = self.rows_seen
            self.ckpt["files_done"] = self.files_done
            self.ckpt["current_workers"] = self.current_workers
            self.ckpt["current_chunk_size"] = self.current_chunk_size
            self.ckpt["completed_ops"] = self.completed_operations
            self.checkpoint_path.write_text(json.dumps(self.ckpt, indent=2), encoding="utf-8")
        except Exception:
            if force:
                raise

    def setup_ui(self):
        """Setup UI"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        header = tk.Frame(self.root, bg='#2c3e50', height=50)
        header.grid(row=0, column=0, sticky='ew')
        tk.Label(header, text='CV Optimization V3 - Grid Search Hyperparamètres',
                 font=('Arial', 12, 'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=20, pady=12)
        
        container = tk.Frame(self.root, bg='#f0f0f0')
        container.grid(row=1, column=0, sticky='nsew', padx=8, pady=8)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=2)
        container.columnconfigure(1, weight=1)
        
        live_frame = tk.LabelFrame(container, text='LIVE Output',
                                   font=('Arial', 10, 'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        live_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=0)
        live_frame.rowconfigure(0, weight=1)
        live_frame.columnconfigure(0, weight=1)
        self.live_text = scrolledtext.ScrolledText(live_frame, font=('Courier', 8),
                                                   bg='#1a1a1a', fg='#00ff00')
        self.live_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        stats_frame = tk.Frame(container, bg='#f0f0f0')
        stats_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=0)
        stats_frame.rowconfigure(5, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        
        ram_frame = tk.LabelFrame(stats_frame, text='RAM', font=('Arial', 9, 'bold'),
                                  bg='white', relief=tk.SUNKEN, bd=2)
        ram_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=3)
        self.ram_label = tk.Label(ram_frame, text='0%', font=('Arial', 10, 'bold'), bg='white', fg='#e74c3c')
        self.ram_label.pack(fill=tk.X, padx=8, pady=3)
        self.ram_progress = ttk.Progressbar(ram_frame, mode='determinate', maximum=100)
        self.ram_progress.pack(fill=tk.X, padx=8, pady=3)
        
        cpu_frame = tk.LabelFrame(stats_frame, text='CPU', font=('Arial', 9, 'bold'),
                                  bg='white', relief=tk.SUNKEN, bd=2)
        cpu_frame.grid(row=1, column=0, sticky='ew', padx=0, pady=3)
        self.cpu_label = tk.Label(cpu_frame, text='0%', font=('Arial', 10, 'bold'), bg='white', fg='#3498db')
        self.cpu_label.pack(fill=tk.X, padx=8, pady=3)
        self.cpu_progress = ttk.Progressbar(cpu_frame, mode='determinate', maximum=100)
        self.cpu_progress.pack(fill=tk.X, padx=8, pady=3)
        
        progress_frame = tk.LabelFrame(stats_frame, text='Avancée', font=('Arial', 9, 'bold'),
                                       bg='white', relief=tk.SUNKEN, bd=2)
        progress_frame.grid(row=2, column=0, sticky='ew', padx=0, pady=3)
        self.progress_label = tk.Label(progress_frame, text='0/0', font=('Arial', 9), bg='white')
        self.progress_label.pack(fill=tk.X, padx=8, pady=3)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=8, pady=3)
        
        eta_frame = tk.LabelFrame(stats_frame, text='ETA', font=('Arial', 9, 'bold'),
                                  bg='white', relief=tk.SUNKEN, bd=2)
        eta_frame.grid(row=3, column=0, sticky='ew', padx=0, pady=3)
        self.eta_label = tk.Label(eta_frame, text='--:--:--', font=('Arial', 10, 'bold'), bg='white', fg='#9b59b6')
        self.eta_label.pack(fill=tk.X, padx=8, pady=3)
        
        alerts_frame = tk.LabelFrame(stats_frame, text='STATUS', font=('Arial', 9, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
        alerts_frame.grid(row=4, column=0, sticky='ew', padx=0, pady=3)
        alerts_frame.rowconfigure(0, weight=1)
        alerts_frame.columnconfigure(0, weight=1)
        self.alerts_text = scrolledtext.ScrolledText(alerts_frame, height=6, font=('Courier', 8),
                                                     bg='#f8f8f8', fg='#333')
        self.alerts_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        footer = tk.Frame(self.root, bg='#ecf0f1', height=60)
        footer.grid(row=2, column=0, sticky='ew')
        
        btn_frame = tk.Frame(footer, bg='#ecf0f1')
        btn_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.start_btn = tk.Button(btn_frame, text='Demarrer',
                                   command=self.start_optimization,
                                   bg='#27ae60', fg='white',
                                   font=('Arial', 11, 'bold'),
                                   padx=15, pady=8)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text='Arreter',
                                  command=self.stop_optimization,
                                  bg='#e74c3c', fg='white',
                                  font=('Arial', 11, 'bold'),
                                  padx=15, pady=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Pas de bouton "Voir Graphiques" (fenêtre graphes lancée autrement si besoin)
        self.graphs_btn = None
        
        self.status_label = tk.Label(footer, text='Prêt',
                                     font=('Arial', 10, 'bold'),
                                     fg='#27ae60', bg='#ecf0f1')
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)

    def log_live(self, msg, tag='info'):
        """Thread-safe log to the live console + shell log."""
        def _append():
            try:
                self.live_text.insert(tk.END, msg + '\n', tag)
                self.live_text.see(tk.END)
                level = "INFO" if tag == "info" else str(tag).upper()
                self.ui_shell.log(msg, level=level)
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            try:
                self.root.after(0, _append)
            except Exception:
                pass

    def add_alert(self, msg, level: str = "INFO"):
        """Thread-safe alert entry."""
        def _append():
            try:
                self.alerts_text.insert(tk.END, f'• {msg}\n')
                self.alerts_text.see(tk.END)
                self.ui_shell.add_alert(msg, level.upper() if isinstance(level, str) else "INFO")
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            try:
                self.root.after(0, _append)
            except Exception:
                pass

    def update_stats(self):
        try:
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.1)
            # update only CPU; RAM stays silent (no alert, no log)
            self.ram_label.config(text=f'{ram:.1f}%')
            self.ram_progress['value'] = ram
            self.cpu_label.config(text=f'{cpu:.1f}%')
            self.cpu_progress['value'] = min(cpu, 100)
            
            if self.start_time and self.completed_operations > 0:
                elapsed = time.time() - self.start_time
                avg = elapsed / self.completed_operations
                remaining = (self.total_operations - self.completed_operations) * avg
                eta = datetime.now() + timedelta(seconds=remaining)
                self.eta_label.config(text=eta.strftime('%H:%M:%S'))
            
            percent = (self.completed_operations / self.total_operations * 100) if self.total_operations > 0 else 0
            self.progress_bar['value'] = percent
            self.progress_label.config(text=f'{self.completed_operations}/{self.total_operations}')
            self.ui_shell.set_overall_progress(percent)
            self.ui_shell.set_stage_progress("overall", percent)
            rowsps = self.rows_seen / max(time.time() - self.start_time, 1e-3) if self.start_time else 0.0
            self.ui_shell.set_metrics(ram, cpu, self.rows_seen, self.files_done, rowsps)
            # push periodic metrics to AI server (no chunk update here, just heartbeat)
            try:
                self.ai_server.send_metrics(
                    AIMetrics(
                        timestamp=time.time(),
                        num_workers=int(self.current_workers),
                        chunk_size=int(self.current_chunk_size),
                        rows_processed=0,
                        ram_percent=float(ram),
                        cpu_percent=float(cpu),
                        throughput=0.0,
                    )
                )
            except Exception:
                pass
            
            self.root.after(500, self.update_stats)
        except:
            self.root.after(500, self.update_stats)

    def start_optimization(self):
        if self.running:
            messagebox.showwarning('Attention', 'Deja en cours')
            return
        # load checkpoint if exists
        if self.checkpoint_path.exists():
            try:
                self.ckpt = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self.completed_operations = self.ckpt.get("completed_ops", 0)
                self.current_workers = self.ckpt.get("current_workers", self.current_workers)
                self.current_chunk_size = self.ckpt.get("current_chunk_size", self.current_chunk_size)
                self.ui_shell.log(f"[CHKPT] Resume at model_idx={self.ckpt.get('model_idx',0)} combo_idx={self.ckpt.get('combo_idx',0)}", "INFO")
            except Exception:
                self.ckpt = {"model_idx": 0, "combo_idx": 0, "best_score": 0.0}
        else:
            self.ckpt = {"model_idx": 0, "combo_idx": 0, "best_score": 0.0}
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text='En cours...', fg='#f57f17')
        self.ui_shell.set_status("Running")
        for k in ("load", "prep", "grid", "overall"):
            self.ui_shell.set_stage_progress(k, 0.0)
        self.ui_shell.set_overall_progress(0.0)
        self.ui_shell.set_best_score("--")
        self.ui_shell.set_ai_recommendation("Waiting...")
        self._ui_tasks("Load → Prep → Grid → Reports")
        self._reset_thread_bars()
        for key in ("load", "prep", "grid", "reports"):
            self.stage_start[key] = time.time()
            self.stage_eta[key] = "ETA: --"
            self._ui_eta(key, "ETA: --")
        # init model bars
        for name in ("Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest"):
            try:
                self._ui_model(name, 0.0)
            except Exception:
                pass
        self.live_text.delete(1.0, tk.END)
        self.alerts_text.delete(1.0, tk.END)
        
        self.log_live('CV OPTIMIZATION V3 - GRID SEARCH\n', 'info')
        self.log_live('Hyperparamètres variables par algo\n\n', 'info')
        
        self.start_time = time.time()
        # open graph window immediately (even if results empty)
        try:
            self.show_graphs()
        except Exception:
            pass
        # Launch optimization in background to keep UI responsive
        self.worker_thread = threading.Thread(target=self.run_optimization, daemon=True, name="CVOptiWorker")
        self.worker_thread.start()
        self.root.after(500, self.update_stats)

    def stop_optimization(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text='Arrete', fg='#e74c3c')
        self.ui_shell.set_status("Stopped")
        self._write_checkpoint(force=False)

    def load_data(self):
        try:
            self.log_live('ETAPE 1: Chargement CSV\n', 'info')
            files = ['fusion_train_smart6.csv', 'fusion_test_smart6.csv']
            fichier = next((f for f in files if os.path.exists(f)), None)
            
            if not fichier:
                self.log_live('Erreur: CSV non trouve\n', 'info')
                return False
            
            self.log_live(f'Fichier: {fichier}\n', 'info')
            
            chunks = []
            total_rows = 0
            chunk_size = max(50_000, self.current_chunk_size)
            t_last = time.time()
            next_metrics_time = t_last
            file_size = os.path.getsize(fichier)
            est_total_rows = None
            avg_bytes_per_row = None
            
            for chunk in pd.read_csv(fichier, low_memory=False, chunksize=chunk_size, encoding='utf-8'):
                if not self.running:
                    return False
                chunks.append(chunk)
                total_rows += len(chunk)
                self.log_live(f'+{len(chunk):,} (total {total_rows:,})\n', 'info')
                # estimate total rows to drive progress/ETA
                if avg_bytes_per_row is None:
                    try:
                        avg_bytes_per_row = max(float(chunk.memory_usage(deep=True).sum()) / max(len(chunk), 1), 1.0)
                        est_total_rows = max(int(file_size / avg_bytes_per_row), len(chunk))
                    except Exception:
                        est_total_rows = None
                # UI progress during load
                if est_total_rows:
                    pct = min(99.0, 100.0 * total_rows / max(est_total_rows, 1))
                    self._ui_stage("load", pct)
                    self._ui_overall(pct * 0.2)  # early weight so user sees motion
                    self._ui_thread(0, pct, f"Load chunk {len(chunks)} ({total_rows:,}/{est_total_rows:,})", total_rows, est_total_rows)
                    self._update_stage_eta("load", total_rows, est_total_rows)
                
                # metrics to AI server (adaptive cadence)
                now = time.time()
                dt = max(now - t_last, 1e-3)
                tp = len(chunk) / dt
                send_interval = max(0.5, min(5.0, chunk_size / 300_000.0))
                if now >= next_metrics_time:
                    ram = psutil.virtual_memory().percent
                    cpu = psutil.cpu_percent(interval=0.0)
                    try:
                        self.ai_server.send_metrics(
                            AIMetrics(
                                timestamp=now,
                                num_workers=int(self.current_workers),
                                chunk_size=int(chunk_size),
                                rows_processed=len(chunk),
                                ram_percent=float(ram),
                                cpu_percent=float(cpu),
                                throughput=float(tp),
                            )
                        )
                        rec = self.ai_server.get_recommendation(timeout=0.02)
                        if rec:
                            new_chunk = int(chunk_size * rec.chunk_mult)
                            chunk_size = max(20_000, min(1_000_000, new_chunk))
                            self.current_chunk_size = chunk_size
                            new_workers = max(1, min(NUM_CORES, self.current_workers + rec.d_workers))
                            if new_workers != self.current_workers:
                                self.current_workers = new_workers
                                self.log_live(f"[AI] workers->{new_workers}", "info")
                            self.log_live(f"[AI] chunk->{chunk_size:,} (reason: {rec.reason})", "info")
                    except Exception:
                        pass
                    next_metrics_time = now + send_interval
                t_last = now
                
                if not MemoryManager.check_memory():
                    self.log_live(f'[WARN] RAM critique, attente...\n', 'info')
                    time.sleep(2)
            
            self.df = pd.concat(chunks, ignore_index=True)
            self.df = _normalize_label_column(self.df)
            self.log_live(f'OK: {len(self.df):,} lignes\n\n', 'info')
            self.rows_seen += len(self.df)
            self.files_done += 1
            self._ui_stage("load", 100.0)
            self._update_stage_eta("load", 1, 1)
            return True
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')
            # also push to alert canvas
            try:
                self.add_alert(f'Erreur: {e}', 'ERROR')
            except Exception:
                pass
            return False

    def prepare_data(self):
        """ETAPE 2: Preparation des donnees (FIXED)"""
        try:
            self.log_live('ETAPE 2: Preparation\n', 'info')
            
            # Step 1: Normalize Label column name
            if 'Label' not in self.df.columns:
                for col in self.df.columns:
                    if str(col).strip().lower() == 'label':
                        self.df = self.df.rename(columns={col: 'Label'})
                        self.log_live(f'Renamed column {col} -> Label\n', 'info')
                        break
            
            # Step 2: Select numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            if not numeric_cols:
                self.log_live('ERROR: No numeric columns found!\n', 'error')
                self.add_alert('No numeric columns found', 'error')
                return False
            
            self.log_live(f'Dataset: {len(self.df):,} rows | {len(numeric_cols)} numeric features\n', 'info')
            
            # Step 3: Drop rows without Label
            if 'Label' in self.df.columns:
                original_rows = len(self.df)
                self.df = self.df.dropna(subset=['Label'])
                self.df['Label'] = self.df['Label'].astype(str)
                dropped = original_rows - len(self.df)
                if dropped > 0:
                    self.log_live(f'Dropped {dropped:,} rows without Label\n', 'info')
                self.log_live(f'Labels: {self.df["Label"].nunique()} unique classes\n', 'info')
            else:
                self.log_live('ERROR: Label column not found!\n', 'error')
                self.add_alert('Label column not found', 'error')
                return False
            
            # Step 4: Stratified sampling (only if >=2 classes), else simple head sample
            n_samples = int(len(self.df) * STRATIFIED_SAMPLE_RATIO)
            unique_classes = self.df["Label"].unique()
            if len(unique_classes) >= 2 and len(self.df) > n_samples:
                self.log_live(f'Sampling (stratified): {n_samples:,} / {len(self.df):,} rows\n', 'info')
                stratifier = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                try:
                    for train_idx, _ in stratifier.split(self.df, self.df['Label']):
                        self.df = self.df.iloc[train_idx[:n_samples]]
                        break
                except Exception as e:
                    self.log_live(f'[WARN] Stratified sample failed: {e}; fallback to simple sample\n', 'info')
                    self.df = self.df.iloc[:n_samples].copy()
            elif len(self.df) > n_samples:
                self.log_live(f'Sampling (simple): {n_samples:,} / {len(self.df):,} rows\n', 'info')
                self.df = self.df.iloc[:n_samples].copy()
            
            self.log_live(f'After sampling: {len(self.df):,} rows\n', 'info')
            try:
                self._send_ai_metric(rows=len(self.df), chunk_size=max(1, self.current_chunk_size))
            except Exception:
                pass
            
            # ✅ CRITICAL STEP: Clean infinity and extreme values FIRST
            self.log_live('Cleaning infinity and extreme values...\n', 'info')
            for col in numeric_cols:
                try:
                    # Replace infinity with NaN (must be BEFORE dtype conversion)
                    self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                    # Clip extreme values to ±1e6
                    self.df[col] = self.df[col].clip(-1e6, 1e6)
                except Exception as e:
                    self.log_live(f'WARN: Error cleaning column {col}: {e}\n', 'warn')
            
            # Step 5: Extract features X
            X = self.df[numeric_cols].copy()
            
            # ✅ FILL NaN with column mean (REQUIRED)
            self.log_live('Filling NaN values with column mean...\n', 'info')
            for col in X.columns:
                col_mean = X[col].mean()
                if pd.isna(col_mean):
                    # If entire column is NaN, use 0
                    X[col] = X[col].fillna(0.0)
                else:
                    X[col] = X[col].fillna(col_mean)
            
            # Step 6: Convert to float64 then float32 (safe conversion)
            self.log_live('Converting data types...\n', 'info')
            try:
                X = X.astype(np.float64)
                X = X.astype(NPZ_FLOAT_DTYPE)  # float32
            except Exception as e:
                self.log_live(f'ERROR converting types: {e}\n', 'error')
                self.add_alert(f'Type conversion failed: {e}', 'error')
                return False
            
            # ✅ VALIDATE: Check no infinity/NaN remain
            n_inf = np.isinf(X.values).sum()
            n_nan = np.isnan(X.values).sum()
            
            if n_inf > 0 or n_nan > 0:
                self.log_live(f'WARN: Found {n_inf} inf and {n_nan} NaN after cleaning!\n', 'warn')
                # Final force fill
                X = X.fillna(0.0)
                X = X.replace([np.inf, -np.inf], 1e6)
                self.log_live(f'Force-filled NaN/inf\n', 'info')
            try:
                self._send_ai_metric(rows=len(X), chunk_size=max(1, self.current_chunk_size))
            except Exception:
                pass
            
            self.log_live(f'Features validated (no inf/nan) ✓\n', 'info')
            
            # Step 7: Encode Label
            self.label_encoder = LabelEncoder()
            try:
                self.y = self.label_encoder.fit_transform(self.df['Label'].astype(str))
            except Exception as e:
                self.log_live(f'ERROR encoding labels: {e}\n', 'error')
                self.add_alert(f'Label encoding failed: {e}', 'error')
                return False
            
            self.log_live(f'Labels encoded: {len(self.label_encoder.classes_)} classes\n', 'info')
            
            # Step 8: Standardize features
            scaler = StandardScaler()
            try:
                self.X_scaled = scaler.fit_transform(X).astype(NPZ_FLOAT_DTYPE)
            except Exception as e:
                self.log_live(f'ERROR standardizing: {e}\n', 'error')
                self.add_alert(f'Scaling failed: {e}', 'error')
                return False
            
            self.log_live(f'Data standardized: X shape {self.X_scaled.shape}\n', 'info')
            
            # Step 9: Save NPZ
            try:
                np.savez_compressed(
                    'preprocessed_dataset.npz',
                    X=self.X_scaled,
                    y=self.y,
                    classes=self.label_encoder.classes_
                )
                self.log_live(f'NPZ saved successfully\n\n', 'info')
            except Exception as e:
                self.log_live(f'WARN: NPZ save failed: {e}\n', 'warn')
            
            self.rows_seen += len(self.X_scaled)
            self._ui_stage("prep", 100.0)
            self._update_stage_eta("prep", 1, 1)
            
            # Cleanup
            del self.df, X
            gc.collect()
            
            return True
            
        except Exception as e:
            self.log_live(f'ERREUR PREP: {e}\n{traceback.format_exc()}\n', 'error')
            self.add_alert(f'Prep failed: {e}', 'error')
            return False

    def generate_param_combinations(self, model_name):
        """Génère toutes les combinaisons de paramètres"""
        grid = PARAM_GRIDS.get(model_name, {})
        param_names = list(grid.keys())
        param_values = [grid[p] for p in param_names]
        
        combinations = []
        for values in product(*param_values):
            combinations.append(dict(zip(param_names, values)))
        
        return combinations

    def run_optimization(self):
        try:
            if not self.load_data():
                return
            if not self.running:
                return
            
            if not self.prepare_data():
                return
            if not self.running:
                return
            
            self.log_live('ETAPE 3: Grid Search\n\n', 'info')
            
            model_configs = {
                'Logistic Regression': LogisticRegression,
                'Naive Bayes': GaussianNB,
                'Decision Tree': DecisionTreeClassifier,
                'Random Forest': RandomForestClassifier,
            }
            
            self.total_operations = sum(
                len(self.generate_param_combinations(name)) * K_FOLD 
                for name in model_configs.keys()
            )
            
            start_model_idx = int(self.ckpt.get("model_idx", 0))
            start_combo_idx = int(self.ckpt.get("combo_idx", 0))

            for i, (name, ModelClass) in enumerate(model_configs.items(), 1):
                if i < start_model_idx:
                    continue
                self.log_live(f'\n{i}/4. {name}\n', 'info')
                self._ui_tasks(f"{name} grid search")
                self._ui_model(name, 0.0)
                
                combinations = self.generate_param_combinations(name)
                self.log_live(f'  Testage: {len(combinations)} combinaisons\n', 'info')
                
                best_score = 0
                best_params = None
                all_results = []
                
                for combo_idx, params in enumerate(combinations, 1):
                    if i == start_model_idx and combo_idx <= start_combo_idx:
                        continue
                    if not self.running:
                        return
                
                def run_fold(fold):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.X_scaled, self.y,
                            test_size=0.2 if name != 'Decision Tree' else 0.3,
                            random_state=42 + fold,
                            stratify=self.y
                        )
                        kwargs = params.copy()
                        # n_jobs has no effect for LogisticRegression in new sklearn; keep only for RF
                        if name == 'Random Forest':
                            kwargs['n_jobs'] = max(1, int(self.current_workers))
                        if name != 'Naive Bayes':
                            kwargs['random_state'] = 42
                        if name == 'Logistic Regression':
                            kwargs.setdefault('solver', 'lbfgs')
                        model = ModelClass(**kwargs)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        rows_proc = len(X_train) + len(X_test)
                        return f1, rows_proc
                    except Exception:
                        return 0, 0

                results = Parallel(n_jobs=int(self.current_workers), backend="threading")(delayed(run_fold)(fold) for fold in range(K_FOLD))
                f1_runs = []
                for f1_val, rows_proc in results:
                    f1_runs.append(f1_val)
                    self.completed_operations += 1
                    progress_grid = (self.completed_operations / self.total_operations * 100) if self.total_operations > 0 else 0
                    self._ui_stage("grid", progress_grid)
                    self._ui_overall(progress_grid)
                    self._ui_stage("overall", progress_grid)
                    self._send_ai_metric(rows_proc, chunk_size=self.current_chunk_size)
                    # update thread bars (round-robin mapping) and ETA grid
                    tid = (self.completed_operations - 1) % self.thread_slots
                    pct = min(100.0, (combo_idx / max(len(combinations), 1)) * 100.0)
                    try:
                        self._ui_thread(
                            tid, pct, f"{name} combo {combo_idx}/{len(combinations)}", combo_idx, len(combinations)
                        )
                    except Exception:
                        self.ui_shell.update_thread(tid, pct, f"{name} combo {combo_idx}/{len(combinations)}")
                    self._update_stage_eta("grid", combo_idx, len(combinations))
                    # model-level progress
                    try:
                        self._ui_model(name, pct)
                    except Exception:
                        pass
                
                mean_f1 = np.mean(f1_runs) if f1_runs else 0
                params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                self.log_live(
                    f'    [{combo_idx}/{len(combinations)}] {params_str}: F1={mean_f1:.4f} '
                    f'| workers={self.current_workers} | chunk={self.current_chunk_size:,} '
                    f'| f1_runs={["%.4f" % x for x in f1_runs]}',
                    'info'
                )
                all_results.append({'params': params, 'f1': mean_f1})
                
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_params = params
                self._ui_best_score(f"{best_score:.4f}")
                self.add_alert(f'{name}: {combo_idx}/{len(combinations)} - F1={mean_f1:.4f}')
                # pull AI recommendation after each combo
                rec = self.ai_server.get_recommendation(timeout=0.02)
                if rec:
                    new_workers = max(1, min(NUM_CORES, self.current_workers + rec.d_workers))
                    if new_workers != self.current_workers:
                        self.current_workers = new_workers
                        self.log_live(f"[AI] workers->{new_workers} (reason: {rec.reason})", "info")
                    new_chunk = int(self.current_chunk_size * rec.chunk_mult)
                    self.current_chunk_size = max(20_000, min(1_000_000, new_chunk))
                    self.log_live(f"[AI] chunk->{self.current_chunk_size:,} (reason: {rec.reason})", "info")
                    self._ui_ai_rec(rec.reason)
                # checkpoint after each combo
                self.ckpt["model_idx"] = i
                self.ckpt["combo_idx"] = combo_idx
                self.ckpt["best_score"] = best_score
                self.ckpt["current_workers"] = self.current_workers
                self.ckpt["current_chunk_size"] = self.current_chunk_size
                self.ckpt["completed_ops"] = self.completed_operations
                self._write_checkpoint(force=False)
            # persist results for this model
            self.results[name] = {
                'all_results': all_results,
                'best_params': best_params,
                'best_f1': best_score,
            }
            self._reset_thread_bars()
            self.optimal_configs[name] = {
                'params': best_params,
                'f1_score': float(best_score),
            }
            # checkpoint at end of model
            self.ckpt["model_idx"] = i
            self.ckpt["combo_idx"] = 0
            self.ckpt["best_score"] = best_score
            self.ckpt["current_workers"] = self.current_workers
            self.ckpt["current_chunk_size"] = self.current_chunk_size
            self.ckpt["completed_ops"] = self.completed_operations
            self._write_checkpoint(force=True)

            self.log_live(f'  BEST: F1={best_score:.4f}\n', 'info')
            self._ui_best_score(f"{best_score:.4f}")
        
            # Final report + status after all models
            self.log_live('\nETAPE 4: Rapports\n', 'info')
            self.generate_reports()
            self.log_live('\n' + '='*60 + '\n', 'info')
            self.log_live('GRID SEARCH TERMINEE\n', 'info')
            self.root.after(0, lambda: self.ui_shell.set_status("Completed"))
            self.add_alert('GRID SEARCH COMPLETE')
            self._ui_stage("grid", 100.0)
            self._ui_overall(100.0)

        except Exception as e:
            self.log_live(f'Erreur: {e}\n{traceback.format_exc()}\n', 'info')
            self.root.after(0, lambda: self.ui_shell.set_status("Erreur"))
        
        finally:
            self._write_checkpoint(force=True)
            self.running = False
            try:
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            except Exception:
                pass

    def generate_reports(self):
        try:
            with open('cv_results_summary.txt', 'w', encoding='utf-8') as f:
                f.write('='*80 + '\n')
                f.write('CV OPTIMIZATION V3 - GRID SEARCH\n')
                f.write('='*80 + '\n\n')
                
                for name in sorted(self.optimal_configs.keys()):
                    cfg = self.optimal_configs[name]
                    f.write(f"{name:<25} F1:{cfg['f1_score']:>7.4f}\n")
                    f.write(f"  Params: {str(cfg['params'])}\n\n")
                
                f.write('='*80 + '\n')
            
            self.log_live('OK: cv_results_summary.txt\n', 'info')
            
            with open('cv_optimal_splits.json', 'w', encoding='utf-8') as jf:
                json.dump(self.optimal_configs, jf, ensure_ascii=False, indent=2, default=str)
            
            self.log_live('OK: cv_optimal_splits.json\n', 'info')
        
        except Exception as e:
            self.log_live(f'Erreur rapports: {e}\n', 'info')

    def show_graphs(self):
        """Affiche les graphiques avancés - Version 2"""
        # Reuse existing graph window if still alive
        try:
            if self.graph_ui and getattr(self.graph_ui, "window", None) and self.graph_ui.window.winfo_exists():
                self.graph_ui.window.lift()
                self.graph_ui.window.focus_force()
                return
        except Exception:
            pass

        # Utiliser la nouvelle fenêtre graphique (ouvre même si résultats vides)
        try:
            self.graph_ui = CVGraphicsWindowV2(
                parent=self.root,
                results=self.results,
                optimal_configs=self.optimal_configs
            )
            self.graph_window = getattr(self.graph_ui, "window", None)
        except Exception as e:
            messagebox.showerror('Erreur', f'Impossible d\'afficher les graphiques:\n{e}')


def main():
    try:
        root = tk.Tk()
        app = CVOptimizationGUI(root)
        root.mainloop()
    except Exception as e:
        print(f'Erreur: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
