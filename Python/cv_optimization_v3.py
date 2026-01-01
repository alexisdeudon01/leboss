#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV OPTIMIZATION V3 - AMÉLIORÉ & CORRIGÉ
======================================
✅ Grid Search: Hyperparamètres variables
✅ Graphiques scrollables (paramètres vs scores)
✅ Gestion RAM dynamique (<90%)
✅ Tkinter GUI avancée
✅ Visualisation complète résultats
✅ FIXE ERREUR #1: Indentation def run_fold
✅ FIXE ERREUR #2: Nettoyage infinity prepare_data
======================================
"""

import os
import sys
import time
import gc
import json
import traceback
import re
import psutil
import threading
import multiprocessing
import math
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from consolidation_style_shell import ConsolidationStyleShell
from ai_optimization_server_with_sessions_v4 import AIOptimizationServer, Metrics as AIMetrics

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
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score, recall_score, precision_score
except ImportError:
    print("Erreur: sklearn non installé")
    sys.exit(1)

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
MIN_CHUNK_SIZE = 10_000

# GRID SEARCH CONFIGURATION
PARAM_GRIDS = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 2000],
        'penalty': ['l2'],
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
    def get_optimal_chunk_size(total_size=None, min_chunk=MIN_CHUNK_SIZE, max_chunk=1000000):
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
                ("transit_load", "Transit L→P"),
                ("prep", "Prep"),
                ("transit_mid", "Transit P→G"),
                ("grid", "Grid"),
                ("transit_grid", "Transit G→R"),
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

        # single graphs window (no extra placeholders)
        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Graphs")
        try:
            self.graph_window.geometry("800x600")
        except Exception:
            pass
        # Single live graph (F1 progression par modèle)
        live_tab = ttk.Frame(self.graph_window)
        live_tab.pack(fill="both", expand=True)

        # Live graph (F1 progression per modèle)
        self.fig_live = Figure(figsize=(7, 4), dpi=100)
        self.ax_live = self.fig_live.add_subplot(111)
        self.ax_live.set_title("F1 en temps réel")
        self.ax_live.set_xlabel("Combinaison #")
        self.ax_live.set_ylabel("F1 pondéré")
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=live_tab)
        self.canvas_live.get_tk_widget().pack(fill="both", expand=True)
        self.live_curves: dict[str, list[float]] = {}
        self.live_params_label = ttk.Label(self.graph_window, text="Params: --", anchor="w", justify="left")
        self.live_params_label.pack(fill="x", padx=6, pady=4)


        # Extra Tk canvas (zone libre pour overlays / mini-visuels)
        self.extra_canvas = tk.Canvas(self.graph_window, height=120, bg='white', highlightthickness=1)
        self.extra_canvas.pack(fill='x', padx=6, pady=(0, 6))
        try:
            self.extra_canvas.create_text(10, 10, anchor='nw', text='Canvas: prêt', fill='black')
        except Exception:
            pass

        # Force graphs window visible (sinon tu peux la rater derrière)
        try:
            self.graph_window.deiconify()
            self.graph_window.lift()
            self.graph_window.attributes("-topmost", True)
            self.graph_window.after(250, lambda: self.graph_window.attributes("-topmost", False))
        except Exception:
            pass

        # Canvas d'estimation dans la fenêtre principale
        self.main_estimate_canvas = None
        self._attach_main_estimate_canvas()
        # AI optimization server (headless)
        self.ai_server = AIOptimizationServer(
            max_workers=1,
            max_chunk_size=1_000_000,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_ram_percent=90.0,
            with_gui=False,
        )
        self.ai_server_thread = threading.Thread(target=self.ai_server.run, daemon=True, name="AIOptimizationServer")
        self.ai_server_thread.start()
        self.rows_seen = 0
        self.current_chunk_size = MemoryManager.get_optimal_chunk_size(min_chunk=MIN_CHUNK_SIZE)
        self.last_ckpt_ts = time.time()

    # -------------------- Lifecycle controls --------------------
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

    def _set_live_params(self, model: str, params: dict):
        """Render current hyperparameters below the F1 live chart."""
        try:
            pretty = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "--"
            if len(pretty) > 220:
                pretty = pretty[:217] + "..."
            msg = f"{model}: {pretty}"
            self.root.after(0, lambda: self.live_params_label.config(text=msg))
        except Exception:
            pass

        # Keep AI server alive with stage heartbeat
        try:
            self._send_ai_metric(rows=0, chunk_size=int(self.current_chunk_size))
        except Exception:
            pass

    def _update_live_graph(self, model: str, f1_values: list[float]):
        """Update live graph window with latest F1 progression (thread-safe)."""
        try:
            self.live_curves[model] = list(f1_values)

            def _draw():
                try:
                    self.ax_live.clear()
                    for name, vals in self.live_curves.items():
                        xs = list(range(1, len(vals) + 1))
                        if not xs:
                            continue
                        self.ax_live.plot(xs, vals, marker="o", label=name)
                    self.ax_live.set_title("F1 en temps réel")
                    self.ax_live.set_xlabel("Combinaison #")
                    self.ax_live.set_ylabel("F1 pondéré")
                    self.ax_live.legend(loc="lower right", fontsize=8)
                    self.ax_live.grid(alpha=0.3)
                    self.canvas_live.draw_idle()
                except Exception:
                    pass

            try:
                self.root.after(0, _draw)
            except Exception:
                _draw()
        except Exception:
            pass

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
                self.current_chunk_size = max(MIN_CHUNK_SIZE, min(1_000_000, new_chunk))
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
            self.last_ckpt_ts = time.time()
        except Exception:
            if force:
                raise

    def _maybe_checkpoint(self, force: bool = False):
        """Checkpoint every 10 minutes or when forced."""
        if force or (time.time() - getattr(self, "last_ckpt_ts", 0) >= 600):
            try:
                self._write_checkpoint(force=False)
            except Exception:
                pass

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
        
        self.graphs_btn = tk.Button(btn_frame, text='Voir Graphiques',
                                    command=self.show_graphs,
                                    bg='#3498db', fg='white',
                                    font=('Arial', 11, 'bold'),
                                    padx=15, pady=8, state=tk.DISABLED)
        self.graphs_btn.pack(side=tk.LEFT, padx=5)
        
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

    def add_alert(self, msg, level="INFO"):
        """Thread-safe alert entry + audible bell on warnings/errors."""
        def _append():
            try:
                self.alerts_text.insert(tk.END, f'• {msg}\n')
                self.alerts_text.see(tk.END)
                self.ui_shell.add_alert(msg, level)
                if level.upper() in {"WARN", "ERROR"} or "error" in msg.lower():
                    try:
                        self.root.bell()
                    except Exception:
                        pass
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
            if ram >= 90.0:
                self.add_alert(f"RAM >=90% ({ram:.1f}%)", "WARN")
            if cpu >= 95.0:
                self.add_alert(f"CPU >=95% ({cpu:.1f}%)", "WARN")
            
            self.ram_label.config(text=f'{ram:.1f}%')
            self.ram_progress['value'] = ram
            self.cpu_label.config(text=f'{cpu:.1f}%')
            self.cpu_progress['value'] = min(cpu, 100)
            
                        # ETA: prefer pre-estimation if available, fallback to avg-per-op
            if self.start_time and self.completed_operations > 0:
                if getattr(self, "estimated_total_seconds", None) and self.total_operations > 0:
                    pct = self.completed_operations / max(self.total_operations, 1)
                    remaining = max(float(self.estimated_total_seconds) * (1.0 - pct), 0.0)
                    eta = datetime.now() + timedelta(seconds=remaining)
                    self.eta_label.config(text=eta.strftime('%H:%M:%S'))
                else:
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
        finally:
            try:
                self._maybe_checkpoint()
            except Exception:
                pass


    def start_optimization(self):
        """Start button handler: opens a preflight window (bench + ETA), then launches the real run on OK."""
        try:
            if getattr(self, "running", False):
                try:
                    messagebox.showwarning("Attention", "Déjà en cours")
                except Exception:
                    pass
                return
            self._open_preflight_window()
        except Exception as e:
            # fallback: if preflight crashes, still allow run
            try:
                self.log_live(f"[WARN] Préflight KO: {e} → lancement direct", "info")
            except Exception:
                pass
            self._start_main_run()

    def _open_preflight_window(self):
        """Popup that benchmarks the machine quickly and estimates total runtime."""
        # close previous preflight if any
        try:
            if getattr(self, "pref_win", None) is not None and self.pref_win.winfo_exists():
                self.pref_win.destroy()
        except Exception:
            pass

        self.pref_done = False
        self.pref_cancelled = False

        w = tk.Toplevel(self.root)
        self.pref_win = w
        w.title("Préflight - Perf PC + Estimation")
        try:
            w.geometry("900x620")
        except Exception:
            pass

        # Keep on top shortly so user sees it
        try:
            w.attributes("-topmost", True)
            w.after(250, lambda: w.attributes("-topmost", False))
        except Exception:
            pass

        top = ttk.Frame(w, padding=10)
        top.pack(fill="x")

        self.pref_cpu_var = tk.StringVar(value="CPU: --")
        self.pref_ram_var = tk.StringVar(value="RAM: --")
        ttk.Label(top, textvariable=self.pref_cpu_var, font=("Arial", 11, "bold")).pack(side="left", padx=(0, 12))
        ttk.Label(top, textvariable=self.pref_ram_var, font=("Arial", 11, "bold")).pack(side="left")

        mid = ttk.Frame(w, padding=(10, 0))
        mid.pack(fill="both", expand=True)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right = ttk.Frame(mid)
        right.pack(side="right", fill="both", expand=False)

        ttk.Label(left, text="Infos dataset + réglages", font=("Arial", 11, "bold")).pack(anchor="w")
        self.pref_info = scrolledtext.ScrolledText(left, height=14, font=("Consolas", 9))
        self.pref_info.pack(fill="both", expand=False, pady=(6, 10))

        ttk.Label(left, text="Estimation temps (minutes) par modèle", font=("Arial", 11, "bold")).pack(anchor="w")

        # Matplotlib bar chart
        self.pref_fig = Figure(figsize=(7.0, 3.2), dpi=100)
        self.pref_ax = self.pref_fig.add_subplot(111)
        self.pref_canvas = FigureCanvasTkAgg(self.pref_fig, master=left)
        self.pref_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Small canvas as “gauge”
        self.pref_gauge = tk.Canvas(right, width=220, height=220, bg="white", highlightthickness=1, highlightbackground="#ddd")
        self.pref_gauge.pack(pady=(22, 8))
        self.pref_gauge_text = ttk.Label(right, text="Benchmark: en cours...", font=("Arial", 10, "bold"))
        self.pref_gauge_text.pack()

        self.pref_status_var = tk.StringVar(value="→ Scan dataset + mini bench (~10–30s)")
        ttk.Label(right, textvariable=self.pref_status_var, wraplength=220).pack(pady=(10, 0))

        btns = ttk.Frame(w, padding=10)
        btns.pack(fill="x")

        self.pref_ok_btn = ttk.Button(btns, text="OK (lancer)", state="disabled", command=self._pref_on_ok)
        self.pref_ok_btn.pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="Annuler", command=self._pref_on_cancel).pack(side="right")

        # start live sys monitor
        self._pref_update_sys()

        # start benchmark thread
        threading.Thread(target=self._pref_run_benchmark, daemon=True, name="PreflightBench").start()

        # close hook
        try:
            w.protocol("WM_DELETE_WINDOW", self._pref_on_cancel)
        except Exception:
            pass

    def _pref_on_cancel(self):
        self.pref_cancelled = True
        try:
            if getattr(self, "pref_win", None) is not None and self.pref_win.winfo_exists():
                self.pref_win.destroy()
        except Exception:
            pass

    def _pref_on_ok(self):
        # Apply recommended settings if present
        try:
            rec = getattr(self, "pref_reco", None) or {}
            if rec.get("workers"):
                self.current_workers = int(rec["workers"])
            if rec.get("chunk_size"):
                self.current_chunk_size = int(rec["chunk_size"])
        except Exception:
            pass
        try:
            if getattr(self, "pref_win", None) is not None and self.pref_win.winfo_exists():
                self.pref_win.destroy()
        except Exception:
            pass
        self._start_main_run()

    def _pref_update_sys(self):
        if getattr(self, "pref_cancelled", False):
            return
        try:
            cpu = psutil.cpu_percent(interval=0.0)
            ram = psutil.virtual_memory().percent
            self.pref_cpu_var.set(f"CPU: {cpu:.0f}%")
            self.pref_ram_var.set(f"RAM: {ram:.0f}%")
            # gauge drawing
            self.pref_gauge.delete("all")
            self.pref_gauge.create_oval(10, 10, 210, 210, outline="#ddd")
            # CPU arc
            self.pref_gauge.create_arc(10, 10, 210, 210, start=90, extent=-3.6*min(cpu,100), style="arc", width=12, outline="#3498db")
            # RAM arc (inner)
            self.pref_gauge.create_arc(28, 28, 192, 192, start=90, extent=-3.6*min(ram,100), style="arc", width=12, outline="#e74c3c")
            self.pref_gauge.create_text(110, 110, text=f"{cpu:.0f}% CPU\n{ram:.0f}% RAM", font=("Arial", 10, "bold"))
        except Exception:
            pass
        try:
            if getattr(self, "pref_win", None) is not None and self.pref_win.winfo_exists():
                self.pref_win.after(500, self._pref_update_sys)
        except Exception:
            pass

    def _can_stratify(self, y, test_size: float):
        """Return (ok, reason, min_required) for using stratify in train_test_split.

        Stratified split requires enough samples per class AND enough room in train/test
        to place at least one sample per class in each split.
        """
        try:
            y_arr = np.asarray(y, dtype=int)
            n = int(y_arr.shape[0])
            if n < 2:
                return False, "n<2", 2
            counts_all = np.bincount(y_arr)
            present = counts_all > 0
            n_classes = int(present.sum())
            if n_classes <= 1:
                return False, "n_classes<=1", 2
            min_count = int(counts_all[present].min())
            # Minimum needed to guarantee at least 1 sample per class in both train and test
            if not (0.0 < float(test_size) < 1.0):
                return False, "bad test_size", 2
            min_required = max(2, int(np.ceil(1.0 / float(test_size))), int(np.ceil(1.0 / float(1.0 - float(test_size)))))
            n_test = int(np.ceil(n * float(test_size)))
            n_train = n - n_test
            if min_count < min_required:
                return False, f"min_count={min_count} < {min_required}", min_required
            if n_test < n_classes or n_train < n_classes:
                return False, f"too many classes for split (classes={n_classes}, test={n_test}, train={n_train})", min_required
            return True, "ok", min_required
        except Exception as e:
            return False, f"check_failed: {e}", 2


    def _pref_run_benchmark(self):
        """Runs a quick benchmark + produces per-model ETA in minutes."""
        try:
            if getattr(self, "pref_cancelled", False):
                return

            # Detect dataset file (same logic as load_data)
            files = ['fusion_train_smart6.csv', 'fusion_test_smart6.csv']
            fichier = next((f for f in files if os.path.exists(f)), None)

            if not fichier:
                self.root.after(0, lambda: self.pref_info.insert(tk.END, "Dataset: CSV non trouvé (fusion_train_smart6.csv / fusion_test_smart6.csv)\n"))
                self.root.after(0, lambda: self.pref_status_var.set("❌ CSV non trouvé → OK dispo (lancement direct possible)"))
                self.root.after(0, lambda: self.pref_ok_btn.config(state="normal"))
                return

            file_size = os.path.getsize(fichier)
            sample_n = 5000
            # Read small sample for columns + avg bytes/row
            df_s = pd.read_csv(fichier, nrows=sample_n, low_memory=False, encoding='utf-8')
            df_s = _normalize_label_column(df_s)

            # estimate rows (rough)
            try:
                avg_bytes_per_row = max(float(df_s.memory_usage(deep=True).sum()) / max(len(df_s), 1), 1.0)
                est_total_rows = max(int(file_size / avg_bytes_per_row), len(df_s))
            except Exception:
                est_total_rows = None

            numeric_cols = df_s.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            D = len(numeric_cols)

            # grid sizes
            grid_sizes = {name: len(self.generate_param_combinations(name)) for name in PARAM_GRIDS.keys()}
            G_total = sum(grid_sizes.values())
            total_ops = sum(v * K_FOLD for v in grid_sizes.values())

            # Recommend workers & chunk
            avail_gb = MemoryManager.get_available_ram_gb()
            base_workers = max(1, min(NUM_CORES, 4))
            # be conservative if low RAM
            if avail_gb < 6:
                base_workers = max(1, min(base_workers, 2))
            rec_chunk = MemoryManager.get_optimal_chunk_size(min_chunk=MIN_CHUNK_SIZE)

            self.pref_reco = {"workers": int(base_workers), "chunk_size": int(rec_chunk)}

            # Write info box
            def _info(text):
                try:
                    self.pref_info.insert(tk.END, text)
                    self.pref_info.see(tk.END)
                except Exception:
                    pass

            lines = []
            lines.append(f"Dataset: {fichier}\n")
            lines.append(f"Taille fichier: {file_size/1024/1024:.1f} MB\n")
            if est_total_rows:
                lines.append(f"Estimation lignes S≈{est_total_rows:,}\n")
            lines.append(f"Sample bench: n={len(df_s):,} lignes\n")
            lines.append(f"Features numériques D={D}\n")
            lines.append(f"K={K_FOLD}\n")
            lines.append("Grid combos par modèle:\n")
            for k, v in grid_sizes.items():
                lines.append(f"  - {k}: {v} combos\n")
            lines.append(f"Total folds (combos×K): {total_ops:,}\n")
            lines.append(f"Reco workers≈{base_workers} | chunk≈{rec_chunk:,}\n\n")

            self.root.after(0, lambda: (self.pref_info.delete(1.0, tk.END), _info(''.join(lines))))

            # Prepare sample arrays for bench
            if 'Label' not in df_s.columns:
                self.root.after(0, lambda: _info("WARN: colonne Label introuvable → bench impossible\n"))
                self.root.after(0, lambda: self.pref_ok_btn.config(state="normal"))
                return

            df_s = df_s.dropna(subset=['Label']).copy()
            df_s['Label'] = df_s['Label'].astype(str)


            # ---- Label distribution (FULL DATASET) + benign/attack aggregate ----
            full_label_counts = None
            full_total_rows = None
            benign_labels_detected = []

            try:
                label_col = 'Label'
                if 'Label' not in df_s.columns:
                    # find case-insensitive label col
                    for c in df_s.columns:
                        if str(c).lower() == 'label':
                            label_col = c
                            break

                self.root.after(0, lambda: self.pref_status_var.set("Scan labels (dataset complet)…"))

                counts: dict[str, int] = {}
                total_rows = 0
                last_ui = time.time()

                # read only label column in chunks to get exact counts without loading everything
                for chunk in pd.read_csv(
                    fichier,
                    usecols=[label_col],
                    chunksize=max(200_000, int(self.current_chunk_size // 5) or 200_000),
                    low_memory=False,
                    encoding='utf-8',
                ):
                    if getattr(self, "pref_cancelled", False):
                        return
                    s = chunk[label_col].astype(str)
                    vc_full = s.value_counts(dropna=False)
                    for k, v in vc_full.items():
                        ks = str(k)
                        counts[ks] = counts.get(ks, 0) + int(v)
                    total_rows += int(len(chunk))

                    # lightweight progress updates
                    now = time.time()
                    if now - last_ui >= 0.6:
                        if est_total_rows:
                            pct = min(99.0, 100.0 * (total_rows / max(int(est_total_rows), 1)))
                            self.root.after(0, lambda p=pct, tr=total_rows: self.pref_status_var.set(f"Scan labels: {p:.1f}% ({tr:,} lignes lues)"))
                        else:
                            self.root.after(0, lambda tr=total_rows: self.pref_status_var.set(f"Scan labels: {tr:,} lignes lues"))
                        last_ui = now

                full_label_counts = counts
                full_total_rows = total_rows

                # Detect BENIGN label(s) heuristically
                keys = list(counts.keys())
                all_int_like = True
                for k in keys[:50]:
                    if not re.fullmatch(r"-?\d+", str(k).strip()):
                        all_int_like = False
                        break

                if all_int_like and "0" in counts:
                    benign_labels_detected = ["0"]
                else:
                    # common in IDS datasets: BENIGN / NORMAL
                    for k in keys:
                        kl = str(k).strip().lower()
                        if kl == "benign" or kl == "normal" or "benign" in kl or kl.startswith("normal"):
                            benign_labels_detected.append(str(k))

                benign_count = sum(counts.get(lbl, 0) for lbl in benign_labels_detected) if benign_labels_detected else 0
                attack_count = int(total_rows - benign_count)

                # Write summary
                _info("\n=== Répartition Labels (dataset complet) ===\n")
                _info(f"Total lignes (compté) S={total_rows:,}\n")
                if benign_labels_detected:
                    _info(f"Benign labels détectés: {benign_labels_detected}\n")
                    _info(f"BENIGN: {benign_count:,} ({100.0*benign_count/max(total_rows,1):.2f}%)\n")
                    _info(f"ATTACK/OTHER: {attack_count:,} ({100.0*attack_count/max(total_rows,1):.2f}%)\n")
                else:
                    _info("Benign label non détecté automatiquement (labels non standard).\n")

                # Top / rare labels + % (limit to keep readable)
                items_sorted = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                top_n = min(5, len(items_sorted))
                rare_n = min(5, len(items_sorted))

                def _fmt_rows(items):
                    out = []
                    for lbl, c in items:
                        out.append(f"{lbl:<35} {c:>12,}  ({100.0*c/max(total_rows,1):>6.2f}%)")
                    return "\n".join(out) + ("\n" if out else "")

                _info("\nTop labels (dataset complet):\n")
                _info(_fmt_rows(items_sorted[:top_n]))
                # 'Others' bucket for readability
                try:
                    top_sum = sum(c for _lbl, c in items_sorted[:top_n])
                    rest = int(total_rows - top_sum)
                    if rest > 0:
                        _info(f"{'OTHER_LABELS':<35} {rest:>12,}  ({100.0*rest/max(total_rows,1):>6.2f}%)\n")
                except Exception:
                    pass
                _info("\nLabels rares (dataset complet):\n")
                _info(_fmt_rows(sorted(items_sorted, key=lambda kv: (kv[1], kv[0]))[:rare_n]))

                # Export full distribution for later inspection
                try:
                    import csv
                    out_path = Path('label_distribution_full.csv')
                    with out_path.open('w', newline='', encoding='utf-8') as _csvf:
                        w = csv.writer(_csvf)
                        w.writerow(['label', 'count', 'percent'])
                        for _lbl, _c in items_sorted:
                            w.writerow([_lbl, int(_c), 100.0*int(_c)/max(total_rows,1)])
                    _info(f"\nFull distribution saved: {out_path.resolve()}\n")
                except Exception:
                    pass

                # Show problematic singletons for stratify on FULL dataset
                singletons = [lbl for lbl, c in items_sorted if c < 2]
                if singletons:
                    _info(f"\n⚠️ Classes avec <2 occurrences (dataset complet): {len(singletons)}\n")
                    preview = singletons[:30]
                    _info("Exemples: " + ", ".join(preview) + (" ..." if len(singletons) > 30 else "") + "\n")

            except Exception as _e:
                _info(f"\n[WARN] Scan labels complet impossible: {_e}\n")

            # ---- Label distribution (sample) + problematic classes for stratified split ----
            try:
                vc = df_s['Label'].value_counts()
                n_classes = int(len(vc))
                min_count = int(vc.min()) if n_classes else 0
                max_count = int(vc.max()) if n_classes else 0
                _info("\n=== Répartition Labels (échantillon bench) ===\n")
                _info("⚠️ NB: échantillon = premières lignes du CSV → si le fichier est trié, ça peut être TRÈS biaisé.\n")
                _info(f"Classes (sample): {n_classes} | min={min_count} | max={max_count}\n")
                try:
                    if full_label_counts is not None:
                        _full_min = int(full_label_counts.min()) if len(full_label_counts) else 0
                        _full_max = int(full_label_counts.max()) if len(full_label_counts) else 0
                        _info(f"Classes (dataset complet): {len(full_label_counts)} | min={_full_min} | max={_full_max} | total={int(full_total_rows or 0):,}\n")
                        if full_benign_count is not None:
                            _info(f"Binaire (dataset complet) → BENIGN={int(full_benign_count):,} | OTHER/ATTACK={int(full_attack_count):,}\n")
                except Exception:
                    pass


                # Show a quick preview
                head_n = min(10, n_classes)
                tail_n = min(10, n_classes)
                if n_classes:
                    _info("\nTop classes (sample):\n")
                    _info(vc.head(head_n).to_string() + "\n")
                    _info("\nRare classes (sample):\n")
                    _info(vc.tail(tail_n).to_string() + "\n")

                    # Always-failing for stratify: count < 2
                    bad2 = vc[vc < 2]
                    if len(bad2):
                        _info(f"\n⚠️ Classes avec <2 exemples (stratify IMPOSSIBLE): {len(bad2)}\n")
                        _info(bad2.head(30).to_string() + ("\n...\n" if len(bad2) > 30 else "\n"))

                    # Risky for specific test_size values used in this script
                    for ts in (0.2, 0.3):
                        min_req = max(2, int(np.ceil(1.0 / ts)), int(np.ceil(1.0 / (1.0 - ts))))
                        bad = vc[vc < min_req]
                        if len(bad):
                            _info(f"\n⚠️ Trop rare pour stratify avec test_size={ts} (min_req={min_req}): {len(bad)}\n")
                            _info(bad.head(30).to_string() + ("\n...\n" if len(bad) > 30 else "\n"))
            except Exception as _e:
                _info(f"[WARN] analyse classes sample impossible: {_e}\n")


            # Clean numeric
            for col in numeric_cols:
                df_s[col] = df_s[col].replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)

            X = df_s[numeric_cols].apply(pd.to_numeric, errors="coerce")
            for col in X.columns:
                m = X[col].mean()
                X[col] = X[col].fillna(0.0 if pd.isna(m) else m)

            le = LabelEncoder()
            y = le.fit_transform(df_s['Label'])

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X).astype(NPZ_FLOAT_DTYPE)

            # Choose one "representative" params per model (middle of grid)
            def pick_params(model_name):
                combos = self.generate_param_combinations(model_name)
                if not combos:
                    return {}
                return combos[len(combos)//2]

            bench_models = {
                'Logistic Regression': LogisticRegression,
                'Naive Bayes': GaussianNB,
                'Decision Tree': DecisionTreeClassifier,
                'Random Forest': RandomForestClassifier,
            }

            # Benchmark: 1 fold per model (fast), with small workers
            t0 = {}
            S0 = len(Xs)
            p = 0.2

            self.root.after(0, lambda: self.pref_status_var.set("Benchmark: fit+predict (sample)…"))

            for name, ModelClass in bench_models.items():
                if getattr(self, "pref_cancelled", False):
                    return
                t_start = time.perf_counter()
                ts = (0.3 if name=='Decision Tree' else 0.2)
                ok_strat, reason_strat, min_req = self._can_stratify(y, ts)
                if not ok_strat:
                    _info(f"[bench] {name}: stratify OFF ({reason_strat})\\n")
                try:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            Xs, y, test_size=(0.3 if name=='Decision Tree' else 0.2), random_state=42,
                            stratify=(y if ok_strat else None)
                        )
                    except ValueError as ve:
                        _info(f"[bench] {name}: split stratifié impossible ({ve}) -> stratify OFF\\n")
                        X_train, X_test, y_train, y_test = train_test_split(
                            Xs, y, test_size=(0.3 if name=='Decision Tree' else 0.2), random_state=42,
                            stratify=None
                        )
                    params = pick_params(name)
                    kwargs = dict(params)
                    if name in ('Logistic Regression', 'Random Forest'):
                        kwargs['n_jobs'] = 1
                    if name != 'Naive Bayes':
                        kwargs['random_state'] = 42
                    model = ModelClass(**kwargs)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    _ = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    t_end = time.perf_counter()
                    t0[name] = max(t_end - t_start, 1e-6)
                    self.root.after(0, lambda n=name, dt=t0[name]: _info(f"[bench] {n}: {dt:.3f}s / fold (sample)\n"))
                except Exception as e:
                    t0[name] = None
                    self.root.after(0, lambda n=name, er=e: _info(f"[bench] {n}: FAIL ({er})\n"))

            # Extrapolate to full dataset size (seconds)
            S_full = est_total_rows or S0
            log_full = math.log(max(S_full, 2), 2)
            log0 = math.log(max(S0, 2), 2)
            W_eff = max(1.0, 0.75 * float(base_workers))

            def scale_nb(t):  # ~ S
                return t * (S_full / S0)

            def scale_lr(t):  # ~ S (I absorbed in t0)
                return t * (S_full / S0)

            def scale_dt(t):  # ~ S log S
                return t * ((S_full * log_full) / (S0 * log0))

            def scale_rf(t):  # ~ S log S (n_estimators absorbed in t0)
                return t * ((S_full * log_full) / (S0 * log0))

            scalers = {
                'Naive Bayes': scale_nb,
                'Logistic Regression': scale_lr,
                'Decision Tree': scale_dt,
                'Random Forest': scale_rf,
            }

            eta_min = {}
            eta_min_opt = {}
            eta_min_pess = {}

            for name in bench_models.keys():
                base = t0.get(name)
                if not base:
                    continue
                t_fold_full = scalers[name](base)
                Gm = grid_sizes.get(name, 1)
                t_total_sec = (Gm * K_FOLD / W_eff) * t_fold_full
                mins = t_total_sec / 60.0
                eta_min[name] = mins
                eta_min_opt[name] = mins * 0.8
                eta_min_pess[name] = mins * 1.3

            # Plot bars
            def draw_plot():
                self.pref_ax.clear()
                labels = list(bench_models.keys())
                x = np.arange(len(labels))
                real = [eta_min.get(l, 0.0) for l in labels]
                opt = [eta_min_opt.get(l, 0.0) for l in labels]
                pess = [eta_min_pess.get(l, 0.0) for l in labels]
                width = 0.25
                self.pref_ax.bar(x - width, opt, width, label="Opt")
                self.pref_ax.bar(x, real, width, label="Réaliste")
                self.pref_ax.bar(x + width, pess, width, label="Pess")
                self.pref_ax.set_xticks(x)
                self.pref_ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
                self.pref_ax.set_ylabel("Minutes")
                self.pref_ax.grid(axis="y", alpha=0.3)
                self.pref_ax.legend(fontsize=9)
                self.pref_fig.tight_layout()
                self.pref_canvas.draw_idle()

            self.root.after(0, draw_plot)

            total_real = sum(eta_min.values())
            total_opt = sum(eta_min_opt.values())
            total_pess = sum(eta_min_pess.values())
            self.root.after(0, lambda: _info(f"\nTOTAL estimé: opt≈{total_opt:.1f} min | réal≈{total_real:.1f} min | pess≈{total_pess:.1f} min\n"))
            self.root.after(0, lambda: self.pref_gauge_text.config(text=f"Total réal≈{total_real:.1f} min"))
            self.root.after(0, lambda: self.pref_status_var.set("✅ OK pour lancer (réglages recommandés appliqués)"))
            self.root.after(0, lambda: self.pref_ok_btn.config(state="normal"))
            self.pref_done = True

        except Exception as e:
            try:
                self.root.after(0, lambda: self.pref_status_var.set(f"Préflight crash: {e}"))
                self.root.after(0, lambda: self.pref_ok_btn.config(state="normal"))
            except Exception:
                pass

    def _start_main_run(self):
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
        for k in ("load", "transit_load", "prep", "transit_mid", "grid", "transit_grid", "overall"):
            self.ui_shell.set_stage_progress(k, 0.0)
        self.ui_shell.set_overall_progress(0.0)
        self.ui_shell.set_best_score("--")
        self.ui_shell.set_ai_recommendation("Waiting...")
        self._ui_tasks("Load → Prep → Grid → Reports")
        self._reset_thread_bars()
        for key in ("load", "transit_load", "prep", "transit_mid", "grid", "transit_grid", "reports"):
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
        # open graph window immediately so it is visible during the run
        try:
            self.show_graphs()
            self.graph_auto_opened = True
        except Exception:
            self.graph_auto_opened = False
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
            chunk_size = max(MIN_CHUNK_SIZE, self.current_chunk_size)
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
                            chunk_size = max(MIN_CHUNK_SIZE, min(1_000_000, new_chunk))
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
            self._maybe_checkpoint()
            # push final load metrics to AI
            try:
                self.ai_server.send_metrics(
                    AIMetrics(
                        timestamp=time.time(),
                        num_workers=int(self.current_workers),
                        chunk_size=int(self.current_chunk_size),
                        rows_processed=int(total_rows),
                        ram_percent=float(psutil.virtual_memory().percent),
                        cpu_percent=float(psutil.cpu_percent(interval=0.0)),
                        throughput=float(self.rows_seen / max(time.time() - self.start_time, 1e-3)),
                    )
                )
            except Exception:
                pass
            return True
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')
            self.add_alert(f'Load failed: {e}', "ERROR")
            try:
                self.root.bell()
            except Exception:
                pass
            return False

    def prepare_data(self):
        """ETAPE 2: Preparation (VERSION FIXÉE - ERREUR #2 CORRIGÉE)"""
        try:
            self.log_live('ETAPE 2: Preparation\n', 'info')
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            self.df = self.df.dropna(subset=['Label'])
            # ensure labels are strings before any split
            self.df['Label'] = self.df['Label'].astype(str)
            
            n_samples = int(len(self.df) * STRATIFIED_SAMPLE_RATIO)
            stratifier = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            try:
                y_labels = self.df['Label'].to_numpy()
                for train_idx, _ in stratifier.split(self.df, y_labels):
                    self.df = self.df.iloc[train_idx[:n_samples]]
                    break
            except Exception as e:
                self.log_live(f'[WARN] Stratified split failed: {e} -> simple sample\n', 'info')
                self.df = self.df.iloc[:n_samples].copy()
            
            self.log_live(f'Dataset: {len(self.df):,} lignes\n', 'info')
            
            # ✅ CRITICAL STEP FIX #2: Clean infinity and extreme values FIRST
            self.log_live('Cleaning infinity and extreme values...\n', 'info')
            for col in numeric_cols:
                try:
                    # Replace infinity with NaN (must be BEFORE dtype conversion)
                    self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                    # Clip extreme values to ±1e6
                    self.df[col] = self.df[col].clip(-1e6, 1e6)
                except Exception as e:
                    self.log_live(f'WARN: Error cleaning column {col}: {e}\n', 'info')
            
            # Extract features X
            X = self.df[numeric_cols].copy()
            try:
                X = X.apply(pd.to_numeric, errors="coerce")
            except Exception as e:
                self.log_live(f'WARN: to_numeric failed {e}\n', 'info')
            
            # ✅ FILL NaN with column mean (REQUIRED)
            self.log_live('Filling NaN values with column mean...\n', 'info')
            for col in X.columns:
                col_mean = X[col].mean()
                if pd.isna(col_mean):
                    # If entire column is NaN, use 0
                    X[col] = X[col].fillna(0.0)
                else:
                    X[col] = X[col].fillna(col_mean)
            
            # Convert to float64 then float32 (safe conversion)
            self.log_live('Converting data types...\n', 'info')
            try:
                X = X.astype(np.float64)
                X = X.astype(NPZ_FLOAT_DTYPE)  # float32
            except Exception as e:
                self.log_live(f'ERROR converting types: {e}\n', 'info')
                self.add_alert(f'Type conversion failed: {e}')
                return False
            
            # ✅ VALIDATE: Check no infinity/NaN remain
            n_inf = np.isinf(X.values).sum()
            n_nan = np.isnan(X.values).sum()
            
            if n_inf > 0 or n_nan > 0:
                self.log_live(f'WARN: Found {n_inf} inf and {n_nan} NaN after cleaning!\n', 'info')
                # Final force fill
                X = X.fillna(0.0)
                X = X.replace([np.inf, -np.inf], 1e6)
                self.log_live(f'Force-filled NaN/inf\n', 'info')
            
            self.log_live(f'Features validated (no inf/nan) ✓\n', 'info')
            
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.df['Label'])
            
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X).astype(NPZ_FLOAT_DTYPE)
            
            self.log_live(f'Data standardized: X shape {self.X_scaled.shape}\n', 'info')
            try:
                self._maybe_show_time_estimate()
            except Exception:
                pass
            
            np.savez_compressed('preprocessed_dataset.npz',
                               X=self.X_scaled,
                               y=self.y,
                               classes=self.label_encoder.classes_)
            
            self.log_live(f'NPZ saved successfully\n\n', 'info')
            self.rows_seen += len(self.X_scaled)
            self._ui_stage("prep", 100.0)
            self._update_stage_eta("prep", 1, 1)
            try:
                self._send_ai_metric(rows=0, chunk_size=int(self.current_chunk_size))
            except Exception:
                pass
            self._maybe_checkpoint()
            del self.df, X
            gc.collect()
            return True
        except Exception as e:
            self.log_live(f'Erreur: {e}\n{traceback.format_exc()}\n', 'info')
            self.add_alert(f'Prep failed: {e}')
            try:
                self.root.bell()
            except Exception:
                pass
            return False



    # -------------------- Runtime estimation (hardware + dataset) --------------------
    def _human_time(self, seconds: float) -> str:
        try:
            seconds = float(seconds)
        except Exception:
            return "--"
        if seconds < 0:
            seconds = 0
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h}h {m:02d}m {sec:02d}s"
        if m > 0:
            return f"{m}m {sec:02d}s"
        return f"{sec}s"


    def _attach_main_estimate_canvas(self):
        """Ajoute un canvas d'estimation dans la fenêtre principale (dans 'Monitoring + Progress' si possible)."""
        try:
            monitor = None
            for w in self.root.winfo_children():
                try:
                    if isinstance(w, tk.LabelFrame) and str(w.cget("text")) == "Monitoring + Progress":
                        monitor = w
                        break
                except Exception:
                    continue
            parent = monitor if monitor is not None else self.root

            self.main_estimate_canvas = tk.Canvas(parent, height=90, bg="white", highlightthickness=1)
            self.main_estimate_canvas.pack(fill="x", padx=6, pady=6)
            try:
                self.main_estimate_canvas.create_text(10, 10, anchor="nw", text="Estimation: en attente…", fill="black")
            except Exception:
                pass
        except Exception:
            self.main_estimate_canvas = None

    def _draw_estimate_bars(self, canvas):
        """Dessine la barre opt/real/pess sur un canvas Tk."""
        try:
            if canvas is None:
                return
            canvas.delete("all")
            wpx = int(canvas.winfo_width() or 900)
            pad = 12
            text_space = 10
            bar_w = max(wpx - 2 * pad, 200)
            y0 = 42

            opt_s = float(getattr(self, "estimated_total_seconds_opt", 0.0) or 0.0)
            real_s = float(getattr(self, "estimated_total_seconds", 0.0) or 0.0)
            pess_s = float(getattr(self, "estimated_total_seconds_pess", 0.0) or 0.0)
            denom = max(pess_s, 1.0)
            opt_r = min(max(opt_s / denom, 0.0), 1.0)
            real_r = min(max(real_s / denom, 0.0), 1.0)

            canvas.create_text(pad, 10, anchor="nw", text="Temps estimé (opt / réaliste / pess)", fill="black")
            canvas.create_rectangle(pad, y0, pad + bar_w, y0 + 18, outline="black")

            # opt (top band)
            canvas.create_rectangle(pad, y0, pad + int(bar_w * opt_r), y0 + 6, outline="", fill="#7bd389")
            # real (middle band)
            canvas.create_rectangle(pad, y0 + 6, pad + int(bar_w * real_r), y0 + 12, outline="", fill="#4ea8de")
            # pess (bottom band full)
            canvas.create_rectangle(pad, y0 + 12, pad + bar_w, y0 + 18, outline="", fill="#f77f7f")

            txt = f"opt: {self._human_time(opt_s)} | real: {self._human_time(real_s)} | pess: {self._human_time(pess_s)}"
            canvas.create_text(pad, y0 + 26, anchor="nw", text=txt, fill="black")
        except Exception:
            pass
    def _hardware_profile(self) -> dict:
        try:
            vm = psutil.virtual_memory()
            cpu_freq = getattr(psutil, "cpu_freq", lambda: None)()
            freq_mhz = getattr(cpu_freq, "current", None) if cpu_freq else None
            return {
                "cores_logical": int(multiprocessing.cpu_count()),
                "cores_physical": int(getattr(psutil, "cpu_count", lambda logical=False: None)(logical=False) or 0),
                "ram_total_gb": float(vm.total / (1024**3)),
                "ram_available_gb": float(vm.available / (1024**3)),
                "cpu_freq_mhz": float(freq_mhz) if freq_mhz else None,
            }
        except Exception:
            return {"cores_logical": int(multiprocessing.cpu_count())}

    def _benchmark_one_fold(self, model_name: str, ModelClass, params: dict, bench_rows: int) -> float:
        """Mesure rapide (1 fit/predict) sur un sous-échantillon. Renvoie seconds."""
        try:
            n = int(min(max(bench_rows, 2000), len(self.X_scaled)))
            if n <= 0:
                return 0.0
            Xb = self.X_scaled[:n]
            yb = self.y[:n]
            t0 = time.perf_counter()
            X_train, X_test, y_train, y_test = train_test_split(
                Xb, yb,
                test_size=0.2 if model_name != 'Decision Tree' else 0.3,
                random_state=123,
                stratify=yb if len(set(yb)) > 1 else None
            )
            kwargs = dict(params or {})
            if model_name in ('Logistic Regression', 'Random Forest'):
                kwargs['n_jobs'] = int(self.current_workers)
            if model_name != 'Naive Bayes':
                kwargs['random_state'] = 42
            model = ModelClass(**kwargs)
            model.fit(X_train, y_train)
            _ = model.predict(X_test)
            return max(time.perf_counter() - t0, 1e-6)
        except Exception:
            return 0.0

    def estimate_total_runtime(self) -> dict:
        """Estime le temps total (optimiste/réaliste/pessimiste) basé sur matos + dataset (best-effort)."""
        info = {
            "ok": False,
            "total_seconds": None,          # realistic
            "total_seconds_opt": None,
            "total_seconds_pess": None,
            "by_model_seconds": {},         # realistic
            "by_model_seconds_opt": {},
            "by_model_seconds_pess": {},
            "bench_rows": None,
            "workers": int(getattr(self, "current_workers", 1) or 1),
            "dataset_shape": None,
            "hardware": self._hardware_profile(),
        }
        try:
            if self.X_scaled is None or self.y is None:
                return info

            n_rows, n_feats = int(self.X_scaled.shape[0]), int(self.X_scaled.shape[1])
            info["dataset_shape"] = (n_rows, n_feats)
            bench_rows = int(min(5000, n_rows))
            info["bench_rows"] = bench_rows

            model_configs = {
                'Logistic Regression': LogisticRegression,
                'Naive Bayes': GaussianNB,
                'Decision Tree': DecisionTreeClassifier,
                'Random Forest': RandomForestClassifier,
            }

            def pick_params(name: str, heavy: bool) -> dict:
                g = PARAM_GRIDS.get(name, {}) or {}
                p = {}
                for k, vals in g.items():
                    if not vals:
                        continue
                    if all(isinstance(v, (int, float)) for v in vals):
                        p[k] = max(vals) if heavy else min(vals)
                    else:
                        p[k] = vals[-1] if heavy else vals[0]
                # Safety fallbacks
                if name == "Random Forest":
                    p.setdefault("n_estimators", 200 if heavy else 50)
                    p.setdefault("max_depth", 20 if heavy else 15)
                if name == "Decision Tree":
                    p.setdefault("max_depth", 20 if heavy else 10)
                if name == "Logistic Regression":
                    p.setdefault("max_iter", 2000 if heavy else 1000)
                return p

            # Parallelism: folds run in Parallel() per combo
            workers = max(1, int(getattr(self, "current_workers", 1) or 1))
            waves = int(math.ceil(K_FOLD / workers))

            scale_rows = max(n_rows / max(bench_rows, 1), 1.0)

            def overhead_multiplier(name: str) -> float:
                # UI/log/joblib overhead differs slightly per model; keep simple
                if name == "Random Forest":
                    return 1.30
                if name == "Decision Tree":
                    return 1.20
                return 1.15

            total_opt = 0.0
            total_real = 0.0
            total_pess = 0.0

            for model_name, ModelClass in model_configs.items():
                combos = len(self.generate_param_combinations(model_name))

                light_params = pick_params(model_name, heavy=False)
                heavy_params = pick_params(model_name, heavy=True)

                t_fold_light = self._benchmark_one_fold(model_name, ModelClass, light_params, bench_rows)
                t_fold_heavy = self._benchmark_one_fold(model_name, ModelClass, heavy_params, bench_rows)

                # Guard: if one bench fails, reuse the other
                if t_fold_light <= 0 and t_fold_heavy > 0:
                    t_fold_light = t_fold_heavy
                if t_fold_heavy <= 0 and t_fold_light > 0:
                    t_fold_heavy = t_fold_light
                if t_fold_light <= 0 and t_fold_heavy <= 0:
                    # can't benchmark => bail
                    return info

                # Extrapolate to full dataset
                fold_opt = t_fold_light * scale_rows
                fold_pess = t_fold_heavy * scale_rows
                fold_real = (fold_opt + fold_pess) / 2.0

                # Per-combo wall time (K folds in waves)
                combo_opt = fold_opt * waves
                combo_real = fold_real * waves
                combo_pess = fold_pess * waves

                mult = overhead_multiplier(model_name)
                model_opt = combo_opt * combos * mult
                model_real = combo_real * combos * mult
                model_pess = combo_pess * combos * mult

                info["by_model_seconds_opt"][model_name] = float(model_opt)
                info["by_model_seconds"][model_name] = float(model_real)
                info["by_model_seconds_pess"][model_name] = float(model_pess)

                total_opt += model_opt
                total_real += model_real
                total_pess += model_pess

            # Global safety margin
            total_opt *= 1.05
            total_real *= 1.10
            total_pess *= 1.15

            info["total_seconds_opt"] = float(total_opt)
            info["total_seconds"] = float(total_real)
            info["total_seconds_pess"] = float(total_pess)
            info["ok"] = True
            return info
        except Exception:
            return info



    def _maybe_show_time_estimate(self):
        """Log + surface l'estimation (appelable après prepare_data)."""
        try:
            # Soft guard sur workers selon RAM dispo
            try:
                avail = MemoryManager.get_available_ram_gb()
                if avail < 4:
                    self.current_workers = min(int(self.current_workers), 1)
                elif avail < 8:
                    self.current_workers = min(int(self.current_workers), 2)
            except Exception:
                pass

            info = self.estimate_total_runtime()
            if not info.get("ok"):
                return

            self.estimated_total_seconds = float(info.get("total_seconds") or 0.0)
            self.estimated_total_seconds_opt = float(info.get("total_seconds_opt") or 0.0)
            self.estimated_total_seconds_pess = float(info.get("total_seconds_pess") or 0.0)

            n_rows, n_feats = info.get("dataset_shape", ("?", "?"))
            hw = info.get("hardware", {})
            cores = hw.get("cores_logical", "?")
            ram_total = hw.get("ram_total_gb", None)
            ram_total_s = f"{ram_total:.1f}GB" if isinstance(ram_total, (int, float)) else "?"
            w = info.get("workers", self.current_workers)

            self.log_live(f"[ESTIMATE] Matos: {cores} cores | RAM {ram_total_s} | workers={w}", "info")
            self.log_live(f"[ESTIMATE] Dataset: {n_rows:,} lignes × {n_feats} features (bench={info.get('bench_rows')})", "info")

            for k, v in info.get("by_model_seconds", {}).items():
                self.log_live(f"[ESTIMATE] {k}: ~{self._human_time(v)}", "info")

            self.log_live(f"[ESTIMATE] TOTAL (opt): ~{self._human_time(self.estimated_total_seconds_opt)}", "info")
            self.log_live(f"[ESTIMATE] TOTAL (real): ~{self._human_time(self.estimated_total_seconds)}", "info")
            self.log_live(f"[ESTIMATE] TOTAL (pess): ~{self._human_time(self.estimated_total_seconds_pess)}", "info")

            try:
                self._ui_tasks(f"Estimation: ~{self._human_time(self.estimated_total_seconds)}")
            except Exception:
                pass

            # Dessin sur le canvas de la fenêtre Graphs + celui de la fenêtre principale
            def _draw_all():
                try:
                    self._draw_estimate_bars(getattr(self, "extra_canvas", None))
                except Exception:
                    pass
                try:
                    self._draw_estimate_bars(getattr(self, "main_estimate_canvas", None))
                except Exception:
                    pass

            try:
                self.root.after(0, _draw_all)
            except Exception:
                _draw_all()
        except Exception:
            pass


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

            # Transit entre load et prep
            self.log_live('Transit: Load → Prep\n', 'info')
            self._ui_stage("transit_load", 50.0)
            self._ui_overall(10.0)
            try:
                self.ai_server.send_metrics(
                    AIMetrics(
                        timestamp=time.time(),
                        num_workers=int(self.current_workers),
                        chunk_size=int(self.current_chunk_size),
                        rows_processed=0,
                        ram_percent=float(psutil.virtual_memory().percent),
                        cpu_percent=float(psutil.cpu_percent(interval=0.0)),
                        throughput=0.0,
                    )
                )
            except Exception:
                pass
            self._ui_stage("transit_load", 100.0)
            self._update_stage_eta("transit_load", 1, 1)
            self._maybe_checkpoint()
            self._ui_stage("transit_load", 0.0)
            
            if not self.prepare_data():
                return
            if not self.running:
                return
            
            # Transit phase (UI + AI heartbeat) Prep → Grid
            self.log_live('ETAPE 2.5: Transit Prep → Grid\n', 'info')
            self._ui_stage("transit_mid", 50.0)
            self._ui_overall(50.0)
            try:
                self.ai_server.send_metrics(
                    AIMetrics(
                        timestamp=time.time(),
                        num_workers=int(self.current_workers),
                        chunk_size=int(self.current_chunk_size),
                        rows_processed=0,
                        ram_percent=float(psutil.virtual_memory().percent),
                        cpu_percent=float(psutil.cpu_percent(interval=0.0)),
                        throughput=0.0,
                    )
                )
            except Exception:
                pass
            self._ui_stage("transit_mid", 100.0)
            self._update_stage_eta("transit_mid", 1, 1)
            self._maybe_checkpoint()
            # hide transit bar after completion
            self._ui_stage("transit_mid", 0.0)

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
                
                # ✅ FIX #1: THIS LOOP AND ALL INNER CODE IS NOW PROPERLY INDENTED
                for combo_idx, params in enumerate(combinations, 1):
                    if i == start_model_idx and combo_idx <= start_combo_idx:
                        continue
                    if not self.running:
                        return
                    
                    # ✅ FIX #1: def run_fold IS NOW INSIDE THE LOOP (5 indents instead of 4)
                    def run_fold(fold):
                        try:
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    self.X_scaled, self.y,
                                    test_size=0.2 if name != 'Decision Tree' else 0.3,
                                    random_state=42 + fold,
                                    stratify=self.y
                                )
                            except ValueError:
                                # Fallback if some classes are too rare for a stratified split
                                X_train, X_test, y_train, y_test = train_test_split(
                                    self.X_scaled, self.y,
                                    test_size=0.2 if name != 'Decision Tree' else 0.3,
                                    random_state=42 + fold,
                                    stratify=None
                                )
                            kwargs = params.copy()
                            if name in ('Logistic Regression', 'Random Forest'):
                                kwargs['n_jobs'] = int(self.current_workers)
                            if name != 'Naive Bayes':
                                kwargs['random_state'] = 42
                            model = ModelClass(**kwargs)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            rows_proc = len(X_train) + len(X_test)
                            return f1, rows_proc
                        except Exception:
                            return 0, 0

                    params_str_verbose = ", ".join(f"{k}={v}" for k, v in params.items())
                    self.log_live(f"[{name}] combo {combo_idx}/{len(combinations)} | params: {params_str_verbose}", "info")
                    results = Parallel(n_jobs=int(self.current_workers), backend="threading")(delayed(run_fold)(fold) for fold in range(K_FOLD))
                    f1_runs = []
                    for f1_val, rows_proc in results:
                        f1_runs.append(f1_val)
                        try:
                            self.rows_seen += int(rows_proc)
                        except Exception:
                            pass
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
                    self.log_live(f'    [{combo_idx}/{len(combinations)}] {params_str}: F1={mean_f1:.4f}\n', 'info')
                    # show live params under the F1 chart
                    self._set_live_params(name, params)
                    all_results.append({'params': params, 'f1': mean_f1})
                    # live graph update
                    try:
                        self._update_live_graph(name, [r['f1'] for r in all_results])
                    except Exception:
                        pass
                    
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
                        self.current_chunk_size = max(MIN_CHUNK_SIZE, min(1_000_000, new_chunk))
                        self.log_live(f"[AI] chunk->{self.current_chunk_size:,} (reason: {rec.reason})", "info")
                        self._ui_ai_rec(rec.reason)
                    # checkpoint after each combo
                    self.ckpt["model_idx"] = i
                    self.ckpt["combo_idx"] = combo_idx
                    self.ckpt["best_score"] = best_score
                    self.ckpt["current_workers"] = self.current_workers
                    self.ckpt["current_chunk_size"] = self.current_chunk_size
                    self.ckpt["completed_ops"] = self.completed_operations
                    self._maybe_checkpoint()
                
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

            # Transit Grid → Reports visualization
            self._ui_stage("transit_grid", 50.0)
            self._ui_overall(90.0)
            self._ui_stage("transit_grid", 100.0)
            self._update_stage_eta("transit_grid", 1, 1)
            try:
                self._send_ai_metric(rows=0, chunk_size=int(self.current_chunk_size))
            except Exception:
                pass
            self._maybe_checkpoint()
            self._ui_stage("transit_grid", 0.0)
            self.root.after(0, lambda: self.ui_shell.set_status("Completed"))
            self.add_alert('GRID SEARCH COMPLETE')
            self._ui_stage("grid", 100.0)
            self._ui_overall(100.0)
            self.root.after(0, lambda: self.graphs_btn.config(state=tk.NORMAL))

        except Exception as e:
            self.log_live(f'Erreur: {e}\n{traceback.format_exc()}\n', 'info')
            self.root.after(0, lambda: self.ui_shell.set_status("Erreur"))
            self.add_alert(f'Run crashed: {e}', "ERROR")
            try:
                self.root.bell()
            except Exception:
                pass
        
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
        """Affiche les graphiques scrollables"""
        # Just raise the live window (already drawing incremental curves)
        try:
            if self.graph_window and self.graph_window.winfo_exists():
                self.graph_window.deiconify()
                self.graph_window.lift()
                return
        except Exception:
            pass



class RegressionVisualizerLite:
    """Interactive regression playground embedded as second tab."""

    def __init__(self, parent):
        self.parent = parent
        self.seed = 42
        np.random.seed(self.seed)
        self.X = np.linspace(0, 10, 100).reshape(-1, 1)
        self.y = 2.5 * self.X.flatten() + 5

        self.alpha_var = tk.DoubleVar(value=0.1)
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.noise_var = tk.DoubleVar(value=2.0)
        self.model_type_var = tk.StringVar(value="Linear")

        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        container = ttk.Frame(self.parent)
        container.pack(fill="both", expand=True)
        left = ttk.Frame(container, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(container)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Type de modèle", font=("Arial", 10, "bold")).pack(anchor="w", pady=4)
        for model in ["Linear", "Ridge", "Lasso"]:
            ttk.Radiobutton(left, text=model, variable=self.model_type_var,
                            value=model, command=self._update_plot).pack(anchor="w")

        ttk.Label(left, text="Alpha", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Scale(left, from_=0.001, to=10.0, orient=tk.HORIZONTAL,
                  variable=self.alpha_var, command=lambda _e: self._update_plot()).pack(fill="x")
        self.alpha_lab = ttk.Label(left, text="")
        self.alpha_lab.pack(anchor="w")

        ttk.Label(left, text="Test size", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Scale(left, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                  variable=self.test_size_var, command=lambda _e: self._update_plot()).pack(fill="x")
        self.test_lab = ttk.Label(left, text="")
        self.test_lab.pack(anchor="w")

        ttk.Label(left, text="Bruit σ", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Scale(left, from_=0.0, to=10.0, orient=tk.HORIZONTAL,
                  variable=self.noise_var, command=lambda _e: self._update_noise()).pack(fill="x")
        self.noise_lab = ttk.Label(left, text="")
        self.noise_lab.pack(anchor="w")

        ttk.Button(left, text="Régénérer données", command=self._regen).pack(fill="x", pady=6)

        self.fig_reg = Figure(figsize=(8, 6), dpi=100)
        self.canvas_reg = FigureCanvasTkAgg(self.fig_reg, master=right)
        self.canvas_reg.get_tk_widget().pack(fill="both", expand=True)

    def _regen(self):
        self.seed += 1
        np.random.seed(self.seed)
        self._update_plot()

    def _update_noise(self):
        self._update_plot()

    def _generate_data(self):
        self.y = 2.5 * self.X.flatten() + 5 + np.random.randn(len(self.X)) * self.noise_var.get()

    def _update_plot(self):
        try:
            self.alpha_lab.config(text=f"α = {self.alpha_var.get():.4f}")
            self.test_lab.config(text=f"Test size = {self.test_size_var.get():.2f}")
            self.noise_lab.config(text=f"Bruit σ = {self.noise_var.get():.2f}")
            self._generate_data()

            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size_var.get(), random_state=42
            )
            model_type = self.model_type_var.get()
            alpha = self.alpha_var.get()
            if model_type == "Linear":
                model = LinearRegression()
            elif model_type == "Ridge":
                model = Ridge(alpha=alpha)
            else:
                model = Lasso(alpha=alpha, max_iter=10000)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_all_pred = model.predict(self.X)

            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)

            self.fig_reg.clear()
            ax1 = self.fig_reg.add_subplot(2, 2, 1)
            ax2 = self.fig_reg.add_subplot(2, 2, 2)
            ax3 = self.fig_reg.add_subplot(2, 2, 3)
            ax4 = self.fig_reg.add_subplot(2, 2, 4)

            ax1.scatter(X_train, y_train, color='blue', s=30, alpha=0.6, label='Train')
            ax1.scatter(X_test, y_test, color='orange', s=30, alpha=0.6, label='Test')
            ax1.plot(self.X, y_all_pred, color='red', linewidth=2, label='Modèle')
            ax1.set_title(f'{model_type} (α={alpha:.3f})')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            residuals_train = y_train - y_train_pred
            ax2.scatter(X_train, residuals_train, color='green', s=30, alpha=0.6)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_title('Résidus Train')
            ax2.grid(True, alpha=0.3)

            residuals_all = self.y - y_all_pred
            ax3.hist(residuals_all, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax3.set_title('Distribution résidus')
            ax3.grid(True, alpha=0.3, axis='y')

            ax4.axis('off')
            metrics_text = (
                f"MSE Train: {mse_train:.4f}\n"
                f"MSE Test:  {mse_test:.4f}\n"
                f"R2 Train :  {r2_train:.4f}\n"
                f"R2 Test  :  {r2_test:.4f}\n"
                f"Slope    :  {model.coef_[0]:.4f}\n"
                f"Intercept:  {model.intercept_:.4f}\n"
            )
            ax4.text(0.05, 0.5, metrics_text, transform=ax4.transAxes,
                     fontfamily='monospace', fontsize=10, va='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.fig_reg.tight_layout()
            self.canvas_reg.draw_idle()
        except Exception:
            # keep silent to avoid crashing main UI
            pass

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