#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV OPTIMIZATION V3 - COMPLETE (FIX + 2nd TAB REGRESSION LIVE)
============================================================
✅ Grid Search: Hyperparamètres variables
✅ Graphiques: onglets (F1 live + Régression live)
✅ Gestion RAM dynamique (<90%)
✅ Tkinter GUI avancée (ConsolidationStyleShell)
✅ Visualisation complète résultats
✅ FIX: imports manquants regression
✅ FIX: _send_ai_metric bien dans CVOptimizationGUI
✅ FIX: dtype float32 réel
✅ FIX: évite oversubscription (n_jobs=1 dans modèles)
============================================================
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
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from consolidation_style_shell import ConsolidationStyleShell
from ai_optimization_server_with_sessions_v4 import AIOptimizationServer, Metrics as AIMetrics

# ✅ vrai float32 si objectif RAM
NPZ_FLOAT_DTYPE = np.float32

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

    # ✅ missing imports for regression tab
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, r2_score

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
        # solver could be added if needed
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
        # tip: add max_samples/bootstrap if you want speed on huge data
    }
}


class MemoryManager:
    """Gère la mémoire dynamiquement"""

    @staticmethod
    def get_ram_usage():
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 50.0

    @staticmethod
    def get_available_ram_gb():
        try:
            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 8.0

    @staticmethod
    def get_optimal_chunk_size(total_size=None, min_chunk=MIN_CHUNK_SIZE, max_chunk=1_000_000):
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
            ttk.Radiobutton(
                left, text=model, variable=self.model_type_var,
                value=model, command=self._update_plot
            ).pack(anchor="w")

        ttk.Label(left, text="Alpha", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Scale(
            left, from_=0.001, to=10.0, orient=tk.HORIZONTAL,
            variable=self.alpha_var, command=lambda _e: self._update_plot()
        ).pack(fill="x")
        self.alpha_lab = ttk.Label(left, text="")
        self.alpha_lab.pack(anchor="w")

        ttk.Label(left, text="Test size", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Scale(
            left, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
            variable=self.test_size_var, command=lambda _e: self._update_plot()
        ).pack(fill="x")
        self.test_lab = ttk.Label(left, text="")
        self.test_lab.pack(anchor="w")

        ttk.Label(left, text="Bruit σ", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Scale(
            left, from_=0.0, to=10.0, orient=tk.HORIZONTAL,
            variable=self.noise_var, command=lambda _e: self._update_plot()
        ).pack(fill="x")
        self.noise_lab = ttk.Label(left, text="")
        self.noise_lab.pack(anchor="w")

        # ✅ Diagnostic UI
        ttk.Label(left, text="Diagnostic", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.diag_lab = tk.Label(left, text="--", justify="left", wraplength=220, anchor="w")
        self.diag_lab.pack(anchor="w", fill="x")
        self.diag_bar = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.diag_bar.pack(fill="x", pady=(4, 0))

        ttk.Button(left, text="Régénérer données", command=self._regen).pack(fill="x", pady=6)

        self.fig_reg = Figure(figsize=(8, 6), dpi=100)
        self.canvas_reg = FigureCanvasTkAgg(self.fig_reg, master=right)
        self.canvas_reg.get_tk_widget().pack(fill="both", expand=True)

    def _regen(self):
        self.seed += 1
        np.random.seed(self.seed)
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
            alpha = float(self.alpha_var.get())

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

            # ✅ Diagnostic logic
            gap = r2_train - r2_test
            risk = 0
            if gap > 0:
                risk += min(60, int(gap * 200))
            risk += min(40, int(max(0.0, 0.8 - r2_test) * 50))
            risk = max(0, min(100, risk))

            if r2_test < 0.2 and r2_train < 0.2:
                verdict = "💤 Underfitting: modèle trop simple / bruit trop fort."
                color = "#666666"
            elif gap > 0.15 and r2_train > 0.8:
                verdict = f"🔥 Overfitting: train >> test (gap R²={gap:.2f})."
                color = "#e74c3c"
            elif r2_test > 0.8 and abs(gap) <= 0.10:
                verdict = f"✅ Bon fit: généralise bien (R² test={r2_test:.2f})."
                color = "#27ae60"
            else:
                verdict = f"👌 OK: R² train={r2_train:.2f}, test={r2_test:.2f}, gap={gap:.2f}."
                color = "#f39c12"

            try:
                self.diag_lab.config(text=verdict, fg=color)
                self.diag_bar["value"] = risk
            except Exception:
                pass

            # ---- plots ----
            self.fig_reg.clear()
            ax1 = self.fig_reg.add_subplot(2, 2, 1)
            ax2 = self.fig_reg.add_subplot(2, 2, 2)
            ax3 = self.fig_reg.add_subplot(2, 2, 3)
            ax4 = self.fig_reg.add_subplot(2, 2, 4)

            ax1.scatter(X_train, y_train, s=30, alpha=0.6, label="Train")
            ax1.scatter(X_test, y_test, s=30, alpha=0.6, label="Test")
            ax1.plot(self.X, y_all_pred, linewidth=2, label="Modèle")
            ax1.set_title(f"{model_type} (α={alpha:.3f})")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            residuals_train = y_train - y_train_pred
            ax2.scatter(X_train, residuals_train, s=30, alpha=0.6)
            ax2.axhline(y=0, linestyle="--", linewidth=2)
            ax2.set_title("Résidus Train")
            ax2.grid(True, alpha=0.3)

            residuals_all = self.y - y_all_pred
            ax3.hist(residuals_all, bins=20, alpha=0.7, edgecolor="black")
            ax3.axvline(x=0, linestyle="--", linewidth=2)
            ax3.set_title("Distribution résidus")
            ax3.grid(True, alpha=0.3, axis="y")

            ax4.axis("off")
            slope = float(model.coef_[0]) if hasattr(model, "coef_") else float("nan")
            intercept = float(model.intercept_) if hasattr(model, "intercept_") else float("nan")
            metrics_text = (
                f"MSE Train: {mse_train:.4f}\n"
                f"MSE Test:  {mse_test:.4f}\n"
                f"R2 Train :  {r2_train:.4f}\n"
                f"R2 Test  :  {r2_test:.4f}\n"
                f"Slope    :  {slope:.4f}\n"
                f"Intercept:  {intercept:.4f}\n"
            )
            ax4.text(
                0.05, 0.5, metrics_text, transform=ax4.transAxes,
                fontfamily="monospace", fontsize=10, va="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            )

            self.fig_reg.tight_layout()
            self.canvas_reg.draw_idle()

        except Exception:
            pass


class CVOptimizationGUI:
    """Interface Tkinter avancée avec graphiques (tabs)"""

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
        self.worker_thread = None
        self._last_ai_ts = time.time()

        # Core data containers
        self.df = None
        self.X_scaled = None
        self.y = None
        self.label_encoder = None

        # Bind shell controls
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
        self.graphs_btn = self.ui_shell.reset_btn  # placeholder

        # Graph window with tabs (F1 Live + Regression Live)
        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Graphs")
        try:
            self.graph_window.geometry("900x700")
        except Exception:
            pass

        notebook = ttk.Notebook(self.graph_window)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Live F1
        live_tab = ttk.Frame(notebook)
        notebook.add(live_tab, text="F1 Live")

        self.fig_live = Figure(figsize=(8, 4.5), dpi=100)
        self.ax_live = self.fig_live.add_subplot(111)
        self.ax_live.set_title("F1 en temps réel")
        self.ax_live.set_xlabel("Combinaison #")
        self.ax_live.set_ylabel("F1 pondéré")

        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=live_tab)
        self.canvas_live.get_tk_widget().pack(fill="both", expand=True)

        self.live_curves = {}
        self.live_params_label = ttk.Label(live_tab, text="Params: --", anchor="w", justify="left")
        self.live_params_label.pack(fill="x", padx=6, pady=6)

        # Tab 2: Regression live
        reg_tab = ttk.Frame(notebook)
        notebook.add(reg_tab, text="Régression Live")
        self.regression_viz = RegressionVisualizerLite(reg_tab)

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

        self.current_chunk_size = MemoryManager.get_optimal_chunk_size(min_chunk=MIN_CHUNK_SIZE)
        self.last_ckpt_ts = time.time()

    # -------------------- UI safe setters --------------------
    def _ui_stage(self, key: str, val: float):
        self.root.after(0, lambda: self.ui_shell.set_stage_progress(key, val))

    def _ui_overall(self, val: float):
        self.root.after(0, lambda: self.ui_shell.set_overall_progress(val))

    def _ui_eta(self, key: str, text: str):
        self.root.after(0, lambda: self.ui_shell.set_stage_eta(key, text))

    def _ui_model(self, key: str, val: float):
        self.root.after(0, lambda: self.ui_shell.set_model_progress(key, val, key))

    def _ui_thread(self, tid: int, val: float, msg: str, done=None, total=None):
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

    # ✅ FIX: AI metric sender is in CVOptimizationGUI (not RegressionVisualizerLite)
    def _send_ai_metric(self, rows: int, chunk_size: int):
        try:
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.0)
            now = time.time()
            dt = max(now - self._last_ai_ts, 1e-6)
            tp = (rows / dt) if rows else 0.0
            self._last_ai_ts = now

            self.ai_server.send_metrics(
                AIMetrics(
                    timestamp=now,
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
                self._ui_ai_rec(rec.reason)
        except Exception:
            pass

    def _set_live_params(self, model: str, params: dict):
        try:
            pretty = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "--"
            if len(pretty) > 220:
                pretty = pretty[:217] + "..."
            msg = f"{model}: {pretty}"
            self.root.after(0, lambda: self.live_params_label.config(text=msg))
        except Exception:
            pass

        try:
            self._send_ai_metric(rows=0, chunk_size=int(self.current_chunk_size))
        except Exception:
            pass

    def _update_live_graph(self, model: str, f1_values: list[float]):
        try:
            self.live_curves[model] = list(f1_values)
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
        if force or (time.time() - getattr(self, "last_ckpt_ts", 0) >= 600):
            try:
                self._write_checkpoint(force=False)
            except Exception:
                pass

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

            if self.start_time and self.completed_operations > 0:
                elapsed = time.time() - self.start_time
                avg = elapsed / self.completed_operations
                remaining = (self.total_operations - self.completed_operations) * avg
                eta = datetime.now() + timedelta(seconds=remaining)
                self.ui_shell.set_stage_eta("overall", eta.strftime('%H:%M:%S'))

            percent = (self.completed_operations / self.total_operations * 100) if self.total_operations > 0 else 0
            self.ui_shell.set_overall_progress(percent)
            self.ui_shell.set_stage_progress("overall", percent)

            rowsps = self.rows_seen / max(time.time() - self.start_time, 1e-3) if self.start_time else 0.0
            self.ui_shell.set_metrics(ram, cpu, self.rows_seen, self.files_done, rowsps)

            # heartbeat to AI server
            try:
                self.ai_server.send_metrics(
                    AIMetrics(
                        timestamp=time.time(),
                        num_workers=int(self.current_workers),
                        chunk_size=int(self.current_chunk_size),
                        rows_processed=0,
                        ram_percent=float(ram),
                        cpu_percent=float(cpu),
                        throughput=float(rowsps),
                    )
                )
            except Exception:
                pass

            self.root.after(500, self.update_stats)
        except Exception:
            self.root.after(500, self.update_stats)
        finally:
            try:
                self._maybe_checkpoint()
            except Exception:
                pass

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

        for name in ("Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest"):
            try:
                self._ui_model(name, 0.0)
            except Exception:
                pass

        try:
            self.live_text.delete(1.0, tk.END)
            self.alerts_text.delete(1.0, tk.END)
        except Exception:
            pass

        self.log_live('CV OPTIMIZATION V3 - GRID SEARCH\n', 'info')
        self.log_live('Hyperparamètres variables par algo\n\n', 'info')

        self.start_time = time.time()

        # show graphs window
        try:
            self.show_graphs()
        except Exception:
            pass

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

                if avg_bytes_per_row is None:
                    try:
                        avg_bytes_per_row = max(float(chunk.memory_usage(deep=True).sum()) / max(len(chunk), 1), 1.0)
                        est_total_rows = max(int(file_size / avg_bytes_per_row), len(chunk))
                    except Exception:
                        est_total_rows = None

                if est_total_rows:
                    pct = min(99.0, 100.0 * total_rows / max(est_total_rows, 1))
                    self._ui_stage("load", pct)
                    self._ui_overall(pct * 0.2)
                    self._ui_thread(0, pct, f"Load chunk {len(chunks)} ({total_rows:,}/{est_total_rows:,})", total_rows, est_total_rows)
                    self._update_stage_eta("load", total_rows, est_total_rows)

                now = time.time()
                dt = max(now - t_last, 1e-3)
                tp = len(chunk) / dt
                send_interval = max(0.5, min(5.0, chunk_size / 300_000.0))

                if now >= next_metrics_time:
                    try:
                        self.ai_server.send_metrics(
                            AIMetrics(
                                timestamp=now,
                                num_workers=int(self.current_workers),
                                chunk_size=int(chunk_size),
                                rows_processed=len(chunk),
                                ram_percent=float(psutil.virtual_memory().percent),
                                cpu_percent=float(psutil.cpu_percent(interval=0.0)),
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
        """ETAPE 2: Preparation (clean inf/nan + scale)"""
        try:
            self.log_live('ETAPE 2: Preparation\n', 'info')

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')

            self.df = self.df.dropna(subset=['Label'])
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

            self.log_live('Cleaning infinity and extreme values...\n', 'info')
            for col in numeric_cols:
                try:
                    self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                    self.df[col] = self.df[col].clip(-1e6, 1e6)
                except Exception as e:
                    self.log_live(f'WARN: Error cleaning column {col}: {e}\n', 'info')

            X = self.df[numeric_cols].copy()
            try:
                X = X.apply(pd.to_numeric, errors="coerce")
            except Exception as e:
                self.log_live(f'WARN: to_numeric failed {e}\n', 'info')

            self.log_live('Filling NaN values with column mean...\n', 'info')
            for col in X.columns:
                col_mean = X[col].mean()
                if pd.isna(col_mean):
                    X[col] = X[col].fillna(0.0)
                else:
                    X[col] = X[col].fillna(col_mean)

            self.log_live('Converting data types...\n', 'info')
            try:
                X = X.astype(np.float64)
                X = X.astype(NPZ_FLOAT_DTYPE)  # ✅ float32
            except Exception as e:
                self.log_live(f'ERROR converting types: {e}\n', 'info')
                self.add_alert(f'Type conversion failed: {e}', "ERROR")
                return False

            n_inf = np.isinf(X.values).sum()
            n_nan = np.isnan(X.values).sum()

            if n_inf > 0 or n_nan > 0:
                self.log_live(f'WARN: Found {n_inf} inf and {n_nan} NaN after cleaning!\n', 'info')
                X = X.fillna(0.0)
                X = X.replace([np.inf, -np.inf], 1e6)
                self.log_live(f'Force-filled NaN/inf\n', 'info')

            self.log_live(f'Features validated (no inf/nan) ✓\n', 'info')

            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.df['Label'])

            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X).astype(NPZ_FLOAT_DTYPE)

            self.log_live(f'Data standardized: X shape {self.X_scaled.shape}\n', 'info')

            np.savez_compressed(
                'preprocessed_dataset.npz',
                X=self.X_scaled,
                y=self.y,
                classes=self.label_encoder.classes_
            )

            self.log_live(f'NPZ saved successfully\n\n', 'info')
            self.rows_seen += len(self.X_scaled)
            self._ui_stage("prep", 100.0)
            self._update_stage_eta("prep", 1, 1)
            self._send_ai_metric(rows=0, chunk_size=int(self.current_chunk_size))
            self._maybe_checkpoint()

            del self.df, X
            gc.collect()
            return True

        except Exception as e:
            self.log_live(f'Erreur: {e}\n{traceback.format_exc()}\n', 'info')
            self.add_alert(f'Prep failed: {e}', "ERROR")
            try:
                self.root.bell()
            except Exception:
                pass
            return False

    def generate_param_combinations(self, model_name):
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

            self.log_live('Transit: Load → Prep\n', 'info')
            self._ui_stage("transit_load", 50.0)
            self._ui_overall(10.0)
            self._ui_stage("transit_load", 100.0)
            self._update_stage_eta("transit_load", 1, 1)
            self._maybe_checkpoint()
            self._ui_stage("transit_load", 0.0)

            if not self.prepare_data():
                return
            if not self.running:
                return

            self.log_live('ETAPE 2.5: Transit Prep → Grid\n', 'info')
            self._ui_stage("transit_mid", 50.0)
            self._ui_overall(50.0)
            self._ui_stage("transit_mid", 100.0)
            self._update_stage_eta("transit_mid", 1, 1)
            self._maybe_checkpoint()
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

                best_score = 0.0
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

                            # ✅ avoid oversubscription: model uses 1 thread, joblib handles folds
                            if name in ('Logistic Regression', 'Random Forest'):
                                kwargs['n_jobs'] = 1

                            if name != 'Naive Bayes':
                                kwargs['random_state'] = 42

                            model = ModelClass(**kwargs)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            rows_proc = len(X_train) + len(X_test)
                            return f1, rows_proc
                        except Exception:
                            return 0.0, 0

                    params_str_verbose = ", ".join(f"{k}={v}" for k, v in params.items())
                    self.log_live(f"[{name}] combo {combo_idx}/{len(combinations)} | params: {params_str_verbose}", "info")

                    results = Parallel(n_jobs=int(self.current_workers), backend="threading")(
                        delayed(run_fold)(fold) for fold in range(K_FOLD)
                    )

                    f1_runs = []
                    rows_total = 0
                    for f1_val, rows_proc in results:
                        f1_runs.append(float(f1_val))
                        rows_total += int(rows_proc)
                        self.rows_seen += int(rows_proc)

                    # update progress
                    self.completed_operations += K_FOLD
                    progress_grid = (self.completed_operations / self.total_operations * 100) if self.total_operations > 0 else 0.0
                    self._ui_stage("grid", progress_grid)
                    self._ui_overall(progress_grid)
                    self._ui_stage("overall", progress_grid)

                    self._send_ai_metric(rows_total, chunk_size=self.current_chunk_size)

                    tid = (combo_idx - 1) % self.thread_slots
                    pct = min(100.0, (combo_idx / max(len(combinations), 1)) * 100.0)
                    self._ui_thread(tid, pct, f"{name} combo {combo_idx}/{len(combinations)}", combo_idx, len(combinations))
                    self._update_stage_eta("grid", combo_idx, len(combinations))
                    self._ui_model(name, pct)

                    mean_f1 = float(np.mean(f1_runs)) if f1_runs else 0.0
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    self.log_live(f'    [{combo_idx}/{len(combinations)}] {params_str}: F1={mean_f1:.4f}\n', 'info')

                    self._set_live_params(name, params)
                    all_results.append({'params': params, 'f1': mean_f1})

                    try:
                        self._update_live_graph(name, [r['f1'] for r in all_results])
                    except Exception:
                        pass

                    if mean_f1 > best_score:
                        best_score = mean_f1
                        best_params = params

                    self._ui_best_score(f"{best_score:.4f}")
                    self.add_alert(f'{name}: {combo_idx}/{len(combinations)} - F1={mean_f1:.4f}')

                    # checkpoint after each combo
                    self.ckpt["model_idx"] = i
                    self.ckpt["combo_idx"] = combo_idx
                    self.ckpt["best_score"] = best_score
                    self.ckpt["current_workers"] = self.current_workers
                    self.ckpt["current_chunk_size"] = self.current_chunk_size
                    self.ckpt["completed_ops"] = self.completed_operations
                    self._maybe_checkpoint()

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

                self.ckpt["model_idx"] = i
                self.ckpt["combo_idx"] = 0
                self.ckpt["best_score"] = best_score
                self.ckpt["current_workers"] = self.current_workers
                self.ckpt["current_chunk_size"] = self.current_chunk_size
                self.ckpt["completed_ops"] = self.completed_operations
                self._write_checkpoint(force=True)

                self.log_live(f'  BEST: F1={best_score:.4f}\n', 'info')
                self._ui_best_score(f"{best_score:.4f}")

            self.log_live('\nETAPE 4: Rapports\n', 'info')
            self.generate_reports()
            self.log_live('\n' + '='*60 + '\n', 'info')
            self.log_live('GRID SEARCH TERMINEE\n', 'info')

            self._ui_stage("transit_grid", 50.0)
            self._ui_overall(90.0)
            self._ui_stage("transit_grid", 100.0)
            self._update_stage_eta("transit_grid", 1, 1)
            self._maybe_checkpoint()
            self._ui_stage("transit_grid", 0.0)

            self.root.after(0, lambda: self.ui_shell.set_status("Completed"))
            self.add_alert('GRID SEARCH COMPLETE')
            self._ui_stage("grid", 100.0)
            self._ui_overall(100.0)

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
        try:
            if self.graph_window and self.graph_window.winfo_exists():
                self.graph_window.deiconify()
                self.graph_window.lift()
                return
        except Exception:
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
