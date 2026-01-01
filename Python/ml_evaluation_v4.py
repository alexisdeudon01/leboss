#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML EVALUATION V3 - AVEC TES MODULES + RAM SAFE + CV TRAIN ONLY
==============================================================
✅ Garde tes modules (ConsolidationStyleShell + AIOptimizationServer + progress_gui si dispo)
✅ RAM-safe: float32 partout + mmap NPZ
✅ KFold sur X_train uniquement (pas de leakage)
✅ 1 éval finale sur X_test (holdout réel)
✅ Rapports + graphs (matplotlib; seaborn optionnel)
==============================================================
"""

import os
import sys
import time
import gc
import json
import threading
import multiprocessing

import numpy as np
import pandas as pd
import psutil

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# ---------- Optional internal modules (keep yours, but no-crash if missing) ----------
try:
    from consolidation_style_shell import ConsolidationStyleShell
except ModuleNotFoundError:
    ConsolidationStyleShell = None

try:
    from ai_optimization_server_with_sessions_v4 import AIOptimizationServer, Metrics as AIMetrics
except ModuleNotFoundError:
    AIOptimizationServer = None
    AIMetrics = None

try:
    from progress_gui import GenericProgressGUI
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    GenericProgressGUI = None

try:
    from pipeline_ui_template import PipelineWindowTemplate
except ModuleNotFoundError:
    PipelineWindowTemplate = None

NUM_CORES = multiprocessing.cpu_count()
DTYPE = np.float32  # ✅ RAM-safe


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


# ============= MEMORY MANAGER =============
class MemoryManager:
    """Gestion mémoire dynamique"""
    RAM_THRESHOLD = 90.0

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
    def check_and_cleanup():
        """Nettoie si RAM trop haute"""
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > MemoryManager.RAM_THRESHOLD:
            gc.collect()
            return False
        return True


# ============= ML EVALUATION RUNNER =============
class MLEvaluationRunner:
    def __init__(self, ui=None, shell=None):
        self.ui = ui
        self.shell = shell
        self.results = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.classes = None

        self.current_workers = max(1, min(NUM_CORES, 4))

        # AI server optional
        self.ai_server = None
        self.ai_thread = None
        if AIOptimizationServer is not None and AIMetrics is not None:
            self.ai_server = AIOptimizationServer(
                max_workers=4,
                max_chunk_size=1_000_000,
                min_chunk_size=50_000,
                max_ram_percent=90.0,
                with_gui=False,
            )
            self.ai_thread = threading.Thread(target=self.ai_server.run, daemon=True, name="AIOptimizationServer")
            self.ai_thread.start()

        # UI stages
        if self.ui:
            self.ui.add_stage("load", "Chargement train/test")
            self.ui.add_stage("train", "CV train + évaluation holdout")
            self.ui.add_stage("reports", "Rapports")
            self.ui.add_stage("graphs", "Graphiques")

        if self.shell:
            # if your ConsolidationStyleShell expects add_stage, keep it; else ignore
            try:
                for key, label in [
                    ("overall", "Overall"),
                    ("load", "Chargement"),
                    ("train", "Train/Eval"),
                    ("reports", "Rapports"),
                    ("graphs", "Graphiques"),
                ]:
                    if hasattr(self.shell, "add_stage"):
                        self.shell.add_stage(key, label)
                if hasattr(self.shell, "set_status"):
                    self.shell.set_status("Idle")
            except Exception:
                pass

    def log(self, msg, level="INFO"):
        """Log compatible GUI + console"""
        if self.ui:
            try:
                self.ui.log(msg, level=level)
            except Exception:
                pass
        if self.shell:
            try:
                self.shell.log(msg, level=level)
            except Exception:
                pass
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

    def log_alert(self, msg, level="ERROR"):
        if self.ui:
            try:
                self.ui.log_alert(msg, level=level.lower())
            except Exception:
                pass
        if self.shell:
            try:
                self.shell.add_alert(msg, level)
            except Exception:
                pass
        else:
            print(f"[ALERT] [{level}] {msg}")

    def _send_metrics(self, rows: int, chunk_size: int = 100_000, throughput: float | None = None):
        if self.ai_server is None or AIMetrics is None:
            return
        try:
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.0)
            tp = throughput if throughput is not None else float(rows)
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
        except Exception:
            pass

    def _ai_try_update_workers(self):
        if self.ai_server is None:
            return
        try:
            rec = self.ai_server.get_recommendation(timeout=0.02)
            if rec:
                new_workers = max(1, min(NUM_CORES, self.current_workers + rec.d_workers))
                if new_workers != self.current_workers:
                    self.current_workers = new_workers
                    self.log(f"[AI] workers->{new_workers} (reason: {rec.reason})", level="info")
        except Exception:
            pass

    def load_data(self):
        """Charge train + test avec RAM safe"""
        npz_files = ["preprocessed_dataset.npz", "tensor_data.npz"]
        npz_file = next((f for f in npz_files if os.path.exists(f)), None)

        if not npz_file:
            self.log_alert("NPZ introuvable (preprocessed_dataset.npz ou tensor_data.npz)", level="ERROR")
            return False

        try:
            if self.shell and hasattr(self.shell, "set_status"):
                self.shell.set_status("Chargement")
            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("load", 0.0)

            t0 = time.time()
            gc.collect()

            # ✅ mmap + float32 = RAM safe
            data = np.load(npz_file, allow_pickle=True, mmap_mode="r")
            self.X_train = np.asarray(data["X"], dtype=DTYPE)
            self.y_train = np.asarray(data["y"])
            self.classes = np.asarray(data["classes"])

            self.log(f"Train NPZ chargé ({len(self.y_train):,} échantillons) en {time.time()-t0:.1f}s", level="OK")
            self.log(f"X_train={self.X_train.shape} dtype={self.X_train.dtype}", level="DETAIL")
            self.log(f"RAM: {MemoryManager.get_ram_usage():.1f}%", level="DETAIL")
            self._send_metrics(rows=len(self.y_train), chunk_size=min(len(self.y_train), 200_000))
            self._ai_try_update_workers()

            # ---- Load test holdout CSV ----
            if not os.path.exists("fusion_test_smart4.csv"):
                self.log_alert("fusion_test_smart4.csv introuvable", level="ERROR")
                return False

            df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False, encoding='utf-8')
            df_test = _normalize_label_column(df_test)
            self.log(f"Test chargé: {len(df_test):,} lignes", level="OK")

            if not MemoryManager.check_and_cleanup():
                self.log("RAM critique, nettoyage", level="WARN")

            numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            X_test_raw = df_test[numeric_cols].astype(DTYPE, copy=False)

            # fill NaNs
            X_test_raw = X_test_raw.fillna(X_test_raw.mean(numeric_only=True))

            if X_test_raw.shape[1] != self.X_train.shape[1]:
                self.log_alert(f"Mismatch features: test={X_test_raw.shape[1]} vs train={self.X_train.shape[1]}", level="ERROR")
                return False

            # normalize test with train stats
            mean = self.X_train.mean(axis=0)
            std = self.X_train.std(axis=0) + DTYPE(1e-8)
            self.X_test = ((X_test_raw - mean) / std).astype(DTYPE, copy=False)

            # labels encoding using train classes
            lbl = LabelEncoder()
            lbl.classes_ = self.classes
            y_test_raw = df_test["Label"].astype(str).values
            self.y_test = lbl.transform(y_test_raw)

            self.log(f"Test normalisé: X={self.X_test.shape} dtype={self.X_test.dtype} | y={len(self.y_test):,}", level="OK")
            self._send_metrics(rows=len(self.y_test), chunk_size=min(len(self.y_test), 200_000))
            self._ai_try_update_workers()

            if self.ui:
                self.ui.update_stage("load", 1, 1, "Train/Test chargés")
                self.ui.update_global(1, 4, f"Train {len(self.X_train):,} | Test {len(self.X_test):,}")

            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("load", 100.0)
            if self.shell and hasattr(self.shell, "set_overall_progress"):
                self.shell.set_overall_progress(25.0)
            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("overall", 25.0)

            del df_test, X_test_raw
            gc.collect()

            return True

        except Exception as e:
            self.log_alert(f"Erreur load_data: {e}", level="ERROR")
            return False

    def train_eval(self):
        """
        ✅ KFold sur X_train uniquement (pas de leakage)
        ✅ Puis fit full train + eval holdout X_test
        """
        try:
            if self.shell and hasattr(self.shell, "set_status"):
                self.shell.set_status("Train/Eval")
            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("train", 0.0)

            models = [
                ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, n_jobs=min(self.current_workers, NUM_CORES))),
                ("Naive Bayes", GaussianNB()),
                ("Decision Tree", DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=10)),
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=min(self.current_workers, NUM_CORES))),
            ]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            total = len(models)

            for i, (name, model) in enumerate(models, 1):
                self.log(f"\n[MODEL {i}/{total}] {name}", level="OK")

                if self.shell and hasattr(self.shell, "set_stage_progress"):
                    progress = i / max(total, 1) * 100.0
                    self.shell.set_stage_progress("train", progress)
                if self.shell and hasattr(self.shell, "set_overall_progress"):
                    overall = 25.0 + 50.0 * (i / max(total, 1))
                    self.shell.set_overall_progress(overall)
                if self.shell and hasattr(self.shell, "set_stage_progress"):
                    self.shell.set_stage_progress("overall", overall)

                f1_runs, recall_runs, precision_runs = [], [], []
                cm_sum_cv = None

                # ---- CV on TRAIN only ----
                for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
                    if not MemoryManager.check_and_cleanup():
                        self.log("RAM critique, pause", level="WARN")
                        time.sleep(0.5)

                    X_tr = self.X_train[train_idx]
                    y_tr = self.y_train[train_idx]
                    X_val = self.X_train[val_idx]
                    y_val = self.y_train[val_idx]

                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)

                    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
                    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)

                    f1_runs.append(float(f1))
                    recall_runs.append(float(rec))
                    precision_runs.append(float(prec))

                    labels_used = np.unique(self.y_train)
                    cm = confusion_matrix(y_val, y_pred, labels=labels_used)
                    cm_sum_cv = cm if cm_sum_cv is None else (cm_sum_cv + cm)

                    self.log(f"  CV Fold {fold}/5: F1={f1:.4f} | Recall={rec:.4f} | Precision={prec:.4f}", level="info")

                    if self.ui:
                        self.ui.update_file_progress(f"{name}", int(fold / 5 * 100), f"CV Fold {fold}/5 F1={f1:.4f}")

                    self._send_metrics(rows=len(y_tr) + len(y_val), chunk_size=min(len(y_tr), 200_000))
                    self._ai_try_update_workers()

                    del X_tr, y_tr, X_val, y_val, y_pred
                    gc.collect()

                cv_f1 = float(np.mean(f1_runs)) if f1_runs else 0.0
                cv_f1_std = float(np.std(f1_runs)) if f1_runs else 0.0
                cv_recall = float(np.mean(recall_runs)) if recall_runs else 0.0
                cv_precision = float(np.mean(precision_runs)) if precision_runs else 0.0

                # ---- Final fit on FULL TRAIN + HOLDOUT TEST ----
                model.fit(self.X_train, self.y_train)
                y_test_pred = model.predict(self.X_test)

                test_f1 = float(f1_score(self.y_test, y_test_pred, average="weighted", zero_division=0))
                test_recall = float(recall_score(self.y_test, y_test_pred, average="weighted", zero_division=0))
                test_precision = float(precision_score(self.y_test, y_test_pred, average="weighted", zero_division=0))
                cm_test = confusion_matrix(self.y_test, y_test_pred, labels=np.unique(self.y_test))

                self.results[name] = {
                    "cv_f1": cv_f1,
                    "cv_f1_std": cv_f1_std,
                    "cv_recall": cv_recall,
                    "cv_precision": cv_precision,
                    "cv_cm": cm_sum_cv.tolist() if cm_sum_cv is not None else [],

                    "test_f1": test_f1,
                    "test_recall": test_recall,
                    "test_precision": test_precision,
                    "test_cm": cm_test.tolist(),
                }

                self.log(f"✅ {name} CV: F1={cv_f1:.4f}±{cv_f1_std:.4f}", level="OK")
                self.log(f"🎯 {name} HOLDOUT: F1={test_f1:.4f} | Recall={test_recall:.4f} | Precision={test_precision:.4f}", level="OK")

                if self.ui:
                    self.ui.update_stage("train", i, total, f"{name} testF1={test_f1:.4f}")
                    self.ui.update_global(1 + i / total, 4, f"Modèle {i}/{total}")

                del y_test_pred
                gc.collect()

            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("train", 100.0)
            if self.shell and hasattr(self.shell, "set_overall_progress"):
                self.shell.set_overall_progress(75.0)
            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("overall", 75.0)

            return True

        except Exception as e:
            self.log_alert(f"Erreur train_eval: {e}", level="ERROR")
            return False

    def save_reports(self):
        """Sauvegarder rapports (CV + TEST)"""
        try:
            with open("evaluation_results_summary.txt", "w", encoding="utf-8") as f:
                f.write("=" * 110 + "\n")
                f.write("ML EVALUATION V3 - CV(train) + HOLDOUT(test) - RAM SAFE\n")
                f.write("=" * 110 + "\n\n")
                f.write("✅ CV: KFold sur X_train uniquement\n")
                f.write("✅ HOLDOUT: 1 évaluation finale sur X_test\n")
                f.write(f"✅ dtype: {DTYPE}\n\n")

                for name, res in sorted(self.results.items(), key=lambda x: x[1]["test_f1"], reverse=True):
                    f.write(
                        f"{name:<25} "
                        f"CV_F1={res['cv_f1']:.4f}±{res['cv_f1_std']:.4f}  "
                        f"TEST_F1={res['test_f1']:.4f}  "
                        f"TEST_Recall={res['test_recall']:.4f}  "
                        f"TEST_Precision={res['test_precision']:.4f}\n"
                    )

                f.write("\n" + "=" * 110 + "\n")

            self.log("Rapport texte: evaluation_results_summary.txt", level="OK")

            with open("ml_evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

            self.log("Rapport JSON: ml_evaluation_results.json", level="OK")

            if self.ui:
                self.ui.update_stage("reports", 1, 1, "Rapports OK")
                self.ui.update_global(3, 4, "Rapports")
            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("reports", 100.0)
            if self.shell and hasattr(self.shell, "set_overall_progress"):
                self.shell.set_overall_progress(90.0)
            if self.shell and hasattr(self.shell, "set_stage_progress"):
                self.shell.set_stage_progress("overall", 90.0)

            return True

        except Exception as e:
            self.log_alert(f"Rapports: {e}", level="ERROR")
            return False

    def save_graphs(self):
        """Générer graphiques (matplotlib; seaborn optionnel)"""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            self.log("matplotlib absent -> pas de graph", level="WARN")
            return True

        try:
            import seaborn as sns
            HAS_SNS = True
        except Exception:
            HAS_SNS = False
            sns = None

        for name, res in self.results.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Graph 1: metrics (test)
            metrics = ["TEST_F1", "TEST_Recall", "TEST_Precision"]
            values = [res["test_f1"], res["test_recall"], res["test_precision"]
                      ]
            axes[0].bar(metrics, values)
            axes[0].set_ylim([0, 1])
            axes[0].set_title(f"{name} - Métriques (TEST)")
            axes[0].set_ylabel("Score")

            # Graph 2: confusion matrix (test)
            cm = np.array(res.get("test_cm", []))
            if cm.size:
                if HAS_SNS:
                    sns.heatmap(cm, annot=False, fmt="d", ax=axes[1])
                else:
                    axes[1].imshow(cm, interpolation="nearest")
                axes[1].set_title(f"{name} - Confusion Matrix (TEST)")
                axes[1].set_xlabel("Prédiction")
                axes[1].set_ylabel("Réalité")
            else:
                axes[1].axis("off")

            plt.tight_layout()
            fname = f"graph_eval_{name.replace(' ', '_').lower()}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            self.log(f"Graphique: {fname}", level="OK")
            gc.collect()

        if self.ui:
            self.ui.update_stage("graphs", 1, 1, "Graphiques OK")
            self.ui.update_global(4, 4, "Terminé")
        if self.shell and hasattr(self.shell, "set_stage_progress"):
            self.shell.set_stage_progress("graphs", 100.0)
        if self.shell and hasattr(self.shell, "set_overall_progress"):
            self.shell.set_overall_progress(100.0)
        if self.shell and hasattr(self.shell, "set_stage_progress"):
            self.shell.set_stage_progress("overall", 100.0)
        if self.shell and hasattr(self.shell, "set_status"):
            self.shell.set_status("Completed")

        return True


def main():
    print("\n" + "="*80)
    print("ML EVALUATION V3 - AVEC MODULES + RAM SAFE")
    print("="*80 + "\n")

    # ---- GUI mode if available ----
    if HAS_GUI:
        print("[INFO] Mode GUI activé\n")
        ui = GenericProgressGUI(
            title="ML Evaluation V3 - RAM SAFE",
            header_info=f"Cores: {NUM_CORES}, RAM free: {MemoryManager.get_available_ram_gb():.1f}GB",
            max_workers=4
        )

        shell = None
        if ConsolidationStyleShell is not None:
            try:
                shell = ConsolidationStyleShell(
                    title="ML Evaluation V3",
                    stages=[
                        ("overall", "Overall"),
                        ("load", "Load"),
                        ("train", "Train/Eval"),
                        ("reports", "Reports"),
                        ("graphs", "Graphs"),
                    ],
                    thread_slots=4,
                )
            except Exception:
                shell = None

        runner = MLEvaluationRunner(ui=ui, shell=shell)

        def job():
            try:
                if shell and hasattr(shell, "set_status"):
                    shell.set_status("Running")
                if shell and hasattr(shell, "set_overall_progress"):
                    shell.set_overall_progress(0.0)

                ui.update_global(0, 4, "Initialisation")

                if not runner.load_data():
                    ui.log_alert("Échec chargement données", level="error")
                    if shell and hasattr(shell, "add_alert"):
                        shell.add_alert("Échec chargement données", level="ERROR")
                    return

                if not runner.train_eval():
                    ui.log_alert("Échec entraînement", level="error")
                    if shell and hasattr(shell, "add_alert"):
                        shell.add_alert("Échec entraînement", level="ERROR")
                    return

                if not runner.save_reports():
                    ui.log_alert("Échec rapports", level="error")
                    if shell and hasattr(shell, "add_alert"):
                        shell.add_alert("Échec rapports", level="ERROR")
                    return

                if not runner.save_graphs():
                    ui.log_alert("Échec graphiques", level="error")
                    if shell and hasattr(shell, "add_alert"):
                        shell.add_alert("Échec graphiques", level="ERROR")
                    return

                ui.log_alert("Evaluation terminée avec succès!", level="success")
                if shell and hasattr(shell, "add_alert"):
                    shell.add_alert("Evaluation terminée avec succès!", level="OK")
                if shell and hasattr(shell, "set_status"):
                    shell.set_status("Completed")

            except Exception as e:
                ui.log_alert(f"Erreur: {e}", level="error")
                if shell and hasattr(shell, "add_alert"):
                    shell.add_alert(f"Erreur: {e}", level="ERROR")

        # If shell supports binding, use it, else auto-run
        if shell and hasattr(shell, "bind_start"):
            shell.bind_start(lambda: threading.Thread(target=job, daemon=True).start())
            if hasattr(shell, "bind_stop"):
                shell.bind_stop(lambda: shell.add_alert("Arrêt non implémenté", level="WARN"))
            threading.Thread(target=job, daemon=True).start()
        else:
            threading.Thread(target=job, daemon=True).start()

        ui.start()
        return True

    # ---- Console mode ----
    print("[INFO] Mode console\n")

    shell = None
    if ConsolidationStyleShell is not None:
        try:
            shell = ConsolidationStyleShell(
                title="ML Evaluation V3",
                stages=[
                    ("overall", "Overall"),
                    ("load", "Load"),
                    ("train", "Train/Eval"),
                    ("reports", "Reports"),
                    ("graphs", "Graphs"),
                ],
                thread_slots=4,
            )
        except Exception:
            shell = None

    runner = MLEvaluationRunner(ui=None, shell=shell)

    def console_job():
        if shell and hasattr(shell, "set_status"):
            shell.set_status("Running")

        if not runner.load_data():
            print("[ERROR] Chargement données échoué")
            if shell and hasattr(shell, "add_alert"):
                shell.add_alert("Chargement données échoué", level="ERROR")
            return

        if not runner.train_eval():
            print("[ERROR] Entraînement échoué")
            if shell and hasattr(shell, "add_alert"):
                shell.add_alert("Entraînement échoué", level="ERROR")
            return

        if not runner.save_reports():
            print("[ERROR] Rapports échoué")
            if shell and hasattr(shell, "add_alert"):
                shell.add_alert("Rapports échoué", level="ERROR")
            return

        if not runner.save_graphs():
            print("[ERROR] Graphiques échoué")
            if shell and hasattr(shell, "add_alert"):
                shell.add_alert("Graphiques échoué", level="ERROR")
            return

        print("\n✅ Evaluation complétée avec succès!")
        if shell and hasattr(shell, "add_alert"):
            shell.add_alert("Evaluation complétée avec succès!", level="OK")
        if shell and hasattr(shell, "set_status"):
            shell.set_status("Completed")

    if shell and hasattr(shell, "bind_start"):
        shell.bind_start(lambda: threading.Thread(target=console_job, daemon=True).start())
        threading.Thread(target=console_job, daemon=True).start()
        return True

    console_job()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
