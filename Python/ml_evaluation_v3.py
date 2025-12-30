#!/usr/bin/env python3
"""
ML EVALUATION V3 - PROGRESS GUI (fusion de l'adaptateur et du script principal)
Pipeline:
  1) Chargement NPZ (preprocessed_dataset.npz ou tensor_data.npz)
  2) Entraînement + KFold sur les modèles (LogReg, NB, DT, RF)
  3) Rapports texte et graphiques
UI: progress_gui.GenericProgressGUI (aucun flag, mode unique)
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import multiprocessing
import threading
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from progress_gui import GenericProgressGUI
except ImportError:
    print("progress_gui manquant (progress_gui.py)")
    sys.exit(1)

NUM_CORES = multiprocessing.cpu_count()
MODELS_CONFIG = {
    "Logistic Regression": {"n_jobs": -1},
    "Naive Bayes": {"n_jobs": 1},
    "Decision Tree": {"n_jobs": -1},
    "Random Forest": {"n_jobs": -1},
}


class MLEvaluationRunner:
    def __init__(self, ui):
        self.ui = ui
        self.results = {}
        self.X = None
        self.y = None
        self._init_ui()

    def _init_ui(self):
        self.ui.add_stage("load", "Chargement NPZ")
        self.ui.add_stage("train", "Entraînement/KFold")
        self.ui.add_stage("reports", "Rapports")
        self.ui.add_stage("graphs", "Graphiques")

    def load_data(self):
        npz_files = ["preprocessed_dataset.npz", "tensor_data.npz"]
        npz_file = next((f for f in npz_files if os.path.exists(f)), None)
        if not npz_file:
            self.ui.log_alert("NPZ introuvable", level="error")
            return False
        t0 = time.time()
        data = np.load(npz_file, allow_pickle=True)
        self.X = data["X"]
        self.y = data["y"]
        self.ui.log(f"[OK] NPZ {npz_file} chargé ({len(self.y):,} échantillons) en {time.time()-t0:.1f}s", level="OK")
        self.ui.update_stage("load", len(self.y), len(self.y), "NPZ chargé")
        self.ui.update_global(1, 4, "NPZ chargé")
        return True

    def train_eval(self):
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42,
                                                       n_jobs=MODELS_CONFIG["Logistic Regression"]["n_jobs"])),
            ("Naive Bayes", GaussianNB()),
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42,
                                                     n_jobs=MODELS_CONFIG["Random Forest"]["n_jobs"])),
        ]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        total = len(models)
        for i, (name, model) in enumerate(models, 1):
            f1_runs = []
            recall_runs = []
            precision_runs = []
            cm_sum = None
            for fold, (train_idx, test_idx) in enumerate(kf.split(self.X), 1):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1_runs.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                recall_runs.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                precision_runs.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                cm = confusion_matrix(y_test, y_pred, labels=np.unique(self.y))
                cm_sum = cm if cm_sum is None else cm_sum + cm
                if fold % 2 == 1:
                    self.ui.update_file_progress(f"{name}", int(fold / kf.n_splits * 100),
                                                 f"Fold {fold}/{kf.n_splits} F1={f1_runs[-1]:.3f}")
            self.results[name] = {
                "f1": float(np.mean(f1_runs)),
                "f1_std": float(np.std(f1_runs)),
                "recall": float(np.mean(recall_runs)),
                "precision": float(np.mean(precision_runs)),
                "cm": cm_sum.tolist() if cm_sum is not None else [],
            }
            self.ui.update_stage("train", i, total, f"{name} F1={self.results[name]['f1']:.3f}")
            self.ui.update_global(1 + i / total, 4, f"Modèle {i}/{total}")
        return True

    def save_reports(self):
        try:
            with open("evaluation_results_summary.txt", "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("ML EVALUATION V3 - PROGRESS GUI\n")
                f.write("=" * 100 + "\n\n")
                for name, res in sorted(self.results.items(), key=lambda x: x[1]["f1"], reverse=True):
                    f.write(f"{name:<25} F1={res['f1']:.4f}±{res['f1_std']:.4f} Recall={res['recall']:.4f} Precision={res['precision']:.4f}\n")
                f.write("=" * 100 + "\n")
            self.ui.update_stage("reports", 1, 1, "Rapports OK")
            self.ui.update_global(3, 4, "Rapports")
            self.ui.log("[OK] Rapports sauvegardés", level="OK")
        except Exception as e:
            self.ui.log_alert(f"Rapports: {e}", level="error")

    def save_graphs(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("darkgrid")
            for name, res in self.results.items():
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].bar(["F1", "Recall", "Precision"],
                            [res["f1"], res["recall"], res["precision"]],
                            color=["#3498db", "#e74c3c", "#f39c12"])
                axes[0].set_ylim([0, 1])
                axes[0].set_title(name)
                if res.get("cm"):
                    cm = np.array(res["cm"])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
                    axes[1].set_title("Confusion matrix")
                plt.tight_layout()
                fname = f"graph_eval_{name.replace(' ', '_').lower()}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
            self.ui.update_stage("graphs", 1, 1, "Graphiques OK")
            self.ui.update_global(4, 4, "Terminé")
            self.ui.log("[OK] Graphiques générés", level="OK")
        except Exception as e:
            self.ui.log_alert(f"Graphiques: {e}", level="error")


def main():
    ui = GenericProgressGUI(title="ML Evaluation V3", header_info=f"Cores: {NUM_CORES}", max_workers=4)
    runner = MLEvaluationRunner(ui=ui)

    def job():
        if not runner.load_data():
            return
        runner.train_eval()
        runner.save_reports()
        runner.save_graphs()
        ui.log_alert("Evaluation terminée", level="success")

    threading.Thread(target=job, daemon=True).start()
    ui.start()


if __name__ == "__main__":
    main()
