#!/usr/bin/env python3
"""
Cross-Validation Optimization (progress_gui)
Remplace l'ancienne UI Tk : pipeline headless + GUI générique.
Étapes:
  1) Chargement CSV fusion
  2) Préparation (float32, stratified sample 50%)
  3) CV multi modèles (LogReg, NB, DT<=80%, RF) KFold=5 sur plusieurs tailles
  4) Rapports (txt + csv) et graphiques
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score

try:
    from progress_gui import GenericProgressGUI
except ImportError:
    print("progress_gui manquant (progress_gui.py)")
    sys.exit(1)

NUM_CORES = multiprocessing.cpu_count()
TRAIN_SIZES = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
K_FOLD = 5
MODELS_CONFIG = {
    "Logistic Regression": {"n_jobs": -1},
    "Naive Bayes": {"n_jobs": 1},
    "Decision Tree": {"n_jobs": -1},
    "Random Forest": {"n_jobs": -1},
}


class CrossValRunner:
    def __init__(self, ui):
        self.ui = ui
        self.results = {}
        self.optimal_configs = {}
        self.df = None
        self.X_scaled = None
        self.y = None
        self.label_encoder = None
        self._init_ui()

    def _init_ui(self):
        self.ui.add_stage("load", "Chargement")
        self.ui.add_stage("prep", "Préparation")
        self.ui.add_stage("cv", "Cross-Validation")
        self.ui.add_stage("reports", "Rapports")
        self.ui.add_stage("graphs", "Graphiques")

    def _log(self, msg, level="INFO"):
        self.ui.log(msg, level=level)

    def load_data(self):
        fichiers = [
            "fusion_ton_iot_cic_final_smart.csv",
            "fusion_ton_iot_cic_final_smart4.csv",
            "fusion_ton_iot_cic_final_smart3.csv",
        ]
        fichier_trouve = None
        for f in fichiers:
            if os.path.exists(f):
                fichier_trouve = f
                break
        if not fichier_trouve:
            self._log("Aucun CSV fusion trouvé", level="ERROR")
            return False
        t0 = time.time()
        self.df = pd.read_csv(fichier_trouve, low_memory=False)
        self._log(f"[OK] Chargé {len(self.df):,} lignes ({fichier_trouve}) en {time.time()-t0:.1f}s", "OK")
        self.ui.update_stage("load", len(self.df), len(self.df), "Lecture terminée")
        self.ui.update_global(1, 5, "Chargement terminé")
        return True

    def prepare_data(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "Label" in numeric_cols:
            numeric_cols.remove("Label")
        if "Label" not in self.df.columns:
            self._log("Colonne Label absente", "ERROR")
            return False
        self.df = self.df.dropna(subset=["Label"])
        n_samples = int(len(self.df) * 0.5)
        stratifier = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        for train_idx, _ in stratifier.split(self.df, self.df["Label"]):
            self.df = self.df.iloc[train_idx[:n_samples]]
            break
        X = self.df[numeric_cols].astype(np.float32).copy()
        X = X.fillna(X.mean())
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df["Label"])
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X).astype(np.float32)
        del self.df, X
        gc.collect()
        self._log(f"[OK] Données prêtes: {self.X_scaled.shape}", "OK")
        self.ui.update_stage("prep", self.X_scaled.shape[0], self.X_scaled.shape[0], "Préparation OK")
        self.ui.update_global(2, 5, "Préparation terminée")
        return True

    def run_cv_for_model(self, model_name, model, train_sizes):
        res = {
            "train_sizes": [],
            "f1_scores": [],
            "recall_scores": [],
            "precision_scores": [],
            "avt_scores": [],
            "f1_std": [],
            "recall_std": [],
            "precision_std": [],
        }
        n_samples = len(self.y)
        for idx, train_size in enumerate(train_sizes, 1):
            sss = StratifiedShuffleSplit(n_splits=K_FOLD, train_size=train_size, test_size=1 - train_size, random_state=42)
            f1s = np.zeros(K_FOLD, dtype=np.float32)
            recs = np.zeros(K_FOLD, dtype=np.float32)
            pres = np.zeros(K_FOLD, dtype=np.float32)
            avts = np.zeros(K_FOLD, dtype=np.float32)
            for fold, (train_idx, val_idx) in enumerate(sss.split(self.X_scaled, self.y), 1):
                Xtr, Xva = self.X_scaled[train_idx], self.X_scaled[val_idx]
                ytr, yva = self.y[train_idx], self.y[val_idx]
                model.fit(Xtr, ytr)
                t_pred = time.time()
                ypred = model.predict(Xva)
                pred_time = time.time() - t_pred
                f1s[fold - 1] = f1_score(yva, ypred, average="weighted", zero_division=0)
                recs[fold - 1] = recall_score(yva, ypred, average="weighted", zero_division=0)
                pres[fold - 1] = precision_score(yva, ypred, average="weighted", zero_division=0)
                avts[fold - 1] = len(Xva) / pred_time if pred_time > 0 else 0
            mean_f1 = float(np.mean(f1s))
            std_f1 = float(np.std(f1s))
            res["train_sizes"].append(int(train_size * 100))
            res["f1_scores"].append(mean_f1)
            res["f1_std"].append(std_f1)
            res["recall_scores"].append(float(np.mean(recs)))
            res["recall_std"].append(float(np.std(recs)))
            res["precision_scores"].append(float(np.mean(pres)))
            res["precision_std"].append(float(np.std(pres)))
            res["avt_scores"].append(float(np.mean(avts)))
            self.ui.update_file_progress(f"{model_name}", int(idx / len(train_sizes) * 100), f"{int(train_size*100)}% F1={mean_f1:.4f}")
        best_idx = int(np.argmax(np.array(res["f1_scores"]) - np.array(res["f1_std"])))
        best_ts = train_sizes[best_idx]
        self.optimal_configs[model_name] = {
            "train_size": float(best_ts),
            "test_size": float(1 - best_ts),
            "f1_score": float(res["f1_scores"][best_idx]),
            "f1_std": float(res["f1_std"][best_idx]),
            "recall": float(res["recall_scores"][best_idx]),
            "precision": float(res["precision_scores"][best_idx]),
            "avt": float(res["avt_scores"][best_idx]),
            "n_jobs": MODELS_CONFIG[model_name]["n_jobs"],
        }
        self.results[model_name] = res

    def run_cv(self):
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, n_jobs=MODELS_CONFIG["Logistic Regression"]["n_jobs"])),
            ("Naive Bayes", GaussianNB()),
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=MODELS_CONFIG["Random Forest"]["n_jobs"])),
        ]
        total = len(models)
        for i, (name, model) in enumerate(models, 1):
            train_sizes_to_test = TRAIN_SIZES[TRAIN_SIZES <= 0.80] if name == "Decision Tree" else TRAIN_SIZES
            self._log(f"[CV] {i}/{total} {name}", "INFO")
            self.run_cv_for_model(name, model, train_sizes_to_test)
            self.ui.update_stage("cv", i, total, f"{name} OK")
            self.ui.update_global(2 + i / total, 5, f"CV {i}/{total}")
        return True

    def save_reports(self):
        try:
            with open("cv_results_summary.txt", "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("CROSS-VALIDATION OPTIMIZATION RESULTS\n")
                f.write("=" * 100 + "\n\n")
                for name in sorted(self.optimal_configs.keys()):
                    cfg = self.optimal_configs[name]
                    f.write(f"{name:<25} Train:{cfg['train_size']*100:>5.0f}% F1:{cfg['f1_score']:>7.4f} ± {cfg.get('f1_std',0):>6.4f}\n")
                f.write("=" * 100 + "\n")
            pd.DataFrame(self.results).T.to_csv("cv_detailed_metrics.csv")
            self.ui.update_stage("reports", 1, 1, "Rapports OK")
            self.ui.update_global(4, 5, "Rapports")
            self._log("[OK] Rapports générés", "OK")
        except Exception as e:
            self._log(f"[ERROR] Rapports: {e}", "ERROR")

    def save_graphs(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("darkgrid")
            for algo, res in self.results.items():
                ts = np.array(res["train_sizes"])
                f1 = np.array(res["f1_scores"])
                f1s = np.array(res["f1_std"])
                recall = np.array(res["recall_scores"])
                precision = np.array(res["precision_scores"])
                avt_ms = np.array([1000.0/x if x>0 else np.nan for x in res["avt_scores"]])
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"{algo}", fontsize=14, fontweight="bold")
                axes[0, 0].plot(ts, f1, "o-", linewidth=2.5, markersize=8, color="#3498db")
                axes[0, 0].fill_between(ts, f1 - f1s, f1 + f1s, alpha=0.2)
                axes[0, 0].set_title("F1"); axes[0, 0].set_ylim([0, 1]); axes[0, 0].set_ylabel("Score")
                axes[0, 1].plot(ts, recall, "s-", linewidth=2.5, markersize=8, color="#e74c3c")
                axes[0, 1].set_title("Recall"); axes[0, 1].set_ylim([0, 1]); axes[0, 1].set_ylabel("Score")
                axes[1, 0].plot(ts, precision, "^-", linewidth=2.5, markersize=8, color="#f39c12")
                axes[1, 0].set_title("Precision"); axes[1, 0].set_ylim([0, 1]); axes[1, 0].set_ylabel("Score")
                axes[1, 0].set_xlabel("Train size (%)")
                axes[1, 1].plot(ts, avt_ms, "d-", linewidth=2.5, markersize=8, color="#27ae60")
                axes[1, 1].set_title("AVT (ms)"); axes[1, 1].set_xlabel("Train size (%)")
                plt.tight_layout()
                fname = f"graph_cv_{algo.replace(' ','_').lower()}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
            self.ui.update_stage("graphs", 1, 1, "Graphiques OK")
            self.ui.update_global(5, 5, "Terminé")
            self._log("[OK] Graphiques générés", "OK")
        except Exception as e:
            self._log(f"[ERROR] Graphiques: {e}", "ERROR")


def main():
    ui = GenericProgressGUI(title="Cross-Validation Optimisée", header_info=f"Cores: {NUM_CORES}", max_workers=4)
    runner = CrossValRunner(ui=ui)

    def job():
        if not runner.load_data():
            return
        if not runner.prepare_data():
            return
        runner.run_cv()
        runner.save_reports()
        runner.save_graphs()
        ui.log_alert("CV terminée", level="success")

    import threading
    threading.Thread(target=job, daemon=True).start()
    ui.start()


if __name__ == "__main__":
    main()
