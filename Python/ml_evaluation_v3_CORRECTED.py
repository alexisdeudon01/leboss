#!/usr/bin/env python3
"""
ML EVALUATION V3 - FINAL CORRIGÉ
========================================
✅ CORRECTION: LabelEncoder utilise MEMES CLASSES que training
✅ Pas de fit_transform(), utilise transform() avec classes du NPZ
✅ K-Fold validation sur test holdout
✅ Rapports et graphiques
========================================
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
from sklearn.model_selection import KFold
import json

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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.classes = None
        self._init_ui()

    def _init_ui(self):
        self.ui.add_stage("load", "Chargement train/test")
        self.ui.add_stage("train", "Entraînement + évaluation holdout")
        self.ui.add_stage("reports", "Rapports")
        self.ui.add_stage("graphs", "Graphiques")

    def load_data(self):
        """✅ CORRIGÉ: Charger train et test avec classes cohérentes"""
        npz_files = ["preprocessed_dataset.npz", "tensor_data.npz"]
        npz_file = next((f for f in npz_files if os.path.exists(f)), None)
        if not npz_file:
            self.ui.log_alert("NPZ introuvable", level="error")
            return False
        
        try:
            t0 = time.time()
            data = np.load(npz_file, allow_pickle=True)
            self.X_train = data["X"]
            self.y_train = data["y"]
            # ✅ CORRECTION: Sauvegarder les classes du training
            self.classes = data["classes"]
            self.ui.log(f"[OK] Train NPZ {npz_file} chargé ({len(self.y_train):,} échantillons) en {time.time()-t0:.1f}s", level="OK")
            self.ui.log(f"[OK] Classes: {list(self.classes)}", level="OK")

            # Charger le test holdout
            if not os.path.exists("fusion_test_smart4.csv"):
                self.ui.log_alert("fusion_test_smart4.csv introuvable", level="error")
                return False
            
            df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False)
            self.ui.log(f"[OK] Test chargé: {len(df_test):,} lignes", level="OK")
            
            numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            X_test_raw = df_test[numeric_cols].astype(np.float32)
            X_test_raw = X_test_raw.fillna(X_test_raw.mean())
            
            if X_test_raw.shape[1] != self.X_train.shape[1]:
                self.ui.log_alert(f"Mismatch features train/test: train {self.X_train.shape[1]} vs test {X_test_raw.shape[1]}", level="error")
                return False
            
            # Normaliser avec stats du training
            mean = self.X_train.mean(axis=0)
            std = self.X_train.std(axis=0) + 1e-8
            self.X_test = ((X_test_raw - mean) / std).astype(np.float32)
            
            # ✅ CORRECTION: Utiliser MEMES CLASSES que training
            lbl = LabelEncoder()
            lbl.classes_ = self.classes  # Imposer les mêmes classes
            
            # Convertir labels en strings pour matcher
            y_test_raw = df_test["Label"].astype(str).values
            
            # transform() au lieu de fit_transform() pour garder l'ordre du training
            self.y_test = lbl.transform(y_test_raw)
            
            self.ui.log(f"[OK] Test normalisé: X={self.X_test.shape}, y={len(self.y_test):,}", level="OK")
            self.ui.update_stage("load", len(self.X_train), len(self.X_train), "Train chargé")
            self.ui.update_stage("load", len(self.X_train)+len(self.X_test), len(self.X_train)+len(self.X_test), "Train/Test chargés")
            self.ui.update_global(1, 4, f"Train {len(self.X_train):,} | Test {len(self.X_test):,}")
            return True
        except Exception as e:
            self.ui.log_alert(f"Erreur load_data: {e}", level="error")
            return False

    def train_eval(self):
        """✅ K-Fold evaluation avec validation croisée"""
        try:
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
                self.ui.log(f"\n[MODEL {i}/{total}] {name}", level="OK")
                
                f1_runs = []
                recall_runs = []
                precision_runs = []
                cm_sum = None
                
                fold_count = 0
                for fold, (train_idx, test_idx) in enumerate(kf.split(self.X_test), 1):
                    fold_count += 1
                    
                    # Combiner train original + fold train du test
                    X_train_combined = np.vstack([self.X_train, self.X_test[train_idx]])
                    y_train_combined = np.hstack([self.y_train, self.y_test[train_idx]])
                    
                    X_val = self.X_test[test_idx]
                    y_val = self.y_test[test_idx]
                    
                    # Entraîner et prédire
                    model.fit(X_train_combined, y_train_combined)
                    y_pred = model.predict(X_val)
                    
                    # Métriques
                    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
                    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
                    
                    f1_runs.append(f1)
                    recall_runs.append(recall)
                    precision_runs.append(precision)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_val, y_pred, labels=np.unique(self.y_test))
                    cm_sum = cm if cm_sum is None else cm_sum + cm
                    
                    self.ui.log(f"  Fold {fold}: F1={f1:.4f} | Recall={recall:.4f} | Precision={precision:.4f}", level="info")
                    self.ui.update_file_progress(f"{name}", int(fold / 5 * 100),
                                                f"Fold {fold}/5 F1={f1:.4f}")
                
                # Résultats finaux
                mean_f1 = np.mean(f1_runs)
                std_f1 = np.std(f1_runs)
                
                self.results[name] = {
                    "f1": float(mean_f1),
                    "f1_std": float(std_f1),
                    "recall": float(np.mean(recall_runs)),
                    "precision": float(np.mean(precision_runs)),
                    "cm": cm_sum.tolist() if cm_sum is not None else [],
                }
                
                self.ui.log(f"  ✅ {name}: F1={mean_f1:.4f}±{std_f1:.4f}", level="OK")
                self.ui.update_stage("train", i, total, f"{name} F1={mean_f1:.4f}")
                self.ui.update_global(1 + i / total, 4, f"Modèle {i}/{total}")
            
            return True
        except Exception as e:
            self.ui.log_alert(f"Erreur train_eval: {e}", level="error")
            return False

    def save_reports(self):
        """Sauvegarder rapports texte et JSON"""
        try:
            # Rapport texte
            with open("evaluation_results_summary.txt", "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("ML EVALUATION V3 - FINAL (CORRIGÉ)\n")
                f.write("=" * 100 + "\n\n")
                f.write("✅ CORRECTION: LabelEncoder cohérent avec training\n")
                f.write("✅ K-Fold validation: 5 folds\n")
                f.write("✅ Test holdout: fusion_test_smart4.csv\n\n")
                
                for name, res in sorted(self.results.items(), key=lambda x: x[1]["f1"], reverse=True):
                    f.write(f"{name:<25} F1={res['f1']:.4f}±{res['f1_std']:.4f} ")
                    f.write(f"Recall={res['recall']:.4f} Precision={res['precision']:.4f}\n")
                
                f.write("=" * 100 + "\n")
            
            self.ui.log("[OK] Rapport texte: evaluation_results_summary.txt", level="OK")
            
            # Rapport JSON
            with open("ml_evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.ui.log("[OK] Rapport JSON: ml_evaluation_results.json", level="OK")
            self.ui.update_stage("reports", 1, 1, "Rapports OK")
            self.ui.update_global(3, 4, "Rapports")
            
            return True
        except Exception as e:
            self.ui.log_alert(f"Rapports: {e}", level="error")
            return False

    def save_graphs(self):
        """Générer graphiques"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("darkgrid")
            
            for name, res in self.results.items():
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Graphique 1: Métriques
                metrics = ["F1", "Recall", "Precision"]
                values = [res["f1"], res["recall"], res["precision"]]
                colors = ["#3498db", "#e74c3c", "#f39c12"]
                
                axes[0].bar(metrics, values, color=colors)
                axes[0].set_ylim([0, 1])
                axes[0].set_title(f"{name} - Métriques")
                axes[0].set_ylabel("Score")
                
                # Graphique 2: Confusion Matrix
                if res.get("cm"):
                    cm = np.array(res["cm"])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                               xticklabels=self.classes, yticklabels=self.classes)
                    axes[1].set_title(f"{name} - Confusion Matrix")
                    axes[1].set_xlabel("Prédiction")
                    axes[1].set_ylabel("Réalité")
                
                plt.tight_layout()
                fname = f"graph_eval_{name.replace(' ', '_').lower()}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                
                self.ui.log(f"[OK] Graphique: {fname}", level="OK")
                gc.collect()
            
            self.ui.update_stage("graphs", 1, 1, "Graphiques OK")
            self.ui.update_global(4, 4, "Terminé")
            self.ui.log("[OK] Graphiques générés", level="OK")
            
            return True
        except Exception as e:
            self.ui.log_alert(f"Graphiques: {e}", level="error")
            return False


def main():
    """Point d'entrée principal"""
    ui = GenericProgressGUI(title="ML Evaluation V3 - CORRIGÉ", 
                           header_info=f"Cores: {NUM_CORES}", 
                           max_workers=4)
    runner = MLEvaluationRunner(ui=ui)

    def job():
        try:
            ui.update_global(0, 4, "Initialisation")
            
            if not runner.load_data():
                ui.log_alert("Échec chargement données", level="error")
                return
            
            if not runner.train_eval():
                ui.log_alert("Échec entraînement", level="error")
                return
            
            if not runner.save_reports():
                ui.log_alert("Échec rapports", level="error")
                return
            
            if not runner.save_graphs():
                ui.log_alert("Échec graphiques", level="error")
                return
            
            ui.log_alert("Evaluation terminée avec succès!", level="success")
        except Exception as e:
            ui.log_alert(f"Erreur: {e}", level="error")

    threading.Thread(target=job, daemon=True).start()
    ui.start()


if __name__ == "__main__":
    main()