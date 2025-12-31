#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML EVALUATION V3 - ADAPTÉ + OPTIMISÉ RAM
=====================================
✅ Gestion RAM dynamique (<90%)
✅ Memory optimization automatique
✅ Fallback console
✅ K-Fold validation
=====================================
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd

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

import multiprocessing
import threading
import psutil
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
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    GenericProgressGUI = None

NUM_CORES = multiprocessing.cpu_count()


# ============= MEMORY MANAGER =============
class MemoryManager:
    """Gestion mémoire dynamique"""
    RAM_THRESHOLD = 90.0
    
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
    def check_and_cleanup():
        """Nettoie si RAM trop haute"""
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > MemoryManager.RAM_THRESHOLD:
            gc.collect()
            return False
        return True


# ============= ML EVALUATION RUNNER =============
class MLEvaluationRunner:
    def __init__(self, ui=None):
        self.ui = ui
        self.results = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.classes = None
        
        if self.ui:
            self.ui.add_stage("load", "Chargement train/test")
            self.ui.add_stage("train", "Entraînement + évaluation holdout")
            self.ui.add_stage("reports", "Rapports")
            self.ui.add_stage("graphs", "Graphiques")

    def log(self, msg, level="INFO"):
        """Log compatible GUI + console"""
        if self.ui:
            self.ui.log(msg, level=level)
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

    def log_alert(self, msg, level="error"):
        if self.ui:
            self.ui.log_alert(msg, level=level)
        else:
            print(f"[ALERT] {msg}")

    def load_data(self):
        """Charge train et test avec gestion RAM"""
        npz_files = ["preprocessed_dataset.npz", "tensor_data.npz"]
        npz_file = next((f for f in npz_files if os.path.exists(f)), None)
        
        if not npz_file:
            self.log_alert("NPZ introuvable", level="error")
            return False
        
        try:
            t0 = time.time()
            
            # Nettoyer avant charge
            gc.collect()
            
            data = np.load(npz_file, allow_pickle=True)
            self.X_train = data["X"]
            self.y_train = data["y"]
            self.classes = data["classes"]
            
            self.log(f"Train NPZ chargé ({len(self.y_train):,} échantillons) en {time.time()-t0:.1f}s", level="OK")
            self.log(f"Classes: {list(self.classes)}", level="OK")
            self.log(f"RAM: {MemoryManager.get_ram_usage():.1f}%", level="DETAIL")

            # Charger test holdout
            if not os.path.exists("fusion_test_smart4.csv"):
                self.log_alert("fusion_test_smart4.csv introuvable", level="error")
                return False
            
            # Charger test en chunks si grand
            df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False, encoding='utf-8')
            df_test = _normalize_label_column(df_test)
            self.log(f"Test chargé: {len(df_test):,} lignes", level="OK")
            
            # Vérifier RAM
            if not MemoryManager.check_and_cleanup():
                self.log("RAM critique, nettoyage", level="WARN")
            
            numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            X_test_raw = df_test[numeric_cols].astype(np.float32)
            X_test_raw = X_test_raw.fillna(X_test_raw.mean())
            
            if X_test_raw.shape[1] != self.X_train.shape[1]:
                self.log_alert(f"Mismatch features", level="error")
                return False
            
            # Normaliser avec stats training
            mean = self.X_train.mean(axis=0)
            std = self.X_train.std(axis=0) + 1e-8
            self.X_test = ((X_test_raw - mean) / std).astype(np.float32)
            
            # Encoder labels
            lbl = LabelEncoder()
            lbl.classes_ = self.classes
            y_test_raw = df_test["Label"].astype(str).values
            self.y_test = lbl.transform(y_test_raw)
            
            self.log(f"Test normalisé: X={self.X_test.shape}, y={len(self.y_test):,}", level="OK")
            
            if self.ui:
                self.ui.update_stage("load", 1, 1, "Train/Test chargés")
                self.ui.update_global(1, 4, f"Train {len(self.X_train):,} | Test {len(self.X_test):,}")
            
            # Nettoyer
            del df_test, X_test_raw
            gc.collect()
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur load_data: {e}", level="error")
            return False

    def train_eval(self):
        """K-Fold evaluation avec gestion RAM"""
        try:
            models = [
                ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
                ("Naive Bayes", GaussianNB()),
                ("Decision Tree", DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=10)),
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            total = len(models)
            
            for i, (name, model) in enumerate(models, 1):
                self.log(f"\n[MODEL {i}/{total}] {name}", level="OK")
                
                f1_runs = []
                recall_runs = []
                precision_runs = []
                cm_sum = None
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(self.X_test), 1):
                    # Vérifier RAM avant split
                    if not MemoryManager.check_and_cleanup():
                        self.log("RAM critique, pause", level="WARN")
                        time.sleep(1)
                    
                    X_train_combined = np.vstack([self.X_train, self.X_test[train_idx]])
                    y_train_combined = np.hstack([self.y_train, self.y_test[train_idx]])
                    
                    X_val = self.X_test[test_idx]
                    y_val = self.y_test[test_idx]
                    
                    # Entraîner
                    model.fit(X_train_combined, y_train_combined)
                    y_pred = model.predict(X_val)
                    
                    # Métriques
                    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
                    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
                    
                    f1_runs.append(f1)
                    recall_runs.append(recall)
                    precision_runs.append(precision)
                    
                    cm = confusion_matrix(y_val, y_pred, labels=np.unique(self.y_test))
                    cm_sum = cm if cm_sum is None else cm_sum + cm
                    
                    self.log(f"  Fold {fold}: F1={f1:.4f} | Recall={recall:.4f} | Precision={precision:.4f}", level="info")
                    
                    # Cleanup
                    del X_train_combined, y_train_combined, X_val, y_val, y_pred
                    gc.collect()
                    
                    if self.ui:
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
                
                self.log(f"✅ {name}: F1={mean_f1:.4f}±{std_f1:.4f}", level="OK")
                
                if self.ui:
                    self.ui.update_stage("train", i, total, f"{name} F1={mean_f1:.4f}")
                    self.ui.update_global(1 + i / total, 4, f"Modèle {i}/{total}")
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur train_eval: {e}", level="error")
            return False

    def save_reports(self):
        """Sauvegarder rapports"""
        try:
            with open("evaluation_results_summary.txt", "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("ML EVALUATION V3 - ADAPTÉ\n")
                f.write("=" * 100 + "\n\n")
                f.write("✅ K-Fold validation: 5 folds\n")
                f.write("✅ Gestion RAM dynamique\n\n")
                
                for name, res in sorted(self.results.items(), key=lambda x: x[1]["f1"], reverse=True):
                    f.write(f"{name:<25} F1={res['f1']:.4f}±{res['f1_std']:.4f} ")
                    f.write(f"Recall={res['recall']:.4f} Precision={res['precision']:.4f}\n")
                
                f.write("=" * 100 + "\n")
            
            self.log("Rapport texte: evaluation_results_summary.txt", level="OK")
            
            with open("ml_evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.log("Rapport JSON: ml_evaluation_results.json", level="OK")
            
            if self.ui:
                self.ui.update_stage("reports", 1, 1, "Rapports OK")
                self.ui.update_global(3, 4, "Rapports")
            
            return True
        except Exception as e:
            self.log_alert(f"Rapports: {e}", level="error")
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
                
                self.log(f"Graphique: {fname}", level="OK")
                gc.collect()
            
            if self.ui:
                self.ui.update_stage("graphs", 1, 1, "Graphiques OK")
                self.ui.update_global(4, 4, "Terminé")
            
            return True
        except Exception as e:
            self.log_alert(f"Graphiques: {e}", level="error")
            return False


def main():
    """Point d'entrée"""
    print("\n" + "="*80)
    print("ML EVALUATION V3 - ADAPTÉ (Gestion RAM Optimale)")
    print("="*80 + "\n")

    if HAS_GUI:
        print("[INFO] Mode GUI activé\n")
        ui = GenericProgressGUI(title="ML Evaluation V3 - ADAPTÉ", 
                               header_info=f"Cores: {NUM_CORES}, RAM: {MemoryManager.get_available_ram_gb():.1f}GB", 
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
        return True
    else:
        print("[INFO] Mode console (progress_gui non disponible)\n")
        runner = MLEvaluationRunner(ui=None)
        
        if not runner.load_data():
            print("[ERROR] Chargement données échouée")
            return False
        
        if not runner.train_eval():
            print("[ERROR] Entraînement échoué")
            return False
        
        if not runner.save_reports():
            print("[ERROR] Rapports échoué")
            return False
        
        if not runner.save_graphs():
            print("[ERROR] Graphiques échoué")
            return False
        
        print("\n✅ Evaluation complétée avec succès!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)