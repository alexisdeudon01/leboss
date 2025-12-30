#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST DECISION TREE SPLITS - ADAPTÉ + OPTIMISÉ RAM
================================================
✅ progress_gui optionnel (console fallback)
✅ Détection overfitting automatique
✅ Gestion RAM dynamique (<90%)
================================================
"""

import os
import sys
import json
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

try:
    from progress_gui import GenericProgressGUI
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    GenericProgressGUI = None

USE_GUI = HAS_GUI

# ============= MEMORY MANAGER =============
class MemoryManager:
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
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > MemoryManager.RAM_THRESHOLD:
            gc.collect()
            return False
        return True


# ============= DT SPLITS TESTER =============
class DTSplitsTester:
    """Test Decision Tree avec gestion RAM"""
    
    def __init__(self, ui=None):
        self.ui = ui
        self.X = None
        self.y = None
        self.classes = None
        self.results = {
            'test_sizes': [],
            'f1_means': [],
            'f1_stds': [],
            'recall_means': [],
            'precision_means': [],
            'all_f1_runs': {}
        }
        
        if self.ui:
            self.ui.add_stage("load", "Chargement données")
            self.ui.add_stage("test", "Évaluation DT")
            self.ui.add_stage("analysis", "Analyse overfitting")
            self.ui.add_stage("graph", "Graphiques")

    def log(self, msg, level="INFO"):
        """Log compatible GUI + console"""
        if self.ui:
            self.ui.log(msg, level=level)
        else:
            import time
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

    def log_alert(self, msg, level="error"):
        if self.ui:
            self.ui.log_alert(msg, level=level)
        else:
            print(f"[ALERT] {msg}")

    def load_data(self):
        """Charge les données"""
        try:
            if not os.path.exists("preprocessed_dataset.npz"):
                self.log_alert("Données manquantes", level="error")
                return False
            
            # Nettoyer avant charge
            gc.collect()
            
            data = np.load("preprocessed_dataset.npz", allow_pickle=True)
            self.X = data["X"]
            self.y = data["y"]
            self.classes = data["classes"]
            
            self.log(f"Données chargées: X={self.X.shape}, y={len(self.y):,}", level="OK")
            self.log(f"Classes: {list(self.classes)}", level="OK")
            self.log(f"RAM: {MemoryManager.get_ram_usage():.1f}%", level="DETAIL")
            
            if self.ui:
                self.ui.update_stage("load", 1, 1, "Données chargées")
                self.ui.update_global(1, 4, "Données chargées")
            
            # Cleanup
            del data
            gc.collect()
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur: {e}", level="error")
            return False

    def test_splits(self):
        """Teste DT avec différentes tailles"""
        try:
            test_sizes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
            num_runs = 5
            total_tests = len(test_sizes) * num_runs
            current_test = 0
            
            self.log(f"\nÉvaluation Decision Tree avec différentes tailles", level="INFO")
            self.log(f"Test sizes: {test_sizes}", level="INFO")
            self.log(f"Runs par taille: {num_runs}", level="INFO")
            self.log(f"Total: {total_tests} évaluations\n", level="INFO")
            
            for test_size in test_sizes:
                self.log(f"\n{test_size*100:.0f}% test", level="INFO")
                
                f1_runs = []
                recall_runs = []
                precision_runs = []
                
                for run in range(num_runs):
                    current_test += 1
                    
                    # Vérifier RAM avant split
                    if not MemoryManager.check_and_cleanup():
                        self.log("RAM critique, pause", level="WARN")
                        import time
                        time.sleep(1)
                    
                    # Split stratifié
                    X_train, X_test, y_train, y_test = train_test_split(
                        self.X, self.y,
                        test_size=test_size,
                        random_state=42 + run,
                        stratify=self.y
                    )
                    
                    # Entraîner DT
                    model = DecisionTreeClassifier(random_state=42 + run, 
                                                  max_depth=20, 
                                                  min_samples_split=10)
                    model.fit(X_train, y_train)
                    
                    # Prédire
                    y_pred = model.predict(X_test)
                    
                    # Métriques
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    
                    f1_runs.append(f1)
                    recall_runs.append(recall)
                    precision_runs.append(precision)
                    
                    self.log(f"  Run {run+1}: F1={f1:.4f} | Recall={recall:.4f} | Precision={precision:.4f}", level="info")
                    
                    if self.ui:
                        self.ui.update_file_progress(
                            f"{test_size*100:.0f}% test",
                            int((run+1)/num_runs*100),
                            f"Run {run+1}/{num_runs}: F1={f1:.4f}"
                        )
                        self.ui.update_stage("test", current_test, total_tests,
                                            f"Test {test_size*100:.0f}% run {run+1}")
                        self.ui.update_global(1 + current_test/total_tests, 4,
                                            f"Évaluation {current_test}/{total_tests}")
                    
                    # Cleanup
                    del X_train, X_test, y_train, y_test, y_pred, model
                    gc.collect()
                
                # Statistiques par taille
                mean_f1 = np.mean(f1_runs)
                std_f1 = np.std(f1_runs)
                
                self.results['test_sizes'].append(test_size)
                self.results['f1_means'].append(mean_f1)
                self.results['f1_stds'].append(std_f1)
                self.results['recall_means'].append(np.mean(recall_runs))
                self.results['precision_means'].append(np.mean(precision_runs))
                self.results['all_f1_runs'][f'{test_size*100:.0f}%'] = f1_runs
                
                self.log(f"RÉSUMÉ: F1={mean_f1:.4f}±{std_f1:.4f}", level="OK")
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur: {e}", level="error")
            return False

    def analyze_overfitting(self):
        """Analyse overfitting"""
        try:
            self.log(f"\nDétection d'overfitting\n", level="INFO")
            
            f1_scores = np.array(self.results['f1_means'])
            f1_stds = np.array(self.results['f1_stds'])
            test_sizes = np.array(self.results['test_sizes'])
            
            self.log(f"F1 Score par taille de test:", level="INFO")
            for ts, f1, std in zip(test_sizes, f1_scores, f1_stds):
                self.log(f"  {ts*100:>5.0f}% test: F1={f1:.4f}±{std:.4f}", level="info")
            
            # Coefficients de variation
            cv = f1_stds / f1_scores
            self.log(f"\nCoefficients de variation:", level="INFO")
            for ts, c in zip(test_sizes, cv):
                self.log(f"  {ts*100:>5.0f}% test: CV={c:.4f}", level="info")
            
            # Décision
            mean_cv = np.mean(cv)
            self.log(f"\nMoyenne CV: {mean_cv:.4f}", level="INFO")
            
            if mean_cv < 0.02:
                verdict = "EXCELLENT - Très stable (pas d'overfitting)"
            elif mean_cv < 0.05:
                verdict = "BON - Stable (peu d'overfitting)"
            elif mean_cv < 0.10:
                verdict = "ACCEPTABLE - Peu stable (léger overfitting)"
            else:
                verdict = "INSTABLE - Très variable (overfitting probable)"
            
            self.log(f"\nVERDICT: {verdict}\n", level="OK")
            
            if self.ui:
                self.ui.log(f"[ANALYSIS] {verdict}", level="OK")
                self.ui.update_stage("analysis", 1, 1, "Analyse complète")
                self.ui.update_global(3, 4, "Analyse")
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur analyse: {e}", level="error")
            return False

    def generate_graph(self):
        """Génère graphiques"""
        try:
            self.log(f"\nGénération graphique", level="INFO")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Graphique 1: F1 vs test size
            test_sizes_pct = np.array(self.results['test_sizes']) * 100
            f1_means = np.array(self.results['f1_means'])
            f1_stds = np.array(self.results['f1_stds'])
            
            axes[0].plot(test_sizes_pct, f1_means, 'o-', linewidth=2.5, markersize=8, color='#3498db')
            axes[0].fill_between(test_sizes_pct, f1_means-f1_stds, f1_means+f1_stds, alpha=0.2, color='#3498db')
            axes[0].set_xlabel('Test Size (%)', fontsize=11)
            axes[0].set_ylabel('F1 Score', fontsize=11)
            axes[0].set_title('Decision Tree - Stabilité en fonction de la taille de test', fontsize=12, fontweight='bold')
            axes[0].set_ylim([0, 1])
            axes[0].grid(True, alpha=0.3)
            
            for x, y, std in zip(test_sizes_pct, f1_means, f1_stds):
                axes[0].text(x, y+0.03, f'{y:.3f}', ha='center', fontsize=9)
            
            # Graphique 2: Coefficient de variation
            f1_stds_arr = np.array(self.results['f1_stds'])
            f1_means_arr = np.array(self.results['f1_means'])
            cv = f1_stds_arr / f1_means_arr
            
            colors = ['#27ae60' if c < 0.05 else '#f39c12' if c < 0.10 else '#e74c3c' for c in cv]
            axes[1].bar(test_sizes_pct, cv, width=3, color=colors, edgecolor='black', linewidth=1.5)
            axes[1].set_xlabel('Test Size (%)', fontsize=11)
            axes[1].set_ylabel('Coefficient de Variation', fontsize=11)
            axes[1].set_title('Stabilité - Coefficient de Variation', fontsize=12, fontweight='bold')
            axes[1].axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Seuil bon')
            axes[1].axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Seuil acceptable')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            
            for x, c in zip(test_sizes_pct, cv):
                axes[1].text(x, c+0.002, f'{c:.4f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('test_dt_splits.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"Graphique sauvegardé: test_dt_splits.png", level="OK")
            
            if self.ui:
                self.ui.update_stage("graph", 1, 1, "Graphique généré")
                self.ui.update_global(4, 4, "Terminé")
            
            gc.collect()
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur graphique: {e}", level="error")
            return False

    def save_results(self):
        """Sauvegarde résultats et modèle"""
        try:
            with open('dt_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=float)
            
            self.log(f"Résultats sauvegardés: dt_test_results.json", level="OK")
            
            # Entraîner et sauvegarder meilleur modèle
            import joblib
            
            # Meilleur split basé sur résultats
            best_idx = np.argmax(self.results['f1_means'])
            best_test_size = self.results['test_sizes'][best_idx]
            
            self.log(f"Entraînement modèle final (test_size={best_test_size})...", level="INFO")
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=best_test_size,
                random_state=42,
                stratify=self.y
            )
            
            final_model = DecisionTreeClassifier(random_state=42, max_depth=20, 
                                               min_samples_split=10)
            final_model.fit(X_train, y_train)
            
            joblib.dump(final_model, 'ddos_detector_final.pkl')
            
            self.log(f"Modèle sauvegardé: ddos_detector_final.pkl", level="OK")
            
            if self.ui:
                self.ui.log("Résultats et modèle sauvegardés", level="OK")
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur sauvegarde: {e}", level="error")
            return False

    def run(self):
        """Exécute tous les tests"""
        if not self.load_data():
            return False
        
        if not self.test_splits():
            return False
        
        if not self.analyze_overfitting():
            return False
        
        if not self.generate_graph():
            return False
        
        if not self.save_results():
            return False
        
        return True


def run_with_gui():
    """Mode avec GUI"""
    ui = GenericProgressGUI(title="Test Decision Tree Splits",
                           header_info="Overfitting Detection",
                           max_workers=2)
    tester = DTSplitsTester(ui=ui)

    def worker():
        try:
            ui.update_global(0, 4, "Initialisation")
            if tester.run():
                ui.log_alert("Test DT Splits complété!", level="success")
            else:
                ui.log_alert("Erreur pendant les tests", level="error")
        except Exception as e:
            ui.log_alert(f"Erreur: {e}", level="error")

    import threading
    threading.Thread(target=worker, daemon=True).start()
    ui.start()


def main():
    """Point d'entrée"""
    print("\n" + "="*80)
    print("TEST DECISION TREE SPLITS - OVERFITTING DETECTION")
    print("="*80 + "\n")

    if USE_GUI:
        print("[INFO] Mode GUI activé\n")
        run_with_gui()
        return True
    
    # Mode console
    print("[INFO] Mode console (progress_gui non disponible)\n")
    tester = DTSplitsTester()
    success = tester.run()
    
    if success:
        print("\n" + "="*80)
        print("TEST COMPLÉTÉ")
        print("="*80 + "\n")
        return True
    else:
        print("\n[ERROR] Tests échoués")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
