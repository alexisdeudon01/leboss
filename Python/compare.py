#!/usr/bin/env python3
"""
COMPARAISON DES ALGORITHMES - ÉTAPE 2

Utilise le fichier de fusion généré par Harmonizer
Teste 4 modèles et trouve le meilleur

Modèles:
  1. KMeans (baseline)
  2. Random Forest (rapide, bon)
  3. XGBoost (puissant)
  4. Isolation Forest (anomalies)

Entrée:  fusion_ton_iot_cic_final_smart.csv
Sorties:
  - comparison_results.txt (rapport)
  - models_comparison.csv (métriques)
  - best_model_comparison.png (graphique)
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import time
import pickle
import threading
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Optional GUI integration
try:
    from progress_gui import GenericProgressGUI
except ImportError:
    GenericProgressGUI = None
USE_GUI = os.getenv("USE_PROGRESS_GUI", "1") == "1"

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'INPUT_FILE': 'fusion_ton_iot_cic_final_smart.csv',
    'OUTPUT_RESULTS': 'comparison_results.txt',
    'OUTPUT_METRICS': 'models_comparison.csv',
    'OUTPUT_BEST_MODEL': 'best_model_comparison.pkl',
    'OUTPUT_CHART': 'models_comparison.png',
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
}

# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class AlgorithmsComparison:
    """Comparaison d'algorithmes"""
    
    def __init__(self, ui=None):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.ui = ui
        self.total_steps = 6  # load, prepare, train, eval, best, save/plot
        
        if self.ui:
            self.ui.add_stage("load", "Chargement")
            self.ui.add_stage("prep", "Préparation")
            self.ui.add_stage("train", "Entraînement")
            self.ui.add_stage("eval", "Évaluation")
            self.ui.add_stage("best", "Meilleur modèle")
            self.ui.add_stage("save", "Sauvegarde/Graphiques")
            self.ui.log("UI prête pour comparaison", level="INFO")
        else:
            self.print_header("COMPARAISON DES ALGORITHMES")
    
    def print_header(self, text):
        print("\n" + "═" * 80)
        print(f"  {text}")
        print("═" * 80 + "\n")
    
    def print_section(self, text):
        print(f"\n{'─' * 80}\n  ► {text}\n{'─' * 80}\n")
    
    def print_success(self, text):
        print(f"  ✅ {text}")
    
    def print_error(self, text):
        print(f"  ❌ {text}")
    
    def print_info(self, text):
        print(f"  ℹ️  {text}")
    
    # ========================================================================
    # CHARGEMENT ET PRÉPARATION
    # ========================================================================
    
    def _log_global(self, step_index, msg, eta=None):
        if self.ui:
            self.ui.update_global(step_index, self.total_steps, msg, eta)
        else:
            self.print_info(msg)

    def _update_stage(self, key, current, total, msg="", eta=None):
        if self.ui:
            self.ui.update_stage(key, current, total, msg, eta)

    def load_data(self):
        """Charger les données"""
        self.print_section("CHARGEMENT DES DONNÉES")
        
        try:
            # Vérifier fichier
            fichiers_possibles = [
                'fusion_ton_iot_cic_final_smart.csv',
                'fusion_ton_iot_cic_final_smart4.csv',
                'fusion_ton_iot_cic_final_smart3.csv',
            ]
            
            fichier_trouve = None
            for f in fichiers_possibles:
                if os.path.exists(f):
                    fichier_trouve = f
                    CONFIG['INPUT_FILE'] = f
                    break
            
            if not fichier_trouve:
                self.print_error(f"Aucun fichier fusion trouvé!")
                self.print_info(f"Cherchait: {fichiers_possibles}")
                return False
            
            self.print_info(f"Chargement: {fichier_trouve}")
            t0 = time.time()
            self.df = pd.read_csv(fichier_trouve, low_memory=False)
            self._update_stage("load", len(self.df), max(1, len(self.df)), "Lecture terminée")
            self._log_global(1, f"Chargé {len(self.df):,} lignes", eta=None)
            self.print_success(f"Chargé: {len(self.df):,} lignes × {len(self.df.columns)} colonnes ({time.time()-t0:.1f}s)")
            return True
        except Exception as e:
            self.print_error(f"Erreur chargement: {e}")
            return False
    
    def prepare_data(self):
        """Préparer les données"""
        self.print_section("PRÉPARATION DES DONNÉES")
        
        try:
            # Features numériques
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            self.print_info(f"Features numériques: {len(numeric_cols)}")
            
            X = self.df[numeric_cols].copy()
            
            # Remplir NaN
            missing = X.isnull().sum().sum()
            if missing > 0:
                self.print_info(f"Valeurs manquantes: {missing}")
                X = X.fillna(X.mean())
                self.print_success("Remplies avec la moyenne")
            
            # Label
            if 'Label' not in self.df.columns:
                self.print_error("Colonne 'Label' manquante!")
                return False
            
            y = self.label_encoder.fit_transform(self.df['Label'])
            self.print_info(f"Classes: {self.label_encoder.classes_}")
            
            # Split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=CONFIG['TEST_SIZE'], random_state=CONFIG['RANDOM_STATE']
            )
            
            # Normaliser
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            self.print_success(f"Train: {len(self.X_train):,} | Test: {len(self.X_test):,}")
            self.print_success("Features normalisées")
            self._update_stage("prep", len(self.X_train), max(1, len(self.df)), "Préparation ok")
            self._log_global(2, "Préparation terminée")
            return True
        except Exception as e:
            self.print_error(f"Erreur préparation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # ENTRAÎNEMENT
    # ========================================================================
    
    def train_models(self):
        """Entraîner les modèles"""
        self.print_section("ENTRAÎNEMENT DES MODÈLES")
        
        try:
            total = 4
            # 1. KMeans
            self.print_info("1. KMeans...")
            start = time.time()
            km = KMeans(n_clusters=2, random_state=CONFIG['RANDOM_STATE'], n_init=10, max_iter=300)
            km.fit(self.X_train_scaled)
            km_time = time.time() - start
            self.models['KMeans'] = km
            self.print_success(f"   KMeans en {km_time:.2f}s")
            self._update_stage("train", 1, total, "KMeans")
            
            # 2. Random Forest
            self.print_info("2. Random Forest...")
            start = time.time()
            rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['RANDOM_STATE'], 
                                       n_jobs=-1, verbose=0)
            rf.fit(self.X_train_scaled, self.y_train)
            rf_time = time.time() - start
            self.models['Random Forest'] = rf
            self.print_success(f"   Random Forest en {rf_time:.2f}s")
            self._update_stage("train", 2, total, "Random Forest")
            
            # 3. XGBoost (si disponible)
            if xgb:
                self.print_info("3. XGBoost...")
                start = time.time()
                xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=CONFIG['RANDOM_STATE'],
                                            n_jobs=-1, verbosity=0)
                xgb_model.fit(self.X_train_scaled, self.y_train, verbose=False)
                xgb_time = time.time() - start
                self.models['XGBoost'] = xgb_model
                self.print_success(f"   XGBoost en {xgb_time:.2f}s")
                self._update_stage("train", 3, total, "XGBoost")
            else:
                self.print_info("3. XGBoost... (non installé, installation: pip install xgboost)")
                total -= 1  # ne compte pas dans la progression si absent
            
            # 4. Isolation Forest
            self.print_info("4. Isolation Forest...")
            start = time.time()
            iso = IsolationForest(random_state=CONFIG['RANDOM_STATE'], n_jobs=-1)
            iso.fit(self.X_train_scaled)
            iso_time = time.time() - start
            self.models['Isolation Forest'] = iso
            self.print_success(f"   Isolation Forest en {iso_time:.2f}s")
            self._update_stage("train", total, total, "Isolation Forest")
            self._log_global(3, "Entraînement terminé")
            
            return True
        except Exception as e:
            self.print_error(f"Erreur entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # ÉVALUATION
    # ========================================================================
    
    def evaluate_models(self):
        """Évaluer les modèles"""
        self.print_section("ÉVALUATION DES MODÈLES")
        
        try:
            for name, model in self.models.items():
                self.print_info(f"Évaluation {name}...")
                
                # Prédictions
                start = time.time()
                y_pred = model.predict(self.X_test_scaled)
                pred_time = time.time() - start
                avg_time_ms = (pred_time / len(self.X_test)) * 1000
                
                # Métriques
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                self.results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Time (ms)': avg_time_ms,
                }
                
                print(f"     Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                      f"Recall: {recall:.4f} | F1: {f1:.4f} | Time: {avg_time_ms:.4f}ms")
            self._update_stage("eval", len(self.models), max(1, len(self.models)), "Évaluations terminées")
            self._log_global(4, "Évaluation OK")
            return True
        except Exception as e:
            self.print_error(f"Erreur évaluation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # MEILLEUR MODÈLE
    # ========================================================================
    
    def find_best_model(self):
        """Trouver le meilleur modèle"""
        self.print_section("SÉLECTION DU MEILLEUR MODÈLE")
        
        try:
            # Score composite: F1 * 70% + (1 - Time/MaxTime) * 30%
            max_time = max([v['Time (ms)'] for v in self.results.values()])
            
            scores = {}
            for name, metrics in self.results.items():
                f1 = metrics['F1-Score']
                time_score = 1 - (metrics['Time (ms)'] / max_time)
                composite = f1 * 0.7 + time_score * 0.3
                scores[name] = composite
            
            self.best_model_name = max(scores, key=scores.get)
            best_metrics = self.results[self.best_model_name]
            
            self.print_success(f"Meilleur modèle: {self.best_model_name}")
            self.print_info(f"  Accuracy:  {best_metrics['Accuracy']:.4f}")
            self.print_info(f"  Precision: {best_metrics['Precision']:.4f}")
            self.print_info(f"  Recall:    {best_metrics['Recall']:.4f}")
            self.print_info(f"  F1-Score:  {best_metrics['F1-Score']:.4f}")
            self.print_info(f"  Time (ms): {best_metrics['Time (ms)']:.4f}")
            self._update_stage("best", 1, 1, self.best_model_name)
            self._log_global(5, "Meilleur modèle sélectionné")
            
            return True
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            return False
    
    # ========================================================================
    # AFFICHAGE ET SAUVEGARDE
    # ========================================================================
    
    def display_results(self):
        """Afficher résultats"""
        self.print_section("RÉSUMÉ COMPARATIF")
        
        print(f"{'Modèle':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Time(ms)':>12}")
        print("─" * 90)
        
        for name in sorted(self.results.keys()):
            m = self.results[name]
            print(f"{name:<20} {m['Accuracy']:>12.4f} {m['Precision']:>12.4f} "
                  f"{m['Recall']:>12.4f} {m['F1-Score']:>12.4f} {m['Time (ms)']:>12.4f}")
        
        print("\n" + "═" * 90)
        print(f"🏆 MEILLEUR: {self.best_model_name}")
        print("═" * 90)
    
    def save_results(self):
        """Sauvegarder résultats"""
        self.print_section("SAUVEGARDE DES RÉSULTATS")
        
        try:
            # Fichier texte
            with open(CONFIG['OUTPUT_RESULTS'], 'w', encoding='utf-8') as f:
                f.write("═" * 80 + "\n")
                f.write("COMPARAISON DES ALGORITHMES\n")
                f.write("═" * 80 + "\n\n")
                
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Fichier: {CONFIG['INPUT_FILE']}\n")
                f.write(f"Train set: {len(self.X_train):,} | Test set: {len(self.X_test):,}\n\n")
                
                f.write("─" * 80 + "\n")
                f.write("RÉSULTATS\n")
                f.write("─" * 80 + "\n\n")
                
                f.write(f"{'Modèle':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Time(ms)':>12}\n")
                f.write("─" * 80 + "\n")
                
                for name in sorted(self.results.keys()):
                    m = self.results[name]
                    f.write(f"{name:<20} {m['Accuracy']:>12.4f} {m['Precision']:>12.4f} "
                           f"{m['Recall']:>12.4f} {m['F1-Score']:>12.4f} {m['Time (ms)']:>12.4f}\n")
                
                f.write("\n" + "═" * 80 + "\n")
                f.write(f"🏆 MEILLEUR MODÈLE: {self.best_model_name}\n")
                f.write(f"   Accuracy:  {self.results[self.best_model_name]['Accuracy']:.4f}\n")
                f.write(f"   Precision: {self.results[self.best_model_name]['Precision']:.4f}\n")
                f.write(f"   Recall:    {self.results[self.best_model_name]['Recall']:.4f}\n")
                f.write(f"   F1-Score:  {self.results[self.best_model_name]['F1-Score']:.4f}\n")
                f.write(f"   Time (ms): {self.results[self.best_model_name]['Time (ms)']:.4f}\n")
                f.write("═" * 80 + "\n")
            
            self.print_success(f"Rapport sauvegardé: {CONFIG['OUTPUT_RESULTS']}")
            
            # Fichier CSV
            df_results = pd.DataFrame(self.results).T
            df_results.to_csv(CONFIG['OUTPUT_METRICS'])
            self.print_success(f"Métriques sauvegardées: {CONFIG['OUTPUT_METRICS']}")
            self._update_stage("save", 2, 3, "CSV sauvegardé")
            # Meilleur modèle
            best_model_obj = self.models[self.best_model_name]
            with open(CONFIG['OUTPUT_BEST_MODEL'], 'wb') as f:
                pickle.dump(best_model_obj, f)
            self.print_success(f"Modèle sauvegardé: {CONFIG['OUTPUT_BEST_MODEL']}")
            self._update_stage("save", 3, 3, "Sauvegardes terminées")
            self._log_global(6, "Sauvegarde OK")
            
            return True
        except Exception as e:
            self.print_error(f"Erreur sauvegarde: {e}")
            return False
    
    def plot_results(self):
        """Créer graphiques"""
        self.print_section("CRÉATION DES GRAPHIQUES")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            models = list(self.results.keys())
            accuracies = [self.results[m]['Accuracy'] for m in models]
            precisions = [self.results[m]['Precision'] for m in models]
            recalls = [self.results[m]['Recall'] for m in models]
            f1s = [self.results[m]['F1-Score'] for m in models]
            
            colors = ['#27ae60' if m == self.best_model_name else '#3498db' for m in models]
            
            # Accuracy
            axes[0, 0].bar(models, accuracies, color=colors)
            axes[0, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Precision
            axes[0, 1].bar(models, precisions, color=colors)
            axes[0, 1].set_title('Precision', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Recall
            axes[1, 0].bar(models, recalls, color=colors)
            axes[1, 0].set_title('Recall', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # F1-Score
            axes[1, 1].bar(models, f1s, color=colors)
            axes[1, 1].set_title('F1-Score', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.suptitle(f'Comparaison des Algorithmes - Meilleur: {self.best_model_name}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(CONFIG['OUTPUT_CHART'], dpi=300, bbox_inches='tight')
            plt.close()
            
            self.print_success(f"Graphique sauvegardé: {CONFIG['OUTPUT_CHART']}")
            return True
        except Exception as e:
            self.print_error(f"Erreur graphique: {e}")
            return False
    
    def run(self):
        """Exécuter le pipeline"""
        try:
            if not self.load_data():
                return False
            if not self.prepare_data():
                return False
            if not self.train_models():
                return False
            if not self.evaluate_models():
                return False
            if not self.find_best_model():
                return False
            
            self.display_results()
            self.save_results()
            self.plot_results()
            
            self.print_header("✅ COMPARAISON TERMINÉE AVEC SUCCÈS!")
            
            return True
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        if GenericProgressGUI and USE_GUI:
            ui = GenericProgressGUI(title="Comparaison Algorithmes",
                                    header_info=f"Fichier: {CONFIG['INPUT_FILE']}",
                                    max_workers=4)
            comp = AlgorithmsComparison(ui=ui)

            def worker():
                success = comp.run()
                if success:
                    ui.log_alert("Comparaison terminée", level="success")
                else:
                    ui.log_alert("Erreur durant la comparaison", level="error")
            threading.Thread(target=worker, daemon=True).start()
            ui.start()
        else:
            comp = AlgorithmsComparison(ui=None)
            success = comp.run()
            
            if success:
                print("\n" + "=" * 80)
                print("📁 FICHIERS GÉNÉRÉS:")
                print("=" * 80)
                print(f"  1. {CONFIG['OUTPUT_RESULTS']}")
                print(f"  2. {CONFIG['OUTPUT_METRICS']}")
                print(f"  3. {CONFIG['OUTPUT_BEST_MODEL']}")
                print(f"  4. {CONFIG['OUTPUT_CHART']}")
                print("=" * 80 + "\n")
                sys.exit(0)
            else:
                sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
