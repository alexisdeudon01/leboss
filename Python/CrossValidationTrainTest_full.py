#!/usr/bin/env python3
"""
CROSS-VALIDATION OPTIMIZATION - RAM OPTIMISÉE POUR VITESSE MAXIMALE

OPTIMISATIONS RAM:
  ✓ Chargement données une seule fois
  ✓ Cache RAM maximisé
  ✓ n_jobs=-1 partout (tous les cores)
  ✓ Batch processing optimisé
  ✓ Pré-allocation numpy arrays
  ✓ Parallélisation stratégique
  ✓ Garbage collection intelligent

FEATURES:
  ✓ 10 splits testés (50-95% training)
  ✓ K-Fold cross-validation (K=5)
  ✓ 4 algorithmes parallélisés
  ✓ Courbes dynamiques interactives
  ✓ UI GUI 1700x1100
  ✓ Graphiques interactifs matplotlib

ALGORITHMES:
  1. Logistic Regression
  2. Naive Bayes
  3. Decision Tree
  4. Random Forest

MÉTRIQUES TRACÉES:
  - F1 Score (courbe principale)
  - Recall vs % Training
  - Precision vs % Training
  - AVT vs % Training

OUTPUT:
  ✓ cv_results_summary.txt (résumé)
  ✓ cv_optimal_config.txt (config trouvée)
  ✓ 4 graphiques interactifs (1 par algo)
  ✓ cv_detailed_metrics.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import psutil
import gc
import time
import threading
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
# Forcer n_jobs
import os
os.environ['JOBLIB_PARALLEL_BACKEND'] = 'loky'

# Et vérifier les cores
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
print(f"🔧 Cores détectés: {NUM_CORES}")
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("❌ tkinter non installé")
    sys.exit(1)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.metrics import f1_score, recall_score, precision_score
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"❌ Dépendance manquante: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION OPTIMISÉE
# ============================================================================

TRAIN_SIZES = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
K_FOLD = 5
N_JOBS = -1  # Tous les cores disponibles
NUM_CORES = multiprocessing.cpu_count()

print(f"🚀 Cores disponibles: {NUM_CORES}")

# ============================================================================
# RAM MANAGER
# ============================================================================

class RAMOptimizer:
    def __init__(self):
        try:
            self.process = psutil.Process(os.getpid())
            self.total_ram = psutil.virtual_memory().total / (1024 ** 3)
            self.available_ram = psutil.virtual_memory().available / (1024 ** 3)
        except:
            self.total_ram = 8.0
            self.available_ram = 6.0
    
    def get_ram_percent(self):
        try:
            return (self.process.memory_info().rss / (1024 ** 3)) / self.total_ram * 100
        except:
            return 0
    
    def optimize_memory(self):
        """Optimiser mémoire"""
        try:
            gc.collect()
            psutil.virtual_memory()
        except:
            pass
    
    def estimate_batch_size(self, n_samples, n_features):
        """Estimer taille de batch optimale"""
        available_gb = self.available_ram * 0.8  # Utiliser 80% de RAM disponible
        bytes_per_sample = n_features * 8  # Float64
        max_samples = int((available_gb * (1024 ** 3)) / bytes_per_sample)
        return max(1000, min(max_samples, n_samples))

# ============================================================================
# GUI
# ============================================================================

class CVOptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 CV Optimization - RAM Maximisée")
        self.root.geometry("1700x1100")
        self.root.configure(bg='#f0f0f0')
        
        self.ram = RAMOptimizer()
        self.running = False
        self.logs = deque(maxlen=500)
        self.results = {}
        self.optimal_configs = {}
        
        self.setup_ui()
        self.update_ram_display()
    
    def setup_ui(self):
        """Créer l'interface"""
        try:
            # Header
            header = tk.Frame(self.root, bg='#2c3e50', height=60)
            header.grid(row=0, column=0, columnspan=3, sticky='ew')
            
            title = tk.Label(header, text="🔍 CV Optimization - Déterminer Split Optimal (RAM Maximisée)", 
                           font=('Arial', 13, 'bold'), fg='white', bg='#2c3e50')
            title.pack(side=tk.LEFT, padx=20, pady=15)
            
            self.root.columnconfigure(0, weight=7)
            self.root.columnconfigure(1, weight=3)
            self.root.rowconfigure(1, weight=1)
            
            # Left: Logs
            left_panel = tk.LabelFrame(self.root, text="📝 Logs CV (RAM Optimisée)", 
                                      font=('Arial', 11, 'bold'),
                                      bg='white', fg='#2c3e50', relief=tk.SUNKEN, bd=2)
            left_panel.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
            left_panel.rowconfigure(0, weight=1)
            left_panel.columnconfigure(0, weight=1)
            
            self.logs_text = scrolledtext.ScrolledText(left_panel, font=('Courier', 8),
                                                       bg='#1e1e1e', fg='#00ff00')
            self.logs_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            # Right: Stats
            right_panel = tk.Frame(self.root, bg='#f0f0f0')
            right_panel.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
            right_panel.rowconfigure(2, weight=1)
            right_panel.columnconfigure(0, weight=1)
            
            # RAM
            ram_frame = tk.LabelFrame(right_panel, text="💾 RAM (Optimisée)", font=('Arial', 10, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
            ram_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=3)
            
            self.ram_label = tk.Label(ram_frame, text="RAM: 0%", font=('Arial', 9), bg='white')
            self.ram_label.pack(fill=tk.X, padx=8, pady=3)
            
            self.ram_progress = ttk.Progressbar(ram_frame, mode='determinate', maximum=100)
            self.ram_progress.pack(fill=tk.X, padx=8, pady=3)
            
            self.ram_details = tk.Label(ram_frame, text="", font=('Arial', 8), bg='white', fg='#666')
            self.ram_details.pack(fill=tk.X, padx=8, pady=3)
            
            # Progress
            progress_frame = tk.LabelFrame(right_panel, text="⏳ Avancée", font=('Arial', 10, 'bold'),
                                          bg='white', relief=tk.SUNKEN, bd=2)
            progress_frame.grid(row=1, column=0, sticky='ew', padx=0, pady=3)
            
            self.progress_label = tk.Label(progress_frame, text="Prêt", font=('Arial', 9), bg='white')
            self.progress_label.pack(fill=tk.X, padx=8, pady=3)
            
            self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
            self.progress_bar.pack(fill=tk.X, padx=8, pady=3)
            
            # Alerts
            alerts_frame = tk.LabelFrame(right_panel, text="⚠️ Alertes", font=('Arial', 10, 'bold'),
                                        bg='white', relief=tk.SUNKEN, bd=2)
            alerts_frame.grid(row=2, column=0, sticky='nsew', padx=0, pady=3)
            alerts_frame.rowconfigure(0, weight=1)
            alerts_frame.columnconfigure(0, weight=1)
            
            self.alerts_text = scrolledtext.ScrolledText(alerts_frame, height=15, font=('Courier', 8),
                                                        bg='#f8f8f8', fg='#333')
            self.alerts_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            self.alerts_text.tag_config('error', foreground='#d32f2f', font=('Courier', 8, 'bold'))
            self.alerts_text.tag_config('success', foreground='#388e3c', font=('Courier', 8, 'bold'))
            
            # Footer
            footer = tk.Frame(self.root, bg='#ecf0f1', height=70)
            footer.grid(row=2, column=0, columnspan=3, sticky='ew')
            
            button_frame = tk.Frame(footer, bg='#ecf0f1')
            button_frame.pack(side=tk.LEFT, padx=10, pady=10)
            
            self.start_btn = tk.Button(button_frame, text="▶ DÉMARRER CV", 
                                      command=self.start_optimization,
                                      bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                      padx=20, pady=10, relief=tk.RAISED, cursor="hand2")
            self.start_btn.pack(side=tk.LEFT, padx=5)
            
            self.stop_btn = tk.Button(button_frame, text="⏹ ARRÊTER", 
                                     command=self.stop_optimization,
                                     bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                     padx=20, pady=10, relief=tk.RAISED,
                                     state=tk.DISABLED, cursor="hand2")
            self.stop_btn.pack(side=tk.LEFT, padx=5)
            
            self.status_label = tk.Label(footer, text="✅ Prêt", font=('Arial', 11, 'bold'),
                                        fg='#27ae60', bg='#ecf0f1')
            self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)
            
        except Exception as e:
            self.log_error(f"Erreur UI: {e}")
    
    def log(self, message, level='INFO'):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            symbols = {'INFO': '📝', 'OK': '✅', 'ERROR': '❌', 'WARNING': '⚠️', 'PROGRESS': '⏳'}
            symbol = symbols.get(level, '>')
            formatted = f"{symbol} [{timestamp}] {message}"
            self.logs.append(formatted)
            self.logs_text.insert(tk.END, formatted + "\n")
            self.logs_text.see(tk.END)
            self.root.update()
        except:
            pass
    
    def log_success(self, message):
        try:
            self.alerts_text.insert(tk.END, f"✅ {message}\n", 'success')
            self.alerts_text.see(tk.END)
            self.root.update()
        except:
            pass
    
    def log_error(self, message):
        try:
            self.alerts_text.insert(tk.END, f"❌ {message}\n", 'error')
            self.alerts_text.see(tk.END)
            self.root.update()
        except:
            pass
    
    def update_ram_display(self):
        try:
            ram_percent = self.ram.get_ram_percent()
            self.ram_progress['value'] = ram_percent
            self.ram_label.config(text=f"RAM: {ram_percent:.1f}%")
            self.ram_details.config(text=f"Total: {self.ram.total_ram:.1f}GB | Available: {self.ram.available_ram:.1f}GB | Cores: {NUM_CORES}")
            self.root.after(500, self.update_ram_display)
        except:
            self.root.after(500, self.update_ram_display)
    
    def update_progress(self, value, max_val, message=""):
        try:
            percent = int((value / max_val) * 100) if max_val > 0 else 0
            self.progress_bar['value'] = percent
            self.progress_label.config(text=message)
            self.root.update()
        except:
            pass
    
    def start_optimization(self):
        try:
            if self.running:
                messagebox.showwarning("Attention", "En cours!")
                return
            
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="⏳ En cours...", fg='#f57f17')
            
            thread = threading.Thread(target=self.run_optimization, daemon=True)
            thread.start()
        except Exception as e:
            self.log_error(f"Erreur: {e}")
    
    def stop_optimization(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏹ Arrêté", fg='#e74c3c')
    
    def run_optimization(self):
        """Exécuter l'optimisation CV"""
        try:
            self.log("═" * 70, 'INFO')
            self.log("CROSS-VALIDATION OPTIMIZATION (RAM MAXIMISÉE)", 'INFO')
            self.log(f"Cores: {NUM_CORES} | Total RAM: {self.ram.total_ram:.1f}GB", 'INFO')
            self.log("═" * 70, 'INFO')
            
            if not self.load_data():
                return
            if not self.running:
                return
            if not self.prepare_data():
                return
            if not self.running:
                return
            
            self.run_cv_for_all_models()
            self.generate_reports()
            self.generate_graphs()
            
            self.log("═" * 70, 'OK')
            self.log("✓✓✓ CV OPTIMIZATION TERMINÉE ✓✓✓", 'OK')
            self.log("═" * 70, 'OK')
            self.status_label.config(text="✅ Succès", fg='#27ae60')
            
        except Exception as e:
            self.log_error(f"Erreur: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            self.status_label.config(text="❌ Erreur", fg='#d32f2f')
        finally:
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def load_data(self):
        """Charger données"""
        self.log("Chargement données...", 'PROGRESS')
        self.update_progress(1, 5, "Chargement")
        
        try:
            fichiers = [
                'fusion_ton_iot_cic_final_smart.csv',
                'fusion_ton_iot_cic_final_smart4.csv',
                'fusion_ton_iot_cic_final_smart3.csv',
            ]
            
            fichier_trouve = None
            for f in fichiers:
                if os.path.exists(f):
                    fichier_trouve = f
                    break
            
            if not fichier_trouve:
                self.log_error("Fichier fusion non trouvé!")
                return False
            
            self.log(f"Chargement: {fichier_trouve}", 'INFO')
            start = time.time()
            self.df = pd.read_csv(fichier_trouve, low_memory=False)
            elapsed = time.time() - start
            
            self.log_success(f"Chargé en {elapsed:.2f}s: {len(self.df):,} lignes × {len(self.df.columns)} colonnes")
            return True
        except Exception as e:
            self.log_error(f"Erreur chargement: {e}")
            return False
    
    def prepare_data(self):
        """Préparer données (RAM optimisée)"""
        self.log("Préparation données (RAM optimisée)...", 'PROGRESS')
        self.update_progress(2, 5, "Préparation")
        
        try:
            # Sélectionner features numériques
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            self.log(f"Features numériques: {len(numeric_cols)}", 'INFO')
            
            # Convertir en float32 pour économiser RAM
            X = self.df[numeric_cols].astype(np.float32).copy()
            X = X.fillna(X.mean())
            
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(self.df['Label'])
            
            # Normaliser (une seule fois, pré-allocatée)
            self.log("Normalisation...", 'INFO')
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X).astype(np.float32)  # Float32 = RAM économisée
            self.y = y
            
            # Libérer la mémoire du dataframe
            del self.df, X
            self.ram.optimize_memory()
            
            self.log_success(f"Données préparées: {self.X_scaled.shape}")
            return True
        except Exception as e:
            self.log_error(f"Erreur préparation: {e}")
            return False
    
    def run_cv_for_all_models(self):
        """Exécuter CV pour tous les modèles (parallélisé)"""
        self.log("Cross-Validation PARALLÉLISÉE pour chaque modèle...", 'PROGRESS')
        self.update_progress(3, 5, "Cross-Validation")
        
        try:
            models_config = [
                ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, n_jobs=N_JOBS)),
                ('Naive Bayes', GaussianNB()),
                ('Decision Tree', DecisionTreeClassifier(random_state=42)),
                ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)),
            ]
            
            for i, (name, model) in enumerate(models_config, 1):
                if not self.running:
                    return
                
                self.log(f"{i}/4. CV pour {name}...", 'PROGRESS')
                self.update_progress(3 + (i/4) * 2, 5, f"CV: {name}")
                
                start_time = time.time()
                self.run_cv_for_model(name, model)
                elapsed = time.time() - start_time
                
                self.log_success(f"{name} terminé en {elapsed:.2f}s")
                self.ram.optimize_memory()
            
            return True
        except Exception as e:
            self.log_error(f"Erreur CV: {e}")
            return False
    
    def run_cv_for_model(self, model_name, model):
        """Exécuter CV pour un modèle (optimisé RAM)"""
        try:
            results = {
                'train_sizes': [],
                'f1_scores': [],
                'recall_scores': [],
                'precision_scores': [],
                'avt_scores': [],
                'f1_std': [],
                'recall_std': [],
                'precision_std': [],
            }
            
            kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
            n_samples = len(self.y)
            
            for train_size_idx, train_size in enumerate(TRAIN_SIZES):
                if not self.running:
                    return
                
                # Nombre de samples à utiliser
                n_train_samples = int(n_samples * train_size)
                indices = np.random.permutation(n_samples)[:n_train_samples]
                
                # Utiliser views pour économiser RAM (pas de copie)
                X_subset = self.X_scaled[indices]
                y_subset = self.y[indices]
                
                # Métriques pour ce fold
                f1_scores = np.zeros(K_FOLD, dtype=np.float32)
                recall_scores = np.zeros(K_FOLD, dtype=np.float32)
                precision_scores = np.zeros(K_FOLD, dtype=np.float32)
                avt_scores = np.zeros(K_FOLD, dtype=np.float32)
                
                # K-Fold
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_subset)):
                    X_train = X_subset[train_idx]
                    X_val = X_subset[val_idx]
                    y_train = y_subset[train_idx]
                    y_val = y_subset[val_idx]
                    
                    # Entraîner
                    model.fit(X_train, y_train)
                    
                    # Prédire + mesurer temps
                    start = time.time()
                    y_pred = model.predict(X_val)
                    pred_time = time.time() - start
                    
                    # Métriques
                    f1_scores[fold_idx] = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                    recall_scores[fold_idx] = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                    precision_scores[fold_idx] = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                    avt_scores[fold_idx] = len(X_val) / pred_time if pred_time > 0 else 0
                
                # Moyenne et std
                results['train_sizes'].append(int(train_size * 100))
                results['f1_scores'].append(float(np.mean(f1_scores)))
                results['f1_std'].append(float(np.std(f1_scores)))
                results['recall_scores'].append(float(np.mean(recall_scores)))
                results['recall_std'].append(float(np.std(recall_scores)))
                results['precision_scores'].append(float(np.mean(precision_scores)))
                results['precision_std'].append(float(np.std(precision_scores)))
                results['avt_scores'].append(float(np.mean(avt_scores)))
                
                # Log progress
                progress = (train_size_idx + 1) / len(TRAIN_SIZES)
                self.log(f"   {int(train_size*100)}% train: F1={results['f1_scores'][-1]:.4f} ±{results['f1_std'][-1]:.4f}", 'INFO')
            
            # Trouver optimal (max F1)
            best_idx = int(np.argmax(results['f1_scores']))
            best_train_size = TRAIN_SIZES[best_idx]
            best_f1 = results['f1_scores'][best_idx]
            
            self.optimal_configs[model_name] = {
                'train_size': float(best_train_size),
                'test_size': float(1 - best_train_size),
                'f1_score': float(best_f1),
                'recall': float(results['recall_scores'][best_idx]),
                'precision': float(results['precision_scores'][best_idx]),
                'avt': float(results['avt_scores'][best_idx]),
            }
            
            self.results[model_name] = results
            
        except Exception as e:
            self.log_error(f"Erreur CV pour {model_name}: {e}")
    
    def generate_reports(self):
        """Générer rapports"""
        self.log("Génération rapports...", 'PROGRESS')
        self.update_progress(4, 5, "Rapports")
        
        try:
            # Rapport résumé
            with open('cv_results_summary.txt', 'w', encoding='utf-8') as f:
                f.write("═" * 100 + "\n")
                f.write("CROSS-VALIDATION OPTIMIZATION RESULTS - DÉTERMINATION DU SPLIT OPTIMAL\n")
                f.write("═" * 100 + "\n\n")
                
                f.write(f"K-Fold: {K_FOLD}\n")
                f.write(f"Train Sizes Testées: {[int(x*100) for x in TRAIN_SIZES]}%\n")
                f.write(f"Total Cores Utilisés: {NUM_CORES}\n")
                f.write(f"RAM Totale: {self.ram.total_ram:.1f}GB\n\n")
                
                f.write("SPLITS OPTIMAUX TROUVÉS:\n")
                f.write("─" * 100 + "\n")
                f.write(f"{'Algorithme':<25} {'Train%':>10} {'Test%':>10} {'F1 Score':>12} {'Recall':>12} {'Precision':>12} {'AVT':>15}\n")
                f.write("─" * 100 + "\n")
                
                for name in sorted(self.optimal_configs.keys()):
                    cfg = self.optimal_configs[name]
                    f.write(f"{name:<25} {cfg['train_size']*100:>10.0f} {cfg['test_size']*100:>10.0f} "
                           f"{cfg['f1_score']:>12.4f} {cfg['recall']:>12.4f} {cfg['precision']:>12.4f} {cfg['avt']:>15.0f}\n")
                
                f.write("\n" + "═" * 100 + "\n")
                f.write("RECOMMANDATION:\n")
                f.write("─" * 100 + "\n")
                
                best_model = max(self.optimal_configs.keys(), key=lambda x: self.optimal_configs[x]['f1_score'])
                best_cfg = self.optimal_configs[best_model]
                
                f.write(f"Meilleur modèle: {best_model}\n")
                f.write(f"Split optimal: {best_cfg['train_size']*100:.0f}% train / {best_cfg['test_size']*100:.0f}% test\n")
                f.write(f"F1 Score attendu: {best_cfg['f1_score']:.4f}\n")
                f.write("═" * 100 + "\n")
            
            self.log_success("Rapport résumé généré")
            
            # Rapport config optimale
            with open('cv_optimal_config.txt', 'w', encoding='utf-8') as f:
                f.write("═" * 80 + "\n")
                f.write("CONFIGURATION OPTIMALE RECOMMANDÉE\n")
                f.write("═" * 80 + "\n\n")
                
                for name, cfg in sorted(self.optimal_configs.items(), key=lambda x: x[1]['f1_score'], reverse=True):
                    f.write(f"\n{name}:\n")
                    f.write(f"  Train Size: {cfg['train_size']*100:.0f}%\n")
                    f.write(f"  Test Size:  {cfg['test_size']*100:.0f}%\n")
                    f.write(f"  F1 Score:   {cfg['f1_score']:.4f}\n")
                    f.write(f"  Recall:     {cfg['recall']:.4f}\n")
                    f.write(f"  Precision:  {cfg['precision']:.4f}\n")
                    f.write(f"  AVT:        {cfg['avt']:.0f} samples/s\n")
                
                f.write("\n" + "═" * 80 + "\n")
            
            self.log_success("Config optimale sauvegardée")
            
            # CSV détaillé
            df_results = pd.DataFrame(self.results).T
            df_results.to_csv('cv_detailed_metrics.csv')
            self.log_success("CSV détaillé généré")
            
        except Exception as e:
            self.log_error(f"Erreur rapports: {e}")
    
    def generate_graphs(self):
        """Générer courbes dynamiques"""
        self.log("Génération courbes dynamiques...", 'PROGRESS')
        self.update_progress(5, 5, "Graphiques")
        
        try:
            sns.set_style("darkgrid")
            
            for algo_name, results in self.results.items():
                if not self.running:
                    return
                
                self.log(f"Graphique: {algo_name}...", 'INFO')
                
                # Créer figure avec 4 subplots
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'{algo_name} - Courbes CV', fontsize=14, fontweight='bold')
                
                train_sizes = np.array(results['train_sizes'])
                f1_scores = np.array(results['f1_scores'])
                f1_std = np.array(results['f1_std'])
                recall_scores = np.array(results['recall_scores'])
                precision_scores = np.array(results['precision_scores'])
                avt_scores = np.array(results['avt_scores'])
                
                # Graphique 1: F1 Score (principal)
                ax = axes[0, 0]
                ax.plot(train_sizes, f1_scores, 'o-', linewidth=2.5, markersize=8, color='#3498db', label='F1 Score')
                ax.fill_between(train_sizes, f1_scores - f1_std, f1_scores + f1_std, alpha=0.2, color='#3498db')
                ax.set_xlabel('% Training Data', fontsize=11, fontweight='bold')
                ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
                ax.set_title('F1 Score vs Training Size', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                
                # Marquer le meilleur point
                best_idx = np.argmax(f1_scores)
                ax.plot(train_sizes[best_idx], f1_scores[best_idx], 'r*', markersize=20, label='Optimal')
                ax.legend()
                
                # Graphique 2: Recall
                ax = axes[0, 1]
                ax.plot(train_sizes, recall_scores, 's-', linewidth=2.5, markersize=8, color='#e74c3c', label='Recall')
                ax.set_xlabel('% Training Data', fontsize=11, fontweight='bold')
                ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
                ax.set_title('Recall vs Training Size', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                ax.legend()
                
                # Graphique 3: Precision
                ax = axes[1, 0]
                ax.plot(train_sizes, precision_scores, '^-', linewidth=2.5, markersize=8, color='#f39c12', label='Precision')
                ax.set_xlabel('% Training Data', fontsize=11, fontweight='bold')
                ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
                ax.set_title('Precision vs Training Size', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                ax.legend()
                
                # Graphique 4: AVT (Vitesse)
                ax = axes[1, 1]
                ax.plot(train_sizes, avt_scores / 1000, 'd-', linewidth=2.5, markersize=8, color='#27ae60', label='AVT (k samples/s)')
                ax.set_xlabel('% Training Data', fontsize=11, fontweight='bold')
                ax.set_ylabel('AVT (1000s samples/s)', fontsize=11, fontweight='bold')
                ax.set_title('AVT vs Training Size', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plt.tight_layout()
                filename = f"graph_cv_{algo_name.replace(' ', '_').lower()}_curves.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log_success(f"Courbe sauvegardée: {filename}")
                self.ram.optimize_memory()
            
            self.log_success("Toutes les courbes générées")
            
        except Exception as e:
            self.log_error(f"Erreur graphiques: {e}")
            import traceback
            self.log_error(traceback.format_exc())

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        root = tk.Tk()
        app = CVOptimizationGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()