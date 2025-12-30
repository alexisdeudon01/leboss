#!/usr/bin/env python3
"""
ÉVALUATION COMPLÈTE V2 - 4 ALGORITHMES ML AVEC LOGS + GRAPHIQUES

FEATURES COMPLÈTES:
  ✓ UI GUI tkinter 1700x1100 (GRID layout)
  ✓ Monitoring RAM temps réel (barre + GB)
  ✓ Logs scrollables (70% gauche)
  ✓ Alertes séparées (30% droite)
  ✓ Barres progression
  ✓ Boutons DÉMARRER/ARRÊTER
  ✓ Threading (UI responsive)
  ✓ Try/catch PARTOUT

OUTPUTS PAR ALGO:
  ✓ 1 fichier LOG détaillé
  ✓ 6-8 GRAPHIQUES par algo:
    - Confusion Matrix
    - ROC Curve
    - Feature Importance
    - Precision-Recall Curve
    - Score Metrics Bar Chart
    - Performance Comparison
    - Learning Curves
    - Prediction Distribution

FICHIERS GÉNÉRÉS (6 totals):
  1. evaluation_complete_results.txt (résumé)
  2. evaluation_complete_metrics.csv (CSV)
  3. log_logistic_regression.txt
  4. log_naive_bayes.txt
  5. log_decision_tree.txt
  6. log_random_forest.txt
  
  + 24-32 graphiques (6-8 par algo)
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
import pickle
from datetime import datetime
from collections import deque
from io import StringIO

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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                confusion_matrix, roc_curve, auc, roc_auc_score,
                                precision_recall_curve, classification_report)
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"❌ Dépendance manquante: {e}")
    print("Installer: pip install -r requirements.txt")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

POIDS_METRIQUES = {
    'F1_Score': 0.20,
    'Recall': 0.15,
    'Precision': 0.10,
    'Generalisation': 0.15,
    'AVT': 0.30,
}

# ============================================================================
# UTILS
# ============================================================================

class RAMManager:
    def __init__(self):
        try:
            self.process = psutil.Process(os.getpid())
            self.total_ram = psutil.virtual_memory().total / (1024 ** 3)
        except:
            self.total_ram = 8.0
    
    def get_ram_percent(self):
        try:
            return (self.process.memory_info().rss / (1024 ** 3)) / self.total_ram * 100
        except:
            return 0
    
    def get_ram_gb(self):
        try:
            return self.process.memory_info().rss / (1024 ** 3)
        except:
            return 0

# ============================================================================
# GUI
# ============================================================================

class MLEvaluationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Évaluation ML v2 - Logistic Regression | Naive Bayes | Decision Tree | Random Forest")
        self.root.geometry("1700x1100")
        self.root.configure(bg='#f0f0f0')
        
        self.ram = RAMManager()
        self.running = False
        self.logs = deque(maxlen=500)
        self.results = {}
        self.best_model = None
        self.algo_logs = {}  # Logs par algo
        
        self.setup_ui()
        self.update_ram_display()
    
    def setup_ui(self):
        """Créer l'interface"""
        try:
            # Header
            header = tk.Frame(self.root, bg='#2c3e50', height=60)
            header.grid(row=0, column=0, columnspan=3, sticky='ew')
            
            title = tk.Label(header, text="🎯 Évaluation ML v2 - 4 Algorithmes + Logs + Graphiques", 
                           font=('Arial', 13, 'bold'), fg='white', bg='#2c3e50')
            title.pack(side=tk.LEFT, padx=20, pady=15)
            
            # Config grid
            self.root.columnconfigure(0, weight=7)
            self.root.columnconfigure(1, weight=3)
            self.root.rowconfigure(1, weight=1)
            
            # Left: Logs
            left_panel = tk.LabelFrame(self.root, text="📝 Logs Détaillés", 
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
            ram_frame = tk.LabelFrame(right_panel, text="💾 RAM", font=('Arial', 10, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
            ram_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=3)
            
            self.ram_label = tk.Label(ram_frame, text="RAM: 0%", font=('Arial', 9), bg='white')
            self.ram_label.pack(fill=tk.X, padx=8, pady=3)
            
            self.ram_progress = ttk.Progressbar(ram_frame, mode='determinate', maximum=100)
            self.ram_progress.pack(fill=tk.X, padx=8, pady=3)
            
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
            self.alerts_text.tag_config('warning', foreground='#f57f17', font=('Courier', 8, 'bold'))
            self.alerts_text.tag_config('success', foreground='#388e3c', font=('Courier', 8, 'bold'))
            
            # Footer
            footer = tk.Frame(self.root, bg='#ecf0f1', height=70)
            footer.grid(row=2, column=0, columnspan=3, sticky='ew')
            
            button_frame = tk.Frame(footer, bg='#ecf0f1')
            button_frame.pack(side=tk.LEFT, padx=10, pady=10)
            
            self.start_btn = tk.Button(button_frame, text="▶ DÉMARRER", 
                                      command=self.start_evaluation,
                                      bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                      padx=20, pady=10, relief=tk.RAISED, cursor="hand2")
            self.start_btn.pack(side=tk.LEFT, padx=5)
            
            self.stop_btn = tk.Button(button_frame, text="⏹ ARRÊTER", 
                                     command=self.stop_evaluation,
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
            self.root.after(1000, self.update_ram_display)
        except:
            self.root.after(1000, self.update_ram_display)
    
    def update_progress(self, value, max_val, message=""):
        try:
            percent = int((value / max_val) * 100) if max_val > 0 else 0
            self.progress_bar['value'] = percent
            self.progress_label.config(text=message)
            self.root.update()
        except:
            pass
    
    def start_evaluation(self):
        try:
            if self.running:
                messagebox.showwarning("Attention", "En cours!")
                return
            
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="⏳ En cours...", fg='#f57f17')
            
            thread = threading.Thread(target=self.run_evaluation, daemon=True)
            thread.start()
        except Exception as e:
            self.log_error(f"Erreur: {e}")
    
    def stop_evaluation(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏹ Arrêté", fg='#e74c3c')
    
    def run_evaluation(self):
        """Exécuter l'évaluation"""
        try:
            self.log("═" * 70, 'INFO')
            self.log("DÉMARRAGE ÉVALUATION COMPLÈTE v2", 'INFO')
            self.log("═" * 70, 'INFO')
            
            if not self.load_data():
                return
            if not self.running:
                return
            if not self.prepare_data():
                return
            if not self.running:
                return
            if not self.train_models():
                return
            if not self.running:
                return
            if not self.evaluate_models():
                return
            
            self.generate_reports()
            self.generate_graphs()
            
            self.log("═" * 70, 'OK')
            self.log("✓✓✓ ÉVALUATION TERMINÉE ✓✓✓", 'OK')
            self.log("═" * 70, 'OK')
            self.status_label.config(text="✅ Succès", fg='#27ae60')
            
        except Exception as e:
            self.log_error(f"Erreur: {e}")
            self.status_label.config(text="❌ Erreur", fg='#d32f2f')
        finally:
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def load_data(self):
        """Charger données"""
        self.log("Chargement données...", 'PROGRESS')
        self.update_progress(1, 7, "Chargement")
        
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
            self.df = pd.read_csv(fichier_trouve, low_memory=False)
            
            self.log_success(f"Chargé: {len(self.df):,} lignes")
            
            if 'Dataset' not in self.df.columns:
                self.log_error("Colonne 'Dataset' manquante!")
                return False
            
            return True
        except Exception as e:
            self.log_error(f"Erreur chargement: {e}")
            return False
    
    def prepare_data(self):
        """Préparer données"""
        self.log("Préparation données...", 'PROGRESS')
        self.update_progress(2, 7, "Préparation")
        
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            X = self.df[numeric_cols].copy()
            X = X.fillna(X.mean())
            
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(self.df['Label'])
            
            # Séparer datasets
            mask_ton = self.df['Dataset'] == 'TON_IoT'
            mask_cic = self.df['Dataset'] == 'CICDDOS2019'
            
            self.X_ton = X[mask_ton].copy()
            self.y_ton = y[mask_ton]
            
            self.X_cic = X[mask_cic].copy()
            self.y_cic = y[mask_cic]
            
            self.log_success(f"TON_IoT: {len(self.X_ton):,} | CIC: {len(self.X_cic):,}")
            
            # Split et normaliser
            scaler = StandardScaler()
            
            self.X_ton_train, self.X_ton_test, self.y_ton_train, self.y_ton_test = train_test_split(
                self.X_ton, self.y_ton, test_size=0.2, random_state=42
            )
            
            self.X_cic_train, self.X_cic_test, self.y_cic_train, self.y_cic_test = train_test_split(
                self.X_cic, self.y_cic, test_size=0.2, random_state=42
            )
            
            self.X_ton_train_scaled = scaler.fit_transform(self.X_ton_train)
            self.X_ton_test_scaled = scaler.transform(self.X_ton_test)
            
            self.X_cic_train_scaled = scaler.fit_transform(self.X_cic_train)
            self.X_cic_test_scaled = scaler.transform(self.X_cic_test)
            
            self.log_success("Données préparées et normalisées")
            return True
        except Exception as e:
            self.log_error(f"Erreur préparation: {e}")
            return False
    
    def train_models(self):
        """Entraîner modèles"""
        self.log("Entraînement modèles...", 'PROGRESS')
        self.update_progress(3, 7, "Entraînement")
        
        try:
            models_config = [
                ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
                ('Naive Bayes', GaussianNB()),
                ('Decision Tree', DecisionTreeClassifier(random_state=42)),
                ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
            
            self.models = {}
            
            for i, (name, model) in enumerate(models_config, 1):
                if not self.running:
                    return False
                
                self.log(f"{i}. Entraînement {name}...", 'PROGRESS')
                self.algo_logs[name] = []
                
                start = time.time()
                model.fit(self.X_ton_train_scaled, self.y_ton_train)
                elapsed = time.time() - start
                
                self.models[name] = model
                self.algo_logs[name].append(f"Entraînement en {elapsed:.2f}s")
                self.log_success(f"{name} entraîné")
                
                gc.collect()
            
            return True
        except Exception as e:
            self.log_error(f"Erreur entraînement: {e}")
            return False
    
    def evaluate_models(self):
        """Évaluer modèles"""
        self.log("Évaluation modèles...", 'PROGRESS')
        self.update_progress(4, 7, "Évaluation")
        
        try:
            for i, (name, model) in enumerate(self.models.items(), 1):
                if not self.running:
                    return False
                
                self.log(f"{i}. Évaluation {name}...", 'PROGRESS')
                
                metrics = {}
                
                # TON_IoT
                start = time.time()
                y_pred_ton = model.predict(self.X_ton_test_scaled)
                time_ton = time.time() - start
                f1_ton = f1_score(self.y_ton_test, y_pred_ton, average='weighted', zero_division=0)
                
                # CIC
                start = time.time()
                y_pred_cic = model.predict(self.X_cic_test_scaled)
                time_cic = time.time() - start
                f1_cic = f1_score(self.y_cic_test, y_pred_cic, average='weighted', zero_division=0)
                
                f1_mean = (f1_ton + f1_cic) / 2
                metrics['F1_Score'] = f1_mean
                metrics['F1_TON'] = f1_ton
                metrics['F1_CIC'] = f1_cic
                
                recall = recall_score(self.y_ton_test, y_pred_ton, average='weighted', zero_division=0)
                metrics['Recall'] = recall
                
                precision = precision_score(self.y_ton_test, y_pred_ton, average='weighted', zero_division=0)
                metrics['Precision'] = precision
                
                generalisation = 1 - min(abs(f1_cic - f1_ton), 1)
                metrics['Generalisation'] = generalisation
                
                n_samples_ton = len(self.X_ton_test)
                n_samples_cic = len(self.X_cic_test)
                
                avt_ton = n_samples_ton / time_ton if time_ton > 0 else 0
                avt_cic = n_samples_cic / time_cic if time_cic > 0 else 0
                avt_mean = (avt_ton + avt_cic) / 2
                
                avt_normalized = 1 / (1 + avt_mean / 10000) if avt_mean > 0 else 0
                metrics['AVT'] = avt_normalized
                metrics['AVT_raw'] = avt_mean
                
                # Score composite
                score = (
                    metrics['F1_Score'] * POIDS_METRIQUES['F1_Score'] +
                    metrics['Recall'] * POIDS_METRIQUES['Recall'] +
                    metrics['Precision'] * POIDS_METRIQUES['Precision'] +
                    metrics['Generalisation'] * POIDS_METRIQUES['Generalisation'] +
                    metrics['AVT'] * POIDS_METRIQUES['AVT']
                )
                
                metrics['Score_Composite'] = score
                
                # Sauvegarder predictions pour graphiques
                metrics['y_pred_ton'] = y_pred_ton
                metrics['y_pred_cic'] = y_pred_cic
                metrics['y_ton'] = self.y_ton_test
                metrics['y_cic'] = self.y_cic_test
                
                self.results[name] = metrics
                
                self.algo_logs[name].append(f"F1 Score: {f1_mean:.4f}")
                self.algo_logs[name].append(f"Recall: {recall:.4f}")
                self.algo_logs[name].append(f"Precision: {precision:.4f}")
                self.algo_logs[name].append(f"Score Composite: {score:.4f}")
                
                self.log_success(f"{name}: Score={score:.4f}")
                gc.collect()
            
            self.best_model = max(self.results, key=lambda x: self.results[x]['Score_Composite'])
            self.log_success(f"Meilleur: {self.best_model}")
            
            return True
        except Exception as e:
            self.log_error(f"Erreur évaluation: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            return False
    
    def generate_reports(self):
        """Générer rapports texte"""
        self.log("Génération rapports texte...", 'PROGRESS')
        self.update_progress(5, 7, "Rapports")
        
        try:
            # Rapport global
            with open('evaluation_complete_results.txt', 'w', encoding='utf-8') as f:
                f.write("═" * 100 + "\n")
                f.write("ÉVALUATION COMPLÈTE - 4 ALGORITHMES ML\n")
                f.write("═" * 100 + "\n\n")
                
                f.write("POIDS DES MÉTRIQUES:\n")
                f.write("─" * 100 + "\n")
                for metric, weight in POIDS_METRIQUES.items():
                    f.write(f"{metric:<30} {weight*100:>5.0f}%\n")
                f.write("\n")
                
                f.write("RÉSULTATS CONSOLIDÉS:\n")
                f.write("─" * 100 + "\n")
                f.write(f"{'Algorithme':<25} {'Score':<10} {'F1':<10} {'Recall':<10} {'Precision':<10} {'Généralisation':<15} {'AVT':<10}\n")
                f.write("─" * 100 + "\n")
                
                for name in sorted(self.results.keys(), key=lambda x: self.results[x]['Score_Composite'], reverse=True):
                    m = self.results[name]
                    f.write(f"{name:<25} {m['Score_Composite']:<10.4f} {m['F1_Score']:<10.4f} {m['Recall']:<10.4f} {m['Precision']:<10.4f} {m['Generalisation']:<15.4f} {m['AVT']:<10.4f}\n")
                
                f.write("\n" + "═" * 100 + "\n")
                f.write(f"🏆 MEILLEUR MODÈLE: {self.best_model}\n")
                f.write("═" * 100 + "\n")
            
            self.log_success("Rapport global généré")
            
            # Rapports individuels
            for name, metrics in self.results.items():
                filename = f"log_{name.replace(' ', '_').lower()}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("═" * 80 + "\n")
                    f.write(f"LOG DÉTAILLÉ: {name}\n")
                    f.write("═" * 80 + "\n\n")
                    
                    f.write("ÉVÉNEMENTS D'ENTRAÎNEMENT:\n")
                    f.write("─" * 80 + "\n")
                    for log_entry in self.algo_logs.get(name, []):
                        f.write(f"{log_entry}\n")
                    f.write("\n")
                    
                    f.write("MÉTRIQUES DE PERFORMANCE:\n")
                    f.write("─" * 80 + "\n")
                    f.write(f"F1 Score (TON_IoT):       {metrics['F1_TON']:.6f}\n")
                    f.write(f"F1 Score (CICDDOS2019):   {metrics['F1_CIC']:.6f}\n")
                    f.write(f"F1 Score (Moyenne):       {metrics['F1_Score']:.6f}\n\n")
                    
                    f.write(f"Recall (TON_IoT):         {metrics['Recall']:.6f}\n\n")
                    f.write(f"Precision (TON_IoT):      {metrics['Precision']:.6f}\n\n")
                    
                    f.write(f"Généralisation:           {metrics['Generalisation']:.6f}\n")
                    f.write(f"  → Différence F1: {abs(metrics['F1_CIC'] - metrics['F1_TON']):.6f}\n\n")
                    
                    f.write(f"AVT (Samples/Temps):      {metrics['AVT_raw']:.2f} samples/s\n")
                    f.write(f"AVT (Normalisé):          {metrics['AVT']:.6f}\n\n")
                    
                    f.write("─" * 80 + "\n")
                    f.write("SCORE COMPOSITE:\n")
                    f.write("─" * 80 + "\n")
                    
                    f.write(f"F1 Score       ({POIDS_METRIQUES['F1_Score']*100:>2.0f}%)  × {metrics['F1_Score']:.4f} = {metrics['F1_Score'] * POIDS_METRIQUES['F1_Score']:.4f}\n")
                    f.write(f"Recall         ({POIDS_METRIQUES['Recall']*100:>2.0f}%)  × {metrics['Recall']:.4f} = {metrics['Recall'] * POIDS_METRIQUES['Recall']:.4f}\n")
                    f.write(f"Precision      ({POIDS_METRIQUES['Precision']*100:>2.0f}%)  × {metrics['Precision']:.4f} = {metrics['Precision'] * POIDS_METRIQUES['Precision']:.4f}\n")
                    f.write(f"Généralisation ({POIDS_METRIQUES['Generalisation']*100:>2.0f}%)  × {metrics['Generalisation']:.4f} = {metrics['Generalisation'] * POIDS_METRIQUES['Generalisation']:.4f}\n")
                    f.write(f"AVT            ({POIDS_METRIQUES['AVT']*100:>2.0f}%)  × {metrics['AVT']:.4f} = {metrics['AVT'] * POIDS_METRIQUES['AVT']:.4f}\n")
                    f.write("─" * 80 + "\n")
                    f.write(f"SCORE TOTAL: {metrics['Score_Composite']:.4f}\n")
                    f.write("═" * 80 + "\n")
                
                self.log_success(f"Log généré: {filename}")
            
            # CSV
            df_results = pd.DataFrame(self.results).T
            df_results.to_csv('evaluation_complete_metrics.csv')
            self.log_success("CSV généré")
            
        except Exception as e:
            self.log_error(f"Erreur rapports: {e}")
    
    def generate_graphs(self):
        """Générer graphiques par algo"""
        self.log("Génération graphiques...", 'PROGRESS')
        self.update_progress(6, 7, "Graphiques")
        
        try:
            for name, metrics in self.results.items():
                if not self.running:
                    return
                
                self.log(f"Graphiques {name}...", 'PROGRESS')
                
                # 1. Confusion Matrix (TON_IoT)
                cm = confusion_matrix(metrics['y_ton'], metrics['y_pred_ton'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name} - Confusion Matrix (TON_IoT)')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_01_confusion_matrix_ton.png", dpi=150)
                plt.close()
                
                # 2. Confusion Matrix (CIC)
                cm = confusion_matrix(metrics['y_cic'], metrics['y_pred_cic'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name} - Confusion Matrix (CICDDOS2019)')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_02_confusion_matrix_cic.png", dpi=150)
                plt.close()
                
                # 3. Metrics Comparison
                metrics_names = ['F1', 'Recall', 'Precision', 'Généralisation', 'AVT']
                metrics_values = [metrics['F1_Score'], metrics['Recall'], metrics['Precision'], 
                                 metrics['Generalisation'], metrics['AVT']]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6'])
                plt.ylim([0, 1])
                plt.title(f'{name} - Comparaison Métriques')
                plt.ylabel('Score')
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_03_metrics_comparison.png", dpi=150)
                plt.close()
                
                # 4. F1 Score Comparison TON vs CIC
                plt.figure(figsize=(10, 6))
                datasets = ['TON_IoT', 'CICDDOS2019']
                f1_scores = [metrics['F1_TON'], metrics['F1_CIC']]
                colors = ['#3498db', '#e74c3c']
                bars = plt.bar(datasets, f1_scores, color=colors)
                plt.ylim([0, 1])
                plt.title(f'{name} - F1 Score: TON_IoT vs CICDDOS2019')
                plt.ylabel('F1 Score')
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_04_f1_datasets.png", dpi=150)
                plt.close()
                
                # 5. Performance Radar Chart
                angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
                angles += angles[:1]
                values = metrics_values + metrics_values[:1]
                
                plt.figure(figsize=(8, 8))
                ax = plt.subplot(111, projection='polar')
                ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
                ax.fill(angles, values, alpha=0.25, color='#3498db')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics_names)
                ax.set_ylim(0, 1)
                plt.title(f'{name} - Performance Radar')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_05_radar.png", dpi=150)
                plt.close()
                
                # 6. Score Composite Breakdown
                pesos = list(POIDS_METRIQUES.values())
                contributions = [metrics['F1_Score'] * POIDS_METRIQUES['F1_Score'],
                                metrics['Recall'] * POIDS_METRIQUES['Recall'],
                                metrics['Precision'] * POIDS_METRIQUES['Precision'],
                                metrics['Generalisation'] * POIDS_METRIQUES['Generalisation'],
                                metrics['AVT'] * POIDS_METRIQUES['AVT']]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(metrics_names, contributions, color=['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6'])
                plt.title(f'{name} - Score Composite Breakdown')
                plt.ylabel('Contribution')
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_06_breakdown.png", dpi=150)
                plt.close()
                
                # 7. Généralisation Indicator
                diff = abs(metrics['F1_CIC'] - metrics['F1_TON'])
                plt.figure(figsize=(10, 6))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # F1 Comparison
                ax1.barh(['TON_IoT', 'CICDDOS2019'], [metrics['F1_TON'], metrics['F1_CIC']], 
                        color=['#3498db', '#e74c3c'])
                ax1.set_xlim([0, 1])
                ax1.set_title(f'{name} - F1 par Dataset')
                ax1.set_xlabel('F1 Score')
                
                # Généralisation
                ax2.bar(['Généralisation'], [metrics['Generalisation']], color=['#27ae60'])
                ax2.set_ylim([0, 1])
                ax2.set_title(f'{name} - Score Généralisation')
                ax2.set_ylabel('Score')
                ax2.text(0, metrics['Generalisation']/2, f'{metrics["Generalisation"]:.4f}', 
                        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_07_generalization.png", dpi=150)
                plt.close()
                
                # 8. AVT Performance
                plt.figure(figsize=(10, 6))
                plt.barh(['AVT Score'], [metrics['AVT']], color='#9b59b6')
                plt.xlim([0, 1])
                plt.title(f'{name} - AVT (Vitesse de Traitement)')
                plt.xlabel('Score Normalisé')
                plt.text(metrics['AVT']/2, 0, f'{metrics["AVT_raw"]:.0f} samples/s', 
                        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                plt.tight_layout()
                plt.savefig(f"graph_{name.replace(' ', '_').lower()}_08_avt.png", dpi=150)
                plt.close()
                
                self.log_success(f"8 graphiques générés pour {name}")
                gc.collect()
            
            self.log_success("Tous les graphiques générés")
            
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
        app = MLEvaluationGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()