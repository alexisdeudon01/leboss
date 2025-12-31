#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV OPTIMIZATION V3 - AMÉLIORÉ
======================================
✅ Grid Search: Hyperparamètres variables
✅ Graphiques scrollables (paramètres vs scores)
✅ Gestion RAM dynamique (<90%)
✅ Tkinter GUI avancée
✅ Visualisation complète résultats
======================================
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
from datetime import datetime, timedelta
from itertools import product

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


try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score, recall_score, precision_score
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

# GRID SEARCH CONFIGURATION
PARAM_GRIDS = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 2000],
        'penalty': ['l2'],
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
    }
}


class MemoryManager:
    """Gère la mémoire dynamiquement"""
    
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
    def get_optimal_chunk_size(total_size=None, min_chunk=100000, max_chunk=1000000):
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


class CVOptimizationGUI:
    """Interface Tkinter avancée avec graphiques scrollables"""
    
    def __init__(self, root):
        self.root = root
        self.root.title('CV Optimization V3 - Grid Search')
        self.root.geometry('1600x1000')
        self.root.configure(bg='#f0f0f0')
        
        self.running = False
        self.results = {}
        self.optimal_configs = {}
        self.start_time = None
        self.completed_operations = 0
        self.total_operations = 0
        
        self.df = None
        self.X_scaled = None
        self.y = None
        self.label_encoder = None
        
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        header = tk.Frame(self.root, bg='#2c3e50', height=50)
        header.grid(row=0, column=0, sticky='ew')
        tk.Label(header, text='CV Optimization V3 - Grid Search Hyperparamètres',
                 font=('Arial', 12, 'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=20, pady=12)
        
        container = tk.Frame(self.root, bg='#f0f0f0')
        container.grid(row=1, column=0, sticky='nsew', padx=8, pady=8)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=2)
        container.columnconfigure(1, weight=1)
        
        live_frame = tk.LabelFrame(container, text='LIVE Output',
                                   font=('Arial', 10, 'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        live_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=0)
        live_frame.rowconfigure(0, weight=1)
        live_frame.columnconfigure(0, weight=1)
        self.live_text = scrolledtext.ScrolledText(live_frame, font=('Courier', 8),
                                                   bg='#1a1a1a', fg='#00ff00')
        self.live_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        stats_frame = tk.Frame(container, bg='#f0f0f0')
        stats_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=0)
        stats_frame.rowconfigure(5, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        
        ram_frame = tk.LabelFrame(stats_frame, text='RAM', font=('Arial', 9, 'bold'),
                                  bg='white', relief=tk.SUNKEN, bd=2)
        ram_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=3)
        self.ram_label = tk.Label(ram_frame, text='0%', font=('Arial', 10, 'bold'), bg='white', fg='#e74c3c')
        self.ram_label.pack(fill=tk.X, padx=8, pady=3)
        self.ram_progress = ttk.Progressbar(ram_frame, mode='determinate', maximum=100)
        self.ram_progress.pack(fill=tk.X, padx=8, pady=3)
        
        cpu_frame = tk.LabelFrame(stats_frame, text='CPU', font=('Arial', 9, 'bold'),
                                  bg='white', relief=tk.SUNKEN, bd=2)
        cpu_frame.grid(row=1, column=0, sticky='ew', padx=0, pady=3)
        self.cpu_label = tk.Label(cpu_frame, text='0%', font=('Arial', 10, 'bold'), bg='white', fg='#3498db')
        self.cpu_label.pack(fill=tk.X, padx=8, pady=3)
        self.cpu_progress = ttk.Progressbar(cpu_frame, mode='determinate', maximum=100)
        self.cpu_progress.pack(fill=tk.X, padx=8, pady=3)
        
        progress_frame = tk.LabelFrame(stats_frame, text='Avancée', font=('Arial', 9, 'bold'),
                                       bg='white', relief=tk.SUNKEN, bd=2)
        progress_frame.grid(row=2, column=0, sticky='ew', padx=0, pady=3)
        self.progress_label = tk.Label(progress_frame, text='0/0', font=('Arial', 9), bg='white')
        self.progress_label.pack(fill=tk.X, padx=8, pady=3)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=8, pady=3)
        
        eta_frame = tk.LabelFrame(stats_frame, text='ETA', font=('Arial', 9, 'bold'),
                                  bg='white', relief=tk.SUNKEN, bd=2)
        eta_frame.grid(row=3, column=0, sticky='ew', padx=0, pady=3)
        self.eta_label = tk.Label(eta_frame, text='--:--:--', font=('Arial', 10, 'bold'), bg='white', fg='#9b59b6')
        self.eta_label.pack(fill=tk.X, padx=8, pady=3)
        
        alerts_frame = tk.LabelFrame(stats_frame, text='STATUS', font=('Arial', 9, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
        alerts_frame.grid(row=4, column=0, sticky='ew', padx=0, pady=3)
        alerts_frame.rowconfigure(0, weight=1)
        alerts_frame.columnconfigure(0, weight=1)
        self.alerts_text = scrolledtext.ScrolledText(alerts_frame, height=6, font=('Courier', 8),
                                                     bg='#f8f8f8', fg='#333')
        self.alerts_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        footer = tk.Frame(self.root, bg='#ecf0f1', height=60)
        footer.grid(row=2, column=0, sticky='ew')
        
        btn_frame = tk.Frame(footer, bg='#ecf0f1')
        btn_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.start_btn = tk.Button(btn_frame, text='Demarrer',
                                   command=self.start_optimization,
                                   bg='#27ae60', fg='white',
                                   font=('Arial', 11, 'bold'),
                                   padx=15, pady=8)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text='Arreter',
                                  command=self.stop_optimization,
                                  bg='#e74c3c', fg='white',
                                  font=('Arial', 11, 'bold'),
                                  padx=15, pady=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.graphs_btn = tk.Button(btn_frame, text='Voir Graphiques',
                                    command=self.show_graphs,
                                    bg='#3498db', fg='white',
                                    font=('Arial', 11, 'bold'),
                                    padx=15, pady=8, state=tk.DISABLED)
        self.graphs_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(footer, text='Prêt',
                                     font=('Arial', 10, 'bold'),
                                     fg='#27ae60', bg='#ecf0f1')
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)

    def log_live(self, msg, tag='info'):
        try:
            self.live_text.insert(tk.END, msg + '\n', tag)
            self.live_text.see(tk.END)
            self.root.update_idletasks()
        except:
            pass

    def add_alert(self, msg):
        try:
            self.alerts_text.insert(tk.END, f'• {msg}\n')
            self.alerts_text.see(tk.END)
            self.root.update_idletasks()
        except:
            pass

    def update_stats(self):
        try:
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.1)
            
            self.ram_label.config(text=f'{ram:.1f}%')
            self.ram_progress['value'] = ram
            self.cpu_label.config(text=f'{cpu:.1f}%')
            self.cpu_progress['value'] = min(cpu, 100)
            
            if self.start_time and self.completed_operations > 0:
                elapsed = time.time() - self.start_time
                avg = elapsed / self.completed_operations
                remaining = (self.total_operations - self.completed_operations) * avg
                eta = datetime.now() + timedelta(seconds=remaining)
                self.eta_label.config(text=eta.strftime('%H:%M:%S'))
            
            percent = (self.completed_operations / self.total_operations * 100) if self.total_operations > 0 else 0
            self.progress_bar['value'] = percent
            self.progress_label.config(text=f'{self.completed_operations}/{self.total_operations}')
            
            self.root.after(500, self.update_stats)
        except:
            self.root.after(500, self.update_stats)

    def start_optimization(self):
        if self.running:
            messagebox.showwarning('Attention', 'Deja en cours')
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text='En cours...', fg='#f57f17')
        self.live_text.delete(1.0, tk.END)
        self.alerts_text.delete(1.0, tk.END)
        
        self.log_live('CV OPTIMIZATION V3 - GRID SEARCH\n', 'info')
        self.log_live('Hyperparamètres variables par algo\n\n', 'info')
        
        threading.Thread(target=self.run_optimization, daemon=True).start()
        self.start_time = time.time()
        self.update_stats()

    def stop_optimization(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text='Arrete', fg='#e74c3c')

    def load_data(self):
        try:
            self.log_live('ETAPE 1: Chargement CSV\n', 'info')
            files = ['fusion_train_smart4.csv', 'fusion_ton_iot_cic_final_smart4.csv']
            fichier = next((f for f in files if os.path.exists(f)), None)
            
            if not fichier:
                self.log_live('Erreur: CSV non trouve\n', 'info')
                return False
            
            self.log_live(f'Fichier: {fichier}\n', 'info')
            
            chunks = []
            total_rows = 0
            chunk_size = MemoryManager.get_optimal_chunk_size()
            
            for chunk in pd.read_csv(fichier, low_memory=False, chunksize=chunk_size, encoding='utf-8'):
                if not self.running:
                    return False
                chunks.append(chunk)
                total_rows += len(chunk)
                self.log_live(f'+{len(chunk):,} (total {total_rows:,})\n', 'info')
                
                if not MemoryManager.check_memory():
                    self.log_live(f'[WARN] RAM critique, attente...\n', 'info')
                    time.sleep(2)
            
            self.df = pd.concat(chunks, ignore_index=True)
            self.df = _normalize_label_column(self.df)
            self.log_live(f'OK: {len(self.df):,} lignes\n\n', 'info')
            return True
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')
            return False

    def prepare_data(self):
        try:
            self.log_live('ETAPE 2: Preparation\n', 'info')
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            self.df = self.df.dropna(subset=['Label'])
            
            n_samples = int(len(self.df) * STRATIFIED_SAMPLE_RATIO)
            stratifier = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            for train_idx, _ in stratifier.split(self.df, self.df['Label']):
                self.df = self.df.iloc[train_idx[:n_samples]]
                break
            
            self.log_live(f'Dataset: {len(self.df):,} lignes\n', 'info')
            
            X = self.df[numeric_cols].astype(np.float32).copy()
            X = X.fillna(X.mean())
            
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.df['Label'])
            
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X).astype(np.float32)
            
            self.log_live(f'Data normalisee: X={self.X_scaled.shape}\n', 'info')
            
            np.savez_compressed('preprocessed_dataset.npz',
                               X=self.X_scaled,
                               y=self.y,
                               classes=self.label_encoder.classes_)
            
            self.log_live(f'NPZ sauvegarde\n\n', 'info')
            del self.df, X
            gc.collect()
            return True
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')
            return False

    def generate_param_combinations(self, model_name):
        """Génère toutes les combinaisons de paramètres"""
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
            
            if not self.prepare_data():
                return
            if not self.running:
                return
            
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
            
            for i, (name, ModelClass) in enumerate(model_configs.items(), 1):
                self.log_live(f'\n{i}/4. {name}\n', 'info')
                
                combinations = self.generate_param_combinations(name)
                self.log_live(f'  Testage: {len(combinations)} combinaisons\n', 'info')
                
                best_score = 0
                best_params = None
                all_results = []
                
                for combo_idx, params in enumerate(combinations, 1):
                    if not self.running:
                        return
                    
                    f1_runs = []
                    
                    for fold in range(K_FOLD):
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                self.X_scaled, self.y,
                                test_size=0.2 if name != 'Decision Tree' else 0.3,
                                random_state=42 + fold,
                                stratify=self.y
                            )
                            
                            model = ModelClass(**params, random_state=42) if name != 'Naive Bayes' else ModelClass(**params)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1_runs.append(f1)
                        except:
                            f1_runs.append(0)
                        
                        self.completed_operations += 1
                        
                        if not MemoryManager.check_memory():
                            time.sleep(1)
                    
                    mean_f1 = np.mean(f1_runs) if f1_runs else 0
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    self.log_live(f'    [{combo_idx}/{len(combinations)}] {params_str}: F1={mean_f1:.4f}\n', 'info')
                    
                    all_results.append({'params': params, 'f1': mean_f1})
                    
                    if mean_f1 > best_score:
                        best_score = mean_f1
                        best_params = params
                    
                    self.add_alert(f'{name}: {combo_idx}/{len(combinations)} - F1={mean_f1:.4f}')
                
                self.results[name] = {
                    'all_results': all_results,
                    'best_params': best_params,
                    'best_f1': best_score
                }
                
                self.optimal_configs[name] = {
                    'params': best_params,
                    'f1_score': float(best_score)
                }
                
                self.log_live(f'  BEST: F1={best_score:.4f}\n', 'info')
            
            self.log_live('\nETAPE 4: Rapports\n', 'info')
            self.generate_reports()
            
            self.log_live('\n' + '='*60 + '\n', 'info')
            self.log_live('GRID SEARCH TERMINEE\n', 'info')
            
            self.status_label.config(text='Succes', fg='#27ae60')
            self.add_alert('GRID SEARCH COMPLETE')
            self.graphs_btn.config(state=tk.NORMAL)
        
        except Exception as e:
            self.log_live(f'Erreur: {e}\n{traceback.format_exc()}\n', 'info')
            self.status_label.config(text='Erreur', fg='#d32f2f')
        
        finally:
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

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
        """Affiche les graphiques scrollables"""
        if not self.results:
            messagebox.showinfo('Info', 'Pas de resultats')
            return
        
        graph_window = tk.Toplevel(self.root)
        graph_window.title('Graphiques - Hyperparamètres vs F1 Scores')
        graph_window.geometry('1400x900')
        
        main_frame = tk.Frame(graph_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for model_name, results_data in self.results.items():
            all_results = results_data['all_results']
            
            if not all_results:
                continue
            
            fig = Figure(figsize=(13, 5), dpi=100)
            ax = fig.add_subplot(111)
            
            x_labels = [str(i+1) for i in range(len(all_results))]
            y_scores = [r['f1'] for r in all_results]
            
            ax.plot(x_labels, y_scores, 'o-', linewidth=2.5, markersize=8, color='#3498db')
            ax.fill_between(range(len(y_scores)), y_scores, alpha=0.2, color='#3498db')
            ax.set_xlabel('Combinaison Paramètres (#)', fontsize=11)
            ax.set_ylabel('F1 Score', fontsize=11)
            ax.set_title(f'{model_name} - Hyperparamètres vs F1 Score', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            best_idx = np.argmax(y_scores)
            best_params = all_results[best_idx]['params']
            params_str = '\n'.join([f"{k}={v}" for k, v in list(best_params.items())[:3]])
            
            ax.text(0.02, 0.98, f'BEST (#{best_idx+1}):\n{params_str}...',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9))
            
            fig.tight_layout()
            
            canvas_frame = tk.Frame(scrollable_frame, bg='white')
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
            
            canvas_plot = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas.yview_moveto(0)


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