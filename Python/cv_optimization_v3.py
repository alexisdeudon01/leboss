#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV OPTIMIZATION V3 - FINAL
Corrigé et simplifié avec Tkinter GUI fonctionnelle

FIXES INTÉGRÉS:
  ✓ FIX 1: StratifiedShuffleSplit
  ✓ FIX 2: Decision Tree limite 80%
  ✓ Encoding UTF-8
  ✓ Tkinter UI simple et robuste
  ✓ Threading proper
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
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score, recall_score, precision_score
except ImportError:
    print("Erreur: sklearn non installe")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("Erreur: tkinter non installe")
    sys.exit(1)

os.environ['JOBLIB_PARALLEL_BACKEND'] = 'loky'

NUM_CORES = multiprocessing.cpu_count()
TRAIN_SIZES = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
K_FOLD = 5
STRATIFIED_SAMPLE_RATIO = 0.5
CPU_THRESHOLD = 90.0
RAM_THRESHOLD = 90.0

MODELS_CONFIG = {
    'Logistic Regression': {'n_jobs': -1, 'desc': 'Tres parallelisable'},
    'Naive Bayes': {'n_jobs': 1, 'desc': 'Non parallelisable'},
    'Decision Tree': {'n_jobs': 1, 'desc': 'Non parallelisable'},
    'Random Forest': {'n_jobs': -1, 'desc': 'Tres parallelisable'},
}

class CPUMonitor:
    """Moniteur CPU/RAM"""
    def __init__(self):
        try:
            self.process = psutil.Process(os.getpid())
        except Exception:
            self.process = None

    def get_cpu_percent(self):
        try:
            return self.process.cpu_percent(interval=0.05) if self.process else 0
        except Exception:
            return 0

    def get_num_threads(self):
        try:
            return self.process.num_threads() if self.process else 1
        except Exception:
            return 1

class CVOptimizationGUI:
    """Interface Tkinter pour CV Optimization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title('CV Optimization V3 - FINAL')
        self.root.geometry('1400x900')
        self.root.configure(bg='#f0f0f0')
        
        self.cpu = CPUMonitor()
        self.running = False
        self.results = {}
        self.optimal_configs = {}
        self.start_time = None
        self.completed_operations = 0
        self.total_operations = len(TRAIN_SIZES) * K_FOLD * len(MODELS_CONFIG)
        
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
        tk.Label(header, text='CV Optimization V3 - FINAL (FIX 1 & 2)',
                 font=('Arial', 11, 'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=20, pady=12)
        
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
        self.live_text = scrolledtext.ScrolledText(live_frame, font=('Courier', 9),
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
        
        progress_frame = tk.LabelFrame(stats_frame, text='Avancee', font=('Arial', 9, 'bold'),
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
        self.alerts_text = scrolledtext.ScrolledText(alerts_frame, height=8, font=('Courier', 8),
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
        
        self.status_label = tk.Label(footer, text='Pret',
                                     font=('Arial', 10, 'bold'),
                                     fg='#27ae60', bg='#ecf0f1')
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)

    def log_live(self, msg, tag='info'):
        """Log en direct"""
        try:
            self.live_text.insert(tk.END, msg + '\n', tag)
            self.live_text.see(tk.END)
            self.root.update_idletasks()
        except Exception:
            pass

    def add_alert(self, msg):
        """Ajouter alerte"""
        try:
            self.alerts_text.insert(tk.END, f'• {msg}\n')
            self.alerts_text.see(tk.END)
            self.root.update_idletasks()
        except Exception:
            pass

    def update_stats(self):
        """Mettre a jour stats"""
        try:
            ram = psutil.virtual_memory().percent
            cpu = self.cpu.get_cpu_percent()
            
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
        except Exception:
            self.root.after(500, self.update_stats)

    def start_optimization(self):
        """Demarrer optimization"""
        if self.running:
            messagebox.showwarning('Attention', 'Deja en cours')
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text='En cours...', fg='#f57f17')
        self.live_text.delete(1.0, tk.END)
        self.alerts_text.delete(1.0, tk.END)
        
        self.log_live('CV OPTIMIZATION V3 - FINAL\n', 'info')
        self.log_live('FIX 1: StratifiedShuffleSplit\n', 'info')
        self.log_live('FIX 2: Decision Tree max 80%\n', 'info')
        self.log_live('\n' + '='*60 + '\n\n', 'info')
        
        threading.Thread(target=self.run_optimization, daemon=True).start()
        self.start_time = time.time()
        self.update_stats()

    def stop_optimization(self):
        """Arreter optimization"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text='Arrete', fg='#e74c3c')

    def load_data(self):
        """Charger CSV"""
        try:
            self.log_live('ETAPE 1: Chargement CSV\n', 'info')
            files = ['fusion_train_smart4.csv', 'fusion_ton_iot_cic_final_smart4.csv']
            fichier = None
            for f in files:
                if os.path.exists(f):
                    fichier = f
                    break
            
            if not fichier:
                self.log_live('Erreur: Aucun CSV trouve\n', 'info')
                return False
            
            self.log_live(f'Fichier: {fichier}\n', 'info')
            
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(fichier, low_memory=False, chunksize=500000, encoding='utf-8'):
                if not self.running:
                    return False
                chunks.append(chunk)
                total_rows += len(chunk)
                self.log_live(f'+{len(chunk):,} (total {total_rows:,})\n', 'info')
            
            self.df = pd.concat(chunks, ignore_index=True)
            self.log_live(f'OK: {len(self.df):,} lignes\n\n', 'info')
            return True
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')
            return False

    def prepare_data(self):
        """Preparer donnees"""
        try:
            self.log_live('ETAPE 2: Preparation\n', 'info')
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            if 'Label' not in self.df.columns:
                self.log_live('Erreur: Label absent\n', 'info')
                return False
            
            self.df = self.df.dropna(subset=['Label'])
            
            n_samples = int(len(self.df) * STRATIFIED_SAMPLE_RATIO)
            from sklearn.model_selection import StratifiedKFold
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

    def run_optimization(self):
        """Lancer optimization"""
        try:
            if not self.load_data():
                return
            if not self.running:
                return
            
            if not self.prepare_data():
                return
            if not self.running:
                return
            
            self.log_live('ETAPE 3: Cross-Validation\n\n', 'info')
            
            models = [
                ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
                ('Naive Bayes', GaussianNB()),
                ('Decision Tree', DecisionTreeClassifier(random_state=42)),
                ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ]
            
            for i, (name, model) in enumerate(models, 1):
                if not self.running:
                    return
                
                if name == 'Decision Tree':
                    train_sizes_to_test = TRAIN_SIZES[TRAIN_SIZES <= 0.80]
                    self.log_live(f'\n{i}/4. {name} (LIMIT 80%)\n', 'info')
                else:
                    train_sizes_to_test = TRAIN_SIZES
                    self.log_live(f'\n{i}/4. {name}\n', 'info')
                
                self.run_cv_for_model(name, model, train_sizes_to_test)
                self.add_alert(f'OK: {name}')
            
            self.log_live('\nETAPE 4: Rapports\n', 'info')
            self.generate_reports()
            
            self.log_live('\n' + '='*60 + '\n', 'info')
            self.log_live('CV OPTIMIZATION TERMINEE\n', 'info')
            self.log_live('='*60 + '\n', 'info')
            
            self.status_label.config(text='Succes', fg='#27ae60')
            self.add_alert('CV OPTIMIZATION COMPLETE')
        
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')
            self.status_label.config(text='Erreur', fg='#d32f2f')
        
        finally:
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def run_cv_for_model(self, model_name, model, train_sizes):
        """CV pour un modele"""
        try:
            res = {
                'train_sizes': [],
                'f1_scores': [],
                'recall_scores': [],
                'precision_scores': [],
                'f1_std': [],
                'recall_std': [],
                'precision_std': []
            }
            
            for train_size in train_sizes:
                if not self.running:
                    return
                
                self.log_live(f'  {int(train_size*100)}%: ', 'info')
                
                sss = StratifiedShuffleSplit(n_splits=K_FOLD, train_size=train_size,
                                            test_size=1-train_size, random_state=42)
                
                f1s = np.zeros(K_FOLD, dtype=np.float32)
                recs = np.zeros(K_FOLD, dtype=np.float32)
                pres = np.zeros(K_FOLD, dtype=np.float32)
                
                for fold, (train_idx, val_idx) in enumerate(sss.split(self.X_scaled, self.y), 1):
                    if not self.running:
                        return
                    
                    Xtr, Xva = self.X_scaled[train_idx], self.X_scaled[val_idx]
                    ytr, yva = self.y[train_idx], self.y[val_idx]
                    
                    model.fit(Xtr, ytr)
                    ypred = model.predict(Xva)
                    
                    f1s[fold-1] = f1_score(yva, ypred, average='weighted', zero_division=0)
                    recs[fold-1] = recall_score(yva, ypred, average='weighted', zero_division=0)
                    pres[fold-1] = precision_score(yva, ypred, average='weighted', zero_division=0)
                    
                    self.completed_operations += 1
                
                mean_f1 = float(np.mean(f1s))
                std_f1 = float(np.std(f1s))
                
                res['train_sizes'].append(int(train_size*100))
                res['f1_scores'].append(mean_f1)
                res['f1_std'].append(std_f1)
                res['recall_scores'].append(float(np.mean(recs)))
                res['recall_std'].append(float(np.std(recs)))
                res['precision_scores'].append(float(np.mean(pres)))
                res['precision_std'].append(float(np.std(pres)))
                
                self.log_live(f'F1={mean_f1:.4f}+/-{std_f1:.4f}\n', 'info')
            
            best_idx = int(np.argmax(np.array(res['f1_scores']) - np.array(res['f1_std'])))
            best_ts = train_sizes[best_idx]
            best_f1 = res['f1_scores'][best_idx]
            best_std = res['f1_std'][best_idx]
            
            self.optimal_configs[model_name] = {
                'train_size': float(best_ts),
                'test_size': float(1-best_ts),
                'f1_score': float(best_f1),
                'f1_std': float(best_std),
                'recall': float(res['recall_scores'][best_idx]),
                'precision': float(res['precision_scores'][best_idx]),
                'n_jobs': MODELS_CONFIG[model_name]['n_jobs']
            }
            
            self.results[model_name] = res
            self.log_live(f'  OPTIMAL: {best_ts*100:.0f}% (F1={best_f1:.4f}+/-{best_std:.4f})\n', 'info')
        
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')

    def generate_reports(self):
        """Generer rapports"""
        try:
            with open('cv_results_summary.txt', 'w', encoding='utf-8') as f:
                f.write('='*80 + '\n')
                f.write('CV OPTIMIZATION V3 - FINAL\n')
                f.write('='*80 + '\n\n')
                f.write('FIX 1: StratifiedShuffleSplit\n')
                f.write('FIX 2: Decision Tree max 80%\n\n')
                for name in sorted(self.optimal_configs.keys()):
                    cfg = self.optimal_configs[name]
                    f.write(f"{name:<25} Train:{cfg['train_size']*100:>5.0f}% F1:{cfg['f1_score']:>7.4f}\n")
                f.write('='*80 + '\n')
            
            self.log_live('OK: cv_results_summary.txt\n', 'info')
            
            with open('cv_optimal_splits.json', 'w', encoding='utf-8') as jf:
                json.dump(self.optimal_configs, jf, ensure_ascii=False, indent=2)
            
            self.log_live('OK: cv_optimal_splits.json\n', 'info')
        
        except Exception as e:
            self.log_live(f'Erreur: {e}\n', 'info')


def main():
    """Main"""
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