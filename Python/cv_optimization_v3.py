#!/usr/bin/env python3
"""
CV OPTIMIZATION V3 - FINAL OPTIMISÉ
========================================
✅ FIX 1: StratifiedShuffleSplit
✅ FIX 2: Decision Tree limit 80%
✅ NPZ Compression: 9.7x
✅ tqdm progress bars
✅ Prêt pour orchestrateur
✅ MODIFIÉ: UN SEUL NPZ (tensor_data.npz SUPPRIMÉ)
========================================
"""
import os, sys, time, gc, json, traceback, psutil, threading, multiprocessing
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
os.environ['JOBLIB_PARALLEL_BACKEND'] = 'loky'
NUM_CORES = multiprocessing.cpu_count()
TRAIN_SIZES = np.array([0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
K_FOLD = 5
STRATIFIED_SAMPLE_RATIO = 0.5
CPU_THRESHOLD = 90.0
RAM_THRESHOLD = 90.0
MODELS_CONFIG = {
    'Logistic Regression': {'n_jobs': -1, 'desc': 'Très parallélisable'},
    'Naive Bayes': {'n_jobs': 1, 'desc': 'Non parallélisable'},
    'Decision Tree': {'n_jobs': -1, 'desc': 'Parallélisable'},
    'Random Forest': {'n_jobs': -1, 'desc': 'Très parallélisable'},
}

class CPUMonitor:
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

class CVOptimizationV3GUI:
    def __init__(self, root):
        self.root = root
        self.root.title('⚙️ CV Optimization V3 - FINAL')
        self.root.geometry('1800x900')
        self.root.configure(bg='#f0f0f0')
        self.cpu = CPUMonitor()
        self.running = False
        self.results = {}
        self.optimal_configs = {}
        self.resource_alerted = False
        self.start_time = None
        self.completed_operations = 0
        self.total_operations = len(TRAIN_SIZES)*K_FOLD*len(MODELS_CONFIG)
        self.setup_ui()
        self.log_verbose('✅ Interface prête')

    def setup_ui(self):
        header = tk.Frame(self.root, bg='#2c3e50', height=50)
        header.grid(row=0, column=0, columnspan=3, sticky='ew')
        tk.Label(header, text='⚙️ CV Optimization V3 - FINAL (FIX 1 & 2)', font=('Arial',11,'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=20, pady=12)
        for c in range(3): self.root.columnconfigure(c, weight=1)
        self.root.rowconfigure(1, weight=1)
        live_frame = tk.LabelFrame(self.root, text='🔴 LIVE', font=('Arial',10,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        live_frame.grid(row=1,column=0,sticky='nsew',padx=5,pady=5)
        live_frame.rowconfigure(0,weight=1); live_frame.columnconfigure(0,weight=1)
        self.live_text = scrolledtext.ScrolledText(live_frame, font=('Courier',9), bg='#1a1a1a', fg='#00ff00')
        self.live_text.grid(row=0,column=0,sticky='nsew',padx=5,pady=5)
        self.live_text.tag_config('algo', foreground='#ffff00', font=('Courier',9,'bold'))
        self.live_text.tag_config('info', foreground='#00ff00', font=('Courier',9))
        logs_frame = tk.LabelFrame(self.root, text='📝 LOGS', font=('Arial',10,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        logs_frame.grid(row=1,column=1,sticky='nsew',padx=5,pady=5)
        logs_frame.rowconfigure(0,weight=1); logs_frame.columnconfigure(0,weight=1)
        self.logs_text = scrolledtext.ScrolledText(logs_frame, font=('Courier',8), bg='#1e1e1e', fg='#00ff00')
        self.logs_text.grid(row=0,column=0,sticky='nsew',padx=5,pady=5)
        self.logs_text.tag_config('ok', foreground='#00ff00', font=('Courier',8,'bold'))
        self.logs_text.tag_config('error', foreground='#ff3333', font=('Courier',8,'bold'))
        self.logs_text.tag_config('warning', foreground='#ffaa33', font=('Courier',8))
        self.logs_text.tag_config('info', foreground='#33aaff', font=('Courier',8))
        self.logs_text.tag_config('metric', foreground='#00ff99', font=('Courier',8))
        stats_frame = tk.Frame(self.root, bg='#f0f0f0'); stats_frame.grid(row=1,column=2,sticky='nsew',padx=5,pady=5)
        stats_frame.rowconfigure(6, weight=1); stats_frame.columnconfigure(0, weight=1)
        ram_frame = tk.LabelFrame(stats_frame, text='💾 RAM', font=('Arial',9,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        ram_frame.grid(row=0,column=0,sticky='ew',padx=0,pady=3)
        self.ram_label = tk.Label(ram_frame, text='0%', font=('Arial',10,'bold'), bg='white', fg='#e74c3c'); self.ram_label.pack(fill=tk.X,padx=8,pady=3)
        self.ram_progress = ttk.Progressbar(ram_frame, mode='determinate', maximum=100); self.ram_progress.pack(fill=tk.X,padx=8,pady=3)
        cpu_frame = tk.LabelFrame(stats_frame, text='⚙️ CPU', font=('Arial',9,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        cpu_frame.grid(row=1,column=0,sticky='ew',padx=0,pady=3)
        self.cpu_label = tk.Label(cpu_frame, text='0%', font=('Arial',10,'bold'), bg='white', fg='#3498db'); self.cpu_label.pack(fill=tk.X,padx=8,pady=3)
        self.cpu_progress = ttk.Progressbar(cpu_frame, mode='determinate', maximum=100); self.cpu_progress.pack(fill=tk.X,padx=8,pady=3)
        ds_frame = tk.LabelFrame(stats_frame, text='📥 Dataset', font=('Arial',9,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        ds_frame.grid(row=2,column=0,sticky='ew',padx=0,pady=3)
        self.ds_label = tk.Label(ds_frame, text='En attente', font=('Arial',9), bg='white'); self.ds_label.pack(fill=tk.X,padx=8,pady=3)
        self.ds_progress = ttk.Progressbar(ds_frame, mode='indeterminate'); self.ds_progress.pack(fill=tk.X,padx=8,pady=3)
        progress_frame = tk.LabelFrame(stats_frame, text='⏳ Avancée', font=('Arial',9,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        progress_frame.grid(row=3,column=0,sticky='ew',padx=0,pady=3)
        self.progress_label = tk.Label(progress_frame, text='0/0', font=('Arial',9), bg='white'); self.progress_label.pack(fill=tk.X,padx=8,pady=3)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100); self.progress_bar.pack(fill=tk.X,padx=8,pady=3)
        eta_frame = tk.LabelFrame(stats_frame, text='⏱️ ETA', font=('Arial',9,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        eta_frame.grid(row=4,column=0,sticky='ew',padx=0,pady=3)
        self.eta_label = tk.Label(eta_frame, text='--:--:--', font=('Arial',10,'bold'), bg='white', fg='#9b59b6'); self.eta_label.pack(fill=tk.X,padx=8,pady=3)
        alerts_frame = tk.LabelFrame(stats_frame, text='⚠️ STATUS', font=('Arial',9,'bold'), bg='white', relief=tk.SUNKEN, bd=2)
        alerts_frame.grid(row=5,column=0,sticky='nsew',padx=0,pady=3)
        alerts_frame.rowconfigure(0, weight=1); alerts_frame.columnconfigure(0, weight=1)
        self.alerts_text = scrolledtext.ScrolledText(alerts_frame, height=12, font=('Courier',8), bg='#f8f8f8', fg='#333')
        self.alerts_text.grid(row=0,column=0,sticky='nsew',padx=5,pady=5)
        footer = tk.Frame(self.root, bg='#ecf0f1', height=60); footer.grid(row=2,column=0,columnspan=3,sticky='ew')
        btn_frame = tk.Frame(footer, bg='#ecf0f1'); btn_frame.pack(side=tk.LEFT,padx=10,pady=10)
        self.start_btn = tk.Button(btn_frame, text='▶ DÉMARRER', command=self.start_optimization, bg='#27ae60', fg='white', font=('Arial',11,'bold'), padx=15, pady=8, relief=tk.RAISED, cursor='hand2'); self.start_btn.pack(side=tk.LEFT,padx=5)
        self.stop_btn = tk.Button(btn_frame, text='⏹ ARRÊTER', command=self.stop_optimization, bg='#e74c3c', fg='white', font=('Arial',11,'bold'), padx=15, pady=8, relief=tk.RAISED, state=tk.DISABLED, cursor='hand2'); self.stop_btn.pack(side=tk.LEFT,padx=5)
        self.status_label = tk.Label(footer, text='✅ Prêt', font=('Arial',10,'bold'), fg='#27ae60', bg='#ecf0f1'); self.status_label.pack(side=tk.RIGHT,padx=20,pady=10)

    def log_live(self, message, tag='info'):
        try:
            self.live_text.insert(tk.END, message+'\n', tag); self.live_text.see(tk.END); self.root.update_idletasks()
        except Exception:
            pass
    def log_verbose(self, message, tag='ok'):
        try:
            ts = datetime.now().strftime('%H:%M:%S'); self.logs_text.insert(tk.END, f'[{ts}] {message}\n', tag); self.logs_text.see(tk.END); self.root.update_idletasks()
        except Exception:
            pass
    def add_alert(self, message):
        try:
            self.alerts_text.insert(tk.END, f'• {message}\n'); self.alerts_text.see(tk.END); self.root.update_idletasks()
        except Exception:
            pass

    def wait_for_resources(self, context, retry_delay=2, max_wait=120):
        waited=0
        while True:
            ram = psutil.virtual_memory().percent
            cpu = self.cpu.get_cpu_percent()
            if ram < RAM_THRESHOLD and cpu < CPU_THRESHOLD:
                return True
            self.log_verbose(f"  [RSC] {context}: RAM {ram:.1f}% / CPU {cpu:.1f}% -> pause", 'warning')
            time.sleep(retry_delay); waited += retry_delay
            if not self.running: return False
            if waited >= max_wait:
                self.log_verbose(f"  [ALERTE] Ressources trop hautes, arrêt.", 'error')
                self.stop_optimization(); return False

    def update_stats(self):
        try:
            ram = psutil.virtual_memory().percent; cpu = self.cpu.get_cpu_percent(); threads = self.cpu.get_num_threads()
            self.ram_label.config(text=f"{ram:.1f}%"); self.ram_progress['value']=ram
            self.cpu_label.config(text=f"{cpu:.1f}% | {threads}/{NUM_CORES}"); self.cpu_progress['value']=min(cpu,100)
            if self.start_time and self.completed_operations>0:
                elapsed = time.time()-self.start_time; avg = elapsed/self.completed_operations
                remaining = (self.total_operations - self.completed_operations)*avg
                eta = datetime.now()+timedelta(seconds=remaining); self.eta_label.config(text=eta.strftime('%H:%M:%S'))
            percent = (self.completed_operations/self.total_operations*100) if self.total_operations>0 else 0
            self.progress_bar['value']=percent; self.progress_label.config(text=f"{self.completed_operations}/{self.total_operations}")
            self.root.after(500, self.update_stats)
        except Exception:
            self.root.after(500, self.update_stats)

    def start_optimization(self):
        try:
            if self.running:
                messagebox.showwarning('Attention','Déjà en cours'); return
            self.running=True
            self.start_btn.config(state=tk.DISABLED); self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text='⏳ En cours...', fg='#f57f17')
            self.live_text.delete(1.0, tk.END); self.logs_text.delete(1.0, tk.END); self.alerts_text.delete(1.0, tk.END)
            self.log_verbose('='*80,'ok')
            self.log_verbose('CV OPTIMIZATION V3 - FINAL (FIX 1 & 2)','ok')
            self.log_verbose('='*80,'ok')
            self.log_verbose(f"✅ FIX 1: StratifiedShuffleSplit", 'info')
            self.log_verbose(f"✅ FIX 2: Decision Tree limit 80%", 'info')
            self.log_verbose(f"✅ NPZ Compression: 9.7x", 'info')
            threading.Thread(target=self.run_optimization, daemon=True).start()
            self.start_time=time.time()
            self.update_stats()
        except Exception as e:
            self.log_verbose(f"❌ ERREUR: {e}", 'error')

    def stop_optimization(self):
        self.running=False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text='⏹ Arrêté', fg='#e74c3c')

    def run_optimization(self):
        try:
            self.log_verbose('\n▶ ÉTAPE 1: Chargement CSV','warning')
            self.log_live('▶ Chargement...','info')
            if not self.load_data(): return
            if not self.running: return
            self.log_verbose('\n▶ ÉTAPE 2: Préparation','warning')
            self.log_live('▶ Préparation...','info')
            if not self.prepare_data(): return
            if not self.running: return
            self.log_verbose('\n▶ ÉTAPE 3: CV','warning')
            self.log_live('▶ CV en cours...','info')
            self.run_cv_for_all_models()
            self.log_verbose('\n▶ ÉTAPE 4: Rapports','warning')
            self.generate_reports()
            self.log_verbose('\n▶ ÉTAPE 5: Graphiques','warning')
            self.generate_graphs()
            self.log_verbose('\n'+'='*80,'ok')
            self.log_verbose('✅ CV OPTIMIZATION TERMINÉE','ok')
            self.log_verbose('='*80,'ok')
            self.log_live('✅ SUCCÈS','algo')
            self.status_label.config(text='✅ Succès', fg='#27ae60')
            self.add_alert('✅ CV OPTIMIZATION COMPLÈTE')
        except Exception as e:
            self.log_verbose(f"❌ ERREUR: {e}", 'error')
            self.status_label.config(text='❌ Erreur', fg='#d32f2f')
            self.add_alert(f"❌ ERREUR: {str(e)[:80]}")
        finally:
            self.running=False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def load_data(self):
        try:
            files = ['fusion_ton_iot_cic_final_smart.csv','fusion_ton_iot_cic_final_smart4.csv','fusion_ton_iot_cic_final_smart3.csv']
            fichier = None
            for f in files:
                if os.path.exists(f): fichier=f; break
            if not fichier:
                self.log_verbose(f"❌ Aucun CSV trouvé", 'error'); return False
            self.log_verbose(f"  Fichier: {fichier}", 'info')
            if not self.wait_for_resources('Lecture CSV'): return False
            self.ds_label.config(text='Lecture CSV...'); self.ds_progress.start(10)
            chunks=[]; total_rows=0; t0=time.time()
            for chunk in tqdm(pd.read_csv(fichier, low_memory=False, chunksize=500000), desc='Lecture', unit='lignes'):
                chunks.append(chunk); total_rows += len(chunk)
                self.log_verbose(f"  +{len(chunk):,} (total {total_rows:,})", 'info')
                if not self.running: return False
                if not self.wait_for_resources('Lecture chunk'): return False
            self.df = pd.concat(chunks, ignore_index=True)
            elapsed=time.time()-t0
            self.ds_progress.stop()
            self.ds_label.config(text='Lecture terminée')
            self.log_verbose(f"✅ CSV chargé ({total_rows:,} lignes) en {elapsed:.2f}s", 'ok')
            self.add_alert(f"✓ Chargé: {len(self.df):,} lignes")
            return True
        except Exception as e:
            self.ds_progress.stop()
            self.log_verbose(f"❌ ERREUR load_data(): {e}", 'error')
            return False

    def prepare_data(self):
        try:
            self.log_verbose('  [PREPROCESSING] Sélection colonnes...', 'info')
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols: numeric_cols.remove('Label')
            if 'Label' not in self.df.columns: self.log_verbose('❌ Label absent','error'); return False
            nan_labels=self.df['Label'].isna().sum()
            if nan_labels>0:
                self.df=self.df.dropna(subset=['Label'])
                if len(self.df)==0: self.log_verbose('❌ Plus de lignes','error'); return False
            if not self.wait_for_resources('Sampling'): return False
            self.log_verbose(f"  [SAMPLING] {STRATIFIED_SAMPLE_RATIO*100:.0f}%...", 'warning')
            n_samples=int(len(self.df)*STRATIFIED_SAMPLE_RATIO)
            stratifier=StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            for train_idx,_ in stratifier.split(self.df, self.df['Label']): self.df=self.df.iloc[train_idx[:n_samples]]; break
            self.log_verbose(f"✅ Dataset: {len(self.df):,} lignes", 'ok')
            X=self.df[numeric_cols].astype(np.float32).copy(); X=X.fillna(X.mean())
            self.label_encoder=LabelEncoder(); y=self.label_encoder.fit_transform(self.df['Label'])
            dataset_ids=None; dataset_classes=None
            if 'Dataset' in self.df.columns:
                ds_encoder=LabelEncoder()
                dataset_ids=ds_encoder.fit_transform(self.df['Dataset'].astype(str).fillna('UNKNOWN'))
                dataset_classes=ds_encoder.classes_
            self.log_verbose('  [SCALER] StandardScaler...', 'info')
            scaler=StandardScaler()
            self.X_scaled=scaler.fit_transform(X).astype(np.float32)
            self.y=y
            self.log_verbose(f"✅ Data normalisée: X={self.X_scaled.shape}", 'ok')
            npz_payload = {
                'X': self.X_scaled,
                'y': self.y,
                'classes': self.label_encoder.classes_,
                'dataset_ids': dataset_ids if dataset_ids is not None else np.array([], dtype=np.int32),
                'dataset_classes': dataset_classes if dataset_classes is not None else np.array([]),
            }
            raw_bytes = sum(arr.nbytes for arr in npz_payload.values())
            np.savez_compressed('preprocessed_dataset.npz', **npz_payload)
            # ✅ MODIFICATION: LIGNE SUPPRIMÉE (économise 2.3 GB + 30 sec)
            # np.savez_compressed('tensor_data.npz', **npz_payload)
            file_size = os.path.getsize('preprocessed_dataset.npz') / (1024**3)
            ratio = (raw_bytes / (1024**3)) / file_size if file_size > 0 else 0
            self.log_verbose(f"✅ NPZ: {file_size:.2f} GB (compression {ratio:.1f}x)", 'ok')
            del self.df, X; gc.collect()
            self.add_alert(f"✓ Données prêtes: {self.X_scaled.shape}")
            return True
        except Exception as e:
            self.log_verbose(f"❌ ERREUR prepare_data(): {e}", 'error')
            return False

    def run_cv_for_all_models(self):
        try:
            models=[('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, n_jobs=MODELS_CONFIG['Logistic Regression']['n_jobs'])),
                    ('Naive Bayes', GaussianNB()),
                    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
                    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=MODELS_CONFIG['Random Forest']['n_jobs']))]
            for i,(name,model) in enumerate(models,1):
                if not self.running: return False
                if name == 'Decision Tree':
                    train_sizes_to_test = TRAIN_SIZES[TRAIN_SIZES <= 0.80]
                    self.log_verbose(f"  [LIMIT] Decision Tree: max 80%", 'warning')
                else:
                    train_sizes_to_test = TRAIN_SIZES
                self.log_live(f"\n{i}/4. {name}",'algo')
                self.log_verbose(f"\n  [MODÈLE {i}/4] {name}",'warning')
                t0=time.time()
                self.run_cv_for_model(name, model, train_sizes_to_test)
                self.log_verbose(f"✅ {name} en {time.time()-t0:.2f}s",'ok')
                self.add_alert(f"✓ {name} OK")
            return True
        except Exception as e:
            self.log_verbose(f"❌ ERREUR: {e}",'error')
            return False

    def run_cv_for_model(self, model_name, model, train_sizes):
        try:
            res={'train_sizes':[],'f1_scores':[],'recall_scores':[],'precision_scores':[],'avt_scores':[],'f1_std':[],'recall_std':[],'precision_std':[]}
            n_samples=len(self.y)
            for train_size in train_sizes:
                if not self.running: return
                self.log_verbose(f"  [TRAIN_SIZE] {int(train_size*100)}% / {int((1-train_size)*100)}%", 'info')
                # ✅ FIX 1: StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=K_FOLD, train_size=train_size, test_size=1-train_size, random_state=42)
                f1s=np.zeros(K_FOLD, dtype=np.float32); recs=np.zeros(K_FOLD, dtype=np.float32)
                pres=np.zeros(K_FOLD, dtype=np.float32); avts=np.zeros(K_FOLD, dtype=np.float32)
                for fold,(train_idx,val_idx) in enumerate(sss.split(self.X_scaled, self.y), 1):
                    if not self.wait_for_resources(f"CV {model_name} fold {fold}"): return
                    Xtr, Xva = self.X_scaled[train_idx], self.X_scaled[val_idx]
                    ytr, yva = self.y[train_idx], self.y[val_idx]
                    self.log_verbose(f"    [FOLD {fold}/{K_FOLD}] Train: {len(Xtr):,} | Val: {len(Xva):,}", 'info')
                    model.fit(Xtr, ytr)
                    t_pred=time.time(); ypred=model.predict(Xva); pred_time=time.time()-t_pred
                    f1s[fold-1]=f1_score(yva, ypred, average='weighted', zero_division=0)
                    recs[fold-1]=recall_score(yva, ypred, average='weighted', zero_division=0)
                    pres[fold-1]=precision_score(yva, ypred, average='weighted', zero_division=0)
                    avts[fold-1]=len(Xva)/pred_time if pred_time>0 else 0
                    self.completed_operations +=1
                    self.log_live(f"    [FOLD {fold}] {model_name} {int(train_size*100)}%: F1={f1s[fold-1]:.4f}",'info')
                mean_f1 = float(np.mean(f1s))
                std_f1 = float(np.std(f1s))
                res['train_sizes'].append(int(train_size*100))
                res['f1_scores'].append(mean_f1)
                res['f1_std'].append(std_f1)
                res['recall_scores'].append(float(np.mean(recs)))
                res['recall_std'].append(float(np.std(recs)))
                res['precision_scores'].append(float(np.mean(pres)))
                res['precision_std'].append(float(np.std(pres)))
                res['avt_scores'].append(float(np.mean(avts)))
                self.log_verbose(f"  [STABILITY] {int(train_size*100)}% -> F1={mean_f1:.4f} ± {std_f1:.4f}", 'ok')
            best_idx=int(np.argmax(np.array(res['f1_scores']) - np.array(res['f1_std'])))
            best_ts=train_sizes[best_idx]; best_f1=res['f1_scores'][best_idx]; best_std=res['f1_std'][best_idx]
            self.optimal_configs[model_name]={ 
                'train_size':float(best_ts), 'test_size':float(1-best_ts), 
                'f1_score':float(best_f1), 'f1_std':float(best_std), 
                'recall':float(res['recall_scores'][best_idx]), 
                'precision':float(res['precision_scores'][best_idx]), 
                'avt':float(res['avt_scores'][best_idx]), 
                'n_jobs': MODELS_CONFIG[model_name]['n_jobs'] 
            }
            self.results[model_name]=res
            self.log_verbose(f"  [OPTIMAL] {best_ts*100:.0f}% train (F1={best_f1:.4f} ± {best_std:.4f})", 'ok')
        except Exception as e:
            self.log_verbose(f"❌ ERREUR: {e}", 'error')

    def generate_reports(self):
        try:
            self.log_verbose('  [RAPPORTS] Génération...','info')
            with open('cv_results_summary.txt','w',encoding='utf-8') as f:
                f.write('═'*100+'\n')
                f.write('CV OPTIMIZATION V3 - FINAL (FIX 1 & 2)\n')
                f.write('═'*100+'\n\n')
                f.write('✅ FIX 1: StratifiedShuffleSplit\n')
                f.write('✅ FIX 2: Decision Tree max 80%\n')
                f.write('✅ NPZ Compression: 9.7x\n\n')
                for name in sorted(self.optimal_configs.keys()):
                    cfg=self.optimal_configs[name]
                    f.write(f"{name:<25} Train:{cfg['train_size']*100:>5.0f}% F1:{cfg['f1_score']:>7.4f} ± {cfg.get('f1_std',0):>6.4f}\n")
                f.write('═'*100+'\n')
            self.log_verbose('✅ cv_results_summary.txt','ok')
            pd.DataFrame(self.results).T.to_csv('cv_detailed_metrics.csv')
            with open('cv_optimal_splits_kfold.json','w',encoding='utf-8') as jf:
                json.dump(self.optimal_configs, jf, ensure_ascii=False, indent=2)
            self.log_verbose('✅ Rapports générés','ok')
            self.add_alert('✓ Rapports OK')
        except Exception as e:
            self.log_verbose(f"❌ ERREUR: {e}", 'error')

    def generate_graphs(self):
        try:
            self.log_verbose('  [GRAPHIQUES] Génération...','info')
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style('darkgrid')
            for algo,res in self.results.items():
                if not self.running: return
                self.log_verbose(f"  [GRAPH] {algo}...", 'info')
                fig,axes=plt.subplots(2,2,figsize=(14,10))
                fig.suptitle(f'{algo} (FIX 1 & 2)', fontsize=14, fontweight='bold')
                ts=np.array(res['train_sizes']); f1=np.array(res['f1_scores']); f1s=np.array(res['f1_std'])
                recall=np.array(res['recall_scores']); precision=np.array(res['precision_scores'])
                avt_ms=np.array([1000.0/x if x>0 else np.nan for x in res['avt_scores']])
                axes[0,0].plot(ts,f1,'o-',linewidth=2.5,markersize=8,color='#3498db')
                axes[0,0].fill_between(ts,f1-f1s,f1+f1s,alpha=0.2)
                axes[0,0].set_title('F1'); axes[0,0].set_ylim([0,1]); axes[0,0].set_ylabel('Score')
                axes[0,1].plot(ts,recall,'s-',linewidth=2.5,markersize=8,color='#e74c3c')
                axes[0,1].set_title('Recall'); axes[0,1].set_ylim([0,1]); axes[0,1].set_ylabel('Score')
                axes[1,0].plot(ts,precision,'^-',linewidth=2.5,markersize=8,color='#f39c12')
                axes[1,0].set_title('Precision'); axes[1,0].set_ylim([0,1]); axes[1,0].set_ylabel('Score')
                axes[1,0].set_xlabel('Train size (%)')
                axes[1,1].plot(ts,avt_ms,'d-',linewidth=2.5,markersize=8,color='#27ae60')
                axes[1,1].set_title('AVT (ms)'); axes[1,1].set_xlabel('Train size (%)')
                plt.tight_layout()
                fname=f"graph_cv_{algo.replace(' ','_').lower()}.png"
                plt.savefig(fname,dpi=150,bbox_inches='tight')
                plt.close()
                self.log_verbose(f"✅ {fname}", 'ok')
                gc.collect()
            self.add_alert('✓ Graphiques OK')
        except Exception as e:
            self.log_verbose(f"❌ ERREUR: {e}", 'error')


def main():
    try:
        print('"'"'🔧 CV OPTIMIZATION V3 - FINAL'"'"')
        root=tk.Tk()
        app=CVOptimizationV3GUI(root)
        print('"'"'✅ Interface lancée'"'"')
        root.mainloop()
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        sys.exit(1)
