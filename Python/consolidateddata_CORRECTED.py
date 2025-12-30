#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONSOLIDATION DATASET - INTERFACE TKINTER AVANCÉE
================================================
✅ GUI complète avec Tkinter
✅ Sélection fichiers + boutons de contrôle
✅ Progress bars visuelles chaque étape
✅ Logs en temps réel
✅ Stats RAM/CPU
================================================
"""

import os
import sys
import gc
import time
import psutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from datetime import datetime
import threading
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


class MemoryManager:
    RAM_THRESHOLD = 90.0
    
    @staticmethod
    def get_ram_usage():
        try:
            return psutil.virtual_memory().percent
        except:
            return 50
    
    @staticmethod
    def check_and_cleanup():
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > MemoryManager.RAM_THRESHOLD:
            gc.collect()
            return False
        return True


class ConsolidationGUI:
    """Interface GUI Tkinter pour consolidation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DDoS Detection - Data Consolidation")
        self.root.geometry('1200x800')
        self.root.configure(bg='#f0f0f0')
        
        self.toniot_file = None
        self.cic_dir = None
        self.is_running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup interface"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=60)
        header.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
        
        tk.Label(header, text="🔧 Data Consolidation - TON_IoT + CIC", 
                font=('Arial', 14, 'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=20, pady=15)
        
        # Main container
        container = tk.Frame(self.root, bg='#f0f0f0')
        container.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)
        
        # Section 1: File Selection
        file_frame = tk.LabelFrame(container, text='📁 FILE SELECTION', font=('Arial', 10, 'bold'),
                                   bg='white', relief=tk.SUNKEN, bd=2)
        file_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=(0, 5))
        file_frame.columnconfigure(1, weight=1)
        
        # TON_IoT
        tk.Label(file_frame, text="TON_IoT CSV:", font=('Arial', 9, 'bold'),
                bg='white').grid(row=0, column=0, sticky='w', padx=10, pady=8)
        
        self.toniot_label = tk.Label(file_frame, text="Not selected", font=('Arial', 9),
                                    fg='#e74c3c', bg='white')
        self.toniot_label.grid(row=0, column=1, sticky='w', padx=10, pady=8)
        
        tk.Button(file_frame, text="Browse TON_IoT", command=self.select_toniot,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold'),
                 padx=15, pady=5).grid(row=0, column=2, sticky='e', padx=10, pady=8)
        
        # CIC
        tk.Label(file_frame, text="CIC Folder:", font=('Arial', 9, 'bold'),
                bg='white').grid(row=1, column=0, sticky='w', padx=10, pady=8)
        
        self.cic_label = tk.Label(file_frame, text="Not selected", font=('Arial', 9),
                                 fg='#e74c3c', bg='white')
        self.cic_label.grid(row=1, column=1, sticky='w', padx=10, pady=8)
        
        tk.Button(file_frame, text="Browse CIC", command=self.select_cic,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold'),
                 padx=15, pady=5).grid(row=1, column=2, sticky='e', padx=10, pady=8)
        
        # Section 2: Progress
        progress_frame = tk.LabelFrame(container, text='⏳ PROGRESS', font=('Arial', 10, 'bold'),
                                      bg='white', relief=tk.SUNKEN, bd=2)
        progress_frame.grid(row=1, column=0, sticky='nsew', padx=0, pady=(0, 5))
        progress_frame.rowconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        
        # Logs
        self.log_text = scrolledtext.ScrolledText(progress_frame, font=('Courier', 8),
                                                 bg='#1a1a1a', fg='#00ff00', wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Section 3: Stats
        stats_frame = tk.LabelFrame(container, text='📊 STATS', font=('Arial', 10, 'bold'),
                                   bg='white', relief=tk.SUNKEN, bd=2)
        stats_frame.grid(row=2, column=0, sticky='ew', padx=0, pady=(0, 5))
        stats_frame.columnconfigure(1, weight=1)
        
        tk.Label(stats_frame, text="RAM:", font=('Arial', 9, 'bold'),
                bg='white').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.ram_label = tk.Label(stats_frame, text="-- %", font=('Arial', 9, 'bold'),
                                 fg='#27ae60', bg='white')
        self.ram_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        tk.Label(stats_frame, text="Time:", font=('Arial', 9, 'bold'),
                bg='white').grid(row=0, column=2, sticky='w', padx=10, pady=5)
        self.time_label = tk.Label(stats_frame, text="00:00:00", font=('Arial', 9, 'bold'),
                                  fg='#9b59b6', bg='white')
        self.time_label.grid(row=0, column=3, sticky='w', padx=10, pady=5)
        
        # Section 4: Current Stage
        self.stage_label = tk.Label(stats_frame, text="Ready to start", font=('Arial', 9),
                                   fg='#3498db', bg='white')
        self.stage_label.grid(row=1, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(stats_frame, mode='determinate', maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        
        # Section 5: Buttons
        button_frame = tk.Frame(container, bg='#f0f0f0')
        button_frame.grid(row=3, column=0, sticky='ew', padx=0, pady=(5, 0))
        
        self.start_button = tk.Button(button_frame, text="▶️  START CONSOLIDATION", 
                                     command=self.start_consolidation,
                                     bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
                                     padx=20, pady=10)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="⏹️  STOP", 
                                    command=self.stop_consolidation,
                                    bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                                    padx=20, pady=10, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="❌ EXIT", 
                 command=self.root.quit,
                 bg='#95a5a6', fg='white', font=('Arial', 11, 'bold'),
                 padx=20, pady=10).pack(side=tk.RIGHT, padx=5)
        
        self.start_time = None
        self.update_stats()
    
    def select_toniot(self):
        """Select TON_IoT file"""
        file = filedialog.askopenfilename(
            title="Select train_test_network.csv (TON_IoT)",
            filetypes=[("CSV files", "*.csv")]
        )
        if file:
            self.toniot_file = file
            self.toniot_label.config(text=os.path.basename(file), fg='#27ae60')
            self.log(f"Selected TON_IoT: {os.path.basename(file)}")
    
    def select_cic(self):
        """Select CIC folder"""
        folder = filedialog.askdirectory(
            title="Select CIC folder"
        )
        if folder:
            self.cic_dir = folder
            # Chercher récursivement les CSVs
            csv_count = 0
            for root, dirs, files in os.walk(folder):
                csv_count += len([f for f in files if f.endswith('.csv')])
            self.cic_label.config(text=f"{os.path.basename(folder)} ({csv_count} CSVs)", fg='#27ae60')
            self.log(f"Selected CIC folder with {csv_count} CSV files (recursive search)")
    
    def log(self, msg, level="INFO"):
        """Add log"""
        ts = datetime.now().strftime("%H:%M:%S")
        if level == "HEADER":
            log_msg = f"\n{'='*80}\n{msg}\n{'='*80}\n"
        elif level == "OK":
            log_msg = f"[{ts}] ✅ {msg}"
        elif level == "ERROR":
            log_msg = f"[{ts}] ❌ {msg}"
        elif level == "WARN":
            log_msg = f"[{ts}] ⚠️  {msg}"
        else:
            log_msg = f"[{ts}] ℹ️  {msg}"
        
        self.log_text.insert(tk.END, log_msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_stats(self):
        """Update stats"""
        try:
            ram = MemoryManager.get_ram_usage()
            color = '#27ae60' if ram < 70 else '#f39c12' if ram < 85 else '#e74c3c'
            self.ram_label.config(text=f"{ram:.1f}%", fg=color)
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                hrs = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                secs = int(elapsed % 60)
                self.time_label.config(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")
            
            self.root.after(500, self.update_stats)
        except:
            self.root.after(500, self.update_stats)
    
    def start_consolidation(self):
        """Start consolidation in thread"""
        if not self.toniot_file or not self.cic_dir:
            self.log("ERROR: Select both TON_IoT and CIC first!", "ERROR")
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.start_time = time.time()
        
        threading.Thread(target=self.consolidate_worker, daemon=True).start()
    
    def stop_consolidation(self):
        """Stop consolidation"""
        self.is_running = False
        self.log("Stopping...", "WARN")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def consolidate_worker(self):
        """Worker thread for consolidation"""
        try:
            self.log("CONSOLIDATION STARTED", "HEADER")
            
            # ÉTAPE 1: Load TON_IoT
            self.stage_label.config(text="[1/8] Loading TON_IoT...")
            self.log("\nÉTAPE 1: Chargement TON_IoT", "HEADER")
            
            df_toniot = pd.read_csv(self.toniot_file, low_memory=False)
            self.log(f"  Loaded: {len(df_toniot):,} rows, {len(df_toniot.columns)} columns", "OK")
            self.progress_bar['value'] = 12
            self.root.update_idletasks()
            
            if not self.is_running:
                return
            
            # ÉTAPE 2: Load CIC (RÉCURSIF - cherche dans tous les sous-dossiers)
            self.stage_label.config(text="[2/8] Loading CIC...")
            self.log("\nÉTAPE 2: Chargement CIC", "HEADER")
            
            # Chercher récursivement les CSVs dans CIC et ses sous-dossiers
            cic_files = []
            for root, dirs, files in os.walk(self.cic_dir):
                cic_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
            
            # Trier pour cohérence
            cic_files.sort()
            
            dfs_cic = []
            for idx, csv_file in enumerate(cic_files, 1):
                if not self.is_running:
                    return
                self.log(f"  [{idx}/{len(cic_files)}] {os.path.basename(csv_file)}", "INFO")
                df = pd.read_csv(csv_file, low_memory=False)
                dfs_cic.append(df)
                self.progress_bar['value'] = 12 + (idx / len(cic_files)) * 13
                self.root.update_idletasks()
            
            self.progress_bar['value'] = 25
            
            if not self.is_running:
                return
            
            # ÉTAPE 3: Merge
            self.stage_label.config(text="[3/8] Merging data...")
            self.log("\nÉTAPE 3: Fusion TON_IoT + CIC", "HEADER")
            
            df_combined = pd.concat([df_toniot] + dfs_cic, ignore_index=True)
            self.log(f"  Combined: {len(df_combined):,} rows", "OK")
            self.progress_bar['value'] = 30
            self.root.update_idletasks()
            
            if not self.is_running:
                return
            
            # VÉRIFICATION 1: Label exists
            if 'Label' not in df_combined.columns:
                self.log("ERROR: Label column not found!", "ERROR")
                return
            self.log("  ✅ Label column present", "OK")
            
            # ÉTAPE 4: Clean
            self.stage_label.config(text="[4/8] Cleaning data...")
            self.log("\nÉTAPE 4: Nettoyage Données", "HEADER")
            
            initial = len(df_combined)
            df_combined = df_combined.drop_duplicates()
            dupes = initial - len(df_combined)
            self.log(f"  Duplicates removed: {dupes:,}", "INFO")
            
            df_combined = df_combined.dropna(subset=['Label'])
            self.log(f"  Valid rows: {len(df_combined):,}", "OK")
            self.progress_bar['value'] = 45
            self.root.update_idletasks()
            
            if len(df_combined) == 0:
                self.log("ERROR: No valid rows!", "ERROR")
                return
            
            if not self.is_running:
                return
            
            # ÉTAPE 5: Split
            self.stage_label.config(text="[5/8] Splitting data (60/40)...")
            self.log("\nÉTAPE 5: Split Scientifique 60/40", "HEADER")
            
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
            for train_idx, test_idx in sss.split(df_combined, df_combined['Label']):
                df_train = df_combined.iloc[train_idx].copy()
                df_test = df_combined.iloc[test_idx].copy()
            
            self.log(f"  Train: {len(df_train):,} rows (60%)", "INFO")
            self.log(f"  Test: {len(df_test):,} rows (40%)", "INFO")
            self.progress_bar['value'] = 55
            self.root.update_idletasks()
            
            # VÉRIFICATION 2: Same columns
            if list(df_train.columns) != list(df_test.columns):
                self.log("ERROR: Different columns in train/test!", "ERROR")
                return
            self.log("  ✅ Same columns", "OK")
            
            if not self.is_running:
                return
            
            # ÉTAPE 6: Select features
            self.stage_label.config(text="[6/8] Selecting features...")
            self.log("\nÉTAPE 6: Sélection Features", "HEADER")
            
            numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            if len(numeric_cols) == 0:
                self.log("ERROR: No numeric columns!", "ERROR")
                return
            
            self.log(f"  Numeric features: {len(numeric_cols)}", "OK")
            
            test_numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in test_numeric_cols:
                test_numeric_cols.remove('Label')
            
            if numeric_cols != test_numeric_cols:
                self.log("ERROR: Different numeric columns!", "ERROR")
                return
            self.log("  ✅ Compatible columns", "OK")
            self.progress_bar['value'] = 65
            self.root.update_idletasks()
            
            if not self.is_running:
                return
            
            # ÉTAPE 7: Write
            self.stage_label.config(text="[7/8] Writing CSV files...")
            self.log("\nÉTAPE 7: Écriture Fichiers", "HEADER")
            
            try:
                df_train.to_csv('fusion_train_smart4.csv', index=False, encoding='utf-8')
                train_size = os.path.getsize('fusion_train_smart4.csv') / (1024**3)
                self.log(f"  ✅ fusion_train_smart4.csv ({train_size:.2f} GB)", "OK")
            except Exception as e:
                self.log(f"ERROR writing train CSV: {e}", "ERROR")
                return
            
            self.progress_bar['value'] = 75
            self.root.update_idletasks()
            
            try:
                df_test.to_csv('fusion_test_smart4.csv', index=False, encoding='utf-8')
                test_size = os.path.getsize('fusion_test_smart4.csv') / (1024**3)
                self.log(f"  ✅ fusion_test_smart4.csv ({test_size:.2f} GB)", "OK")
            except Exception as e:
                self.log(f"ERROR writing test CSV: {e}", "ERROR")
                return
            
            if not self.is_running:
                return
            
            # ÉTAPE 8: Normalize and NPZ
            self.stage_label.config(text="[8/8] Creating NPZ...")
            self.log("\nÉTAPE 8: Normalisation et NPZ", "HEADER")
            
            X_train = df_train[numeric_cols].astype(np.float32).fillna(df_train[numeric_cols].mean())
            y_train = df_train['Label'].astype(str)
            
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            
            self.log(f"  Classes: {list(le.classes_)}", "INFO")
            self.log(f"  Distribution: {np.bincount(y_train_encoded)}", "INFO")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
            
            try:
                np.savez_compressed('preprocessed_dataset.npz',
                                   X=X_train_scaled,
                                   y=y_train_encoded,
                                   classes=le.classes_,
                                   numeric_cols=np.array(numeric_cols, dtype=object))
                
                npz_size = os.path.getsize('preprocessed_dataset.npz') / (1024**3)
                self.log(f"  ✅ preprocessed_dataset.npz ({npz_size:.2f} GB)", "OK")
                
                # Verify
                data = np.load('preprocessed_dataset.npz', allow_pickle=True)
                self.log(f"  ✅ Verified: X={data['X'].shape}, y={data['y'].shape}", "OK")
            except Exception as e:
                self.log(f"ERROR creating NPZ: {e}", "ERROR")
                return
            
            self.progress_bar['value'] = 100
            self.log("\nCONSOLIDATION COMPLETED SUCCESSFULLY", "HEADER")
            self.log(f"  ✅ fusion_train_smart4.csv ({train_size:.2f} GB)", "OK")
            self.log(f"  ✅ fusion_test_smart4.csv ({test_size:.2f} GB)", "OK")
            self.log(f"  ✅ preprocessed_dataset.npz ({npz_size:.2f} GB)", "OK")
            
            self.stage_label.config(text="✅ COMPLETED SUCCESSFULLY", fg='#27ae60')
            
        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}", "ERROR")
        
        finally:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = ConsolidationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()