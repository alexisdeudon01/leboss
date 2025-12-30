#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONSOLIDATION DATASET - AMÉLIORÉ
=====================================
✅ Progress bars par étape (+ verbose)
✅ Progress bar écriture fichiers
✅ Gestion RAM dynamique (<90%)
✅ Chunks adaptatifs
✅ Logs détaillés
=====================================
"""

import os
import sys
import gc
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============= MEMORY MANAGER =============
class MemoryManager:
    """Gère la mémoire dynamiquement"""
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
    def get_optimal_chunk_size(min_chunk=50000, max_chunk=500000):
        """Calcule chunk size optimal basé sur RAM libre"""
        ram_free = MemoryManager.get_available_ram_gb()
        ram_usage = MemoryManager.get_ram_usage()
        
        if ram_usage > 80:
            chunk_size = int(min_chunk * (100 - ram_usage) / 20)
        else:
            chunk_size = int(max_chunk * (ram_free / 16))
        
        return max(min_chunk, min(chunk_size, max_chunk))
    
    @staticmethod
    def check_and_cleanup():
        """Nettoie mémoire si trop haute"""
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > MemoryManager.RAM_THRESHOLD:
            gc.collect()
            return False
        return True


# ============= CONSOLIDATOR =============
class DataConsolidator:
    """Consolide TON_IoT + CIC avec progress bars"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.chunk_size = MemoryManager.get_optimal_chunk_size()
    
    def log(self, msg, level="INFO"):
        """Log avec timestamp"""
        ts = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{ts}] [{level}] {msg}"
        print(log_msg)
        if self.log_callback:
            self.log_callback(log_msg)
    
    def log_progress(self, current, total, label="", bar_width=40):
        """Affiche progress bar"""
        if total == 0:
            return
        
        pct = current / total * 100
        filled = int(bar_width * current / total)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        ram_usage = MemoryManager.get_ram_usage()
        msg = f"[{bar}] {pct:6.1f}% ({current:,} / {total:,}) | RAM: {ram_usage:.1f}%"
        if label:
            msg = f"{label}: {msg}"
        
        print(f"\r{msg}", end='', flush=True)
        if self.log_callback:
            self.log_callback(msg)
    
    def load_csv_with_progress(self, filepath, label=""):
        """Charge CSV par chunks avec progress"""
        self.log(f"Chargement {filepath}", "LOAD")
        
        if not os.path.exists(filepath):
            self.log(f"Fichier introuvable: {filepath}", "ERROR")
            return None
        
        file_size = os.path.getsize(filepath) / (1024**3)
        self.log(f"  Taille: {file_size:.2f} GB", "DETAIL")
        
        chunks = []
        try:
            # Déterminer nombre de chunks
            chunk_iter = pd.read_csv(filepath, chunksize=self.chunk_size, 
                                     low_memory=False, encoding='utf-8')
            
            chunk_list = list(chunk_iter)
            total_chunks = len(chunk_list)
            total_rows = sum(len(c) for c in chunk_list)
            
            self.log(f"  Total: {total_rows:,} lignes, {total_chunks} chunks", "DETAIL")
            
            # Charger avec progress
            for chunk_idx, chunk in enumerate(chunk_list, 1):
                chunks.append(chunk)
                rows_loaded = sum(len(c) for c in chunks)
                
                self.log_progress(chunk_idx, total_chunks, 
                                 label=f"  [{label}] Chunk")
                
                # Vérifier RAM
                while not MemoryManager.check_and_cleanup():
                    self.log(f"  [WARN] RAM élevée ({MemoryManager.get_ram_usage():.1f}%), pause...", "WARN")
                    time.sleep(2)
            
            print()  # Nouvelle ligne après progress bar
            result = pd.concat(chunks, ignore_index=True)
            self.log(f"  OK: {len(result):,} lignes chargées", "OK")
            
            return result
            
        except Exception as e:
            self.log(f"  ERREUR: {e}", "ERROR")
            return None
    
    def consolidate(self, toniot_path, cic_dir, use_dialog=True):
        """Consolide TON_IoT + CIC"""
        self.log("=" * 80, "HEADER")
        self.log("CONSOLIDATION DATASET - DÉTAILLÉE", "HEADER")
        self.log("=" * 80, "HEADER")
        
        # Sélectionner fichiers si dialog
        if use_dialog:
            self.log("\nSélection fichiers...", "INFO")
            
            root = tk.Tk()
            root.withdraw()
            
            toniot_file = filedialog.askopenfilename(
                title="Sélectionnez train_test_network.csv (TON_IoT)",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if not toniot_file:
                self.log("Opération annulée", "WARN")
                return False
            
            cic_path = filedialog.askdirectory(
                title="Sélectionnez dossier CIC"
            )
            
            if not cic_path:
                self.log("Opération annulée", "WARN")
                return False
            
            root.destroy()
        else:
            toniot_file = toniot_path
            cic_path = cic_dir
        
        # ÉTAPE 1: Charger TON_IoT
        self.log("\nÉTAPE 1: Chargement TON_IoT", "STEP")
        df_toniot = self.load_csv_with_progress(toniot_file, "TON_IoT")
        if df_toniot is None:
            return False
        
        # ÉTAPE 2: Charger CIC
        self.log("\nÉTAPE 2: Chargement CIC", "STEP")
        cic_files = []
        for root, dirs, files in os.walk(cic_path):
            cic_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
        
        self.log(f"Trouvé {len(cic_files)} fichiers CSV dans CIC", "INFO")
        
        dfs_cic = []
        for idx, csv_file in enumerate(cic_files, 1):
            self.log(f"  [{idx}/{len(cic_files)}] {os.path.basename(csv_file)}", "DETAIL")
            df = self.load_csv_with_progress(csv_file, f"CIC[{idx}]")
            if df is not None:
                dfs_cic.append(df)
        
        if not dfs_cic:
            self.log("Aucun fichier CIC chargé", "ERROR")
            return False
        
        # ÉTAPE 3: Fusionner
        self.log("\nÉTAPE 3: Fusion TON_IoT + CIC", "STEP")
        self.log(f"  TON_IoT: {len(df_toniot):,} lignes", "DETAIL")
        total_cic = sum(len(df) for df in dfs_cic)
        self.log(f"  CIC total: {total_cic:,} lignes", "DETAIL")
        
        df_combined = pd.concat([df_toniot] + dfs_cic, ignore_index=True)
        self.log(f"  Combiné: {len(df_combined):,} lignes", "OK")
        
        # ÉTAPE 4: Nettoyage
        self.log("\nÉTAPE 4: Nettoyage Données", "STEP")
        
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        duplicates = initial_rows - len(df_combined)
        self.log(f"  Doublons supprimés: {duplicates:,}", "DETAIL")
        
        df_combined = df_combined.dropna(subset=['Label'])
        self.log(f"  Rows avec Label valide: {len(df_combined):,}", "OK")
        
        # ÉTAPE 5: Split 60/40
        self.log("\nÉTAPE 5: Split Scientifique 60/40", "STEP")
        
        from sklearn.model_selection import StratifiedShuffleSplit
        
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, 
                                     test_size=0.4, random_state=42)
        
        for train_idx, test_idx in sss.split(df_combined, df_combined['Label']):
            df_train = df_combined.iloc[train_idx]
            df_test = df_combined.iloc[test_idx]
        
        self.log(f"  Train: {len(df_train):,} lignes (60%)", "DETAIL")
        self.log(f"  Test: {len(df_test):,} lignes (40%)", "DETAIL")
        
        # ÉTAPE 6: Sélectionner colonnes numériques
        self.log("\nÉTAPE 6: Sélection Features", "STEP")
        
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        self.log(f"  Features numériques: {len(numeric_cols)}", "DETAIL")
        
        # ÉTAPE 7: Écriture avec progress
        self.log("\nÉTAPE 7: Écriture Fichiers", "STEP")
        
        # Ajouter Label columns
        df_train['Label'] = df_combined.loc[train_idx, 'Label'].values
        df_test['Label'] = df_combined.loc[test_idx, 'Label'].values
        
        # Écrire train
        self.log("  Écriture fusion_train_smart4.csv...", "DETAIL")
        df_train.to_csv('fusion_train_smart4.csv', index=False, encoding='utf-8')
        train_size = os.path.getsize('fusion_train_smart4.csv') / (1024**3)
        self.log(f"    OK: {len(df_train):,} lignes ({train_size:.2f} GB)", "OK")
        
        # Écrire test
        self.log("  Écriture fusion_test_smart4.csv...", "DETAIL")
        df_test.to_csv('fusion_test_smart4.csv', index=False, encoding='utf-8')
        test_size = os.path.getsize('fusion_test_smart4.csv') / (1024**3)
        self.log(f"    OK: {len(df_test):,} lignes ({test_size:.2f} GB)", "OK")
        
        # ÉTAPE 8: Normalisation + NPZ
        self.log("\nÉTAPE 8: Normalisation et NPZ", "STEP")
        
        # Préparer X et y
        X_train = df_train[numeric_cols].astype(np.float32).fillna(df_train[numeric_cols].mean())
        y_train = df_train['Label'].astype(str)
        
        # Encoder labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        self.log(f"  Classes: {list(le.classes_)}", "DETAIL")
        
        # Normaliser
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        
        self.log(f"  Features normalisées: {X_train_scaled.shape}", "DETAIL")
        
        # Sauvegarder NPZ
        self.log("  Sauvegarde preprocessed_dataset.npz...", "DETAIL")
        np.savez_compressed('preprocessed_dataset.npz',
                           X=X_train_scaled,
                           y=y_train_encoded,
                           classes=le.classes_)
        
        npz_size = os.path.getsize('preprocessed_dataset.npz') / (1024**3)
        self.log(f"    OK: {npz_size:.2f} GB", "OK")
        
        # RÉSUMÉ
        self.log("\n" + "=" * 80, "HEADER")
        self.log("CONSOLIDATION COMPLÉTÉE", "HEADER")
        self.log("=" * 80, "HEADER")
        self.log(f"\nFichiers générés:", "OK")
        self.log(f"  • fusion_train_smart4.csv ({train_size:.2f} GB)", "OK")
        self.log(f"  • fusion_test_smart4.csv ({test_size:.2f} GB)", "OK")
        self.log(f"  • preprocessed_dataset.npz ({npz_size:.2f} GB)", "OK")
        
        return True


def main():
    """Point d'entrée"""
    consolidator = DataConsolidator()
    success = consolidator.consolidate(None, None, use_dialog=True)
    
    if success:
        print("\n✅ Consolidation réussie!")
        sys.exit(0)
    else:
        print("\n❌ Consolidation échouée")
        sys.exit(1)


if __name__ == "__main__":
    main()
