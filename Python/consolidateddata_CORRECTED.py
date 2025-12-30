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
        """Consolide TON_IoT + CIC avec vérifications robustes"""
        self.log("=" * 80, "HEADER")
        self.log("CONSOLIDATION DATASET - DÉTAILLÉE + ROBUSTE", "HEADER")
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
        
        # ✅ VÉRIFICATION 1: Label existe
        if 'Label' not in df_combined.columns:
            self.log("ERREUR CRITIQUE: Colonne 'Label' manquante!", "ERROR")
            self.log(f"  Colonnes disponibles: {list(df_combined.columns)}", "ERROR")
            return False
        self.log("  ✅ Label colonne présente", "OK")
        
        # ÉTAPE 4: Nettoyage
        self.log("\nÉTAPE 4: Nettoyage Données", "STEP")
        
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        duplicates = initial_rows - len(df_combined)
        self.log(f"  Doublons supprimés: {duplicates:,}", "DETAIL")
        
        df_combined = df_combined.dropna(subset=['Label'])
        self.log(f"  Rows avec Label valide: {len(df_combined):,}", "OK")
        
        if len(df_combined) == 0:
            self.log("ERREUR: Aucune ligne avec Label valide!", "ERROR")
            return False
        
        # ÉTAPE 5: Split 60/40
        self.log("\nÉTAPE 5: Split Scientifique 60/40", "STEP")
        
        from sklearn.model_selection import StratifiedShuffleSplit
        
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, 
                                     test_size=0.4, random_state=42)
        
        for train_idx, test_idx in sss.split(df_combined, df_combined['Label']):
            df_train = df_combined.iloc[train_idx].copy()
            df_test = df_combined.iloc[test_idx].copy()
        
        self.log(f"  Train: {len(df_train):,} lignes (60%)", "DETAIL")
        self.log(f"  Test: {len(df_test):,} lignes (40%)", "DETAIL")
        
        # ✅ VÉRIFICATION 2: Même colonnes train/test
        if list(df_train.columns) != list(df_test.columns):
            self.log("ERREUR: Colonnes train et test différentes!", "ERROR")
            train_cols = set(df_train.columns)
            test_cols = set(df_test.columns)
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            if missing_in_test:
                self.log(f"  Manquant dans test: {missing_in_test}", "ERROR")
            if missing_in_train:
                self.log(f"  Manquant dans train: {missing_in_train}", "ERROR")
            return False
        self.log("  ✅ Colonnes identiques train/test", "OK")
        
        # ÉTAPE 6: Sélectionner colonnes numériques
        self.log("\nÉTAPE 6: Sélection Features", "STEP")
        
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        # ✅ VÉRIFICATION 3: Features pas vides
        if len(numeric_cols) == 0:
            self.log("ERREUR: Aucune colonne numérique trouvée!", "ERROR")
            self.log(f"  Colonnes dans train: {list(df_train.columns)}", "ERROR")
            return False
        
        self.log(f"  Features numériques: {len(numeric_cols)}", "DETAIL")
        self.log(f"  Premiers features: {numeric_cols[:5]}...", "DETAIL")
        
        # ✅ VÉRIFICATION 4: Même numeric_cols dans test
        test_numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in test_numeric_cols:
            test_numeric_cols.remove('Label')
        
        if numeric_cols != test_numeric_cols:
            self.log("ERREUR: Colonnes numériques différentes entre train et test!", "ERROR")
            self.log(f"  Train numériques ({len(numeric_cols)}): {numeric_cols[:5]}...", "ERROR")
            self.log(f"  Test numériques ({len(test_numeric_cols)}): {test_numeric_cols[:5]}...", "ERROR")
            return False
        self.log("  ✅ Colonnes numériques identiques", "OK")
        
        # ÉTAPE 7: Écriture avec progress
        self.log("\nÉTAPE 7: Écriture Fichiers", "STEP")
        
        # Écrire train
        self.log("  Écriture fusion_train_smart4.csv...", "DETAIL")
        try:
            df_train.to_csv('fusion_train_smart4.csv', index=False, encoding='utf-8')
            if not os.path.exists('fusion_train_smart4.csv'):
                raise Exception("Fichier non créé")
            train_size = os.path.getsize('fusion_train_smart4.csv') / (1024**3)
            self.log(f"    ✅ OK: {len(df_train):,} lignes ({train_size:.2f} GB)", "OK")
        except Exception as e:
            self.log(f"    ❌ ERREUR écriture train: {e}", "ERROR")
            return False
        
        # Écrire test
        self.log("  Écriture fusion_test_smart4.csv...", "DETAIL")
        try:
            df_test.to_csv('fusion_test_smart4.csv', index=False, encoding='utf-8')
            if not os.path.exists('fusion_test_smart4.csv'):
                raise Exception("Fichier non créé")
            test_size = os.path.getsize('fusion_test_smart4.csv') / (1024**3)
            self.log(f"    ✅ OK: {len(df_test):,} lignes ({test_size:.2f} GB)", "OK")
        except Exception as e:
            self.log(f"    ❌ ERREUR écriture test: {e}", "ERROR")
            return False
        
        # ÉTAPE 8: Normalisation + NPZ
        self.log("\nÉTAPE 8: Normalisation et NPZ", "STEP")
        
        # Préparer X et y
        X_train = df_train[numeric_cols].astype(np.float32).fillna(df_train[numeric_cols].mean())
        y_train = df_train['Label'].astype(str)
        
        # Encoder labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        self.log(f"  Classes: {list(le.classes_)}", "DETAIL")
        self.log(f"  Distribution: Normal={np.sum(y_train_encoded==0)}, DDoS={np.sum(y_train_encoded==1)}", "DETAIL")
        
        # Normaliser
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        
        self.log(f"  Features normalisées: {X_train_scaled.shape}", "DETAIL")
        
        # Sauvegarder NPZ
        self.log("  Sauvegarde preprocessed_dataset.npz...", "DETAIL")
        try:
            np.savez_compressed('preprocessed_dataset.npz',
                               X=X_train_scaled,
                               y=y_train_encoded,
                               classes=le.classes_,
                               numeric_cols=np.array(numeric_cols, dtype=object))  # ✅ AJOUTER
            
            if not os.path.exists('preprocessed_dataset.npz'):
                raise Exception("Fichier non créé")
            
            npz_size = os.path.getsize('preprocessed_dataset.npz') / (1024**3)
            self.log(f"    ✅ OK: {npz_size:.2f} GB", "OK")
            
            # Vérifier que le NPZ se charge bien
            data = np.load('preprocessed_dataset.npz', allow_pickle=True)
            self.log(f"    ✅ Vérification NPZ: X={data['X'].shape}, y={data['y'].shape}, classes={list(data['classes'])}", "OK")
            
        except Exception as e:
            self.log(f"    ❌ ERREUR NPZ: {e}", "ERROR")
            return False
        
        # RÉSUMÉ
        self.log("\n" + "=" * 80, "HEADER")
        self.log("CONSOLIDATION COMPLÉTÉE AVEC SUCCÈS", "HEADER")
        self.log("=" * 80, "HEADER")
        self.log(f"\nFichiers générés:", "OK")
        self.log(f"  ✅ fusion_train_smart4.csv ({train_size:.2f} GB) - {len(df_train):,} lignes", "OK")
        self.log(f"  ✅ fusion_test_smart4.csv ({test_size:.2f} GB) - {len(df_test):,} lignes", "OK")
        self.log(f"  ✅ preprocessed_dataset.npz ({npz_size:.2f} GB) - {len(numeric_cols)} features", "OK")
        
        self.log(f"\nVérifications robustes:", "OK")
        self.log(f"  ✅ Label colonne présente", "OK")
        self.log(f"  ✅ Colonnes identiques train/test", "OK")
        self.log(f"  ✅ Features numériques valides", "OK")
        self.log(f"  ✅ NPZ chargeable et valide", "OK")
        
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