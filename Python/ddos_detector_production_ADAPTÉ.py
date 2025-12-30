#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDoS DETECTOR - PRODUCTION - ADAPTÉ + OPTIMISÉ RAM
==================================================
✅ progress_gui optionnel (console fallback)
✅ Classes cohérentes du training
✅ Normalisation training stats
✅ Gestion RAM dynamique (<90%)
==================================================
"""

import joblib
import numpy as np
import pandas as pd
import os
import sys
import threading
import psutil
import gc
from sklearn.preprocessing import LabelEncoder

try:
    from progress_gui import GenericProgressGUI
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    GenericProgressGUI = None

USE_GUI = HAS_GUI and os.getenv("USE_PROGRESS_GUI", "1") == "1"

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
    def check_and_cleanup():
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > MemoryManager.RAM_THRESHOLD:
            gc.collect()
            return False
        return True
    
    @staticmethod
    def get_optimal_batch_size(min_batch=1000, max_batch=100000):
        """Batch size adaptatif"""
        ram_usage = MemoryManager.get_ram_usage()
        if ram_usage > 80:
            batch_size = int(min_batch * (100 - ram_usage) / 20)
        else:
            batch_size = max_batch
        return max(min_batch, batch_size)


# ============= DDoS DETECTOR =============
class DDoSDetector:
    """Détecteur DDoS avec gestion RAM"""
    
    def __init__(self, ui=None):
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.classes = None
        self.label_encoder = None
        self.ui = ui
        
        if self.ui:
            self.ui.add_stage("load_model", "Chargement modèle")
            self.ui.add_stage("load_data", "Chargement données")
            self.ui.add_stage("predict", "Prédiction lot")

    def log(self, msg, level="OK"):
        """Log compatible GUI + console"""
        if self.ui:
            self.ui.log(msg, level=level)
        else:
            import time
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

    def log_alert(self, msg, level="error"):
        """Alert compatible GUI + console"""
        if self.ui:
            self.ui.log_alert(msg, level=level)
        else:
            print(f"[ALERT] {msg}")

    def load_model_and_scaler(self):
        """Charge modèle et paramètres normalisation"""
        try:
            if not os.path.exists('ddos_detector_final.pkl'):
                self.log_alert("ddos_detector_final.pkl manquant", level="error")
                return False
            
            self.model = joblib.load('ddos_detector_final.pkl')
            self.log("Modèle chargé: ddos_detector_final.pkl", level="OK")
            
            if not os.path.exists('preprocessed_dataset.npz'):
                self.log_alert("preprocessed_dataset.npz manquant", level="error")
                return False
            
            # Nettoyer avant charge
            gc.collect()
            
            data = np.load('preprocessed_dataset.npz', allow_pickle=True)
            X_train = data['X']
            
            # Paramètres normalisation
            self.scaler_mean = X_train.mean(axis=0)
            self.scaler_std = X_train.std(axis=0) + 1e-8
            
            # Classes du training
            self.classes = data['classes']
            
            # Label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = self.classes
            
            self.log(f"Scaler et classes chargés (Mean shape: {self.scaler_mean.shape})", level="OK")
            self.log(f"Classes: {list(self.classes)}", level="OK")
            self.log(f"RAM: {MemoryManager.get_ram_usage():.1f}%", level="DETAIL")
            
            if self.ui:
                self.ui.update_stage("load_model", 1, 1, "Modèle et scaler chargés")
            
            # Cleanup
            del data, X_train
            gc.collect()
            
            return True
        except Exception as e:
            self.log_alert(f"Erreur chargement modèle: {e}", level="error")
            return False

    def normalize_features(self, X_raw):
        """Normalise avec stats training"""
        if self.scaler_mean is None or self.scaler_std is None:
            self.log_alert("Scaler non initialisé", level="error")
            return None
        
        try:
            X_normalized = ((X_raw - self.scaler_mean) / self.scaler_std).astype(np.float32)
            return X_normalized
        except Exception as e:
            self.log_alert(f"Erreur normalisation: {e}", level="error")
            return None

    def predict_batch(self, samples, batch_size=None):
        """Prédiction batch avec gestion RAM"""
        if self.model is None:
            self.log_alert("Modèle non chargé", level="error")
            return None

        try:
            if batch_size is None:
                batch_size = MemoryManager.get_optimal_batch_size()
            
            results = []
            total_samples = len(samples)
            
            for batch_idx in range(0, total_samples, batch_size):
                # Vérifier RAM
                if not MemoryManager.check_and_cleanup():
                    self.log("RAM critique, pause", level="WARN")
                    import time
                    time.sleep(1)
                
                batch_end = min(batch_idx + batch_size, total_samples)
                batch = samples[batch_idx:batch_end]
                
                # Normaliser
                batch_norm = self.normalize_features(batch)
                if batch_norm is None:
                    return None
                
                # Prédire
                predictions = self.model.predict(batch_norm)
                probabilities = self.model.predict_proba(batch_norm)
                
                # Formater résultats
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    pred_label = self.classes[pred]
                    
                    result = {
                        'sample_id': batch_idx + i,
                        'prediction': str(pred_label),
                        'prediction_id': int(pred),
                        'confidence': float(max(prob)),
                        'probabilities': {
                            str(self.classes[0]): float(prob[0]),
                            str(self.classes[1]): float(prob[1])
                        }
                    }
                    results.append(result)
                
                # Progress
                pct = batch_end / total_samples * 100
                self.log(f"Batch {batch_idx//batch_size + 1}: {pct:.1f}% (RAM: {MemoryManager.get_ram_usage():.1f}%)", level="DETAIL")
                
                if self.ui:
                    self.ui.update_stage("predict", batch_end, total_samples, 
                                        f"Batch {batch_end}/{total_samples}")
                
                # Cleanup
                del batch, batch_norm, predictions, probabilities
                gc.collect()
            
            if self.ui:
                self.ui.update_stage("predict", total_samples, total_samples, 
                                    "Prédictions terminées")
            
            return results
        except Exception as e:
            self.log_alert(f"Erreur prédiction: {e}", level="error")
            return None


def run_with_gui():
    """Mode avec GUI"""
    ui = GenericProgressGUI(title="DDoS Detector Production", 
                           header_info="Test holdout", 
                           max_workers=2)
    detector = DDoSDetector(ui=ui)

    def worker():
        try:
            ui.update_global(0, 3, "Initialisation")
            
            if not detector.load_model_and_scaler():
                ui.log_alert("Échec chargement modèle", level="error")
                return
            
            ui.update_global(1, 3, "Modèle chargé")
            
            # Charger test
            if not os.path.exists("fusion_test_smart4.csv"):
                ui.log_alert("fusion_test_smart4.csv introuvable", level="error")
                return
            
            df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False, encoding='utf-8')
            numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            X_test_raw = df_test[numeric_cols].astype(np.float32)
            X_test_raw = X_test_raw.fillna(X_test_raw.mean())
            
            ui.update_stage("load_data", len(df_test), len(df_test), "Test chargé")
            
            # Prédire
            results = detector.predict_batch(X_test_raw.values)
            
            if results:
                ui.update_global(3, 3, "Terminé")
                
                ddos_count = sum(1 for r in results if r['prediction'] == 'DDoS')
                normal_count = len(results) - ddos_count
                
                summary = f"Prédictions terminées:\n  DDoS: {ddos_count:,}\n  Normal: {normal_count:,}"
                ui.log_alert(summary, level="success")
        except Exception as e:
            ui.log_alert(f"Erreur: {e}", level="error")

    threading.Thread(target=worker, daemon=True).start()
    ui.start()


def main():
    """Point d'entrée"""
    print("\n" + "="*80)
    print("DDoS DETECTOR - PRODUCTION (TEST HOLDOUT) - Gestion RAM Optimale")
    print("="*80 + "\n")

    detector = DDoSDetector()
    
    if not detector.load_model_and_scaler():
        print("[ERROR] Impossible charger modèle")
        return False

    # Mode GUI
    if USE_GUI:
        print("[INFO] Mode GUI activé\n")
        run_with_gui()
        return True

    # Mode console
    print("[INFO] Mode console\n")
    try:
        print("[INFO] Chargement test holdout...")
        
        if not os.path.exists("fusion_test_smart4.csv"):
            print("[ERROR] fusion_test_smart4.csv introuvable")
            return False
        
        df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False, encoding='utf-8')
        print(f"[OK] Test chargé: {len(df_test):,} lignes")
        
        numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
        X_test_raw = df_test[numeric_cols].astype(np.float32)
        X_test_raw = X_test_raw.fillna(X_test_raw.mean())
        
        print("[INFO] Prédiction sur test holdout...")
        results = detector.predict_batch(X_test_raw.values)
        
        if results:
            ddos_count = sum(1 for r in results if r['prediction'] == 'DDoS')
            normal_count = len(results) - ddos_count
            
            print(f"\n[OK] Prédictions complètes:")
            print(f"    Total: {len(results):,}")
            print(f"    DDoS:  {ddos_count:,} ({ddos_count/len(results)*100:.1f}%)")
            print(f"    Normal: {normal_count:,} ({normal_count/len(results)*100:.1f}%)")
            
            # Sauvegarder
            pred_df = pd.DataFrame(results)
            pred_df.to_csv("ddos_predictions_test_holdout.csv", index=False, encoding='utf-8')
            print(f"\n[OK] Prédictions sauvegardées: ddos_predictions_test_holdout.csv")
            
            return True
        else:
            print("[ERROR] Erreur prédictions")
            return False
            
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
