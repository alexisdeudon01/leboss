#!/usr/bin/env python3
"""
DDoS DETECTOR - PRODUCTION SCRIPT - CORRIGÉ
Charge le modèle final et prédit sur nouvelles données
✅ CORRECTION: Utilise MEMES CLASSES que training
✅ Normalisation avec stats du training
"""

import joblib
import numpy as np
import pandas as pd
import os
import threading
from sklearn.preprocessing import LabelEncoder

# Optional GUI
try:
    from progress_gui import GenericProgressGUI
except ImportError:
    GenericProgressGUI = None

USE_GUI = os.getenv("USE_PROGRESS_GUI", "1") == "1"


class DDoSDetector:
    """✅ CORRIGÉ: Gestion cohérente des classes et normalisation"""
    
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

    def load_model_and_scaler(self):
        """Charger le modèle et les paramètres de normalisation"""
        try:
            # Charger le modèle
            if not os.path.exists('ddos_detector_final.pkl'):
                print("[ERROR] ddos_detector_final.pkl manquant")
                if self.ui:
                    self.ui.log_alert("Modèle manquant", level="error")
                return False
            
            self.model = joblib.load('ddos_detector_final.pkl')
            print("[OK] Modèle chargé: ddos_detector_final.pkl")
            
            # Charger les paramètres de normalisation depuis le dataset d'entraînement
            if not os.path.exists('preprocessed_dataset.npz'):
                print("[ERROR] preprocessed_dataset.npz manquant pour scaler")
                if self.ui:
                    self.ui.log_alert("Données d'entraînement manquantes", level="error")
                return False
            
            data = np.load('preprocessed_dataset.npz', allow_pickle=True)
            X_train = data['X']
            
            # ✅ Sauvegarder les paramètres de normalisation du training
            self.scaler_mean = X_train.mean(axis=0)
            self.scaler_std = X_train.std(axis=0) + 1e-8
            
            # ✅ CORRECTION: Charger les classes du training
            self.classes = data['classes']
            
            # Créer le label encoder avec les MEMES classes
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = self.classes
            
            print(f"[OK] Scaler et classes chargés")
            print(f"    Mean shape: {self.scaler_mean.shape}")
            print(f"    Classes: {list(self.classes)}")
            
            if self.ui:
                self.ui.update_stage("load_model", 1, 1, "Modèle et scaler chargés")
            
            return True
        except Exception as e:
            print(f"[ERROR] Erreur chargement modèle: {e}")
            if self.ui:
                self.ui.log_alert(f"Erreur modèle: {e}", level="error")
            return False

    def normalize_features(self, X_raw):
        """Normaliser les features avec les paramètres du training"""
        if self.scaler_mean is None or self.scaler_std is None:
            print("[ERROR] Scaler non inicalisé")
            return None
        
        try:
            X_normalized = ((X_raw - self.scaler_mean) / self.scaler_std).astype(np.float32)
            return X_normalized
        except Exception as e:
            print(f"[ERROR] Erreur normalisation: {e}")
            return None

    def predict_single(self, sample):
        """Prédire sur un seul échantillon"""
        if self.model is None:
            print("[ERROR] Modèle non chargé")
            return None

        try:
            # Assurer que l'input est 2D
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)

            # Normaliser
            sample_norm = self.normalize_features(sample)
            if sample_norm is None:
                return None

            # Prédire
            prediction = self.model.predict(sample_norm)[0]
            probability = self.model.predict_proba(sample_norm)[0]

            # ✅ CORRECTION: Utiliser les classes correctes
            pred_label = self.classes[prediction]
            
            result = {
                'prediction': pred_label,
                'prediction_id': int(prediction),
                'confidence': float(max(probability)),
                'probabilities': {
                    self.classes[0]: float(probability[0]),
                    self.classes[1]: float(probability[1])
                }
            }

            return result
        except Exception as e:
            print(f"[ERROR] Erreur prédiction: {e}")
            if self.ui:
                self.ui.log_alert(f"Erreur prédiction: {e}", level="error")
            return None

    def predict_batch(self, samples):
        """Prédire sur plusieurs échantillons"""
        if self.model is None:
            print("[ERROR] Modèle non chargé")
            return None

        try:
            # Normaliser
            samples_norm = self.normalize_features(samples)
            if samples_norm is None:
                return None

            # Prédire
            predictions = self.model.predict(samples_norm)
            probabilities = self.model.predict_proba(samples_norm)

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                # ✅ CORRECTION: Utiliser les classes correctes
                pred_label = self.classes[pred]
                
                result = {
                    'sample_id': i,
                    'prediction': pred_label,
                    'prediction_id': int(pred),
                    'confidence': float(max(prob)),
                    'probabilities': {
                        self.classes[0]: float(prob[0]),
                        self.classes[1]: float(prob[1])
                    }
                }
                results.append(result)
                
                if self.ui and i % 100 == 0:
                    self.ui.update_stage("predict", i + 1, len(samples), 
                                        f"Batch {i+1}/{len(samples)}")

            if self.ui:
                self.ui.update_stage("predict", len(samples), len(samples), 
                                    "Prédictions terminées")
            
            return results
        except Exception as e:
            print(f"[ERROR] Erreur batch prédiction: {e}")
            if self.ui:
                self.ui.log_alert(f"Erreur batch: {e}", level="error")
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
            
            # Charger test holdout
            if not os.path.exists("fusion_test_smart4.csv"):
                ui.log_alert("fusion_test_smart4.csv introuvable", level="error")
                return
            
            df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False)
            numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
            X_test_raw = df_test[numeric_cols].astype(np.float32)
            X_test_raw = X_test_raw.fillna(X_test_raw.mean())
            
            ui.update_stage("load_data", len(df_test), len(df_test), "Test chargé")
            
            # Prédire
            results = detector.predict_batch(X_test_raw.values)
            
            if results:
                ui.update_global(3, 3, "Terminé")
                
                # Statistiques
                ddos_count = sum(1 for r in results if r['prediction'] == 'DDoS')
                normal_count = len(results) - ddos_count
                
                summary = f"Prédictions terminées:\n  DDoS: {ddos_count:,}\n  Normal: {normal_count:,}"
                ui.log_alert(summary, level="success")
                print(f"\n[OK] {summary}")
            
        except Exception as e:
            ui.log_alert(f"Erreur: {e}", level="error")

    threading.Thread(target=worker, daemon=True).start()
    ui.start()


def main():
    """Point d'entrée principal"""
    print("\n" + "="*80)
    print("DDoS DETECTOR - PRODUCTION (TEST HOLDOUT)")
    print("="*80 + "\n")

    detector = DDoSDetector()
    
    if not detector.load_model_and_scaler():
        print("[ERROR] Impossible charger modèle")
        return False

    # Mode GUI si disponible
    if GenericProgressGUI and USE_GUI:
        print("[INFO] Lancement mode GUI...")
        run_with_gui()
        return True

    # Mode console
    try:
        print("[INFO] Chargement test holdout...")
        
        if not os.path.exists("fusion_test_smart4.csv"):
            print("[ERROR] fusion_test_smart4.csv introuvable")
            return False
        
        df_test = pd.read_csv("fusion_test_smart4.csv", low_memory=False)
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
            
            # Sauvegarder les prédictions
            pred_df = pd.DataFrame(results)
            pred_df.to_csv("ddos_predictions_test_holdout.csv", index=False)
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
    import sys
    sys.exit(0 if success else 1)