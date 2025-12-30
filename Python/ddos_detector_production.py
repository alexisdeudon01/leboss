#!/usr/bin/env python3
"""
DDoS DETECTOR - PRODUCTION SCRIPT
Charge le modele final et predit sur nouvelles donnees
"""

import joblib
import numpy as np
import os
import threading
from sklearn.preprocessing import StandardScaler

# Optional GUI
try:
    from progress_gui import GenericProgressGUI
except ImportError:
    GenericProgressGUI = None
USE_GUI = os.getenv("USE_PROGRESS_GUI", "1") == "1"


class DDoSDetector:
    def __init__(self, ui=None):
        self.model = None
        self.scaler = None
        self.classes = None
        self.ui = ui
        if self.ui:
            self.ui.add_stage("load_model", "Chargement modele")
            self.ui.add_stage("load_data", "Chargement donnees")
            self.ui.add_stage("predict", "Prediction lot")

    def load_model(self):
        """Charger le modele final"""
        try:
            self.model = joblib.load('ddos_detector_final.pkl')
            print("[OK] Modele charge: ddos_detector_final.pkl")
            if self.ui:
                self.ui.update_stage("load_model", 1, 1, "Modele charge")
            return True
        except Exception as e:
            print(f"[ERROR] Erreur chargement modele: {e}")
            if self.ui:
                self.ui.log_alert(f"Erreur modele: {e}", level="error")
            return False

    def predict_single(self, sample):
        """Predire sur un seul echantillon"""
        if self.model is None:
            print("[ERROR] Modele non charge")
            return None

        try:
            # Assurer que l'input est 2D
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)

            prediction = self.model.predict(sample)[0]
            probability = self.model.predict_proba(sample)[0]

            result = {
                'prediction': 'DDoS' if prediction == 1 else 'Normal',
                'confidence': float(max(probability)),
                'probabilities': {
                    'normal': float(probability[0]),
                    'ddos': float(probability[1])
                }
            }

            return result
        except Exception as e:
            print(f"[ERROR] Erreur prediction: {e}")
            if self.ui:
                self.ui.log_alert(f"Erreur prediction: {e}", level="error")
            return None

    def predict_batch(self, samples):
        """Predire sur plusieurs echantillons"""
        if self.model is None:
            print("[ERROR] Modele non charge")
            return None

        try:
            predictions = self.model.predict(samples)
            probabilities = self.model.predict_proba(samples)

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'sample_id': i,
                    'prediction': 'DDoS' if pred == 1 else 'Normal',
                    'confidence': float(max(prob)),
                    'probabilities': {
                        'normal': float(prob[0]),
                        'ddos': float(prob[1])
                    }
                }
                results.append(result)
                if self.ui and i % 10 == 0:
                    self.ui.update_stage("predict", i + 1, len(samples), f"Batch {i+1}/{len(samples)}")

            if self.ui:
                self.ui.update_stage("predict", len(samples), len(samples), "Predictions terminees")
            return results
        except Exception as e:
            print(f"[ERROR] Erreur batch prediction: {e}")
            if self.ui:
                self.ui.log_alert(f"Erreur batch: {e}", level="error")
            return None


def run_with_gui():
    ui = GenericProgressGUI(title="DDoS Detector", header_info="Production", max_workers=2)
    detector = DDoSDetector(ui=ui)

    def worker():
        ui.update_global(0, 3, "Initialisation")
        if not detector.load_model():
            ui.log_alert("Echec chargement modele", level="error")
            return
        ui.update_global(1, 3, "Modele charge")
        try:
            data = np.load('preprocessed_dataset.npz', allow_pickle=True)
            X_test = data['X'][:100]
            y_test = data['y'][:100]
            ui.update_stage("load_data", len(X_test), len(X_test), "Donnees chargees")
            ui.update_global(2, 3, "Donnees chargees")
            results = detector.predict_batch(X_test)
            correct = 0
            for i, result in enumerate(results[:10]):
                if (y_test[i] == 1 and result['prediction'] == 'DDoS') or \
                   (y_test[i] == 0 and result['prediction'] == 'Normal'):
                    correct += 1
            ui.log(f"Precision sur 10 echantillons: {correct}/10", level="INFO")
            ui.update_global(3, 3, "Termine")
            ui.log_alert("Predictions terminees", level="success")
        except Exception as e:
            ui.log_alert(f"Erreur test: {e}", level="error")

    threading.Thread(target=worker, daemon=True).start()
    ui.start()


def main():
    if GenericProgressGUI and USE_GUI:
        run_with_gui()
        return

    print("[INFO] DDoS Detector - Production")

    detector = DDoSDetector()
    if not detector.load_model():
        return False

    print("[OK] Modele pret pour les predictions")

    try:
        data = np.load('preprocessed_dataset.npz', allow_pickle=True)
        X_test = data['X'][:100]  # Premiers 100 echantillons
        y_test = data['y'][:100]

        # Predire
        results = detector.predict_batch(X_test)

        # Afficher resultats
        correct = 0
        for i, result in enumerate(results[:10]):
            print(f"Sample {i}: Prediction={result['prediction']}, "
                  f"Confidence={result['confidence']:.4f}")
            if (y_test[i] == 1 and result['prediction'] == 'DDoS') or \
               (y_test[i] == 0 and result['prediction'] == 'Normal'):
                correct += 1

        print(f"[OK] Precision sur premiers 10: {correct}/10")

    except Exception as e:
        print(f"[ERROR] Erreur test: {e}")


if __name__ == "__main__":
    main()
