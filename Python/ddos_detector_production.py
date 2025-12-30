#!/usr/bin/env python3
"""
DDoS DETECTOR - PRODUCTION SCRIPT
Charge le modèle final et prédit sur nouvelles données
"""

import joblib
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

class DDoSDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.classes = None
        
    def load_model(self):
        """Charger le modèle final"""
        try:
            self.model = joblib.load('ddos_detector_final.pkl')
            print("[OK] Modele charge: ddos_detector_final.pkl")
            return True
        except Exception as e:
            print(f"[ERROR] Erreur chargement modele: {e}")
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
            
            return results
        except Exception as e:
            print(f"[ERROR] Erreur batch prediction: {e}")
            return None


def main():
    print("[INFO] DDoS Detector - Production")
    
    # Charger le modele
    detector = DDoSDetector()
    if not detector.load_model():
        return False
    
    print("[OK] Modele pret pour les predictions")
    
    # Exemple: charger les donnees de test
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