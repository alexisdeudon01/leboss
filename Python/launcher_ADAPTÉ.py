#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDoS DETECTION PROJECT - LAUNCHER (AMÉLIORÉ)
=============================================
✅ Menu interactif pour choisir mode
✅ Vérification fichiers
✅ Pipeline séquentielle
✅ Scripts ADAPTÉ supportés
=============================================
"""

import os
import sys
import subprocess

def _build_consolidation_env() -> dict:
    env = os.environ.copy()
    # defaults (user can override before launching)
    env.setdefault('FULL_RUN', '0')
    env.setdefault('SAMPLE_ROWS', '1000')
    return env

import time

def print_header():
    print("\n" + "="*80)
    print("DDoS DETECTION PROJECT - LAUNCHER (AMÉLIORÉ)")
    print("="*80)

def print_menu():
    print_header()
    print("\nChoisissez le mode d'exécution:\n")
    print("  [1] CONSOLIDATION DATASET")
    print("      └─ GUI Tkinter pour sélection fichiers")
    print("      └─ Génère fusion_train_smart4.csv + fusion_test_smart4.csv")
    print("      └─ Durée estimée: 10-40 minutes\n")
    print("  [2] ML EVALUATION (ADAPTÉ)")
    print("      └─ GUI si progress_gui disponible, sinon console")
    print("      └─ K-Fold validation + rapports")
    print("      └─ Durée estimée: 1-6 heures\n")
    print("  [3] DDoS DETECTOR (ADAPTÉ)")
    print("      └─ GUI si progress_gui disponible, sinon console")
    print("      └─ Prédictions sur test holdout")
    print("      └─ Durée estimée: 5-15 minutes\n")
    print("  [4] TEST DECISION TREE (ADAPTÉ)")
    print("      └─ GUI si progress_gui disponible, sinon console")
    print("      └─ Détection overfitting automatique")
    print("      └─ Durée estimée: 15-60 minutes\n")
    print("  [5] CV OPTIMIZATION V3 (GRID SEARCH)")
    print("      └─ Hyperparamètres variables avec graphiques")
    print("      └─ Trouvé meilleure config pour chaque algo")
    print("      └─ Durée estimée: 2-10 heures\n")
    print("  [6] PIPELINE COMPLÈTE (SÉQUENTIELLE)")
    print("      └─ Lance tous les scripts dans l'ordre")
    print("      └─ Consolidation → ML Eval → Test DT → DDoS Det")
    print("      └─ Durée estimée: 2-15 heures\n")
    print("  [7] QUITTER\n")
    
    choice = input("Votre choix (1-7): ").strip()
    return choice

def check_files():
    """Vérifie fichiers Python requis"""
    required_files = {
        "consolidateddata_CORRECTED.py": "Consolidation Dataset",
        "ml_evaluation_v3_ADAPTÉ.py": "ML Evaluation (ADAPTÉ)",
        "ddos_detector_production_ADAPTÉ.py": "DDoS Detector (ADAPTÉ)",
        "test_dt_splits_ADAPTÉ.py": "Test DT Splits (ADAPTÉ)",
        "cv_optimization_v3.py": "CV Optimization V3",
    }
    
    optional_files = {
        "progress_gui.py": "Interface GUI (OPTIONNEL)",
    }
    
    print("\n[CHECK] Vérification fichiers requis...\n")
    
    missing = []
    for fname, desc in required_files.items():
        if os.path.exists(fname):
            file_size = os.path.getsize(fname) / 1024
            print(f"  ✅ {fname:<40} ({file_size:>8.1f} KB)")
        else:
            print(f"  ❌ {fname:<40} MANQUANT")
            missing.append(fname)
    
    print("\n[CHECK] Fichiers optionnels:\n")
    for fname, desc in optional_files.items():
        if os.path.exists(fname):
            file_size = os.path.getsize(fname) / 1024
            print(f"  ✅ {fname:<40} ({file_size:>8.1f} KB) - {desc}")
        else:
            print(f"  ⚠️  {fname:<40} absent (mode console utilisé)")
    
    if missing:
        print(f"\n[ERROR] Fichiers manquants: {', '.join(missing)}")
        return False
    
    print("\n  ✅ Tous les fichiers requis présents!")
    return True

def check_dataset():
    """Vérifie présence du dataset"""
    print("\n[CHECK] Vérification dataset...\n")
    
    if os.path.exists("fusion_train_smart4.csv"):
        size_train = os.path.getsize("fusion_train_smart4.csv") / (1024**3)
        print(f"  ✅ fusion_train_smart4.csv ({size_train:.2f} GB)")
        return True
    
    if os.path.exists("../TONIOT/train_test_network.csv"):
        size = os.path.getsize("../TONIOT/train_test_network.csv") / (1024**3)
        print(f"  ✅ train_test_network.csv ({size:.2f} GB)")
    else:
        print(f"  ⚠️  train_test_network.csv manquant")
    
    if os.path.isdir("../CIC"):
        print(f"  ✅ Dossier CIC/ trouvé")
    else:
        print(f"  ⚠️  Dossier CIC/ manquant")
    
    return True

def mode_consolidation():
    """Lancer consolidation"""
    print_header()
    print("\n[1] CONSOLIDATION DATASET\n")
    print("="*80 + "\n")
    
    if not os.path.exists("consolidateddata_CORRECTED.py"):
        print("[ERROR] consolidateddata_CORRECTED.py manquant")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("[INFO] Une fenêtre Tkinter va s'ouvrir")
    print("       1. Sélectionnez train_test_network.csv (TON_IoT)")
    print("       2. Sélectionnez dossier CIC")
    print("       3. Cliquez DEMARRER")
    print("       Durée estimée: 10-40 minutes")
    print("       ⏳ Vous verrez progress bars détaillées pour chaque étape\n")
    
    input("Appuyez sur ENTREE pour lancer...")
    
    result = subprocess.run([sys.executable, "consolidateddata_CORRECTED.py"], env=_build_consolidation_env())
    
    if result.returncode != 0:
        print("\n[WARNING] Consolidation interrompue")
        return False
    
    if not os.path.exists("fusion_train_smart4.csv"):
        print("\n[ERROR] fusion_train_smart4.csv non généré")
        return False
    
    print("\n[OK] Consolidation complétée!")
    input("Appuyez sur ENTREE pour revenir au menu...")
    return True

def mode_ml_evaluation():
    """Lancer ML Evaluation"""
    print_header()
    print("\n[2] ML EVALUATION V3 (ADAPTÉ)\n")
    print("="*80 + "\n")
    
    if not os.path.exists("ml_evaluation_v3_ADAPTÉ.py"):
        print("[ERROR] ml_evaluation_v3_ADAPTÉ.py manquant")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    if not os.path.exists("preprocessed_dataset.npz"):
        print("[ERROR] preprocessed_dataset.npz manquant")
        print("[INFO] Lancez Consolidation en premier")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("[INFO] Lancement ML Evaluation V3")
    print("       - GUI si progress_gui.py disponible")
    print("       - Console sinon (mode fallback)")
    print("       - RAM gestion dynamique (<90%)")
    print("       Durée estimée: 1-6 heures\n")
    
    input("Appuyez sur ENTREE pour lancer...")
    
    result = subprocess.run([sys.executable, "ml_evaluation_v3_ADAPTÉ.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] ML Evaluation interrompue")
        return False
    
    print("\n[OK] ML Evaluation complétée!")
    input("Appuyez sur ENTREE pour revenir au menu...")
    return True

def mode_ddos_detector():
    """Lancer DDoS Detector"""
    print_header()
    print("\n[3] DDoS DETECTOR PRODUCTION (ADAPTÉ)\n")
    print("="*80 + "\n")
    
    if not os.path.exists("ddos_detector_production_ADAPTÉ.py"):
        print("[ERROR] ddos_detector_production_ADAPTÉ.py manquant")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    if not os.path.exists("ddos_detector_final.pkl"):
        print("[ERROR] ddos_detector_final.pkl manquant")
        print("[INFO] Ce fichier est créé lors du test DT Splits")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("[INFO] Lancement DDoS Detector Production")
    print("       - GUI si progress_gui.py disponible")
    print("       - Console sinon")
    print("       - Batch processing avec gestion RAM")
    print("       Durée estimée: 5-15 minutes\n")
    
    input("Appuyez sur ENTREE pour lancer...")
    
    result = subprocess.run([sys.executable, "ddos_detector_production_ADAPTÉ.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] DDoS Detector interrompue")
        return False
    
    print("\n[OK] DDoS Detector complétée!")
    input("Appuyez sur ENTREE pour revenir au menu...")
    return True

def mode_test_dt():
    """Lancer Test Decision Tree"""
    print_header()
    print("\n[4] TEST DECISION TREE SPLITS (ADAPTÉ)\n")
    print("="*80 + "\n")
    
    if not os.path.exists("test_dt_splits_ADAPTÉ.py"):
        print("[ERROR] test_dt_splits_ADAPTÉ.py manquant")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    if not os.path.exists("preprocessed_dataset.npz"):
        print("[ERROR] preprocessed_dataset.npz manquant")
        print("[INFO] Lancez Consolidation en premier")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("[INFO] Lancement Test Decision Tree Splits")
    print("       - Détection overfitting automatique")
    print("       - 30 évaluations (6 tailles × 5 runs)")
    print("       - Gestion RAM dynamique")
    print("       Durée estimée: 15-60 minutes\n")
    
    input("Appuyez sur ENTREE pour lancer...")
    
    result = subprocess.run([sys.executable, "test_dt_splits_ADAPTÉ.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] Test DT Splits interrompue")
        return False
    
    print("\n[OK] Test DT Splits complétée!")
    print("     - dt_test_results.json généré")
    print("     - test_dt_splits.png généré")
    print("     - ddos_detector_final.pkl créé")
    
    input("Appuyez sur ENTREE pour revenir au menu...")
    return True

def mode_cv_optimization():
    """Lancer CV Optimization Grid Search"""
    print_header()
    print("\n[5] CV OPTIMIZATION V3 - GRID SEARCH\n")
    print("="*80 + "\n")
    
    if not os.path.exists("cv_optimization_v3.py"):
        print("[ERROR] cv_optimization_v3.py manquant")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("[INFO] Lancement CV Optimization V3 - Grid Search")
    print("       - Hyperparamètres variables")
    print("       - GUI avec graphiques scrollables")
    print("       - Paramètres vs F1 Scores")
    print("       - Gestion RAM dynamique")
    print("       Durée estimée: 2-10 heures\n")
    
    input("Appuyez sur ENTREE pour lancer...")
    
    result = subprocess.run([sys.executable, "cv_optimization_v3.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] CV Optimization interrompue")
        return False
    
    print("\n[OK] CV Optimization complétée!")
    print("     - cv_results_summary.txt généré")
    print("     - cv_optimal_splits.json généré")
    
    input("Appuyez sur ENTREE pour revenir au menu...")
    return True

def mode_pipeline_complete():
    """Lancer pipeline séquentielle"""
    print_header()
    print("\n[6] PIPELINE COMPLÈTE (SÉQUENTIELLE)\n")
    print("="*80 + "\n")
    
    if not check_files():
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("\nLancement pipeline séquentielle:")
    print("  1. Consolidation (10-40 min)")
    print("  2. ML Evaluation (1-6 heures)")
    print("  3. Test Decision Tree (15-60 min)")
    print("  4. DDoS Detector (5-15 min)")
    print("  ────────────────────────────────")
    print("  TOTAL: 2-15 heures estimées\n")
    
    input("Appuyez sur ENTREE pour lancer...")
    
    # 1. Consolidation
    print("\n[1/4] Lancement Consolidation...")
    if not mode_consolidation():
        print("[ERROR] Pipeline interrompue (Consolidation échouée)")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    # 2. ML Evaluation
    print("\n[2/4] Lancement ML Evaluation...")
    if not mode_ml_evaluation():
        print("[ERROR] Pipeline interrompue (ML Evaluation échouée)")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    # 3. Test DT
    print("\n[3/4] Lancement Test Decision Tree...")
    if not mode_test_dt():
        print("[ERROR] Pipeline interrompue (Test DT échouée)")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    # 4. DDoS Detector
    print("\n[4/4] Lancement DDoS Detector...")
    if not mode_ddos_detector():
        print("[ERROR] Pipeline interrompue (DDoS Detector échouée)")
        input("Appuyez sur ENTREE pour revenir au menu...")
        return False
    
    print("\n" + "="*80)
    print("[OK] PIPELINE COMPLÈTE TERMINÉE AVEC SUCCÈS!")
    print("="*80)
    print("\nFichiers générés:")
    print("  ✅ fusion_train_smart4.csv")
    print("  ✅ fusion_test_smart4.csv")
    print("  ✅ preprocessed_dataset.npz")
    print("  ✅ evaluation_results_summary.txt")
    print("  ✅ ml_evaluation_results.json")
    print("  ✅ graph_eval_*.png (4 graphiques)")
    print("  ✅ dt_test_results.json")
    print("  ✅ test_dt_splits.png")
    print("  ✅ ddos_predictions_test_holdout.csv")
    print("  ✅ ddos_detector_final.pkl")
    
    input("\nAppuyez sur ENTREE pour revenir au menu...")
    return True

def main():
    """Main loop"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if not check_files():
        print("\n[ERROR] Fichiers manquants!")
        input("Appuyez sur ENTREE pour quitter...")
        sys.exit(1)
    
    check_dataset()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        choice = print_menu()
        
        if choice == "1":
            os.system('cls' if os.name == 'nt' else 'clear')
            mode_consolidation()
        
        elif choice == "2":
            os.system('cls' if os.name == 'nt' else 'clear')
            mode_ml_evaluation()
        
        elif choice == "3":
            os.system('cls' if os.name == 'nt' else 'clear')
            mode_ddos_detector()
        
        elif choice == "4":
            os.system('cls' if os.name == 'nt' else 'clear')
            mode_test_dt()
        
        elif choice == "5":
            os.system('cls' if os.name == 'nt' else 'clear')
            mode_cv_optimization()
        
        elif choice == "6":
            os.system('cls' if os.name == 'nt' else 'clear')
            mode_pipeline_complete()
        
        elif choice == "7":
            print("\nAu revoir!")
            sys.exit(0)
        
        else:
            print("\n[ERROR] Choix invalide")
            time.sleep(1)

if __name__ == "__main__":
    main()
