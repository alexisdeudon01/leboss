#!/usr/bin/env python3
"""
DDoS DETECTION PROJECT - LAUNCHER (CORRIGÉ)
Permet de choisir entre Mode GUI et Mode Automatique
✅ UTILISE LES SCRIPTS CORRIGÉS (pas les originaux)
"""

import os
import sys
import subprocess
import time

def print_header():
    print("\n" + "="*80)
    print("DDoS DETECTION PROJECT - LAUNCHER (CORRIGÉ)")
    print("="*80)

def print_menu():
    print_header()
    print("\nChoisissez le mode d'execution:\n")
    print("  [1] MODE INTERACTIF GUI")
    print("      └─ Voir les interfaces graphiques Tkinter")
    print("      └─ Controle manuel avec boutons DEMARRER/ARRETER")
    print("      └─ Logs en temps reel dans les fenêtres\n")
    print("  [2] MODE AUTOMATIQUE (RECOMMANDÉ)")
    print("      └─ Reset complet + Regeneration")
    print("      └─ Tout lance automatiquement en background")
    print("      └─ Rapide et mains-libres\n")
    print("  [3] CONSOLIDATION UNIQUEMENT")
    print("      └─ Lance uniquement consolidateddata_CORRECTED.py\n")
    print("  [4] CV OPTIMIZATION UNIQUEMENT (GUI)")
    print("      └─ Lance uniquement cv_optimization_v3.py\n")
    print("  [5] ML EVALUATION UNIQUEMENT (GUI)")
    print("      └─ Lance uniquement ml_evaluation_v3_CORRECTED.py\n")
    print("  [6] QUITTER\n")
    
    choice = input("Votre choix (1, 2, 3, 4, 5 ou 6): ").strip()
    return choice

def check_files():
    """Vérifier fichiers Python requis (CORRIGÉS)"""
    required_files = {
        "consolidateddata_CORRECTED.py": "Consolidation Dataset",
        "cv_optimization_v3.py": "CV Optimization",
        "ml_evaluation_v3_CORRECTED.py": "ML Evaluation (CORRIGÉ)",
        "test_dt_splits_CORRECTED.py": "Test DT Splits (CORRIGÉ)",
        "ddos_detector_production_CORRECTED.py": "Production (CORRIGÉ)",
        "orchestrator_master_CORRECTED.py": "Orchestrator (CORRIGÉ)"
    }
    
    print("\n[CHECK] Vérification fichiers requis...\n")
    
    missing = []
    for fname, desc in required_files.items():
        if os.path.exists(fname):
            file_size = os.path.getsize(fname) / 1024  # KB
            print(f"  ✅ {fname:<40} ({file_size:>8.1f} KB) - {desc}")
        else:
            print(f"  ❌ {fname:<40} MANQUANT - {desc}")
            missing.append(fname)
    
    if missing:
        print(f"\n[ERROR] Fichiers manquants: {', '.join(missing)}")
        print("[ERROR] Assurez-vous que tous les fichiers CORRIGÉS sont présents")
        return False
    
    print("\n  ✅ Tous les fichiers requis présents!")
    return True

def mode_gui_interactive():
    """Mode GUI Interactif"""
    print_header()
    print("\nMODE INTERACTIF - GUI TKINTER\n")
    print("="*80 + "\n")
    
    if not check_files():
        return False
    
    print("[1/3] Lancement CONSOLIDATION DATASET")
    print("      Une fenetre Tkinter va s'ouvrir avec interface graphique")
    print("      Sélectionnez les fichiers TON_IoT et dossier CIC")
    print("      Cliquez sur [DÉMARRER] pour lancer la consolidation\n")
    input("Appuyez sur ENTREE pour continuer...")
    
    result = subprocess.run([sys.executable, "consolidateddata_CORRECTED.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] Consolidation interrompue ou erreur")
        return False
    
    if not os.path.exists("fusion_train_smart4.csv"):
        print("\n[ERROR] fusion_train_smart4.csv non généré")
        print("La consolidation doit être complétée pour continuer")
        return False
    
    print("\n[OK] Consolidation complétée\n")
    print("="*80 + "\n")
    
    print("[2/3] Lancement CV OPTIMIZATION V3 GUI")
    print("      Une autre fenetre Tkinter va s'ouvrir")
    print("      Cliquez sur [DÉMARRER] pour lancer l'optimisation\n")
    input("Appuyez sur ENTREE pour continuer...")
    
    result = subprocess.run([sys.executable, "cv_optimization_v3.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] CV Optimization interrompue ou erreur")
        return False
    
    if not os.path.exists("cv_optimal_splits_kfold.json"):
        print("\n[ERROR] cv_optimal_splits_kfold.json non généré")
        print("La CV Optimization doit être complétée pour continuer")
        return False
    
    print("\n[OK] CV Optimization complétée\n")
    print("="*80 + "\n")
    
    print("[3/3] Lancement ML EVALUATION V3 GUI (CORRIGÉ)")
    print("      Une autre fenetre Tkinter va s'ouvrir")
    print("      Cliquez sur [DÉMARRER] pour lancer l'évaluation\n")
    input("Appuyez sur ENTREE pour continuer...")
    
    result = subprocess.run([sys.executable, "ml_evaluation_v3_CORRECTED.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] ML Evaluation interrompue ou erreur")
        return False
    
    print("\n[OK] ML Evaluation complétée\n")
    
    return True

def mode_automatique():
    """Mode Automatique - Tout en background"""
    print_header()
    print("\nMODE AUTOMATIQUE - PIPELINE COMPLÈTE (CORRIGÉE)\n")
    print("="*80 + "\n")
    
    if not check_files():
        return False
    
    print("[INFO] Lancement orchestrator complet (CORRIGÉ)...")
    print("       - RESET complet")
    print("       - ✅ Consolidation dataset (NOUVEAU)")
    print("       - CV Optimization")
    print("       - ✅ ML Evaluation (CORRIGÉ)")
    print("       - ✅ Test DT Splits (CORRIGÉ)")
    print("       - Entrainement modèle final")
    print("       - Génération rapports\n")
    
    result = subprocess.run([sys.executable, "orchestrator_master_CORRECTED.py"])
    
    if result.returncode == 0:
        print("\n[OK] Orchestration complétée avec succès!")
        return True
    else:
        print("\n[ERROR] Orchestration échouée")
        return False

def mode_consolidation_only():
    """Lancer uniquement consolidation"""
    print_header()
    print("\nCONSOLIDATION DATASET UNIQUEMENT\n")
    print("="*80 + "\n")
    
    if not os.path.exists("consolidateddata_CORRECTED.py"):
        print("[ERROR] consolidateddata_CORRECTED.py manquant")
        return False
    
    print("[INFO] Lancement Consolidation Dataset GUI...")
    subprocess.run([sys.executable, "consolidateddata_CORRECTED.py"])
    return True

def mode_cv_only():
    """Lancer uniquement CV Optimization"""
    print_header()
    print("\nCV OPTIMIZATION UNIQUEMENT\n")
    print("="*80 + "\n")
    
    if not os.path.exists("cv_optimization_v3.py"):
        print("[ERROR] cv_optimization_v3.py manquant")
        return False
    
    if not os.path.exists("fusion_train_smart4.csv"):
        print("[ERROR] fusion_train_smart4.csv manquant")
        print("[INFO] Lancez Consolidation en premier")
        return False
    
    print("[INFO] Lancement CV Optimization V3 GUI...")
    subprocess.run([sys.executable, "cv_optimization_v3.py"])
    return True

def mode_ml_only():
    """Lancer uniquement ML Evaluation"""
    print_header()
    print("\nML EVALUATION UNIQUEMENT (CORRIGÉ)\n")
    print("="*80 + "\n")
    
    if not os.path.exists("ml_evaluation_v3_CORRECTED.py"):
        print("[ERROR] ml_evaluation_v3_CORRECTED.py manquant")
        return False
    
    if not os.path.exists("cv_optimal_splits_kfold.json"):
        print("[ERROR] cv_optimal_splits_kfold.json manquant")
        print("[INFO] Lancez CV Optimization en premier")
        return False
    
    if not os.path.exists("fusion_test_smart4.csv"):
        print("[ERROR] fusion_test_smart4.csv manquant")
        print("[INFO] Lancez Consolidation en premier")
        return False
    
    print("[INFO] Lancement ML Evaluation V3 GUI (CORRIGÉ)...")
    subprocess.run([sys.executable, "ml_evaluation_v3_CORRECTED.py"])
    return True

def main():
    """Main loop"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
    
    while True:
        choice = print_menu()
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        if choice == "1":
            success = mode_gui_interactive()
            if success:
                print("\n[OK] Mode GUI complété!")
                print("\nVoulez-vous continuer avec Mode Automatique? (y/n): ", end="")
                if input().strip().lower() == "y":
                    os.system('cls' if os.name == 'nt' else 'clear')
                    mode_automatique()
                else:
                    print("\nAu revoir!")
                    break
            else:
                print("\n[ERROR] Mode GUI interrompu")
                print("Appuyez sur ENTREE pour revenir au menu...")
                input()
                os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "2":
            success = mode_automatique()
            if success:
                print("\nAppuyez sur ENTREE pour continuer...")
                input()
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "3":
            mode_consolidation_only()
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "4":
            mode_cv_only()
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "5":
            mode_ml_only()
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "6":
            print("\nAu revoir!")
            sys.exit(0)
        
        else:
            print("\n[ERROR] Choix invalide")
            time.sleep(1)
            os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()