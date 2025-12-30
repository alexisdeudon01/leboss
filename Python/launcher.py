#!/usr/bin/env python3
"""
DDoS DETECTION PROJECT - LAUNCHER
Permet de choisir entre Mode GUI et Mode Automatique
"""

import os
import sys
import subprocess
import time

def print_header():
    print("\n" + "="*80)
    print("DDoS DETECTION PROJECT - LAUNCHER")
    print("="*80)

def print_menu():
    print_header()
    print("\nChoisissez le mode d'execution:\n")
    print("  [1] MODE INTERACTIF GUI")
    print("      └─ Voir les interfaces graphiques Tkinter")
    print("      └─ Controle manuel avec boutons DEMARRER/ARRETER")
    print("      └─ Logs en temps reel dans les fenêtres\n")
    print("  [2] MODE AUTOMATIQUE")
    print("      └─ Reset complet + Regeneration")
    print("      └─ Tout lance automatiquement en background")
    print("      └─ Rapide et mains-libres\n")
    print("  [3] CV OPTIMIZATION UNIQUEMENT (GUI)")
    print("      └─ Lance uniquement cv_optimization_v3.py\n")
    print("  [4] ML EVALUATION UNIQUEMENT (GUI)")
    print("      └─ Lance uniquement ml_evaluation_v3.py\n")
    print("  [5] QUITTER\n")
    
    choice = input("Votre choix (1, 2, 3, 4 ou 5): ").strip()
    return choice

def check_files():
    """Verifier fichiers Python requis"""
    required_files = [
        "cv_optimization_v3.py",
        "ml_evaluation_v3.py",
        "test_dt_splits.py",
        "ddos_detector_production.py",
        "orchestrator_master_FINAL.py"
    ]
    
    missing = []
    for fname in required_files:
        if not os.path.exists(fname):
            missing.append(fname)
    
    if missing:
        print(f"\n[ERROR] Fichiers manquants: {missing}")
        print("Assurez-vous que tous les fichiers Python sont dans le même dossier")
        return False
    
    return True

def mode_gui_interactive():
    """Mode GUI Interactif"""
    print_header()
    print("\nMODE INTERACTIF - GUI TKINTER\n")
    print("="*80 + "\n")
    
    if not check_files():
        return False
    
    print("[1/2] Lancement CV OPTIMIZATION V3 GUI")
    print("      Une fenetre Tkinter va s'ouvrir avec interface graphique")
    print("      Cliquez sur [DEMARRER] pour lancer l'optimisation")
    print("      Cliquez sur [ARRETER] pour arreter\n")
    input("Appuyez sur ENTREE pour continuer...")
    
    result = subprocess.run([sys.executable, "cv_optimization_v3.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] CV Optimization interrompue ou erreur")
        return False
    
    if not os.path.exists("cv_optimal_splits_kfold.json"):
        print("\n[ERROR] cv_optimal_splits_kfold.json non genere")
        print("La CV Optimization doit être completee pour continuer")
        return False
    
    print("\n[OK] CV Optimization completee\n")
    print("="*80 + "\n")
    
    print("[2/2] Lancement ML EVALUATION V3 GUI")
    print("      Une autre fenetre Tkinter va s'ouvrir")
    print("      Cliquez sur [DEMARRER] pour lancer l'evaluation\n")
    input("Appuyez sur ENTREE pour continuer...")
    
    result = subprocess.run([sys.executable, "ml_evaluation_v3.py"])
    
    if result.returncode != 0:
        print("\n[WARNING] ML Evaluation interrompue ou erreur")
        return False
    
    print("\n[OK] ML Evaluation completee\n")
    
    return True

def mode_automatique():
    """Mode Automatique - Tout en background"""
    print_header()
    print("\nMODE AUTOMATIQUE - REGENERATION COMPLETE\n")
    print("="*80 + "\n")
    
    if not check_files():
        return False
    
    print("[INFO] Lancement orchestrator complet...")
    print("       - RESET complet")
    print("       - CV Optimization")
    print("       - ML Evaluation")
    print("       - Test DT Splits")
    print("       - Entrainement modele final")
    print("       - Generation rapports\n")
    
    result = subprocess.run([sys.executable, "orchestrator_master_FINAL.py"])
    
    if result.returncode == 0:
        print("\n[OK] Orchestration completee avec succes!")
        return True
    else:
        print("\n[ERROR] Orchestration echouee")
        return False

def mode_cv_only():
    """Lancer uniquement CV Optimization"""
    print_header()
    print("\nCV OPTIMIZATION UNIQUEMENT\n")
    print("="*80 + "\n")
    
    if not os.path.exists("cv_optimization_v3.py"):
        print("[ERROR] cv_optimization_v3.py manquant")
        return False
    
    print("[INFO] Lancement CV Optimization V3 GUI...")
    subprocess.run([sys.executable, "cv_optimization_v3.py"])
    return True

def mode_ml_only():
    """Lancer uniquement ML Evaluation"""
    print_header()
    print("\nML EVALUATION UNIQUEMENT\n")
    print("="*80 + "\n")
    
    if not os.path.exists("ml_evaluation_v3.py"):
        print("[ERROR] ml_evaluation_v3.py manquant")
        return False
    
    if not os.path.exists("cv_optimal_splits_kfold.json"):
        print("[ERROR] cv_optimal_splits_kfold.json manquant")
        print("[INFO] Lancez CV Optimization en premier")
        return False
    
    print("[INFO] Lancement ML Evaluation V3 GUI...")
    subprocess.run([sys.executable, "ml_evaluation_v3.py"])
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
                print("\n[OK] Mode GUI completé!")
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
            mode_cv_only()
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "4":
            mode_ml_only()
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif choice == "5":
            print("\nAu revoir!")
            sys.exit(0)
        
        else:
            print("\n[ERROR] Choix invalide")
            time.sleep(1)
            os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()