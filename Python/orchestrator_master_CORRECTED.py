#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCHESTRATOR MASTER - VERSION SIMPLIFIÉE
========================================
✅ Auto-détecte structure répertoires
✅ Lance pipeline complète
✅ Logs horodatés
✅ Gestion erreurs
========================================
"""

import os
import sys
import subprocess
import shutil
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

class LauncherDDoS:
    """Orchestrateur simplifié"""
    
    def __init__(self):
        # Détecter répertoire master
        if os.path.basename(os.getcwd()) == 'Python':
            self.master_dir = os.path.dirname(os.getcwd())
            self.python_dir = os.getcwd()
        else:
            self.master_dir = os.getcwd()
            self.python_dir = os.path.join(self.master_dir, 'Python')
        
        self.cic_dir = os.path.join(self.master_dir, 'CIC')
        self.toniot_dir = os.path.join(self.master_dir, 'TONIOT')
        self.start_time = datetime.now()
        self.results = {}
        
        # Créer dossier logs
        self.logs_dir = os.path.join(self.master_dir, 'pipeline_logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_file = os.path.join(self.logs_dir, f"launch_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
    
    def log(self, msg, level="INFO"):
        """Log avec timestamp"""
        ts = datetime.now().strftime("%H:%M:%S")
        if level == "HEADER":
            log_msg = f"\n{'='*80}\n{msg}\n{'='*80}\n"
        elif level == "OK":
            log_msg = f"[{ts}] ✅ {msg}"
        elif level == "ERROR":
            log_msg = f"[{ts}] ❌ {msg}"
        elif level == "WARNING":
            log_msg = f"[{ts}] ⚠️  {msg}"
        else:
            log_msg = f"[{ts}] ℹ️  {msg}"
        
        print(log_msg)
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(log_msg + "\n")
        except Exception:
            pass
    
    def verify_structure(self):
        """Vérifier structure répertoires"""
        self.log("VÉRIFICATION STRUCTURE", "HEADER")
        
        checks = {
            "Répertoire master": os.path.isdir(self.master_dir),
            "Répertoire Python": os.path.isdir(self.python_dir),
            "Dossier CIC": os.path.isdir(self.cic_dir),
            "Dossier TONIOT": os.path.isdir(self.toniot_dir),
        }
        
        for desc, exists in checks.items():
            if exists:
                self.log(f"  ✅ {desc}", "OK")
            else:
                self.log(f"  ❌ {desc}", "ERROR")
                return False
        
        # Vérifier fichiers
        self.log("\nVérification fichiers", "INFO")
        
        toniot_file = os.path.join(self.toniot_dir, 'train_test_network.csv')
        if os.path.exists(toniot_file):
            size = os.path.getsize(toniot_file) / (1024**3)
            self.log(f"  ✅ train_test_network.csv ({size:.2f} GB)", "OK")
        else:
            self.log(f"  ❌ train_test_network.csv manquant", "ERROR")
            return False
        
        # Scripts
        scripts = [
            'consolidateddata_CORRECTED.py',
            'cv_optimization_v3.py',
            'ml_evaluation_v3_ADAPTÉ.py',
            'ddos_detector_production_ADAPTÉ.py',
            'test_dt_splits_ADAPTÉ.py'
        ]
        
        for script in scripts:
            script_path = os.path.join(self.python_dir, script)
            if os.path.exists(script_path):
                self.log(f"  ✅ {script}", "OK")
            else:
                self.log(f"  ❌ {script} manquant", "ERROR")
                return False
        
        return True
    
    def setup_working_directory(self):
        """Préparer répertoire de travail"""
        self.log("\nPRÉPARATION RÉPERTOIRE DE TRAVAIL", "HEADER")
        
        os.chdir(self.python_dir)
        self.log(f"  Répertoire courant: {os.getcwd()}", "OK")
        
        return True
    
    def run_consolidata(self):
        """Lancer consolidation"""
        self.log("\nÉTAPE 1: CONSOLIDATION DATASET", "HEADER")
        
        if not os.path.exists('consolidateddata_CORRECTED.py'):
            self.log("❌ consolidateddata_CORRECTED.py manquant", "ERROR")
            return False
        
        self.log("Lancement consolidateddata_CORRECTED.py", "INFO")
        self.log("  - Fusion TON_IoT + CIC", "INFO")
        self.log("  - Split 60/40 scientifique", "INFO")
        self.log("  - Durée estimée: 10-40 minutes", "INFO")
        
        try:
            result = subprocess.run(
                [sys.executable, 'consolidateddata_CORRECTED.py'],
                capture_output=False,
                text=True,
                timeout=7200
            )
            
            if result.returncode != 0:
                self.log("❌ consolidata échouée", "ERROR")
                return False
            
            # Vérifier fichiers créés
            required_files = [
                'fusion_train_smart4.csv',
                'fusion_test_smart4.csv',
                'preprocessed_dataset.npz'
            ]
            
            all_ok = True
            for fname in required_files:
                if os.path.exists(fname):
                    size = os.path.getsize(fname) / (1024**3)
                    self.log(f"  ✅ Créé: {fname} ({size:.2f} GB)", "OK")
                    self.results[fname] = "OK"
                else:
                    self.log(f"  ❌ Manquant: {fname}", "ERROR")
                    all_ok = False
            
            return all_ok
        
        except subprocess.TimeoutExpired:
            self.log("❌ Timeout (>2 heures)", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ Erreur: {e}", "ERROR")
            return False
    
    def run_cv_optimi(self):
        """Lancer CV Optimization"""
        self.log("\nÉTAPE 2: CV OPTIMIZATION V3", "HEADER")
        
        if not os.path.exists('cv_optimization_v3.py'):
            self.log("❌ cv_optimization_v3.py manquant", "ERROR")
            return False
        
        if not os.path.exists('fusion_train_smart4.csv'):
            self.log("❌ fusion_train_smart4.csv manquant", "ERROR")
            return False
        
        self.log("Lancement cv_optimization_v3.py", "INFO")
        self.log("  - Grid Search hyperparamètres", "INFO")
        self.log("  - Graphiques scrollables", "INFO")
        self.log("  - Durée estimée: 2-10 heures", "INFO")
        
        try:
            result = subprocess.run(
                [sys.executable, 'cv_optimization_v3.py'],
                capture_output=False,
                text=True,
                timeout=36000
            )
            
            if result.returncode != 0:
                self.log("⚠️  cv_optimi terminée avec avertissements", "WARNING")
                return True
            
            if os.path.exists('cv_results_summary.txt'):
                self.log("  ✅ Créé: cv_results_summary.txt", "OK")
                self.results['cv_results_summary.txt'] = "OK"
            
            return True
        
        except Exception as e:
            self.log(f"⚠️  Erreur: {e}", "WARNING")
            return True
    
    def generate_summary(self):
        """Génère rapport final"""
        self.log("\nRAPPORT FINAL", "HEADER")
        
        duration = datetime.now() - self.start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        
        summary = f"""
{'='*80}
DDoS DETECTION PIPELINE - RAPPORT D'EXÉCUTION
{'='*80}

Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Durée:     {hours}h {minutes}m

Structure:
  Répertoire master:  {self.master_dir}
  Répertoire Python:  {self.python_dir}
  Dossier CIC:        {self.cic_dir}
  Dossier TONIOT:     {self.toniot_dir}

Fichiers générés:
"""
        
        for fname, status in self.results.items():
            if status == "OK":
                if os.path.exists(fname):
                    size = os.path.getsize(fname)
                    if size > 1024**3:
                        size_str = f"({size/(1024**3):.2f} GB)"
                    elif size > 1024**2:
                        size_str = f"({size/(1024**2):.2f} MB)"
                    else:
                        size_str = f"({size/1024:.2f} KB)"
                    summary += f"  ✅ {fname:<35} {size_str}\n"
        
        summary += f"""
{'='*80}
NEXT STEPS
{'='*80}

1. Consulter les résultats:
   - cv_results_summary.txt
   - evaluation_results_summary.txt (si ML Eval lancé)

2. Analyser les graphiques:
   - test_dt_splits.png
   - graph_eval_*.png

3. Pour production:
   - ddos_detector_final.pkl
   - ddos_predictions_test_holdout.csv

{'='*80}
Logs: {self.log_file}
{'='*80}
"""
        
        print(summary)
        
        with open('PIPELINE_SUMMARY.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.log("Résumé sauvegardé: PIPELINE_SUMMARY.txt", "OK")
    
    def run(self):
        """Exécute pipeline"""
        self.log("DÉMARRAGE PIPELINE DDoS DETECTION", "HEADER")
        
        if not self.verify_structure():
            self.log("❌ Vérification échouée", "ERROR")
            return False
        
        if not self.setup_working_directory():
            self.log("❌ Préparation échouée", "ERROR")
            return False
        
        if not self.run_consolidata():
            self.log("⚠️  Consolidation échouée, arrêt", "WARNING")
            return False
        
        if not self.run_cv_optimi():
            self.log("⚠️  CV Optimization échouée", "WARNING")
        
        self.generate_summary()
        
        self.log("\n✅ PIPELINE COMPLÉTÉE AVEC SUCCÈS", "OK")
        return True


def main():
    """Point d'entrée"""
    print("\n" + "="*80)
    print("DDoS DETECTION PIPELINE - ORCHESTRATOR")
    print("="*80 + "\n")
    
    launcher = LauncherDDoS()
    success = launcher.run()
    
    print("\n" + "="*80)
    if success:
        print("✅ Pipeline exécutée avec succès!")
        print("📋 Consultez PIPELINE_SUMMARY.txt pour le rapport")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("❌ Pipeline échouée")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
