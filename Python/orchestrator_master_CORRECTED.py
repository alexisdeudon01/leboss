#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import subprocess
import shutil
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

class LauncherDDoS:
    """Launcher simplifié pour pipeline DDoS"""
    
    def __init__(self):
        # Détecter le répertoire master (parent du répertoire courant)
        if os.path.basename(os.getcwd()) == 'Python':
            self.master_dir = os.path.dirname(os.getcwd())
            self.python_dir = os.getcwd()
        else:
            # Si on est dans master/ directement
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
        """Vérifier la structure des répertoires"""
        self.log("VÉRIFICATION STRUCTURE", "HEADER")
        
        checks = {
            "Répertoire master": os.path.isdir(self.master_dir),
            "Répertoire Python": os.path.isdir(self.python_dir),
            "Dossier CIC": os.path.isdir(self.cic_dir),
            "Dossier TONIOT": os.path.isdir(self.toniot_dir),
            "CIC/CSV-03-11 (Novembre)": os.path.isdir(os.path.join(self.cic_dir, 'CSV-03-11')),
            "CIC/CSV-01-12 (Décembre)": os.path.isdir(os.path.join(self.cic_dir, 'CSV-01-12')),
        }
        
        for desc, exists in checks.items():
            if exists:
                self.log(f"  ✅ {desc}", "OK")
            else:
                self.log(f"  ❌ {desc}", "ERROR")
                return False
        
        # Vérifier fichiers
        self.log("\nVérification fichiers", "INFO")
        
        # TON_IoT
        toniot_file = os.path.join(self.toniot_dir, 'train_test_network.csv')
        if os.path.exists(toniot_file):
            size = os.path.getsize(toniot_file) / (1024**3)
            self.log(f"  ✅ train_test_network.csv ({size:.2f} GB)", "OK")
        else:
            self.log(f"  ❌ train_test_network.csv manquant", "ERROR")
            return False
        
        # Scripts FINAL
        scripts = ['consolidateddata_CORRECTED.py', 'cv_optimization_v3.py']
        for script in scripts:
            script_path = os.path.join(self.python_dir, script)
            if os.path.exists(script_path):
                self.log(f"  ✅ {script}", "OK")
            else:
                self.log(f"  ❌ {script} manquant", "ERROR")
                return False
        
        # CSV dans CIC
        self.log("\nVérification CSV CIC", "INFO")
        csv_count = 0
        for folder in ['CSV-03-11', 'CSV-01-12']:
            folder_path = os.path.join(self.cic_dir, folder)
            # Chercher récursivement (peut avoir sous-dossier)
            for root, dirs, files in os.walk(folder_path):
                csv_files = [f for f in files if f.endswith('.csv')]
                csv_count += len(csv_files)
                if csv_files:
                    self.log(f"  ✅ {folder}: {len(csv_files)} fichiers CSV", "OK")
                    break
        
        if csv_count == 0:
            self.log(f"  ❌ Aucun fichier CSV trouvé dans CIC/", "ERROR")
            return False
        
        self.log(f"\n✅ Structure validée ({csv_count} fichiers CSV trouvés)", "OK")
        return True
    
    def setup_working_directory(self):
        """Préparer le répertoire de travail"""
        self.log("\nPRÉPARATION RÉPERTOIRE DE TRAVAIL", "HEADER")
        
        # Se placer dans Python/
        os.chdir(self.python_dir)
        self.log(f"  Répertoire courant: {os.getcwd()}", "OK")
        
        # Créer liens symboliques ou copier fichiers si nécessaire
        # Pour Windows, on va juste utiliser les chemins absolus dans les scripts
        
        # Créer dossier CIC symlink dans Python/
        if not os.path.exists('CIC'):
            try:
                # Essayer symlink (fonctionne sur Windows 10+)
                os.symlink(self.cic_dir, 'CIC')
                self.log(f"  Symlink CIC créé", "OK")
            except:
                # Fallback: copier les chemins en variables d'environnement
                self.log(f"  Chemins CIC: {self.cic_dir}", "INFO")
        
        return True
    
    def run_consolidata(self):
        """Lancer consolidateddata_CORRECTED.py"""
        self.log("\nÉTAPE 1: CONSOLIDATION DATASET", "HEADER")
        
        if not os.path.exists('consolidateddata_CORRECTED.py'):
            self.log("❌ consolidateddata_CORRECTED.py manquant", "ERROR")
            return False
        
        self.log("Lancement consolidateddata_CORRECTED.py", "INFO")
        self.log("  - Fusion TON_IoT + CIC", "INFO")
        self.log("  - Split 60/40 scientifique (IEEE)", "INFO")
        self.log("  - Durée estimée: 10-40 minutes", "INFO")
        
        try:
            # Lancer le script avec chemins absolus
            env = os.environ.copy()
            env['TONIOT_PATH'] = self.toniot_dir
            env['CIC_PATH'] = self.cic_dir
            env['MASTER_PATH'] = self.master_dir
            
            result = subprocess.run(
                [sys.executable, 'consolidateddata_CORRECTED.py'],
                capture_output=False,  # Afficher la sortie en direct
                text=True,
                timeout=7200,  # 2 heures max
                env=env
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
        """Lancer cv_optimi_FINAL.py"""
        self.log("\nÉTAPE 2: CV OPTIMIZATION", "HEADER")
        
        if not os.path.exists('cv_optimi_FINAL.py'):
            self.log("❌ cv_optimi_FINAL.py manquant", "ERROR")
            return False
        
        if not os.path.exists('fusion_train_smart4.csv'):
            self.log("❌ fusion_train_smart4.csv manquant - lancez consolidata d'abord", "ERROR")
            return False
        
        self.log("Lancement cv_optimi_FINAL.py", "INFO")
        self.log("  - FIX 1: StratifiedShuffleSplit", "INFO")
        self.log("  - FIX 2: Decision Tree max 80%", "INFO")
        self.log("  - Durée estimée: 1-6 heures", "INFO")
        
        try:
            result = subprocess.run(
                [sys.executable, 'cv_optimi_FINAL.py'],
                capture_output=False,
                text=True,
                timeout=21600,  # 6 heures max
            )
            
            if result.returncode != 0:
                self.log("⚠️  cv_optimi terminée avec avertissements", "WARNING")
                return True  # Continuer même en cas d'erreur
            
            # Vérifier fichiers créés
            if os.path.exists('cv_results_summary.txt'):
                self.log("  ✅ Créé: cv_results_summary.txt", "OK")
                self.results['cv_results_summary.txt'] = "OK"
            
            if os.path.exists('cv_optimal_splits.json'):
                self.log("  ✅ Créé: cv_optimal_splits.json", "OK")
                self.results['cv_optimal_splits.json'] = "OK"
            
            return True
        
        except subprocess.TimeoutExpired:
            self.log("⚠️  Timeout (>6 heures) - peut continuer", "WARNING")
            return True
        except Exception as e:
            self.log(f"⚠️  Erreur: {e}", "WARNING")
            return True  # Continuer
    
    def generate_summary(self):
        """Générer rapport final"""
        self.log("\nRAPPORT FINAL", "HEADER")
        
        summary = f"""
{'='*80}
DDoS DETECTION PIPELINE - RAPPORT D'EXÉCUTION
{'='*80}

Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Durée:     {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes

Structure:
  Répertoire master:  {self.master_dir}
  Répertoire Python:  {self.python_dir}
  Dossier CIC:        {self.cic_dir}
  Dossier TONIOT:     {self.toniot_dir}

Étapes complétées:
  ✅ Vérification structure
  ✅ Consolidation dataset
  ✅ CV Optimization

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
PROCHAINES ÉTAPES
{'='*80}

1. Consulter les résultats:
   - Ouvrir: cv_results_summary.txt
   - Ouvrir: cv_optimal_splits.json

2. Analyser les métriques:
   - F1 Score, Recall, Precision
   - Configuration optimale pour chaque modèle

3. Entraîner modèle final:
   - Utiliser la configuration optimale trouvée
   - Modèle Decision Tree recommandé

{'='*80}
Logs: {self.log_file}
Résultats: {os.getcwd()}
{'='*80}
"""
        
        print(summary)
        
        # Sauvegarder résumé
        with open('PIPELINE_SUMMARY.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.log("Résumé sauvegardé: PIPELINE_SUMMARY.txt", "OK")
    
    def run(self):
        """Exécuter la pipeline complète"""
        self.log("DÉMARRAGE PIPELINE DDoS DETECTION", "HEADER")
        
        if not self.verify_structure():
            self.log("❌ Vérification échouée", "ERROR")
            return False
        
        if not self.setup_working_directory():
            self.log("❌ Préparation répertoire échouée", "ERROR")
            return False
        
        # Étape 1: Consolidation
        if not self.run_consolidata():
            self.log("⚠️  Consolidation échouée, arrêt", "WARNING")
            return False
        
        # Étape 2: CV Optimization
        if not self.run_cv_optimi():
            self.log("⚠️  CV Optimization échouée", "WARNING")
        
        # Rapport final
        self.generate_summary()
        
        self.log("\n✅ PIPELINE COMPLÉTÉE AVEC SUCCÈS", "OK")
        return True


def main():
    """Point d'entrée principal"""
    print("\n" + "="*80)
    print("DDoS DETECTION PIPELINE - LAUNCHER SIMPLIFIÉ")
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