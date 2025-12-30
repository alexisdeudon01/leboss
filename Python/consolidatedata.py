#!/usr/bin/env python3
"""
SCRIPT CONSOLIDATION DATASET - VERSION FINALE OPTIMISÉE

AMÉLIORATIONS:
  ✓ Recherche RÉCURSIVE des fichiers CIC
  ✓ 3 Canvas (Logs 70%, Stats 30%, Alertes)
  ✓ MULTITHREADING pour traitement parallèle
  ✓ Maximisation RAM: chunks dynamiques, compression
  ✓ Barre de progression DÉTAILLÉE avec ETA
  ✓ Suppression automatique fichiers intermédiaires
  ✓ Monitoring CPU/RAM en temps réel
  ✓ Boutons DÉMARRER/ARRÊTER visibles
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import warnings
import psutil
import gc
import time
import threading
from datetime import datetime, timedelta
from collections import deque
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
except ImportError:
    print("❌ tkinter non installé")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'TON_IOT_INPUT': None,
    'CIC_DIR': None,
    'FUSION_OUTPUT': 'fusion_ton_iot_cic_final_smart4.csv',
    'TEMP_TON': 'temp_ton_iot.csv',
    'TEMP_CIC': 'temp_cic.csv',
}

# ============================================================================
# UTILS
# ============================================================================

def find_csv_files_recursive(root_dir):
    """Trouver TOUS les CSV récursivement"""
    csv_files = []
    try:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv') and not file.startswith('~'):
                    csv_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Erreur recherche: {e}")
    
    return sorted(csv_files)

class RAMManager:
    """Gestionnaire RAM et CPU optimisé"""
    
    def __init__(self):
        try:
            self.process = psutil.Process(os.getpid())
            self.total_ram = psutil.virtual_memory().total
            self.total_ram_gb = self.total_ram / (1024 ** 3)
            self.chunk_size = self.calculate_optimal_chunk_size()
        except Exception as e:
            print(f"Erreur RAM: {e}")
            self.total_ram_gb = 8.0
            self.chunk_size = 100000
    
    def get_available_ram_gb(self):
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except:
            return 0
    
    def get_used_ram_gb(self):
        try:
            return self.process.memory_info().rss / (1024 ** 3)
        except:
            return 0
    
    def get_ram_percent(self):
        try:
            return (self.get_used_ram_gb() / self.total_ram_gb) * 100
        except:
            return 0
    
    def get_cpu_percent(self):
        try:
            return self.process.cpu_percent(interval=0.1)
        except:
            return 0
    
    def calculate_optimal_chunk_size(self):
        """Maximiser utilisation RAM - chunks plus petits pour gros fichiers"""
        try:
            available_gb = self.get_available_ram_gb()
            # Utiliser 50% du RAM disponible (conservateur pour gros fichiers)
            usable_ram = available_gb * 0.5
            bytes_per_row = 200
            rows_per_gb = (1024 ** 3) / bytes_per_row
            chunk_size = int(usable_ram * rows_per_gb)
            # Entre 50K et 300K lignes (plus conservateur)
            return max(50000, min(300000, chunk_size))
        except:
            return 50000


def estimate_csv_rows(csv_path, sample_lines=2000):
    """Estimer rapidement le nombre de lignes d'un CSV pour le pourcentage."""
    try:
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            return 0
        lengths = []
        with open(csv_path, 'rb') as f:
            for _ in range(sample_lines):
                line = f.readline()
                if not line:
                    break
                lengths.append(len(line))
        if not lengths:
            return 0
        avg = sum(lengths) / len(lengths)
        return max(len(lengths), int(file_size / max(1, avg)))
    except Exception:
        return 0


def format_seconds(seconds):
    """Formatter un ETA lisible."""
    try:
        return str(timedelta(seconds=int(seconds)))
    except Exception:
        return "--:--"


# ============================================================================
# GUI: FILE SELECTOR (TON_IoT file + CIC folder)
# ============================================================================

class FileSelectorGUI:
    """Fenêtre pour choisir le CSV TON_IoT et le dossier CIC (scan récursif)."""

    def __init__(self, root):
        self.root = root
        self.root.title("📂 Sélection des Chemins")
        self.root.geometry("700x420")
        self.root.configure(bg="#f0f0f0")

        self.ton_iot_path = None
        self.cic_dir_path = None
        self.cic_files_found = []

        self.build_ui()

    def build_ui(self):
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="Sélection des Chemins",
                 font=("Arial", 14, "bold"), fg="white", bg="#2c3e50").pack(side=tk.LEFT, padx=20, pady=15)

        main = tk.Frame(self.root, bg="white")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(main, text="1. Fichier TON_IoT (CSV)", font=("Arial", 11, "bold"),
                 bg="white", fg="#2c3e50").pack(anchor=tk.W, pady=(0, 5))
        ton_frame = tk.Frame(main, bg="#ecf0f1")
        ton_frame.pack(fill=tk.X, pady=(0, 15))
        self.ton_entry = tk.Entry(ton_frame, font=("Courier", 9))
        self.ton_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=8)
        tk.Button(ton_frame, text="Parcourir", command=self.select_ton,
                  bg="#3498db", fg="white", font=("Arial", 10),
                  padx=12, pady=5, relief=tk.RAISED, cursor="hand2").pack(side=tk.RIGHT, padx=8, pady=8)

        tk.Label(main, text="2. Dossier CIC (CSV, recherche récursive)", font=("Arial", 11, "bold"),
                 bg="white", fg="#2c3e50").pack(anchor=tk.W, pady=(0, 5))
        cic_frame = tk.Frame(main, bg="#ecf0f1")
        cic_frame.pack(fill=tk.X, pady=(0, 15))
        self.cic_entry = tk.Entry(cic_frame, font=("Courier", 9))
        self.cic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=8)
        tk.Button(cic_frame, text="Parcourir", command=self.select_cic,
                  bg="#3498db", fg="white", font=("Arial", 10),
                  padx=12, pady=5, relief=tk.RAISED, cursor="hand2").pack(side=tk.RIGHT, padx=8, pady=8)

        tk.Label(main, text="Fichiers CSV trouvés :", font=("Arial", 10, "bold"),
                 bg="white", fg="#2c3e50").pack(anchor=tk.W, pady=(5, 5))
        files_frame = tk.Frame(main, bg="#f8f8f8", relief=tk.SUNKEN, bd=1)
        files_frame.pack(fill=tk.BOTH, expand=True)
        self.files_text = scrolledtext.ScrolledText(files_frame, height=8, font=("Courier", 8))
        self.files_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.files_text.config(state=tk.DISABLED)

        footer = tk.Frame(self.root, bg="#ecf0f1", height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Button(footer, text="CONTINUER", command=self.validate,
                  bg="#27ae60", fg="white", font=("Arial", 11, "bold"),
                  padx=22, pady=8, relief=tk.RAISED, cursor="hand2").pack(side=tk.RIGHT, padx=10, pady=10)
        tk.Button(footer, text="QUITTER", command=self.root.quit,
                  bg="#e74c3c", fg="white", font=("Arial", 11, "bold"),
                  padx=22, pady=8, relief=tk.RAISED, cursor="hand2").pack(side=tk.RIGHT, padx=5, pady=10)

    def select_ton(self):
        path = filedialog.askopenfilename(
            title="Sélectionner train_test_network.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.ton_entry.delete(0, tk.END)
            self.ton_entry.insert(0, path)
            self.ton_iot_path = path

    def select_cic(self):
        path = filedialog.askdirectory(title="Sélectionner dossier CIC")
        if path:
            self.cic_entry.delete(0, tk.END)
            self.cic_entry.insert(0, path)
            self.cic_dir_path = path
            self.cic_files_found = find_csv_files_recursive(path)
            self.files_text.config(state=tk.NORMAL)
            self.files_text.delete("1.0", tk.END)
            if self.cic_files_found:
                self.files_text.insert(tk.END, f"Total: {len(self.cic_files_found)} fichiers\n\n")
                for f in self.cic_files_found[:400]:
                    self.files_text.insert(tk.END, f"- {os.path.relpath(f, path)}\n")
                if len(self.cic_files_found) > 400:
                    self.files_text.insert(tk.END, f"... {len(self.cic_files_found)-400} supplémentaires\n")
            else:
                self.files_text.insert(tk.END, "Aucun fichier CSV trouvé.")
            self.files_text.config(state=tk.DISABLED)

    def validate(self):
        ton = self.ton_entry.get().strip()
        cic = self.cic_entry.get().strip()

        if not ton or not cic:
            messagebox.showerror("Erreur", "Sélectionner les deux chemins !")
            return
        if not os.path.isfile(ton):
            messagebox.showerror("Erreur", f"Fichier TON_IoT invalide: {ton}")
            return
        if not os.path.isdir(cic):
            messagebox.showerror("Erreur", f"Dossier CIC invalide: {cic}")
            return
        cic_files = find_csv_files_recursive(cic)
        if not cic_files:
            messagebox.showerror("Erreur", f"Aucun CSV trouvé dans {cic}")
            return

        self.ton_iot_path = ton
        self.cic_dir_path = cic
        self.cic_files_found = cic_files
        messagebox.showinfo("Succès", f"{len(cic_files)} fichiers CSV trouvés.")
        self.root.destroy()
# ============================================================================
# GUI: PATH SELECTION
# ============================================================================

class ConsolidatorGUI:
    """Fenêtre de consolidation avec progressions détaillées."""
    
    def __init__(self, root, ton_iot_path, cic_dir_path, cic_files):
        try:
            self.root = root
            self.root.title("Dataset Consolidator - Optimise")
            self.root.geometry("1700x1000")
            self.root.configure(bg='#f0f0f0')
            
            self.ton_iot_path = ton_iot_path
            self.cic_dir_path = cic_dir_path
            self.cic_files = cic_files or []
            self.cic_files_found = self.cic_files
            
            self.ram = RAMManager()
            self.max_workers = max(2, min(8, (os.cpu_count() or 4)))
            self.progress_blocks = {}
            self.file_progress_widgets = {}
            self.file_progress_order = deque()
            self.max_file_bars = self.max_workers
            self.cic_processed = 0
            self.cic_total = len(self.cic_files)
            self.cic_rows_total = 0
            self.ton_rows_total = 0
            self.fusion_rows_total = 0
            self.logs = deque(maxlen=500)
            self.alerts = deque(maxlen=100)
            self.running = False
            self.start_time = None
            
            self.setup_ui()
            self.update_monitoring()
            
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            sys.exit(1)
    
    def _async_ui(self, callback):
        """Exécuter un update UI depuis un thread worker en toute sécurité."""
        try:
            self.root.after(0, callback)
        except Exception:
            pass

    def _progress_percent(self, current, total):
        try:
            if total <= 0:
                return 0
            return max(0, min(100, int((current / total) * 100)))
        except Exception:
            return 0

    def _make_progress_block(self, parent, title):
        frame = tk.Frame(parent, bg='white')
        frame.grid_columnconfigure(0, weight=1)
        label = tk.Label(frame, text=title, font=('Arial', 9, 'bold'), bg='white', fg='#2c3e50')
        label.grid(row=0, column=0, sticky='w', padx=8, pady=(6, 2))
        bar = ttk.Progressbar(frame, mode='determinate', maximum=100)
        bar.grid(row=1, column=0, sticky='ew', padx=8, pady=2)
        detail = tk.Label(frame, text="", font=('Arial', 8), bg='white', fg='#666')
        detail.grid(row=2, column=0, sticky='w', padx=8, pady=(0, 6))
        return {'frame': frame, 'label': label, 'bar': bar, 'detail': detail}

    def update_stage_progress(self, key, current, total, msg="", eta_sec=None):
        block = self.progress_blocks.get(key)
        if not block:
            return

        pct = self._progress_percent(current, total)
        details = f"{current}/{total}" if total else f"{current}"
        if pct:
            details += f" ({pct}%)"
        if eta_sec is not None:
            details += f" | ETA {format_seconds(eta_sec)}"

        def _apply():
            block['label'].config(text=msg or block['label'].cget('text'))
            block['bar']['value'] = pct
            block['detail'].config(text=details)

        self._async_ui(_apply)

    def _ensure_file_progress_widget(self, filename):
        if filename in self.file_progress_widgets:
            return self.file_progress_widgets[filename]

        if len(self.file_progress_order) >= self.max_file_bars:
            oldest = self.file_progress_order.popleft()
            widget = self.file_progress_widgets.pop(oldest, None)
            if widget:
                widget['frame'].destroy()

        frame = tk.Frame(self.file_progress_container, bg='white', bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X, pady=2, padx=2)

        title = tk.Label(frame, text=filename, font=('Arial', 8, 'bold'), bg='white', anchor='w')
        title.pack(fill=tk.X, padx=6, pady=(4, 0))

        bar = ttk.Progressbar(frame, mode='determinate', maximum=100)
        bar.pack(fill=tk.X, padx=6, pady=2)

        status = tk.Label(frame, text="En attente", font=('Arial', 8), bg='white', fg='#666', anchor='w')
        status.pack(fill=tk.X, padx=6, pady=(0, 4))

        widget = {'frame': frame, 'bar': bar, 'status': status}
        self.file_progress_widgets[filename] = widget
        self.file_progress_order.append(filename)
        return widget

    def reset_file_progress(self):
        for widget in self.file_progress_widgets.values():
            try:
                widget['frame'].destroy()
            except Exception:
                pass
        self.file_progress_widgets.clear()
        self.file_progress_order.clear()

    def update_file_progress(self, filename, percent, status_text):
        def _apply():
            widget = self._ensure_file_progress_widget(filename)
            widget['bar']['value'] = max(0, min(100, percent))
            widget['status'].config(text=status_text)
        self._async_ui(_apply)

    def setup_ui(self):
        """Layout sans canvas gauche: focus sur progressions et logs verbeux."""
        try:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)
            
            # HEADER
            header = tk.Frame(self.root, bg="#2c3e50", height=70)
            header.grid(row=0, column=0, sticky='ew')
            
            title_frame = tk.Frame(header, bg="#2c3e50")
            title_frame.pack(side=tk.LEFT, padx=20, pady=15, fill=tk.X, expand=True)
            
            tk.Label(title_frame, text="Dataset Consolidator - Optimise", 
                     font=('Arial', 16, 'bold'), fg='white', bg="#2c3e50").pack(side=tk.LEFT)
            
            ton_label = os.path.basename(self.ton_iot_path) if self.ton_iot_path else "Non défini"
            cic_label = f"{len(self.cic_files)} fichiers" if getattr(self, "cic_files", None) is not None else "Non défini"
            info_text = f"TON_IoT: {ton_label} | CIC: {cic_label} | RAM: {self.ram.total_ram_gb:.1f}GB | Threads: {self.max_workers}"
            tk.Label(title_frame, text=info_text, 
                     font=('Arial', 9), fg='#bdc3c7', bg="#2c3e50").pack(side=tk.LEFT, padx=30)
            
            # MAIN GRID (progress + logs)
            main = tk.Frame(self.root, bg='#f0f0f0')
            main.grid(row=1, column=0, sticky='nsew', padx=8, pady=8)
            main.rowconfigure(0, weight=3)
            main.rowconfigure(1, weight=2)
            main.columnconfigure(0, weight=1)
            
            progress_grid = tk.Frame(main, bg='#f0f0f0')
            progress_grid.grid(row=0, column=0, sticky='nsew')
            progress_grid.columnconfigure(0, weight=1)
            progress_grid.columnconfigure(1, weight=1)
            
            prog_frame = tk.LabelFrame(progress_grid, text="Avancement global",
                                       font=('Arial', 10, 'bold'),
                                       bg='white', relief=tk.SUNKEN, bd=2)
            prog_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 6))
            
            self.prog_label = tk.Label(prog_frame, text="Pret", font=('Arial', 9), bg='white')
            self.prog_label.pack(fill=tk.X, padx=8, pady=(6, 2))
            
            self.prog_bar = ttk.Progressbar(prog_frame, mode='determinate', maximum=100)
            self.prog_bar.pack(fill=tk.X, padx=8, pady=2)
            
            self.prog_details = tk.Label(prog_frame, text="", font=('Arial', 8), bg='white', fg='#666')
            self.prog_details.pack(fill=tk.X, padx=8, pady=2)
            
            self.eta_label = tk.Label(prog_frame, text="ETA: --:--", font=('Arial', 8), bg='white', fg='#666')
            self.eta_label.pack(fill=tk.X, padx=8, pady=(0, 6))
            
            ton_frame = tk.LabelFrame(progress_grid, text="TON_IoT",
                                      font=('Arial', 10, 'bold'),
                                      bg='white', relief=tk.SUNKEN, bd=2)
            ton_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 6), pady=2)
            ton_frame.columnconfigure(0, weight=1)
            
            self.progress_blocks['ton_read'] = self._make_progress_block(ton_frame, "Lecture TON_IoT")
            self.progress_blocks['ton_read']['frame'].grid(row=0, column=0, sticky='ew', padx=6, pady=4)
            self.progress_blocks['ton_features'] = self._make_progress_block(ton_frame, "Features/clean")
            self.progress_blocks['ton_features']['frame'].grid(row=1, column=0, sticky='ew', padx=6, pady=4)
            
            cic_frame = tk.LabelFrame(progress_grid, text="CIC (parallele)",
                                      font=('Arial', 10, 'bold'),
                                      bg='white', relief=tk.SUNKEN, bd=2)
            cic_frame.grid(row=1, column=1, sticky='nsew', padx=(6, 0), pady=2)
            cic_frame.columnconfigure(0, weight=1)
            cic_frame.rowconfigure(2, weight=1)
            
            self.progress_blocks['cic_files'] = self._make_progress_block(cic_frame, "Fichiers CIC")
            self.progress_blocks['cic_files']['frame'].grid(row=0, column=0, sticky='ew', padx=6, pady=4)
            
            tk.Label(cic_frame, text="Threads/fichiers actifs (progression)",
                     font=('Arial', 8), bg='white', fg='#333').grid(row=1, column=0, sticky='w', padx=8)
            
            files_container = tk.Frame(cic_frame, bg='white')
            files_container.grid(row=2, column=0, sticky='nsew', padx=6, pady=(0, 6))
            files_container.columnconfigure(0, weight=1)
            files_container.rowconfigure(0, weight=1)
            
            self.file_progress_container = tk.Frame(files_container, bg='white')
            self.file_progress_container.grid(row=0, column=0, sticky='nsew')
            
            fusion_frame = tk.LabelFrame(progress_grid, text="Fusion finale",
                                         font=('Arial', 10, 'bold'),
                                         bg='white', relief=tk.SUNKEN, bd=2)
            fusion_frame.grid(row=2, column=0, sticky='nsew', padx=(0, 6), pady=2)
            fusion_frame.columnconfigure(0, weight=1)
            
            self.progress_blocks['fusion'] = self._make_progress_block(fusion_frame, "Ecriture fusion")
            self.progress_blocks['fusion']['frame'].grid(row=0, column=0, sticky='ew', padx=6, pady=4)
            
            monitor_frame = tk.LabelFrame(progress_grid, text="Monitoring & Alertes",
                                          font=('Arial', 10, 'bold'),
                                          bg='white', relief=tk.SUNKEN, bd=2)
            monitor_frame.grid(row=2, column=1, sticky='nsew', padx=(6, 0), pady=2)
            monitor_frame.columnconfigure(0, weight=1)
            monitor_frame.rowconfigure(3, weight=1)
            
            ram_frame = tk.Frame(monitor_frame, bg='white')
            ram_frame.grid(row=0, column=0, sticky='ew', padx=6, pady=(6, 2))
            self.ram_label = tk.Label(ram_frame, text="RAM: 0%", font=('Arial', 9), bg='white')
            self.ram_label.pack(fill=tk.X)
            self.ram_progress = ttk.Progressbar(ram_frame, mode='determinate', maximum=100)
            self.ram_progress.pack(fill=tk.X, pady=2)
            self.ram_details = tk.Label(ram_frame, text="", font=('Arial', 8), bg='white', fg='#666')
            self.ram_details.pack(fill=tk.X)
            
            cpu_frame = tk.Frame(monitor_frame, bg='white')
            cpu_frame.grid(row=1, column=0, sticky='ew', padx=6, pady=2)
            self.cpu_label = tk.Label(cpu_frame, text="CPU: 0%", font=('Arial', 9), bg='white')
            self.cpu_label.pack(fill=tk.X)
            self.cpu_progress = ttk.Progressbar(cpu_frame, mode='determinate', maximum=100)
            self.cpu_progress.pack(fill=tk.X, pady=2)
            
            alerts_frame = tk.Frame(monitor_frame, bg='white')
            alerts_frame.grid(row=2, column=0, sticky='ew', padx=6, pady=2)
            tk.Label(alerts_frame, text="Alertes", font=('Arial', 9, 'bold'), bg='white', fg='#2c3e50').pack(anchor='w')
            self.alerts_text = scrolledtext.ScrolledText(alerts_frame, height=6,
                                                         font=('Courier', 8),
                                                         bg='#f8f8f8', fg='#333')
            self.alerts_text.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
            self.alerts_text.tag_config('error', foreground='#d32f2f', font=('Courier', 8, 'bold'))
            self.alerts_text.tag_config('warning', foreground='#f57f17')
            self.alerts_text.tag_config('success', foreground='#388e3c')
            
            logs_frame = tk.LabelFrame(main, text="Logs detailles (verbose)",
                                      font=('Arial', 10, 'bold'),
                                      bg='white', relief=tk.SUNKEN, bd=2)
            logs_frame.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
            logs_frame.rowconfigure(0, weight=1)
            logs_frame.columnconfigure(0, weight=1)
            
            self.logs_text = scrolledtext.ScrolledText(logs_frame,
                                                       font=('Courier', 9),
                                                       bg='#1e1e1e', fg='#00ff00')
            self.logs_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            footer = tk.Frame(self.root, bg='#ecf0f1', height=60)
            footer.grid(row=2, column=0, sticky='ew')
            
            btn_frame = tk.Frame(footer, bg='#ecf0f1')
            btn_frame.pack(side=tk.LEFT, padx=10, pady=10)
            
            self.start_btn = tk.Button(btn_frame, text="Demarrer",
                                       command=self.start_consolidation,
                                       bg='#27ae60', fg='white',
                                       font=('Arial', 12, 'bold'),
                                       padx=20, pady=10, relief=tk.RAISED, cursor="hand2")
            self.start_btn.pack(side=tk.LEFT, padx=5)
            
            self.stop_btn = tk.Button(btn_frame, text="Arreter",
                                      command=self.stop_consolidation,
                                      bg='#e74c3c', fg='white',
                                      font=('Arial', 12, 'bold'),
                                      padx=20, pady=10, relief=tk.RAISED,
                                      state=tk.DISABLED, cursor="hand2")
            self.stop_btn.pack(side=tk.LEFT, padx=5)
            
            self.status_label = tk.Label(footer, text="Pret",
                                         font=('Arial', 11, 'bold'),
                                         fg='#27ae60', bg='#ecf0f1')
            self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)
            
        except Exception as e:
            messagebox.showerror("Erreur UI", str(e))
    
    def log(self, msg, level='INFO'):
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            icons = {'INFO': '[info]', 'OK': '[ok]', 'ERROR': '[err]', 'WARNING': '[warn]', 'PROGRESS': '[..]'}
            icon = icons.get(level, '>')
            formatted = f"{icon} [{ts}] {msg}"
            self.logs.append(formatted)

            def _apply():
                self.logs_text.insert(tk.END, formatted + "\n")
                self.logs_text.see(tk.END)
            self._async_ui(_apply)
        except Exception:
            pass

    def log_alert(self, msg, level='warning'):
        try:
            self.alerts.append(msg)

            def _apply():
                self.alerts_text.insert(tk.END, f"{msg}\n", level)
                self.alerts_text.see(tk.END)
            self._async_ui(_apply)
        except Exception:
            pass

    def log_error(self, msg):
        self.log_alert(f"❌ {msg}", 'error')
    
    def log_success(self, msg):
        self.log_alert(f"✅ {msg}", 'success')
    
    def update_monitoring(self):
        """Mettre à jour RAM/CPU en temps réel"""
        try:
            ram_pct = self.ram.get_ram_percent()
            cpu_pct = self.ram.get_cpu_percent()
            
            self.ram_progress['value'] = ram_pct
            self.cpu_progress['value'] = cpu_pct
            
            used = self.ram.get_used_ram_gb()
            avail = self.ram.get_available_ram_gb()
            
            self.ram_label.config(text=f"RAM: {ram_pct:.1f}%")
            self.cpu_label.config(text=f"CPU: {cpu_pct:.1f}%")
            self.ram_details.config(text=f"U:{used:.1f}|A:{avail:.1f}|T:{self.ram.total_ram_gb:.1f}GB")
            
            self.root.after(500, self.update_monitoring)
        except:
            self.root.after(500, self.update_monitoring)
    
    def update_progress(self, current, total, msg="", eta_sec=None):
        """Mettre à jour progress bar avec ETA"""
        try:
            pct = self._progress_percent(current, total)
            if isinstance(current, float) and not float(current).is_integer():
                current_display = f"{current:.1f}"
            else:
                current_display = str(int(current)) if total else str(current)
            details = f"{current_display}/{total}" if total else current_display
            if pct:
                details += f" ({pct}%)"

            def _apply():
                self.prog_bar['value'] = pct
                self.prog_label.config(text=msg)
                self.prog_details.config(text=details)
                self.eta_label.config(text=f"ETA: {format_seconds(eta_sec)}" if eta_sec is not None else "ETA: --:--")
            self._async_ui(_apply)
        except Exception:
            pass
    
    def start_consolidation(self):
        if getattr(self, "running", False):
            messagebox.showwarning("Attention", "Processus en cours!")
            return
        
        self.running = True
        self.start_time = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="⏳ En cours...", fg='#f57f17')
        
        thread = threading.Thread(target=self.run_consolidation, daemon=True)
        thread.start()
    
    def stop_consolidation(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏹ Arrêté", fg='#e74c3c')
        self.log("Processus arrêté", 'WARNING')
    
    def process_cic_file(self, csv_file):
        """Traiter UN fichier CIC (chunks) avec progression par fichier."""
        try:
            if not self.running:
                return None

            filename = os.path.basename(csv_file)
            file_size_gb = os.path.getsize(csv_file) / (1024 ** 3)
            est_rows = estimate_csv_rows(csv_file, sample_lines=1500)
            self.log(f"[CIC] {filename}: {file_size_gb:.2f} GB | ~{est_rows:,} lignes estimees", 'INFO')
            self.update_file_progress(filename, 0, "Lecture en cours...")

            chunk_size = max(20000, min(150000, self.ram.chunk_size // 3))
            chunks = []
            processed_rows = 0
            chunk_index = 0
            start_time = time.time()

            reader = pd.read_csv(
                csv_file,
                low_memory=False,
                chunksize=chunk_size,
                dtype_backend='numpy_nullable',
                memory_map=True,
            )

            for chunk in reader:
                if not self.running:
                    return None

                chunk_index += 1
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    chunk_features = chunk[numeric_cols].astype(np.float32)
                    chunks.append(chunk_features)
                    processed_rows += len(chunk_features)

                pct = self._progress_percent(processed_rows, est_rows)
                status = f"Chunk {chunk_index}: {processed_rows:,}/{est_rows or '?'}"
                self.update_file_progress(filename, pct, status)

                if chunk_index % 2 == 0:
                    elapsed = time.time() - start_time
                    self.log(f"[CIC][{filename}] chunk {chunk_index} | {processed_rows:,} lignes | {pct}% | {elapsed:.1f}s", 'PROGRESS')

                gc.collect()

            if chunks:
                df_features = pd.concat(chunks, ignore_index=True)
                df_features['Dataset'] = 'CICDDoS2019'
                total_rows = len(df_features)

                self.log(f"[CIC][{filename}] {total_rows:,} lignes numeriques (float32)", 'OK')
                self.update_file_progress(filename, 100, f"Termine: {total_rows:,} lignes")

                del chunks
                gc.collect()
                return {'data': df_features, 'rows': total_rows, 'filename': filename}

            self.update_file_progress(filename, 100, "Aucune colonne numerique")
            return None

        except pd.errors.ParserError as e:
            self.update_file_progress(os.path.basename(csv_file), 100, "Erreur parser")
            self.log_alert(f"Erreur parsing {os.path.basename(csv_file)}: {str(e)[:80]}", 'warning')
            return None
        except Exception as e:
            self.update_file_progress(os.path.basename(csv_file), 100, "Erreur")
            self.log_alert(f"Erreur {os.path.basename(csv_file)}: {str(e)[:100]}", 'warning')
            return None

    def run_consolidation(self):
        """Execution multithreadee optimisee avec barres detaillees."""
        try:
            self.log("=" * 60, 'INFO')
            self.log("DEMARRAGE CONSOLIDATION", 'INFO')
            self.log("=" * 60, 'INFO')

            for temp_path in [CONFIG['TEMP_TON'], CONFIG['TEMP_CIC']]:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        self.log(f"Ancien fichier supprime: {temp_path}", 'WARNING')
                    except Exception as e:
                        self.log_alert(f"Impossible de supprimer {temp_path}: {e}", 'warning')

            steps_total = 3
            self.update_progress(0, steps_total, "Initialisation...")
            self.reset_file_progress()

            if not self.running:
                return

            # --- ETAPE 1: TON_IOT ---
            ton_est = estimate_csv_rows(self.ton_iot_path, sample_lines=4000)
            self.log(f"TON_IoT -> estimation {ton_est:,} lignes", 'INFO')
            self.update_stage_progress('ton_read', 0, max(ton_est, 1), "Lecture TON_IoT")
            self.update_progress(0, steps_total, "Lecture TON_IoT...")

            mapping = {
                'duration': 'Flow Duration',
                'src_pkts': 'Total Fwd Packets',
                'src_bytes': 'Total Length Fwd Packets',
                'src_ip_bytes': 'Fwd Header Length',
                'dst_pkts': 'Total Bwd Packets',
                'dst_bytes': 'Total Length Bwd Packets',
                'dst_ip_bytes': 'Bwd Header Length',
                'label': 'Label',
            }
            wanted_cols = set(mapping.keys())

            try:
                file_size_gb = os.path.getsize(self.ton_iot_path) / (1024 ** 3)
                self.log(f"Taille TON_IoT: {file_size_gb:.2f} GB", 'INFO')

                chunk_size = max(50000, min(150000, self.ram.chunk_size // 3))
                processed_rows = 0
                chunk_idx = 0
                chunks = []

                read_kwargs = {
                    'low_memory': False,
                    'usecols': lambda c: c in wanted_cols,
                }

                if file_size_gb > 1.0:
                    self.log("Fichier >1GB -> lecture par chunks optimisee", 'WARNING')
                    for chunk in pd.read_csv(self.ton_iot_path, chunksize=chunk_size, **read_kwargs):
                        if not self.running:
                            return
                        chunk_idx += 1
                        processed_rows += len(chunk)
                        chunks.append(chunk)
                        self.update_stage_progress('ton_read', processed_rows, max(ton_est, processed_rows), f"Chunk {chunk_idx}")
                        if chunk_idx % 2 == 0:
                            self.log(f"[TON] chunk {chunk_idx} ({processed_rows:,} lignes)", 'PROGRESS')
                        gc.collect()
                    df_ton = pd.concat(chunks, ignore_index=True)
                else:
                    df_ton = pd.read_csv(self.ton_iot_path, **read_kwargs)

                self.ton_rows_total = len(df_ton)
                self.update_stage_progress('ton_read', self.ton_rows_total, max(ton_est, self.ton_rows_total), f"Lecture terminee ({self.ton_rows_total:,})")
                self.log(f"Chargement TON_IoT OK: {self.ton_rows_total:,} lignes", 'OK')
            except Exception as e:
                self.log_error(f"Erreur chargement TON_IoT: {e}")
                return

            if not self.running:
                return

            # Creation features TON_IoT
            self.update_stage_progress('ton_features', 10, 100, "Creation features TON_IoT...")
            try:
                df = pd.DataFrame()
                cols = df_ton.columns.tolist()

                for src_col, dst_col in mapping.items():
                    if src_col in cols:
                        df[dst_col] = pd.to_numeric(df_ton[src_col], errors='coerce').fillna(0).astype(np.float32)

                self.update_stage_progress('ton_features', 40, 100, "Calcul ratios/flows...")
                if 'Total Fwd Packets' in df.columns and 'Total Length Fwd Packets' in df.columns:
                    df['Fwd Packet Length Mean'] = np.where(
                        df['Total Fwd Packets'] > 0,
                        df['Total Length Fwd Packets'] / df['Total Fwd Packets'],
                        0,
                    ).astype(np.float32)

                if 'Total Bwd Packets' in df.columns and 'Total Length Bwd Packets' in df.columns:
                    df['Bwd Packet Length Mean'] = np.where(
                        df['Total Bwd Packets'] > 0,
                        df['Total Length Bwd Packets'] / df['Total Bwd Packets'],
                        0,
                    ).astype(np.float32)

                if 'Flow Duration' in df.columns:
                    df['Flow Bytes/s'] = np.where(
                        df['Flow Duration'] > 0,
                        (df['Total Length Fwd Packets'] + df['Total Length Bwd Packets']) / df['Flow Duration'],
                        0,
                    ).astype(np.float32)

                    df['Flow Packets/s'] = np.where(
                        df['Flow Duration'] > 0,
                        (df['Total Fwd Packets'] + df['Total Bwd Packets']) / df['Flow Duration'],
                        0,
                    ).astype(np.float32)

                    df['Fwd Packets/s'] = np.where(
                        df['Flow Duration'] > 0,
                        df['Total Fwd Packets'] / df['Flow Duration'],
                        0,
                    ).astype(np.float32)

                    df['Bwd Packets/s'] = np.where(
                        df['Flow Duration'] > 0,
                        df['Total Bwd Packets'] / df['Flow Duration'],
                        0,
                    ).astype(np.float32)

                if 'Total Length Fwd Packets' in df.columns and 'Total Length Bwd Packets' in df.columns:
                    df['Down/Up Ratio'] = np.where(
                        df['Total Length Fwd Packets'] > 0,
                        df['Total Length Bwd Packets'] / df['Total Length Fwd Packets'],
                        0,
                    ).astype(np.float32)

                    df['Average Packet Size'] = np.where(
                        (df['Total Fwd Packets'] + df['Total Bwd Packets']) > 0,
                        (df['Total Length Fwd Packets'] + df['Total Length Bwd Packets']) / (df['Total Fwd Packets'] + df['Total Bwd Packets']),
                        0,
                    ).astype(np.float32)

                df['Avg Fwd Segment Size'] = np.where(
                    df.get('Total Fwd Packets', pd.Series([1] * len(df))) > 0,
                    df.get('Total Length Fwd Packets', pd.Series([0] * len(df))) / df.get('Total Fwd Packets', pd.Series([1] * len(df))),
                    0,
                ).astype(np.float32)

                df['Avg Bwd Segment Size'] = np.where(
                    df.get('Total Bwd Packets', pd.Series([1] * len(df))) > 0,
                    df.get('Total Length Bwd Packets', pd.Series([0] * len(df))) / df.get('Total Bwd Packets', pd.Series([1] * len(df))),
                    0,
                ).astype(np.float32)

                df['Dataset'] = 'TON_IoT'
                df = df.fillna(0)

                self.update_stage_progress('ton_features', 80, 100, "Ecriture temporaire TON_IoT")
                df.to_csv(CONFIG['TEMP_TON'], index=False)
                self.ton_rows_total = len(df)
                self.update_stage_progress('ton_features', 100, 100, f"Sauvegarde TON_IoT ({self.ton_rows_total:,})")

                del df_ton, df
                gc.collect()
            except Exception as e:
                self.log_error(f"Erreur features: {e}")
                return

            if not self.running:
                return

            self.update_progress(1, steps_total, "TON_IoT termine")

            # --- ETAPE 2: CIC MULTITHREAD ---
            self.log("TRAITEMENT CIC (MULTI)", 'INFO')
            self.update_stage_progress('cic_files', 0, max(1, len(self.cic_files)), "CIC en attente")
            self.cic_rows_total = 0

            if not self.cic_files:
                self.log_alert("Aucun fichier CIC!", 'warning')
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.process_cic_file, csv_file): csv_file for csv_file in self.cic_files}

                    for idx, future in enumerate(as_completed(futures), start=1):
                        if not self.running:
                            executor.shutdown(wait=False)
                            return

                        csv_file = futures[future]
                        elapsed = time.time() - self.start_time
                        eta = (elapsed / idx) * (len(self.cic_files) - idx) if idx else None
                        self.update_stage_progress('cic_files', idx, len(self.cic_files), f"Fichiers {idx}/{len(self.cic_files)}", eta)
                        self.update_progress(1 + (idx / max(1, len(self.cic_files))), steps_total, f"CIC {idx}/{len(self.cic_files)}", eta)

                        try:
                            res = future.result()
                            if res and res.get('data') is not None:
                                df_res = res['data']
                                rows = res.get('rows', len(df_res))
                                write_header = not os.path.exists(CONFIG['TEMP_CIC'])
                                df_res.to_csv(CONFIG['TEMP_CIC'], mode='a', header=write_header, index=False)
                                self.cic_rows_total += rows
                                self.log(f"[CIC] {os.path.basename(csv_file)} -> {rows:,} lignes ecrites", 'OK')
                                del df_res
                                gc.collect()
                        except Exception as e:
                            self.log_alert(f"{os.path.basename(csv_file)}: {e}", 'warning')

                self.update_stage_progress('cic_files', len(self.cic_files), max(1, len(self.cic_files)), "CIC termine")
                self.log(f"CIC total: {self.cic_rows_total:,} lignes", 'INFO')

            if not self.running:
                return

            # --- ETAPE 3: FUSION FINALE ---
            self.log("FUSION FINALE", 'INFO')
            self.update_progress(2, steps_total, "Fusion finale...")
            fusion_total = self.ton_rows_total + self.cic_rows_total
            self.fusion_rows_total = fusion_total
            self.update_stage_progress('fusion', 0, max(1, fusion_total), "Fusion en cours...")

            try:
                df_ton = pd.read_csv(CONFIG['TEMP_TON'], low_memory=False)
                df_ton = df_ton.astype(np.float32)
                df_ton.to_csv(CONFIG['FUSION_OUTPUT'], index=False)
                written_rows = len(df_ton)
                self.update_stage_progress('fusion', written_rows, max(1, fusion_total), f"TON_IoT {written_rows:,}/{fusion_total or written_rows}")
                self.log(f"TON_IoT ecrit: {written_rows:,} lignes", 'OK')

                del df_ton
                gc.collect()

                if os.path.exists(CONFIG['TEMP_CIC']):
                    cic_chunk_size = max(50000, min(150000, self.ram.chunk_size // 8))
                    cic_written = 0

                    for chunk in pd.read_csv(CONFIG['TEMP_CIC'], chunksize=cic_chunk_size, low_memory=False):
                        if not self.running:
                            return
                        chunk = chunk.astype(np.float32)
                        chunk.to_csv(CONFIG['FUSION_OUTPUT'], mode='a', header=False, index=False)
                        cic_written += len(chunk)
                        written_rows += len(chunk)
                        self.update_stage_progress('fusion', written_rows, max(1, fusion_total), f"CIC {cic_written:,}/{self.cic_rows_total:,}")
                        self.log(f"[Fusion] CIC +{len(chunk):,} lignes (total {written_rows:,})", 'PROGRESS')
                        del chunk
                        gc.collect()

                    self.log(f"CIC ajoute: {cic_written:,} lignes", 'OK')

                self.update_stage_progress('fusion', fusion_total or written_rows, max(1, fusion_total or written_rows), "Fusion terminee")
                self.update_progress(3, steps_total, "Termine!")

                # Nettoyage des fichiers temporaires
                for temp_path in [CONFIG['TEMP_TON'], CONFIG['TEMP_CIC']]:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        self.log(f"Supprime: {temp_path}", 'OK')

                self.log_success(f"SUCCES: {CONFIG['FUSION_OUTPUT']} ({written_rows:,} lignes)")
            except Exception as e:
                self.log_error(f"Erreur fusion: {e}")
                return

            self.log("=" * 60, 'INFO')
            self.log("CONSOLIDATION REUSSIE", 'OK')
            self.log("=" * 60, 'INFO')

            duration = time.time() - self.start_time
            self.log(f"Duree: {str(timedelta(seconds=int(duration)))}", 'OK')

            self.status_label.config(text="Succes", fg='#27ae60')

        except Exception as e:
            self.log_error(f"Erreur globale: {e}")
            self.status_label.config(text="Erreur", fg='#d32f2f')

        finally:
            self.running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        # Fenêtre 1 : sélection des chemins
        root1 = tk.Tk()
        selector = FileSelectorGUI(root1)
        root1.mainloop()

        if not selector.ton_iot_path or not selector.cic_dir_path:
            print("❌ Annulé (chemins non fournis)")
            sys.exit(1)

        # Fenêtre 2 : consolidation avec progressions
        root2 = tk.Tk()
        app = ConsolidatorGUI(root2, selector.ton_iot_path, selector.cic_dir_path, selector.cic_files_found)
        root2.mainloop()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
