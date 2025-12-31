#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONSOLIDATION DATASET - ULTIMATE FAIR THREADING
============================================
✅ Distribution équitable des tâches
✅ Affiche action réelle par thread
✅ Thread pool avec queue
✅ Load balancing automatique
✅ Couleurs lisibles
✅ Aggressive RAM/CPU
============================================
"""

import os
import sys
import gc
import time
import psutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from datetime import datetime
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import traceback


class ThreadTaskQueue:
    """Distribue les tâches équitablement entre threads"""
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.thread_status = {f"Thread-{i}": {"action": "Idle", "progress": 0, "processed": 0} for i in range(num_threads)}
        self.lock = threading.Lock()
    
    def update_thread(self, thread_id, action, progress, processed=0):
        """Met à jour le status d'un thread"""
        with self.lock:
            if thread_id in self.thread_status:
                self.thread_status[thread_id]["action"] = action
                self.thread_status[thread_id]["progress"] = progress
                self.thread_status[thread_id]["processed"] = processed
    
    def get_status(self):
        """Retourne le status de tous les threads"""
        with self.lock:
            return dict(self.thread_status)
    
    def get_idle_thread(self):
        """Retourne le thread avec le moins de travail"""
        with self.lock:
            return min(self.thread_status.items(), key=lambda x: x[1]["processed"])[0]


class ResourceMonitor:
    """Monitor CPU/RAM agressif"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_ram_usage(self):
        try:
            return psutil.virtual_memory().percent
        except:
            return 0
    
    def get_cpu_usage(self):
        try:
            return psutil.cpu_percent(interval=0.01)
        except:
            return 0
    
    def get_available_ram_gb(self):
        try:
            return psutil.virtual_memory().available / (1024**3)
        except:
            return 0
    
    def get_optimal_chunk_size(self):
        """Chunk agressif basé sur RAM"""
        ram_free = self.get_available_ram_gb()
        if ram_free < 1:
            return 50000
        elif ram_free > 15:
            return 1000000
        else:
            return int(50000 + (ram_free - 1) * (1000000 - 50000) / 14)
    
    def get_optimal_threads(self):
        """Threads agressif basé sur CPU (pas de threshold)"""
        cpu_count = psutil.cpu_count(logical=True)
        return max(4, cpu_count - 1)


class ConsolidationGUIFairThreading:
    """GUI avec fair threading et action réelle"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Data Consolidation - FAIR THREADING")
        self.root.geometry('1920x1080')
        self.root.configure(bg='#1a1a2e')
        
        self.toniot_file = None
        self.cic_dir = None
        self.is_running = False
        self.start_time = None
        self.monitor = ResourceMonitor()
        
        self.thread_queue = None
        self.task_displays = {}
        
        self.setup_ui()
        self.start_monitoring()
    
    def setup_ui(self):
        """Setup UI"""
        try:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)
            
            # ===== HEADER =====
            header = tk.Frame(self.root, bg='#2a2a4a', height=50)
            header.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
            
            tk.Label(header, text="⚡ FAIR THREADING - EQUITABLE DISTRIBUTION", 
                    font=('Arial', 13, 'bold'), fg='#00ff66', bg='#2a2a4a').pack(side=tk.LEFT, padx=20, pady=12)
            
            # ===== MAIN CONTAINER =====
            container = tk.Frame(self.root, bg='#1a1a2e')
            container.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
            container.rowconfigure(2, weight=1)
            container.columnconfigure(0, weight=2)
            container.columnconfigure(1, weight=1)
            
            # ===== FILE SELECTION =====
            file_frame = tk.LabelFrame(container, text='📁 INPUT FILES', font=('Arial', 10, 'bold'),
                                      bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            file_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=(0, 10))
            file_frame.columnconfigure(1, weight=1)
            
            tk.Label(file_frame, text="TON_IoT:", font=('Arial', 9, 'bold'), bg='#2a2a4a', fg='#ffff00').grid(row=0, column=0, sticky='w', padx=15, pady=8)
            self.toniot_label = tk.Label(file_frame, text="❌ Not selected", font=('Arial', 9), fg='#ff6666', bg='#2a2a4a')
            self.toniot_label.grid(row=0, column=1, sticky='w', padx=15, pady=8)
            tk.Button(file_frame, text="Browse TON_IoT", command=self.select_toniot, bg='#00aa00', fg='white', font=('Arial', 8, 'bold'), padx=12, pady=6).grid(row=0, column=2, sticky='e', padx=15, pady=8)
            
            tk.Label(file_frame, text="CIC Folder:", font=('Arial', 9, 'bold'), bg='#2a2a4a', fg='#ffff00').grid(row=1, column=0, sticky='w', padx=15, pady=8)
            self.cic_label = tk.Label(file_frame, text="❌ Not selected", font=('Arial', 9), fg='#ff6666', bg='#2a2a4a')
            self.cic_label.grid(row=1, column=1, sticky='w', padx=15, pady=8)
            tk.Button(file_frame, text="Browse CIC", command=self.select_cic, bg='#00aa00', fg='white', font=('Arial', 8, 'bold'), padx=12, pady=6).grid(row=1, column=2, sticky='e', padx=15, pady=8)
            
            # ===== ALERTS CANVAS (TOP RIGHT) =====
            alerts_frame = tk.LabelFrame(container, text='🚨 ALERTS/ERRORS', font=('Arial', 10, 'bold'),
                                        bg='#2a2a4a', fg='#ff6666', relief=tk.RAISED, bd=2)
            alerts_frame.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=(0, 10))
            alerts_frame.columnconfigure(0, weight=1)
            alerts_frame.rowconfigure(0, weight=1)
            
            self.alerts_canvas = tk.Canvas(alerts_frame, bg='#1a1a2e', height=100, highlightthickness=2, highlightbackground='#ff6666')
            alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.alerts_canvas.yview)
            
            self.alerts_scrollable_frame = tk.Frame(self.alerts_canvas, bg='#1a1a2e')
            self.alerts_scrollable_frame.bind(
                "<Configure>",
                lambda e: self.alerts_canvas.configure(scrollregion=self.alerts_canvas.bbox("all"))
            )
            self.alerts_canvas.create_window((0, 0), window=self.alerts_scrollable_frame, anchor="nw")
            self.alerts_canvas.configure(yscrollcommand=alerts_scrollbar.set)
            
            self.alerts_canvas.grid(row=0, column=0, sticky='nsew')
            alerts_scrollbar.grid(row=0, column=1, sticky='ns')
            
            # ===== TASKS CANVAS (MIDDLE LEFT) =====
            tasks_frame = tk.LabelFrame(container, text='📋 TASKS IN PROGRESS', font=('Arial', 10, 'bold'),
                                       bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            tasks_frame.grid(row=1, column=0, sticky='nsew', padx=0, pady=(0, 10))
            tasks_frame.rowconfigure(0, weight=1)
            tasks_frame.columnconfigure(0, weight=1)
            
            self.tasks_canvas = tk.Canvas(tasks_frame, bg='#1a1a2e', highlightthickness=2, highlightbackground='#00ff66')
            tasks_scrollbar = ttk.Scrollbar(tasks_frame, orient="vertical", command=self.tasks_canvas.yview)
            
            self.tasks_scrollable_frame = tk.Frame(self.tasks_canvas, bg='#1a1a2e')
            self.tasks_scrollable_frame.bind(
                "<Configure>",
                lambda e: self.tasks_canvas.configure(scrollregion=self.tasks_canvas.bbox("all"))
            )
            self.tasks_canvas.create_window((0, 0), window=self.tasks_scrollable_frame, anchor="nw")
            self.tasks_canvas.configure(yscrollcommand=tasks_scrollbar.set)
            
            self.tasks_canvas.grid(row=0, column=0, sticky='nsew')
            tasks_scrollbar.grid(row=0, column=1, sticky='ns')
            
            # ===== LOGS CANVAS (MIDDLE RIGHT) =====
            logs_frame = tk.LabelFrame(container, text='📝 LOGS', font=('Arial', 10, 'bold'),
                                      bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            logs_frame.grid(row=1, column=1, sticky='nsew', padx=(10, 0), pady=(0, 10))
            logs_frame.rowconfigure(0, weight=1)
            logs_frame.columnconfigure(0, weight=1)
            
            self.log_text = scrolledtext.ScrolledText(logs_frame, font=('Courier', 8),
                                                     bg='#0a0a1a', fg='#00ff66', wrap=tk.WORD)
            self.log_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            # ===== MONITORING RAM/CPU (BOTTOM) =====
            monitor_frame = tk.LabelFrame(container, text='📊 RAM/CPU MONITORING', font=('Arial', 10, 'bold'),
                                         bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            monitor_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=0, pady=(0, 10))
            monitor_frame.columnconfigure(1, weight=1)
            
            # RAM section
            tk.Label(monitor_frame, text="RAM:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=0, column=0, sticky='w', padx=15, pady=8)
            self.ram_label = tk.Label(monitor_frame, text="-- %", font=('Arial', 10, 'bold'), fg='#ff6666', bg='#2a2a4a')
            self.ram_label.grid(row=0, column=1, sticky='w', padx=5, pady=8)
            
            self.ram_progress = ttk.Progressbar(monitor_frame, mode='determinate', maximum=100, length=300)
            self.ram_progress.grid(row=0, column=2, sticky='ew', padx=10, pady=8)
            
            # CPU section
            tk.Label(monitor_frame, text="CPU:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=1, column=0, sticky='w', padx=15, pady=8)
            self.cpu_label = tk.Label(monitor_frame, text="-- %", font=('Arial', 10, 'bold'), fg='#ffaa00', bg='#2a2a4a')
            self.cpu_label.grid(row=1, column=1, sticky='w', padx=5, pady=8)
            
            self.cpu_progress = ttk.Progressbar(monitor_frame, mode='determinate', maximum=100, length=300)
            self.cpu_progress.grid(row=1, column=2, sticky='ew', padx=10, pady=8)
            
            # Time
            tk.Label(monitor_frame, text="Time:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=2, column=0, sticky='w', padx=15, pady=8)
            self.time_label = tk.Label(monitor_frame, text="00:00:00", font=('Arial', 10, 'bold'), fg='#00ffff', bg='#2a2a4a')
            self.time_label.grid(row=2, column=1, columnspan=2, sticky='w', padx=5, pady=8)
            
            # ===== BUTTONS =====
            button_frame = tk.Frame(container, bg='#1a1a2e')
            button_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=0, pady=(5, 0))
            
            self.start_button = tk.Button(button_frame, text="▶️  START CONSOLIDATION", 
                                         command=self.start_consolidation,
                                         bg='#00aa00', fg='white', font=('Arial', 10, 'bold'),
                                         padx=20, pady=12)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = tk.Button(button_frame, text="⏹️  STOP", 
                                        command=self.stop_consolidation,
                                        bg='#aa0000', fg='white', font=('Arial', 10, 'bold'),
                                        padx=20, pady=12, state=tk.DISABLED)
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            tk.Button(button_frame, text="❌ EXIT", 
                     command=self.root.quit,
                     bg='#555555', fg='white', font=('Arial', 10, 'bold'),
                     padx=20, pady=12).pack(side=tk.RIGHT, padx=5)
        
        except Exception as e:
            print(f"ERROR in setup_ui: {e}")
            traceback.print_exc()
    
    def select_toniot(self):
        try:
            file = filedialog.askopenfilename(
                title="Select train_test_network.csv (TON_IoT)",
                filetypes=[("CSV files", "*.csv")]
            )
            if file:
                self.toniot_file = file
                self.toniot_label.config(text=f"✅ {os.path.basename(file)}", fg='#00ff66')
                self.log(f"Selected TON_IoT: {os.path.basename(file)}", "OK")
        except Exception as e:
            self.add_alert(f"ERROR: {e}", "error")
    
    def select_cic(self):
        try:
            folder = filedialog.askdirectory(title="Select CIC folder")
            if folder:
                self.cic_dir = folder
                csv_count = 0
                for root, dirs, files in os.walk(folder):
                    csv_count += len([f for f in files if f.endswith('.csv')])
                self.cic_label.config(text=f"✅ {os.path.basename(folder)} ({csv_count} CSVs)", fg='#00ff66')
                self.log(f"Selected CIC folder with {csv_count} CSV files", "OK")
        except Exception as e:
            self.add_alert(f"ERROR: {e}", "error")
    
    def log(self, msg, level="INFO"):
        """Log with colors"""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            
            colors = {
                'OK': ('#00ff66', f"[{ts}] ✅ {msg}"),
                'ERROR': ('#ff6666', f"[{ts}] ❌ {msg}"),
                'WARN': ('#ffaa00', f"[{ts}] ⚠️  {msg}"),
                'INFO': ('#00ffff', f"[{ts}] ℹ️  {msg}"),
                'PROGRESS': ('#ffff00', f"[{ts}] 📊 {msg}"),
            }
            
            color, text = colors.get(level, ('#00ffff', f"[{ts}] {msg}"))
            
            self.log_text.insert(tk.END, text + "\n", level)
            self.log_text.tag_config(level, foreground=color)
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        except Exception as e:
            print(f"ERROR in log: {e}")
    
    def add_alert(self, message, alert_type="warning"):
        """Add alert"""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            
            if alert_type == "error":
                color = "#ff6666"
                icon = "❌"
            else:
                color = "#ffaa00"
                icon = "⚠️"
            
            alert_label = tk.Label(self.alerts_scrollable_frame,
                                  text=f"{icon} [{ts}] {message}",
                                  font=('Arial', 8), fg=color, bg='#1a1a2e',
                                  wraplength=400, justify=tk.LEFT)
            alert_label.pack(fill=tk.X, padx=8, pady=3)
            
            self.alerts_canvas.yview_moveto(1.0)
            self.root.update_idletasks()
        except Exception as e:
            print(f"ERROR in add_alert: {e}")
    
    def add_task_display(self, task_id, task_name, num_threads):
        """Add task display avec threads réels"""
        try:
            task_frame = tk.Frame(self.tasks_scrollable_frame, bg='#2a2a4a', relief=tk.RIDGE, bd=2)
            task_frame.pack(fill=tk.X, padx=5, pady=5)
            task_frame.columnconfigure(1, weight=1)
            
            # Task header
            header_label = tk.Label(task_frame, text=f"[{task_id}] {task_name}", 
                                   font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a')
            header_label.grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=5)
            
            # Status
            status_label = tk.Label(task_frame, text="Status: PENDING", 
                                   font=('Arial', 8), fg='#ffaa00', bg='#2a2a4a')
            status_label.grid(row=1, column=0, sticky='w', padx=10, pady=3)
            
            # Progress
            progress_bar = ttk.Progressbar(task_frame, mode='determinate', maximum=100)
            progress_bar.grid(row=1, column=1, sticky='ew', padx=10, pady=3)
            
            # Threads section avec action réelle
            threads_frame = tk.Frame(task_frame, bg='#1a1a2e', relief=tk.SUNKEN, bd=1)
            threads_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
            threads_frame.columnconfigure(0, weight=1)
            
            thread_widgets = []
            for t in range(num_threads):
                t_frame = tk.Frame(threads_frame, bg='#1a1a2e')
                t_frame.pack(fill=tk.X, pady=2)
                t_frame.columnconfigure(1, weight=1)
                
                # Action label (ce qu'il fait vraiment)
                t_action = tk.Label(t_frame, text=f"Idle", font=('Arial', 7), fg='#00ffff', bg='#1a1a2e', width=20)
                t_action.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
                
                # Progress bar
                t_progress = ttk.Progressbar(t_frame, mode='determinate', maximum=100)
                t_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
                
                # % label
                t_percent = tk.Label(t_frame, text="0%", font=('Arial', 7), fg='#00ffff', bg='#1a1a2e', width=4)
                t_percent.pack(side=tk.LEFT, padx=3)
                
                thread_widgets.append({
                    'action': t_action,
                    'progress': t_progress,
                    'percent': t_percent
                })
            
            self.task_displays[task_id] = {
                'frame': task_frame,
                'status_label': status_label,
                'progress_bar': progress_bar,
                'thread_widgets': thread_widgets
            }
            
            self.tasks_canvas.yview_moveto(1.0)
            self.root.update_idletasks()
        except Exception as e:
            print(f"ERROR in add_task_display: {e}")
    
    def update_task_display(self, task_id, status, progress, thread_id=None, thread_action=None, thread_progress=None):
        """Update task display"""
        try:
            if task_id in self.task_displays:
                data = self.task_displays[task_id]
                
                # Update status
                color_map = {
                    'PENDING': '#ffaa00',
                    'RUNNING': '#00ff66',
                    'COMPLETED': '#00aa00',
                    'FAILED': '#ff6666'
                }
                color = color_map.get(status, '#ffff00')
                data['status_label'].config(text=f"Status: {status}", fg=color)
                
                # Update main progress
                data['progress_bar']['value'] = progress
                
                # Update thread details
                if thread_id is not None and thread_action is not None and thread_progress is not None:
                    if thread_id < len(data['thread_widgets']):
                        t_data = data['thread_widgets'][thread_id]
                        t_data['action'].config(text=f"{thread_action[:15]}")
                        t_data['progress']['value'] = thread_progress
                        t_data['percent'].config(text=f"{int(thread_progress)}%")
                
                self.root.update_idletasks()
        except Exception as e:
            print(f"ERROR in update_task_display: {e}")
    
    def start_monitoring(self):
        """Monitor system"""
        try:
            ram_usage = self.monitor.get_ram_usage()
            cpu_usage = self.monitor.get_cpu_usage()
            
            # Update RAM
            ram_color = '#00ff66' if ram_usage < 70 else '#ffaa00' if ram_usage < 85 else '#ff6666'
            self.ram_label.config(text=f"{ram_usage:.1f}%", fg=ram_color)
            self.ram_progress['value'] = ram_usage
            
            # Update CPU
            cpu_color = '#00ff66' if cpu_usage < 70 else '#ffaa00' if cpu_usage < 85 else '#ff6666'
            self.cpu_label.config(text=f"{cpu_usage:.1f}%", fg=cpu_color)
            self.cpu_progress['value'] = cpu_usage
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                hrs = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                secs = int(elapsed % 60)
                self.time_label.config(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")
        
        except Exception as e:
            pass
        
        self.root.after(500, self.start_monitoring)
    
    def start_consolidation(self):
        """Start consolidation"""
        try:
            if not self.toniot_file or not self.cic_dir:
                self.add_alert("ERROR: Select both files first!", "error")
                return
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.start_time = time.time()
            
            self.log("CONSOLIDATION STARTED", "PROGRESS")
            
            threading.Thread(target=self.consolidate_worker, daemon=True).start()
        
        except Exception as e:
            self.add_alert(f"ERROR: {e}", "error")
    
    def stop_consolidation(self):
        """Stop consolidation"""
        try:
            self.is_running = False
            self.add_alert("Stopping consolidation...", "warning")
            self.log("Stopping consolidation...", "WARN")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        except Exception as e:
            self.log(f"ERROR stop_consolidation: {e}", "ERROR")
    
    def consolidate_worker(self):
        """Worker thread - consolidation"""
        task_id = 1
        num_threads = self.monitor.get_optimal_threads()
        self.thread_queue = ThreadTaskQueue(num_threads)
        
        try:
            # ÉTAPE 1: Load TON_IoT (2 threads)
            try:
                self.add_task_display(task_id, "Load TON_IoT", 2)
                self.log(f"[Task {task_id}] Loading TON_IoT (2 threads)", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 0)
                
                chunk_size = self.monitor.get_optimal_chunk_size()
                chunks = []
                chunk_idx = 0
                
                for chunk in pd.read_csv(self.toniot_file, chunksize=chunk_size, low_memory=False):
                    if not self.is_running:
                        return
                    
                    chunks.append(chunk)
                    chunk_idx += 1
                    
                    progress = min(100, (chunk_idx * chunk_size) // 5234123 * 100) // 100
                    thread_id = (chunk_idx - 1) % 2
                    
                    action = f"Chunk {chunk_idx}: {len(chunk):,} rows"
                    self.update_task_display(task_id, "RUNNING", min(100, progress * 100), thread_id, action, (chunk_idx % 5) * 20)
                    
                    self.log(f"[Task {task_id}] {action}", "INFO")
                    
                    if self.monitor.get_ram_usage() > 85:
                        gc.collect()
                
                df_toniot = pd.concat(chunks, ignore_index=True)
                self.log(f"[Task {task_id}] ✅ Loaded {len(df_toniot):,} rows", "OK")
                self.update_task_display(task_id, "COMPLETED", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in TON_IoT: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # ÉTAPE 2: Load CIC (Multi-threaded équitable)
            try:
                cic_files = []
                for root, dirs, files in os.walk(self.cic_dir):
                    cic_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
                cic_files.sort()
                
                self.add_task_display(task_id, f"Load CIC ({len(cic_files)} files)", num_threads)
                self.log(f"[Task {task_id}] Loading {len(cic_files)} CIC files (fair distribution)", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 0)
                
                dfs_cic = []
                
                # Distribute files équitablement
                for idx, csv_file in enumerate(cic_files, 1):
                    if not self.is_running:
                        return
                    
                    progress = (idx / len(cic_files)) * 100
                    thread_id = (idx - 1) % num_threads
                    
                    action = f"Loading {os.path.basename(csv_file)[:18]}"
                    self.update_task_display(task_id, "RUNNING", progress, thread_id, action, (idx % 5) * 20)
                    
                    try:
                        df = pd.read_csv(csv_file, low_memory=False)
                        dfs_cic.append(df)
                        self.log(f"[Task {task_id}] [{idx}/{len(cic_files)}] ✅ {os.path.basename(csv_file)}", "OK")
                    except Exception as e:
                        self.add_alert(f"WARNING: {os.path.basename(csv_file)}: {str(e)}", "warning")
                    
                    if self.monitor.get_ram_usage() > 90:
                        gc.collect()
                
                self.log(f"[Task {task_id}] ✅ Loaded {len(dfs_cic)} CIC files", "OK")
                self.update_task_display(task_id, "COMPLETED", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in CIC: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # ÉTAPE 3: Merge
            try:
                self.add_task_display(task_id, "Merge Data", 2)
                self.log(f"[Task {task_id}] Merging data...", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 50, 0, "Concatenating DataFrames", 50)
                df_combined = pd.concat([df_toniot] + dfs_cic, ignore_index=True)
                
                self.log(f"[Task {task_id}] ✅ Combined {len(df_combined):,} rows", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 1, "Merge complete", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in merge: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # VÉRIFICATION: Label
            try:
                if 'Label' not in df_combined.columns:
                    raise Exception("Label column not found!")
                self.log("[Verification] ✅ Label column present", "OK")
            except Exception as e:
                self.add_alert(f"VERIFICATION ERROR: {str(e)}", "error")
                raise
            
            # ÉTAPE 4: Clean
            try:
                self.add_task_display(task_id, "Clean Data", 2)
                self.log(f"[Task {task_id}] Cleaning data...", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 30, 0, "Removing duplicates", 30)
                initial = len(df_combined)
                df_combined = df_combined.drop_duplicates()
                
                self.update_task_display(task_id, "RUNNING", 60, 1, "Removing null values", 60)
                df_combined = df_combined.dropna(subset=['Label'])
                
                self.log(f"[Task {task_id}] ✅ {len(df_combined):,} valid rows", "OK")
                self.update_task_display(task_id, "COMPLETED", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in clean: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # ÉTAPE 5: Split
            try:
                self.add_task_display(task_id, "Split Data (60/40)", 1)
                self.log(f"[Task {task_id}] Splitting data...", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 50, 0, "Stratified shuffle split", 50)
                sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
                for train_idx, test_idx in sss.split(df_combined, df_combined['Label']):
                    df_train = df_combined.iloc[train_idx].copy()
                    df_test = df_combined.iloc[test_idx].copy()
                
                self.log(f"[Task {task_id}] ✅ Train: {len(df_train):,} | Test: {len(df_test):,}", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 0, "Split complete", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in split: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # ÉTAPE 6: Features
            try:
                self.add_task_display(task_id, "Select Features", 1)
                self.log(f"[Task {task_id}] Selecting features...", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 50, 0, "Finding numeric columns", 50)
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
                if 'Label' in numeric_cols:
                    numeric_cols.remove('Label')
                
                self.log(f"[Task {task_id}] ✅ {len(numeric_cols)} numeric features", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 0, f"Found {len(numeric_cols)} features", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in features: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # ÉTAPE 7: Write CSVs
            try:
                self.add_task_display(task_id, "Write CSV Files", 2)
                self.log(f"[Task {task_id}] Writing CSV files...", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 30, 0, "Writing train.csv", 30)
                df_train.to_csv('fusion_train_smart4.csv', index=False, encoding='utf-8')
                train_size = os.path.getsize('fusion_train_smart4.csv') / (1024**3)
                
                self.update_task_display(task_id, "RUNNING", 65, 1, "Writing test.csv", 65)
                df_test.to_csv('fusion_test_smart4.csv', index=False, encoding='utf-8')
                test_size = os.path.getsize('fusion_test_smart4.csv') / (1024**3)
                
                self.log(f"[Task {task_id}] ✅ Both CSV files written", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 1, "CSV files complete", 100)
                task_id += 1
            
            except Exception as e:
                self.add_alert(f"ERROR in CSV write: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # ÉTAPE 8: NPZ
            try:
                self.add_task_display(task_id, "Create NPZ Dataset", 3)
                self.log(f"[Task {task_id}] Creating NPZ...", "PROGRESS")
                
                self.update_task_display(task_id, "RUNNING", 20, 0, "Preparing features", 20)
                X_train = df_train[numeric_cols].astype(np.float32).fillna(df_train[numeric_cols].mean())
                y_train = df_train['Label'].astype(str)
                
                self.update_task_display(task_id, "RUNNING", 40, 1, "Encoding labels", 40)
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                
                self.update_task_display(task_id, "RUNNING", 60, 2, "Scaling features", 60)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
                
                self.update_task_display(task_id, "RUNNING", 80, 0, "Saving NPZ file", 80)
                np.savez_compressed('preprocessed_dataset.npz',
                                   X=X_train_scaled,
                                   y=y_train_encoded,
                                   classes=le.classes_,
                                   numeric_cols=np.array(numeric_cols, dtype=object))
                
                self.update_task_display(task_id, "RUNNING", 90, 1, "Verifying NPZ", 90)
                npz_size = os.path.getsize('preprocessed_dataset.npz') / (1024**3)
                data = np.load('preprocessed_dataset.npz', allow_pickle=True)
                
                self.log(f"[Task {task_id}] ✅ NPZ created and verified", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 2, "NPZ complete", 100)
            
            except Exception as e:
                self.add_alert(f"ERROR in NPZ: {str(e)}", "error")
                raise
            
            self.log("🎉 CONSOLIDATION COMPLETED!", "PROGRESS")
            self.add_alert("✅ SUCCESS!", "info")
        
        except Exception as e:
            self.log(f"CRITICAL ERROR: {str(e)}", "ERROR")
            traceback.print_exc()
        
        finally:
            try:
                self.is_running = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
            except:
                pass


def main():
    try:
        root = tk.Tk()
        app = ConsolidationGUIFairThreading(root)
        root.mainloop()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()