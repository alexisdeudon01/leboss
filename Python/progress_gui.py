#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROGRESS GUI - Interface Tkinter Générique
==========================================
✅ Interface réutilisable pour tous les scripts
✅ Progress bars, logs, stats
✅ Support multi-threading
==========================================
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime, timedelta


class GenericProgressGUI:
    """Interface GUI générique et réutilisable"""
    
    def __init__(self, title="Processing", header_info="", max_workers=4):
        self.title = title
        self.header_info = header_info
        self.max_workers = max_workers
        
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('1400x800')
        self.root.configure(bg='#f0f0f0')
        
        self.stages = {}
        self.file_progress = {}
        self.global_progress = {'current': 0, 'total': 0, 'status': ''}
        self.start_time = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup interface"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=60)
        header.grid(row=0, column=0, sticky='ew')
        
        tk.Label(header, text=self.title, 
                font=('Arial', 14, 'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=20, pady=15)
        
        if self.header_info:
            tk.Label(header, text=self.header_info,
                    font=('Arial', 10), fg='#bdc3c7', bg='#2c3e50').pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Main container
        container = tk.Frame(self.root, bg='#f0f0f0')
        container.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=1)
        
        # Left: Logs
        left_frame = tk.LabelFrame(container, text='LOGS', font=('Arial', 10, 'bold'),
                                   bg='white', relief=tk.SUNKEN, bd=2)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8), pady=0)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(left_frame, font=('Courier', 8),
                                                  bg='#1a1a1a', fg='#00ff00', wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Right: Stats
        right_frame = tk.Frame(container, bg='#f0f0f0')
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(8, 0), pady=0)
        right_frame.rowconfigure(6, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
        # Global progress
        global_frame = tk.LabelFrame(right_frame, text='GLOBAL', font=('Arial', 9, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
        global_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=(0, 5))
        global_frame.columnconfigure(0, weight=1)
        
        self.global_label = tk.Label(global_frame, text='0%', font=('Arial', 12, 'bold'),
                                    bg='white', fg='#3498db')
        self.global_label.pack(fill=tk.X, padx=8, pady=3)
        
        self.global_progress_bar = ttk.Progressbar(global_frame, mode='determinate', maximum=100)
        self.global_progress_bar.pack(fill=tk.X, padx=8, pady=3)
        
        # Stages
        stages_frame = tk.LabelFrame(right_frame, text='STAGES', font=('Arial', 9, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
        stages_frame.grid(row=1, column=0, sticky='ew', padx=0, pady=(0, 5))
        stages_frame.columnconfigure(0, weight=1)
        stages_frame.rowconfigure(0, weight=1)
        
        self.stages_text = scrolledtext.ScrolledText(stages_frame, height=8, font=('Courier', 8),
                                                     bg='#f8f8f8', fg='#333')
        self.stages_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Time
        time_frame = tk.LabelFrame(right_frame, text='TIME', font=('Arial', 9, 'bold'),
                                   bg='white', relief=tk.SUNKEN, bd=2)
        time_frame.grid(row=2, column=0, sticky='ew', padx=0, pady=(0, 5))
        
        self.elapsed_label = tk.Label(time_frame, text='00:00:00', font=('Arial', 10, 'bold'),
                                     bg='white', fg='#9b59b6')
        self.elapsed_label.pack(fill=tk.X, padx=8, pady=3)
        
        self.eta_label = tk.Label(time_frame, text='ETA: --:--:--', font=('Arial', 9),
                                 bg='white', fg='#7f8c8d')
        self.eta_label.pack(fill=tk.X, padx=8, pady=3)
        
        # Alerts
        alerts_frame = tk.LabelFrame(right_frame, text='ALERTS', font=('Arial', 9, 'bold'),
                                     bg='white', relief=tk.SUNKEN, bd=2)
        alerts_frame.grid(row=3, column=0, sticky='ew', padx=0, pady=(0, 5))
        alerts_frame.columnconfigure(0, weight=1)
        
        self.alert_label = tk.Label(alerts_frame, text='Ready', font=('Arial', 10, 'bold'),
                                   bg='white', fg='#27ae60')
        self.alert_label.pack(fill=tk.X, padx=8, pady=5)
        
        # File progress
        file_frame = tk.LabelFrame(right_frame, text='FILES', font=('Arial', 9, 'bold'),
                                   bg='white', relief=tk.SUNKEN, bd=2)
        file_frame.grid(row=4, column=0, sticky='ew', padx=0, pady=(0, 5))
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)
        
        self.file_text = scrolledtext.ScrolledText(file_frame, height=5, font=('Courier', 8),
                                                   bg='#f8f8f8', fg='#333')
        self.file_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Status bar
        footer = tk.Frame(self.root, bg='#ecf0f1', height=40)
        footer.grid(row=2, column=0, sticky='ew')
        
        self.status_label = tk.Label(footer, text='Ready',
                                    font=('Arial', 9, 'bold'),
                                    fg='#27ae60', bg='#ecf0f1')
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)
    
    def add_stage(self, stage_id, label):
        """Ajoute une étape"""
        self.stages[stage_id] = {
            'label': label,
            'current': 0,
            'total': 1,
            'status': ''
        }
        self.update_stages_display()
    
    def update_stage(self, stage_id, current, total, status=''):
        """Met à jour une étape"""
        if stage_id in self.stages:
            self.stages[stage_id]['current'] = current
            self.stages[stage_id]['total'] = total
            self.stages[stage_id]['status'] = status
            self.update_stages_display()
    
    def update_stages_display(self):
        """Affiche toutes les étapes"""
        try:
            self.stages_text.config(state=tk.NORMAL)
            self.stages_text.delete(1.0, tk.END)
            
            for stage_id, data in self.stages.items():
                current = data['current']
                total = data['total']
                pct = (current / total * 100) if total > 0 else 0
                bar_width = 20
                filled = int(bar_width * pct / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                line = f"{data['label']}\n"
                line += f"[{bar}] {pct:5.1f}% ({current}/{total})\n"
                if data['status']:
                    line += f"→ {data['status']}\n"
                line += "\n"
                
                self.stages_text.insert(tk.END, line)
            
            self.stages_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
        except:
            pass
    
    def update_global(self, current, total, status=''):
        """Met à jour progress global"""
        try:
            self.global_progress['current'] = current
            self.global_progress['total'] = total
            self.global_progress['status'] = status
            
            pct = (current / total * 100) if total > 0 else 0
            self.global_label.config(text=f'{pct:.1f}%')
            self.global_progress_bar['value'] = pct
            
            self.root.update_idletasks()
        except:
            pass
    
    def update_file_progress(self, filename, pct, status=''):
        """Met à jour progress fichier"""
        try:
            self.file_progress[filename] = {'pct': pct, 'status': status}
            
            self.file_text.config(state=tk.NORMAL)
            self.file_text.delete(1.0, tk.END)
            
            for fname, data in self.file_progress.items():
                bar_width = 20
                filled = int(bar_width * data['pct'] / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                line = f"{fname}\n"
                line += f"[{bar}] {data['pct']:3.0f}%\n"
                if data['status']:
                    line += f"→ {data['status']}\n"
                line += "\n"
                
                self.file_text.insert(tk.END, line)
            
            self.file_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
        except:
            pass
    
    def log(self, msg, level="INFO"):
        """Ajoute log"""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            log_line = f"[{ts}] [{level:<6}] {msg}\n"
            
            self.log_text.insert(tk.END, log_line)
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        except:
            pass
    
    def log_alert(self, msg, level="info"):
        """Alert"""
        try:
            colors = {
                'success': '#27ae60',
                'error': '#e74c3c',
                'warning': '#f39c12',
                'info': '#3498db'
            }
            color = colors.get(level, '#3498db')
            
            self.alert_label.config(text=msg, fg=color)
            
            self.log(msg, level=level.upper())
            self.root.update_idletasks()
        except:
            pass
    
    def update_time(self):
        """Met à jour temps"""
        try:
            if self.start_time:
                elapsed = time.time() - self.start_time
                hrs = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                secs = int(elapsed % 60)
                
                self.elapsed_label.config(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")
                
                # ETA
                if self.global_progress['current'] > 0 and self.global_progress['total'] > 0:
                    avg_time = elapsed / self.global_progress['current']
                    remaining = avg_time * (self.global_progress['total'] - self.global_progress['current'])
                    eta = datetime.now() + timedelta(seconds=remaining)
                    self.eta_label.config(text=f"ETA: {eta.strftime('%H:%M:%S')}")
            
            self.root.after(500, self.update_time)
        except:
            self.root.after(500, self.update_time)
    
    def start(self):
        """Démarre GUI"""
        self.start_time = time.time()
        self.update_time()
        self.root.mainloop()
    
    def stop(self):
        """Arrête GUI"""
        try:
            self.root.quit()
        except:
            pass


if __name__ == "__main__":
    # Test
    gui = GenericProgressGUI(title="Test Progress GUI", header_info="Demo")
    
    gui.add_stage("stage1", "Stage 1: Data Loading")
    gui.add_stage("stage2", "Stage 2: Processing")
    gui.add_stage("stage3", "Stage 3: Output")
    
    gui.log("Starting test...", level="INFO")
    gui.log_alert("Test started", level="success")
    
    def demo():
        for i in range(11):
            gui.update_stage("stage1", i, 10, f"Loading {i}/10")
            gui.update_global(i, 30, "Processing...")
            gui.log(f"Progress: {i}/10", level="INFO")
            time.sleep(0.5)
        
        gui.log_alert("Complete!", level="success")
    
    threading.Thread(target=demo, daemon=True).start()
    gui.start()
