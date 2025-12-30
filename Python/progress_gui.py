#!/usr/bin/env python3
"""
Generic progress GUI (Tkinter) inspired by consolidatedata layout.

Objectif: réutiliser la même logique de canvas/progress bars/logs dans les autres scripts
sans réécrire l'UI. Intégration minimale:

    from progress_gui import GenericProgressGUI
    ui = GenericProgressGUI(title="Mon job", header_info="Dataset XYZ", max_workers=8)
    ui.add_stage("step1", "Lecture")
    ui.add_stage("step2", "Traitement")

    def worker():
        ui.update_global(0, 100, "Lecture...", eta=360)
        ui.update_stage("step1", 50, 100, "Mi-parcours")
        ui.update_file_progress("fileA.csv", 30, "Chunk 3/10")
        ui.log("Message détaillé", level="INFO")
        ui.log_alert("Petite alerte", level="warning")
        ui.update_stats_now(cpu=12.3, ram=42.1)  # optionnel
    threading.Thread(target=worker, daemon=True).start()
    ui.start()  # bloque sur mainloop

API principale:
  - update_global(current, total, msg="", eta=None)
  - add_stage(key, title)
  - update_stage(key, current, total, msg="", eta=None)
  - reset_file_progress()
  - update_file_progress(name, percent, status)
  - log(msg, level="INFO")  # levels: INFO, OK, ERROR, WARNING, PROGRESS
  - log_alert(msg, level="warning"|"error"|"success")
  - start() pour lancer la boucle Tk

Option: psutil est utilisé si présent pour l'auto-monitoring CPU/RAM (sinon affiche N/A).
"""

import threading
import time
from collections import deque
from datetime import timedelta
try:
    import psutil
except ImportError:
    psutil = None

import tkinter as tk
from tkinter import ttk, scrolledtext


def _fmt_eta(seconds):
    try:
        return str(timedelta(seconds=int(seconds)))
    except Exception:
        return "--:--"


class GenericProgressGUI:
    def __init__(self, title="Task Monitor", header_info="", max_workers=4, show_monitoring=True):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.max_workers = max_workers
        self.progress_blocks = {}
        self.file_progress_widgets = {}
        self.file_progress_order = deque()
        self.max_file_bars = max_workers
        self.logs = deque(maxlen=500)
        self.alerts = deque(maxlen=200)
        self.show_monitoring = show_monitoring

        self._build_ui(header_info)
        if show_monitoring:
            self._schedule_stats()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, header_info):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(header, text=self.root.title(),
                 font=("Arial", 14, "bold"), fg="white", bg="#2c3e50").pack(side=tk.LEFT, padx=20, pady=15)
        tk.Label(header, text=header_info,
                 font=("Arial", 9), fg="#bdc3c7", bg="#2c3e50").pack(side=tk.LEFT, padx=20)

        container = tk.Frame(self.root, bg="#f0f0f0")
        container.grid(row=1, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, bg="#f0f0f0", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vscroll.set)

        main = tk.Frame(canvas, bg="#f0f0f0")
        win_id = canvas.create_window((0, 0), window=main, anchor="nw")

        def _on_frame_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        main.bind("<Configure>", _on_frame_config)

        def _on_canvas_config(event):
            try:
                canvas.itemconfigure(win_id, width=event.width)
            except Exception:
                pass
        canvas.bind("<Configure>", _on_canvas_config)

        # Layout: progress grid + logs
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)
        main.columnconfigure(0, weight=1)

        progress_grid = tk.Frame(main, bg="#f0f0f0")
        progress_grid.grid(row=0, column=0, sticky="nsew")
        progress_grid.columnconfigure(0, weight=1)
        progress_grid.columnconfigure(1, weight=1)

        # Global progress
        global_frame = tk.LabelFrame(progress_grid, text="Avancement global",
                                     font=("Arial", 10, "bold"),
                                     bg="white", relief=tk.SUNKEN, bd=2)
        global_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.global_label = tk.Label(global_frame, text="Prêt", font=("Arial", 9), bg="white")
        self.global_label.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.global_bar = ttk.Progressbar(global_frame, mode="determinate", maximum=100)
        self.global_bar.pack(fill=tk.X, padx=8, pady=2)
        self.global_details = tk.Label(global_frame, text="", font=("Arial", 8), bg="white", fg="#666")
        self.global_details.pack(fill=tk.X, padx=8, pady=2)
        self.global_eta = tk.Label(global_frame, text="ETA: --:--", font=("Arial", 8), bg="white", fg="#666")
        self.global_eta.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Left: stages
        self.stages_frame = tk.LabelFrame(progress_grid, text="Étapes",
                                          font=("Arial", 10, "bold"),
                                          bg="white", relief=tk.SUNKEN, bd=2)
        self.stages_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=2)
        self.stages_frame.columnconfigure(0, weight=1)

        # Right: per-file/worker bars + monitoring
        right_frame = tk.LabelFrame(progress_grid, text="Tâches parallèles",
                                    font=("Arial", 10, "bold"),
                                    bg="white", relief=tk.SUNKEN, bd=2)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=2)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)

        tk.Label(right_frame, text="Threads/fichiers actifs (scrollable)",
                 font=("Arial", 8), bg="white", fg="#333").grid(row=1, column=0, sticky="w", padx=8)

        files_container = tk.Frame(right_frame, bg="white")
        files_container.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 6))
        files_container.columnconfigure(0, weight=1)
        files_container.rowconfigure(0, weight=1)

        self.files_canvas = tk.Canvas(files_container, bg="white", highlightthickness=0)
        self.files_canvas.grid(row=0, column=0, sticky="nsew")
        file_scroll = ttk.Scrollbar(files_container, orient="vertical", command=self.files_canvas.yview)
        file_scroll.grid(row=0, column=1, sticky="ns")
        self.files_canvas.configure(yscrollcommand=file_scroll.set)
        self.file_progress_container = tk.Frame(self.files_canvas, bg="white")
        win_file = self.files_canvas.create_window((0, 0), window=self.file_progress_container, anchor="nw")

        def _on_files_config(event):
            self.files_canvas.configure(scrollregion=self.files_canvas.bbox("all"))
        self.file_progress_container.bind("<Configure>", _on_files_config)

        def _on_files_canvas(event):
            try:
                self.files_canvas.itemconfigure(win_file, width=event.width)
            except Exception:
                pass
        self.files_canvas.bind("<Configure>", _on_files_canvas)

        # Monitoring
        monitor_frame = tk.LabelFrame(progress_grid, text="Monitoring & Alertes",
                                      font=("Arial", 10, "bold"),
                                      bg="white", relief=tk.SUNKEN, bd=2)
        monitor_frame.grid(row=2, column=1, sticky="nsew", padx=(6, 0), pady=2)
        monitor_frame.columnconfigure(0, weight=1)
        monitor_frame.rowconfigure(2, weight=1)

        self.ram_label = tk.Label(monitor_frame, text="RAM: --", font=("Arial", 9), bg="white")
        self.ram_label.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        self.ram_progress = ttk.Progressbar(monitor_frame, mode="determinate", maximum=100)
        self.ram_progress.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))

        self.cpu_label = tk.Label(monitor_frame, text="CPU: --", font=("Arial", 9), bg="white")
        self.cpu_label.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 2))
        self.cpu_progress = ttk.Progressbar(monitor_frame, mode="determinate", maximum=100)
        self.cpu_progress.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))

        self.alerts_text = scrolledtext.ScrolledText(monitor_frame, height=6,
                                                     font=("Courier", 8),
                                                     bg="#f8f8f8", fg="#333")
        self.alerts_text.grid(row=4, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.alerts_text.tag_config("error", foreground="#d32f2f", font=("Courier", 8, "bold"))
        self.alerts_text.tag_config("warning", foreground="#f57f17")
        self.alerts_text.tag_config("success", foreground="#388e3c")

        # Logs
        logs_frame = tk.LabelFrame(main, text="Logs détaillés (verbose)",
                                   font=("Arial", 10, "bold"),
                                   bg="white", relief=tk.SUNKEN, bd=2)
        logs_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        logs_frame.rowconfigure(0, weight=1)
        logs_frame.columnconfigure(0, weight=1)
        self.logs_text = scrolledtext.ScrolledText(logs_frame,
                                                   font=("Courier", 9),
                                                   bg="#1e1e1e", fg="#00ff00")
        self.logs_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pct(current, total):
        try:
            if total <= 0:
                return 0
            return max(0, min(100, int((current / total) * 100)))
        except Exception:
            return 0

    def _make_block(self, parent, title):
        frame = tk.Frame(parent, bg="white")
        frame.grid_columnconfigure(0, weight=1)
        label = tk.Label(frame, text=title, font=("Arial", 9, "bold"), bg="white", fg="#2c3e50")
        label.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        bar = ttk.Progressbar(frame, mode="determinate", maximum=100)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        detail = tk.Label(frame, text="", font=("Arial", 8), bg="white", fg="#666")
        detail.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 6))
        return {"frame": frame, "label": label, "bar": bar, "detail": detail}

    # ------------------------------------------------------------------ public API
    def add_stage(self, key, title):
        if key in self.progress_blocks:
            return
        block = self._make_block(self.stages_frame, title)
        row = len(self.progress_blocks)
        block["frame"].grid(row=row, column=0, sticky="ew", padx=6, pady=4)
        self.progress_blocks[key] = block

    def update_stage(self, key, current, total, msg="", eta=None):
        block = self.progress_blocks.get(key)
        if not block:
            return
        pct = self._pct(current, total)
        details = f"{current}/{total}" if total else f"{current}"
        if pct:
            details += f" ({pct}%)"
        if eta is not None:
            details += f" | ETA {_fmt_eta(eta)}"
        def _apply():
            block["label"].config(text=msg or block["label"].cget("text"))
            block["bar"]["value"] = pct
            block["detail"].config(text=details)
        self.root.after(0, _apply)

    def update_global(self, current, total, msg="", eta=None):
        pct = self._pct(current, total)
        details = f"{current}/{total}" if total else f"{current}"
        if pct:
            details += f" ({pct}%)"
        def _apply():
            self.global_bar["value"] = pct
            self.global_label.config(text=msg)
            self.global_details.config(text=details)
            self.global_eta.config(text=f"ETA: {_fmt_eta(eta)}" if eta is not None else "ETA: --:--")
        self.root.after(0, _apply)

    def _ensure_file_widget(self, name):
        if name in self.file_progress_widgets:
            return self.file_progress_widgets[name]
        if len(self.file_progress_order) >= self.max_file_bars:
            oldest = self.file_progress_order.popleft()
            widget = self.file_progress_widgets.pop(oldest, None)
            if widget:
                widget["frame"].destroy()
        frame = tk.Frame(self.file_progress_container, bg="white", bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X, pady=2, padx=2)
        title = tk.Label(frame, text=name, font=("Arial", 8, "bold"), bg="white", anchor="w")
        title.pack(fill=tk.X, padx=6, pady=(4, 0))
        bar = ttk.Progressbar(frame, mode="determinate", maximum=100)
        bar.pack(fill=tk.X, padx=6, pady=2)
        status = tk.Label(frame, text="En attente", font=("Arial", 8), bg="white", fg="#666", anchor="w")
        status.pack(fill=tk.X, padx=6, pady=(0, 4))
        widget = {"frame": frame, "bar": bar, "status": status}
        self.file_progress_widgets[name] = widget
        self.file_progress_order.append(name)
        return widget

    def reset_file_progress(self):
        for widget in self.file_progress_widgets.values():
            try:
                widget["frame"].destroy()
            except Exception:
                pass
        self.file_progress_widgets.clear()
        self.file_progress_order.clear()
        try:
            self.files_canvas.yview_moveto(0)
        except Exception:
            pass

    def update_file_progress(self, name, percent, status):
        def _apply():
            widget = self._ensure_file_widget(name)
            widget["bar"]["value"] = max(0, min(100, percent))
            widget["status"].config(text=status)
        self.root.after(0, _apply)

    def log(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        icons = {"INFO": "[info]", "OK": "[ok]", "ERROR": "[err]", "WARNING": "[warn]", "PROGRESS": "[..]"}
        icon = icons.get(level, ">")
        formatted = f"{icon} [{ts}] {msg}"
        self.logs.append(formatted)
        def _apply():
            self.logs_text.insert(tk.END, formatted + "\n")
            self.logs_text.see(tk.END)
        self.root.after(0, _apply)

    def log_alert(self, msg, level="warning"):
        self.alerts.append(msg)
        def _apply():
            self.alerts_text.insert(tk.END, f"{msg}\n", level)
            self.alerts_text.see(tk.END)
        self.root.after(0, _apply)

    # ------------------------------------------------------------------ monitoring
    def _schedule_stats(self):
        def tick():
            self._update_stats_auto()
            self.root.after(500, self._schedule_stats)
        self.root.after(500, tick)

    def _update_stats_auto(self):
        if not psutil:
            return
        try:
            ram_pct = psutil.virtual_memory().percent
            cpu_pct = psutil.cpu_percent(interval=None)
            self.update_stats_now(cpu_pct, ram_pct)
        except Exception:
            pass

    def update_stats_now(self, cpu=None, ram=None):
        def _apply():
            if ram is not None:
                self.ram_label.config(text=f"RAM: {ram:.1f}%")
                self.ram_progress["value"] = max(0, min(100, ram))
            if cpu is not None:
                self.cpu_label.config(text=f"CPU: {cpu:.1f}%")
                self.cpu_progress["value"] = max(0, min(100, cpu))
        self.root.after(0, _apply)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Petite démo locale
    ui = GenericProgressGUI(title="Demo Progress", header_info="Exemple générique", max_workers=5)
    ui.add_stage("stage1", "Étape 1")
    ui.add_stage("stage2", "Étape 2")

    def demo_worker():
        for i in range(0, 101, 5):
            ui.update_global(i, 100, "Avancement général", eta=200 - 2 * i)
            ui.update_stage("stage1", i, 100, f"Step1 {i}%", eta=150 - i)
            ui.update_stage("stage2", i // 2, 50, f"Step2 {i//2}/50", eta=100 - i)
            ui.update_file_progress(f"tâche-{i%3}", i % 100, f"Progress {i}%")
            ui.log(f"Log {i}", level="INFO")
            time.sleep(0.2)
        ui.log_alert("Démo terminée", level="success")
    threading.Thread(target=demo_worker, daemon=True).start()
    ui.start()
