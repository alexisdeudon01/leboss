#!/usr/bin/env python3
"""
Consolidation-style shell UI (no pipeline logic inside).
Layout mirrors consolidation_gui_fixed_v6: header inputs/controls, alerts, monitoring (RAM/CPU/status),
overall bar + stage bars, thread bars, and verbose logs.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Iterable


class ConsolidationStyleShell:
    def __init__(
        self,
        *,
        title: str = "Pipeline",
        stages: Iterable[tuple[str, str]] = (),
        thread_slots: int = 6,
    ) -> None:
        self.root = tk.Toplevel()
        self.root.title(title)
        try:
            self.root.geometry("1500x950")
        except Exception:
            pass

        self.stages = list(stages)
        self.thread_slots = int(thread_slots)

        # Header
        header = tk.Frame(self.root, bg="#eef2f7", pady=6, padx=8)
        header.pack(fill="x")

        input_frame = tk.LabelFrame(header, text="Input", bg="#eef2f7")
        input_frame.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(input_frame, text="Input A:", bg="#eef2f7").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        tk.Label(input_frame, text="(select)", bg="#eef2f7").grid(row=0, column=1, sticky="w", padx=4, pady=2)
        tk.Label(input_frame, text="Input B:", bg="#eef2f7").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        tk.Label(input_frame, text="(select)", bg="#eef2f7").grid(row=1, column=1, sticky="w", padx=4, pady=2)

        controls = tk.LabelFrame(header, text="Controls", bg="#eef2f7")
        controls.pack(side="right", padx=4)
        self.start_btn = tk.Button(controls, text="Start / Resume")
        self.start_btn.grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        self.stop_btn = tk.Button(controls, text="Stop", state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        self.reset_btn = tk.Button(controls, text="Reset state")
        self.reset_btn.grid(row=1, column=0, columnspan=2, padx=4, pady=2, sticky="ew")

        # Alerts
        alerts_frame = tk.LabelFrame(self.root, text="Alerts / Errors", bg="#f8fafc")
        alerts_frame.pack(fill="x", padx=6, pady=4)
        self.alerts_text = scrolledtext.ScrolledText(
            alerts_frame, height=4, bg="#0f172a", fg="#e2e8f0", insertbackground="#e2e8f0"
        )
        self.alerts_text.pack(fill="x")

        # Monitoring + AI info
        monitor = tk.LabelFrame(self.root, text="Monitoring + Progress", bg="#f8fafc")
        monitor.pack(fill="x", padx=6, pady=4)
        top_row = tk.Frame(monitor, bg="#f8fafc")
        top_row.pack(fill="x", padx=4, pady=4)

        self.ram_var = tk.StringVar(value="RAM: 0%")
        self.cpu_var = tk.StringVar(value="CPU: 0%")
        self.rows_var = tk.StringVar(value="Rows seen: 0")
        self.files_var = tk.StringVar(value="Files done: 0")
        self.rowsps_var = tk.StringVar(value="Rows/s: 0")
        self.status_var = tk.StringVar(value="Status: Idle")
        self.ai_rec_var = tk.StringVar(value="AI: --")
        self.ai_score_var = tk.StringVar(value="Best score: --")
        self.task_var = tk.StringVar(value="Tasks: --")

        tk.Label(top_row, textvariable=self.task_var, bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 10, "bold")).pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.ram_var, bg="#f8fafc").pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.cpu_var, bg="#f8fafc").pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.rows_var, bg="#f8fafc").pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.files_var, bg="#f8fafc").pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.rowsps_var, bg="#f8fafc").pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.status_var, bg="#f8fafc", font=("Segoe UI", 10, "bold")).pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.ai_rec_var, bg="#f8fafc", fg="#0ea5e9").pack(side="right", padx=8)
        tk.Label(top_row, textvariable=self.ai_score_var, bg="#f8fafc", fg="#16a34a").pack(side="right", padx=8)

        # Overall + stages
        bars_frame = tk.Frame(monitor, bg="#f8fafc")
        bars_frame.pack(fill="x", padx=4, pady=2)
        tk.Label(bars_frame, text="Overall progress:", bg="#f8fafc").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.overall_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(bars_frame, variable=self.overall_var, maximum=100.0).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        bars_frame.columnconfigure(1, weight=1)

        self.stage_vars: dict[str, tk.DoubleVar] = {}
        self.stage_eta_vars: dict[str, tk.StringVar] = {}
        r = 1
        for key, label in self.stages:
            tk.Label(bars_frame, text=f"{label}:", bg="#f8fafc").grid(row=r, column=0, sticky="w", padx=4, pady=2)
            var = tk.DoubleVar(value=0.0)
            ttk.Progressbar(bars_frame, variable=var, maximum=100.0).grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            eta_var = tk.StringVar(value="ETA: --")
            tk.Label(bars_frame, textvariable=eta_var, bg="#f8fafc", fg="#6b7280").grid(row=r, column=2, sticky="w", padx=4)
            self.stage_vars[key] = var
            self.stage_eta_vars[key] = eta_var
            r += 1

        # Thread progress
        threads_frame = tk.LabelFrame(self.root, text="Thread progress (never empty)", bg="#f8fafc")
        threads_frame.pack(fill="x", padx=6, pady=4)
        self.thread_vars: dict[int, tk.DoubleVar] = {}
        self.thread_labels: dict[int, tk.Label] = {}
        for tid in range(self.thread_slots):
            row = tk.Frame(threads_frame, bg="#f8fafc")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"T{tid}", width=4, bg="#f8fafc").pack(side="left")
            var = tk.DoubleVar(value=0.0)
            ttk.Progressbar(row, variable=var, maximum=100.0).pack(side="left", fill="x", expand=True, padx=4)
            lbl = tk.Label(row, text="Idle", width=40, anchor="w", bg="#f8fafc")
            lbl.pack(side="left", padx=4)
            self.thread_vars[tid] = var
            self.thread_labels[tid] = lbl

        # Logs
        logs_frame = tk.LabelFrame(self.root, text="Logs (VERY verbose)", bg="#0f172a", fg="#e2e8f0")
        logs_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self.log_text = scrolledtext.ScrolledText(
            logs_frame, height=12, bg="#0b1220", fg="#e2e8f0", insertbackground="#e2e8f0"
        )
        self.log_text.pack(fill="both", expand=True)

    # Status/metrics
    def set_status(self, text: str) -> None:
        self.status_var.set(f"Status: {text}")

    def set_metrics(self, ram: float, cpu: float, rows_seen: int, files_done: int, rows_per_s: float) -> None:
        self.ram_var.set(f"RAM: {ram:.1f}%")
        self.cpu_var.set(f"CPU: {cpu:.1f}%")
        self.rows_var.set(f"Rows seen: {rows_seen}")
        self.files_var.set(f"Files done: {files_done}")
        self.rowsps_var.set(f"Rows/s: {rows_per_s:.1f}")

    def set_tasks(self, text: str) -> None:
        self.task_var.set(f"Tasks: {text}")

    # Progress
    def set_overall(self, value: float) -> None:
        self.overall_var.set(max(0.0, min(100.0, value)))

    def set_overall_progress(self, value: float) -> None:
        self.set_overall(value)

    def set_stage(self, key: str, value: float) -> None:
        if key in self.stage_vars:
            self.stage_vars[key].set(max(0.0, min(100.0, value)))

    def set_stage_progress(self, key: str, value: float) -> None:
        self.set_stage(key, value)

    def set_stage_eta(self, key: str, eta_text: str) -> None:
        if key in self.stage_eta_vars:
            self.stage_eta_vars[key].set(eta_text)

    def update_thread(self, tid: int, value: float, msg: str | None = None) -> None:
        if tid in self.thread_vars:
            self.thread_vars[tid].set(max(0.0, min(100.0, value)))
        if msg and tid in self.thread_labels:
            self.thread_labels[tid].configure(text=msg)

    # Alerts/logs
    def add_alert(self, message: str, level: str = "INFO") -> None:
        tag = level.upper()
        self.alerts_text.insert("end", f"[{tag}] {message}\n")
        self.alerts_text.see("end")

    def log(self, message: str, level: str = "INFO") -> None:
        tag = level.upper()
        colors = {
            "ERROR": "#f87171",
            "WARN": "#fbbf24",
            "OK": "#22c55e",
            "DEBUG": "#94a3b8",
            "INFO": "#e2e8f0",
        }
        color = colors.get(tag, "#e2e8f0")
        self.log_text.insert("end", f"[{tag}] {message}\n", tag)
        self.log_text.tag_config(tag, foreground=color)
        self.log_text.see("end")

    # AI info
    def set_ai_recommendation(self, text: str) -> None:
        self.ai_rec_var.set(f"AI: {text}")

    def set_best_score(self, text: str) -> None:
        self.ai_score_var.set(f"Best score: {text}")
