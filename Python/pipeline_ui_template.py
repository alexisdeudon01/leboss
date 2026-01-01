#!/usr/bin/env python3
"""
Reusable, lightweight GUI template for pipeline-style scripts.

Features:
- Start/Stop buttons with callback binding.
- Status label.
- Overall + stage progress bars (add_stage / set_stage_progress).
- Log area with level tags (INFO/WARN/ERROR/OK/DEBUG).
- Optional detached window (Toplevel) so existing UIs are not disrupted.

Usage:
    ui = PipelineWindowTemplate(title="My Pipeline", detached=True)
    ui.bind_start(callback)  # optional
    ui.bind_stop(callback)   # optional
    ui.add_stage("load", "Chargement")
    ui.set_stage_progress("load", 50)
    ui.log("message", level="INFO")
    ui.set_status("Running")
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Callable


class PipelineWindowTemplate:
    def __init__(
        self,
        *,
        title: str = "Pipeline",
        parent: tk.Misc | None = None,
        detached: bool = True,
        width: int = 480,
        height: int = 540,
    ) -> None:
        self._detached = detached
        if detached:
            self.root = tk.Toplevel(parent) if parent else tk.Tk()
        else:
            # Embed as a frame inside the given parent
            self.root = tk.Frame(parent)
        if not isinstance(self.root, tk.Frame):
            self.root.title(title)
            try:
                self.root.geometry(f"{width}x{height}")
            except Exception:
                pass

        self._start_cb: Callable[[], None] | None = None
        self._stop_cb: Callable[[], None] | None = None
        self._stage_bars: dict[str, ttk.Progressbar] = {}

        container = self.root if isinstance(self.root, tk.Frame) else tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        header = tk.Frame(container, bg="#0f172a", padx=8, pady=6)
        header.pack(fill="x")
        self.title_label = tk.Label(
            header, text=title, fg="#e2e8f0", bg="#0f172a", font=("Arial", 12, "bold")
        )
        self.title_label.pack(side="left")

        btns = tk.Frame(header, bg="#0f172a")
        btns.pack(side="right")
        self.start_btn = tk.Button(btns, text="Start", command=self._on_start, bg="#22c55e", fg="#0f172a")
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(btns, text="Stop", command=self._on_stop, bg="#ef4444", fg="#0f172a")
        self.stop_btn.pack(side="left", padx=4)

        status_frame = tk.Frame(container, bg="#0b1220", pady=4, padx=8)
        status_frame.pack(fill="x")
        tk.Label(status_frame, text="Status:", bg="#0b1220", fg="#e2e8f0").pack(side="left")
        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, bg="#0b1220", fg="#e2e8f0")
        self.status_label.pack(side="left", padx=6)

        # Overall progress bar (always present)
        self.overall_var = tk.DoubleVar(value=0.0)
        overall_frame = tk.Frame(container, bg="#0b1220", pady=6, padx=8)
        overall_frame.pack(fill="x")
        tk.Label(overall_frame, text="Overall", bg="#0b1220", fg="#e2e8f0").pack(anchor="w")
        self.overall_bar = ttk.Progressbar(overall_frame, variable=self.overall_var, maximum=100.0)
        self.overall_bar.pack(fill="x", pady=2)

        stages_frame = tk.Frame(container, bg="#0b1220", pady=4, padx=8)
        stages_frame.pack(fill="x")
        self._stages_frame = stages_frame

        log_frame = tk.Frame(container, bg="#0f172a", padx=6, pady=6)
        log_frame.pack(fill="both", expand=True)
        tk.Label(log_frame, text="Logs", bg="#0f172a", fg="#e2e8f0").pack(anchor="w")
        self.log_widget = scrolledtext.ScrolledText(
            log_frame, height=16, wrap="word", bg="#0b1220", fg="#e2e8f0", insertbackground="#e2e8f0"
        )
        self.log_widget.pack(fill="both", expand=True)

        alert_frame = tk.Frame(container, bg="#0f172a", padx=6, pady=6)
        alert_frame.pack(fill="x")
        tk.Label(alert_frame, text="Alerts", bg="#0f172a", fg="#e2e8f0").pack(anchor="w")
        self.alert_var = tk.StringVar(value="")
        self.alert_label = tk.Label(alert_frame, textvariable=self.alert_var, bg="#0f172a", fg="#facc15")
        self.alert_label.pack(fill="x")

    # -------- public API --------
    def bind_start(self, cb: Callable[[], None]) -> None:
        self._start_cb = cb

    def bind_stop(self, cb: Callable[[], None]) -> None:
        self._stop_cb = cb

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def log(self, message: str, level: str = "INFO") -> None:
        tag = level.upper()
        color_map = {
            "ERROR": "#f87171",
            "WARN": "#fbbf24",
            "OK": "#22c55e",
            "DEBUG": "#94a3b8",
            "INFO": "#e2e8f0",
        }
        color = color_map.get(tag, "#e2e8f0")
        self.log_widget.insert("end", f"[{tag}] {message}\n", tag)
        self.log_widget.tag_config(tag, foreground=color)
        self.log_widget.see("end")

    def add_alert(self, message: str, level: str = "INFO") -> None:
        tag = level.upper()
        if tag == "ERROR":
            color = "#f87171"
        elif tag == "WARN":
            color = "#fbbf24"
        else:
            color = "#22c55e"
        self.alert_var.set(f"[{tag}] {message}")
        self.alert_label.configure(fg=color)

    def set_overall_progress(self, value: float) -> None:
        self.overall_var.set(max(0.0, min(100.0, value)))

    def add_stage(self, key: str, label: str) -> None:
        if key in self._stage_bars:
            return
        row = tk.Frame(self._stages_frame, bg="#0b1220")
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, bg="#0b1220", fg="#e2e8f0").pack(anchor="w")
        var = tk.DoubleVar(value=0.0)
        bar = ttk.Progressbar(row, variable=var, maximum=100.0)
        bar.var = var  # attach for convenience
        bar.pack(fill="x")
        self._stage_bars[key] = bar

    def set_stage_progress(self, key: str, value: float) -> None:
        bar = self._stage_bars.get(key)
        if not bar:
            return
        bar.var.set(max(0.0, min(100.0, value)))

    # -------- internal handlers --------
    def _on_start(self) -> None:
        if not self._start_cb:
            return
        threading.Thread(target=self._start_cb, daemon=True).start()
        self.set_status("Running")

    def _on_stop(self) -> None:
        if not self._stop_cb:
            return
        threading.Thread(target=self._stop_cb, daemon=True).start()
        self.set_status("Stopping...")
