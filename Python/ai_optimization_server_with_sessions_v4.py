#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Optimization Server with inline GUI (single process, no multiprocessing).
Bandit contextuel + garde-fous RAM/CPU, reward basé sur score10.
"""

import queue
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, List, Dict, Tuple

import numpy as np

# Tkinter désactivé (serveur headless)
HAS_TKINTER = False
tk = None
ttk = None


# ============================================================
# Data models
# ============================================================

@dataclass
class ProcessSnapshot:
    pid: Any
    chunk_size: int
    rows_5s: int
    ram_mb: float
    cpu_frac: float  # 0..1


@dataclass
class Metrics:
    timestamp: float
    num_workers: int
    chunk_size: int
    rows_processed: int       # delta depuis la dernière métrique
    ram_percent: float        # 0..100
    cpu_percent: float        # 0..100
    throughput: float         # rows/s (optionnel)
    processes: Optional[List[ProcessSnapshot]] = None


@dataclass
class Recommendation:
    d_workers: int
    chunk_mult: float
    reason: str
    score: float
    timestamp: float


# ============================================================
# Bandit LinUCB sur actions discrètes (d_workers, chunk_mult)
# ============================================================

class LinUCBBandit:
    def __init__(self, feature_dim: int = 5, alpha: float = 1.1):
        self.d = int(feature_dim)
        self.alpha = float(alpha)
        self.actions: List[Tuple[int, float]] = [
            (-2, 0.85), (-1, 0.90), (0, 0.95), (0, 1.00), (0, 1.05),
            (+1, 1.10), (+2, 1.15),
        ]
        self.A = [np.eye(self.d, dtype=np.float64) for _ in self.actions]
        self.b = [np.zeros((self.d, 1), dtype=np.float64) for _ in self.actions]
        self.last_action_idx: Optional[int] = None
        self.last_x: Optional[np.ndarray] = None

    def choose_action(self, x: np.ndarray) -> Tuple[int, float]:
        x = x.reshape(self.d, 1).astype(np.float64)
        best_idx = 0
        best_score = -1e18
        for i in range(len(self.actions)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            ucb_term = self.alpha * np.sqrt(max(float((x.T @ A_inv @ x)[0, 0]), 0.0))
            score = float((theta.T @ x)[0, 0]) + ucb_term
            if score > best_score:
                best_score = score
                best_idx = i
        self.last_action_idx = best_idx
        self.last_x = x
        return best_idx, best_score

    def update(self, reward: float) -> None:
        if self.last_action_idx is None or self.last_x is None:
            return
        i = self.last_action_idx
        x = self.last_x
        r = float(reward)
        self.A[i] = self.A[i] + (x @ x.T)
        self.b[i] = self.b[i] + r * x


# ============================================================
# GUI
# ============================================================

class AIServerGUI:
    def __init__(self):
        if not HAS_TKINTER:
            raise RuntimeError("Tkinter not available")

        self.root = tk.Tk()
        self.root.title("AI Optimization Server")
        self.root.geometry("1120x720")

        self.history: Dict[str, List[Any]] = {
            "t": [], "throughput": [], "workers": [], "chunk_mult": [],
            "score10": [], "reward": [], "ram": [], "cpu": [],
        }
        self._log_lines: List[str] = []

        self._build_ui()

    def update_status(self, msg: str):
        try:
            self.metrics_label.config(text=msg)
        except Exception:
            pass

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x", side="top")
        ttk.Label(header, text="AI Optimization Server", font=("Arial", 14, "bold")).pack(anchor="w")

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        self.metrics_label = ttk.Label(left, text="Waiting for metrics...", justify="left", font=("Courier", 10))
        self.metrics_label.pack(anchor="nw", fill="x", padx=5, pady=5)

        log_frame = ttk.LabelFrame(left, text="Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text = tk.Text(log_frame, height=18, wrap="word", state=tk.DISABLED)
        self.log_text.pack(fill="both", expand=True)

        def make_canvas(parent, title: str):
            fr = ttk.LabelFrame(parent, text=title, padding=5)
            fr.pack(fill="both", expand=True, pady=(0, 6))
            cv = tk.Canvas(fr, bg="#f5f5f5", height=120)
            cv.pack(fill="both", expand=True)
            return cv

        self.canvas_tp = make_canvas(right, "Throughput")
        self.canvas_w = make_canvas(right, "Workers")
        self.canvas_c = make_canvas(right, "Chunk Multiplier")
        self.canvas_s = make_canvas(right, "Score (rows/10s)")

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        self._log_lines = self._log_lines[-300:]
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "\n".join(self._log_lines) + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def add_metric(self, d: Dict[str, Any]):
        if "t" in d:
            self.history["t"].append(float(d["t"]))
            self.history["t"] = self.history["t"][-300:]
        for k in ["throughput", "workers", "chunk_mult", "score10", "reward", "ram", "cpu"]:
            if k in d:
                self.history[k].append(float(d[k]))
                self.history[k] = self.history[k][-300:]

    def update_metrics_display(self, metrics: Metrics, score10: float):
        txt = (
            f"Workers:        {metrics.num_workers}\n"
            f"Chunk size:     {metrics.chunk_size:,}\n"
            f"Throughput:     {metrics.throughput:,.0f} rows/s\n"
            f"Score (10s):    {score10:,.0f} rows/10s\n"
            f"RAM:            {metrics.ram_percent:.1f}%\n"
            f"CPU:            {metrics.cpu_percent:.1f}%\n"
            f"Time:           {datetime.fromtimestamp(metrics.timestamp).strftime('%H:%M:%S')}\n"
        )
        self.metrics_label.config(text=txt)

    def _draw_graph(self, canvas, values, color="#22c55e"):
        if not values or len(values) < 2:
            return
        try:
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 50 or h < 50:
                return
            vals = list(values[-120:])
            vmin = min(vals)
            vmax = max(vals)
            if vmax - vmin < 1e-9:
                vmax = vmin + 1.0
            pts = []
            for i, v in enumerate(vals):
                x = int((i / max(len(vals) - 1, 1)) * (w - 20)) + 10
                y = int(h - 10 - ((v - vmin) / (vmax - vmin)) * (h - 20))
                pts.extend([x, y])
            if len(pts) >= 4:
                canvas.create_line(*pts, fill=color, width=2)
        except Exception:
            pass

    def update_gui(self):
        try:
            if self.history.get("throughput"):
                self._draw_graph(self.canvas_tp, self.history["throughput"], "#22c55e")
            if self.history.get("workers"):
                self._draw_graph(self.canvas_w, self.history["workers"], "#3b82f6")
            if self.history.get("chunk_mult"):
                self._draw_graph(self.canvas_c, self.history["chunk_mult"], "#a855f7")
            if self.history.get("score10"):
                self._draw_graph(self.canvas_s, self.history["score10"], "#f97316")
            self.root.after(500, self.update_gui)
        except Exception:
            pass

    def loop(self):
        if HAS_TKINTER:
            self.update_gui()
            self.root.mainloop()


# ============================================================
# AIOptimizationServer
# ============================================================

class AIOptimizationServer:
    def __init__(
        self,
        max_workers: int = 12,
        *,
        max_chunk_size: int = 750_000,
        min_chunk_size: int = 50_000,
        max_ram_percent: float = 90.0,
        with_gui: bool = False,
    ):
        self.max_workers = int(max_workers)
        self.max_chunk_size = int(max_chunk_size)
        self.min_chunk_size = int(min_chunk_size)
        self.max_ram_percent = float(max_ram_percent)

        self.metrics_queue: "queue.Queue[Metrics]" = queue.Queue(maxsize=20)
        self.recommendation_queue: "queue.Queue[Recommendation]" = queue.Queue(maxsize=5)

        self.bandit = LinUCBBandit(feature_dim=5, alpha=1.1)
        self._stop_event = False

        self._rows_window = deque()  # (t, rows_delta)
        self._prev_score10 = 0.0

        self.gui: Optional[AIServerGUI] = None  # headless only

    def stop(self):
        self._stop_event = True

    def send_metrics(self, metrics: Metrics):
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass

    def get_recommendation(self, timeout: float = 0.0) -> Optional[Recommendation]:
        try:
            return self.recommendation_queue.get(timeout=timeout)
        except Exception:
            return None

    def _compute_reward(self, metrics: Metrics, score10: float) -> float:
        prev = float(self._prev_score10)
        gain = (float(score10) - prev) / max(prev + 1.0, 1.0)

        ram_pen = 0.0
        if metrics.ram_percent >= self.max_ram_percent:
            ram_pen = -2.0
        elif metrics.ram_percent >= (self.max_ram_percent - 3):
            ram_pen = -0.7

        cpu_pen = 0.0
        if metrics.cpu_percent >= 98.0:
            cpu_pen = -1.5
        elif metrics.cpu_percent >= 95.0:
            cpu_pen = -0.8

        self._prev_score10 = float(score10)
        return float(gain + ram_pen + cpu_pen)

    def run(self) -> None:
        print("[AIServer] Starting AI Optimization Server")
        if self.gui:
            self.gui.update_status("Ready - Waiting for metrics...")
        else:
            print("[AIServer] Waiting for metrics...")

        last_action_idx = -1

        while not self._stop_event:
            try:
                metrics = self.metrics_queue.get(timeout=0.5)
            except queue.Empty:
                if self.gui:
                    try:
                        self.gui.root.update_idletasks()
                        self.gui.root.update()
                    except Exception:
                        pass
                continue

            try:
                now_t = float(metrics.timestamp)
                self._rows_window.append((now_t, int(metrics.rows_processed)))
                while self._rows_window and (now_t - self._rows_window[0][0]) > 10.0:
                    self._rows_window.popleft()
                score10 = float(sum(r for _t, r in self._rows_window))

                x = np.array([
                    min(metrics.ram_percent / 100.0, 1.0),
                    min(metrics.cpu_percent / 100.0, 1.0),
                    min(score10 / 500_000.0, 1.0),
                    metrics.num_workers / float(max(self.max_workers, 1)),
                    metrics.chunk_size / float(max(self.max_chunk_size, 1)),
                ], dtype=np.float64)

                if last_action_idx >= 0:
                    reward = self._compute_reward(metrics, score10)
                    self.bandit.update(reward)
                else:
                    reward = 0.0

                if metrics.ram_percent >= self.max_ram_percent:
                    d_workers = -(metrics.num_workers - 1)
                    chunk_mult = 0.80
                    reason = f"GUARD: RAM {metrics.ram_percent:.1f}%"
                elif metrics.ram_percent >= (self.max_ram_percent - 3):
                    d_workers = -1
                    chunk_mult = 0.95
                    reason = f"NEAR: RAM {metrics.ram_percent:.1f}%"
                elif metrics.cpu_percent >= 98.0:
                    d_workers = -1
                    chunk_mult = 0.90
                    reason = f"GUARD: CPU {metrics.cpu_percent:.1f}%"
                else:
                    action_idx, _ucb = self.bandit.choose_action(x)
                    last_action_idx = action_idx
                    d_workers, chunk_mult = self.bandit.actions[action_idx]
                    reason = f"AI: dw={d_workers:+d} chunk×={chunk_mult:.2f}"

                new_workers = max(1, min(self.max_workers, metrics.num_workers + d_workers))
                chunk_mult = float(max(0.60, min(1.40, chunk_mult)))

                rec = Recommendation(
                    d_workers=int(new_workers - metrics.num_workers),
                    chunk_mult=chunk_mult,
                    reason=reason,
                    score=float(reward),
                    timestamp=metrics.timestamp,
                )
                try:
                    self.recommendation_queue.put_nowait(rec)
                except queue.Full:
                    pass

                if self.gui:
                    self.gui.add_metric({
                        "t": metrics.timestamp,
                        "throughput": metrics.throughput,
                        "workers": metrics.num_workers,
                        "chunk_mult": chunk_mult,
                        "reward": reward,
                        "ram": metrics.ram_percent,
                        "cpu": metrics.cpu_percent,
                    })
                    self.gui.update_metrics_display(metrics, score10)
                    self.gui.log(f"W:{metrics.num_workers}→{new_workers} chunk×:{chunk_mult:.2f} TP:{metrics.throughput:,.0f} R:{reward:+.3f} {reason}")
                    try:
                        self.gui.root.update_idletasks()
                        self.gui.root.update()
                    except Exception:
                        pass
                else:
                    ts = datetime.fromtimestamp(metrics.timestamp).strftime("%H:%M:%S")
                    print(f"[AI {ts}] W:{metrics.num_workers}→{new_workers} | score10:{score10:,.0f} | RAM:{metrics.ram_percent:.1f}% CPU:{metrics.cpu_percent:.1f}% | R:{reward:+.3f} | {reason}")

            except Exception as e:
                print(f"[AIServer] Error: {e}")
                time.sleep(0.1)

        print("[AIServer] Stopped")


if __name__ == "__main__":
    srv = AIOptimizationServer(max_workers=12, with_gui=True)
    try:
        srv.run()
    except KeyboardInterrupt:
        srv.stop()
