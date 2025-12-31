"""
AI Optimization Server
======================

Standalone class that runs in a separate thread:
- Receives metrics every 5 seconds via queue
- Trains LinUCB bandit model
- Returns recommended actions
- Displays monitoring GUI with real-time graphs

Communication:
  Program → metrics_in_queue → AIOptimizationServer
  AIOptimizationServer → actions_out_queue → Program
"""

import queue
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


# ============================================================
# Data Models
# ============================================================

@dataclass
class Metrics:
    """Metrics from the program every 5 seconds"""
    timestamp: float
    num_workers: int
    chunk_size: int
    rows_processed: int  # In the last 5 seconds
    ram_percent: float
    cpu_percent: float
    throughput: float  # rows/s
    
    def to_features(self, max_workers: int = 12) -> np.ndarray:
        """Convert to normalized feature vector for LinUCB"""
        return np.array([
            min(self.ram_percent / 100.0, 1.0),
            min(self.cpu_percent / 100.0, 1.0),
            min(self.throughput / 50_000.0, 1.0),  # normalized by typical throughput
            self.num_workers / float(max_workers),
            self.chunk_size / 750_000.0,
        ], dtype=np.float64)


@dataclass
class Recommendation:
    """Action recommendation from AI"""
    d_workers: int  # +/- change to worker count
    chunk_mult: float  # multiplier for chunk size
    reason: str
    score: float  # confidence/score of this action
    timestamp: float


# ============================================================
# LinUCB Bandit
# ============================================================

class LinUCBBandit:
    """Contextual bandit for online learning"""
    
    def __init__(self, feature_dim: int = 5, alpha: float = 1.2):
        self.d = feature_dim
        self.alpha = alpha
        
        # Actions: (d_workers, chunk_mult)
        self.actions = [
            (-2, 0.80),
            (-1, 0.90),
            (0, 1.00),
            (+1, 1.10),
            (+2, 1.25),
        ]
        
        # LinUCB state per action
        self.A = [np.eye(self.d, dtype=np.float64) for _ in self.actions]
        self.b = [np.zeros((self.d, 1), dtype=np.float64) for _ in self.actions]
        
        self.last_action_idx: Optional[int] = None
        self.last_x: Optional[np.ndarray] = None
        self.t = 0
        self.cum_reward = 0.0
    
    def choose_action(self, x: np.ndarray) -> tuple[int, float]:
        """
        Choose best action given features x.
        Returns: (action_index, confidence_score)
        """
        x = x.reshape(self.d, 1).astype(np.float64)
        best_idx = 0
        best_score = -1e18
        
        for i in range(len(self.actions)):
            try:
                A_inv = np.linalg.inv(self.A[i] + 1e-6 * np.eye(self.d))
                theta = A_inv @ self.b[i]
                
                # UCB = exploitation + exploration
                exploit = float((theta.T @ x)[0, 0])
                explore = self.alpha * math.sqrt(max(float((x.T @ A_inv @ x)[0, 0]), 0.0))
                score = exploit + explore
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            except Exception:
                pass
        
        self.last_action_idx = best_idx
        self.last_x = x
        return best_idx, best_score
    
    def update(self, reward: float) -> None:
        """Update with observed reward"""
        if self.last_action_idx is None or self.last_x is None:
            return
        
        i = self.last_action_idx
        x = self.last_x
        
        self.A[i] = self.A[i] + (x @ x.T)
        self.b[i] = self.b[i] + float(reward) * x
        
        self.t += 1
        self.cum_reward += float(reward)


# ============================================================
# AI Optimization Server
# ============================================================

class AIOptimizationServer:
    """
    Runs in separate thread. Receives metrics, outputs recommendations.
    
    Usage:
        server = AIOptimizationServer(max_workers=12, max_chunk_size=750_000)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        
        # Send metrics
        server.send_metrics(Metrics(...))
        
        # Get recommendation
        rec = server.get_recommendation(timeout=1.0)
        if rec:
            print(f"Recommended: Δworkers={rec.d_workers}, chunk×={rec.chunk_mult}")
    """
    
    def __init__(
        self,
        max_workers: int = 12,
        max_chunk_size: int = 750_000,
        min_chunk_size: int = 50_000,
        max_ram_percent: float = 90.0,
        show_gui: bool = True,
    ):
        self.max_workers = max_workers
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_ram_percent = max_ram_percent
        self.show_gui = show_gui
        
        # Queues for communication
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=2)
        self.recommendation_queue: queue.Queue = queue.Queue(maxsize=1)
        
        # AI model
        self.bandit = LinUCBBandit(feature_dim=5, alpha=1.1)
        
        # State
        self.current_metrics: Optional[Metrics] = None
        self.last_recommendation: Optional[Recommendation] = None
        self.history = {
            "t": [],
            "throughput": [],
            "workers": [],
            "chunk_mult": [],
            "reward": [],
            "ram": [],
            "cpu": [],
        }
        
        self._stop_event = threading.Event()
        self._prev_reward = 0.0
        self._prev_throughput = 0.0
        
        # GUI
        self.root: Optional[tk.Tk] = None
        self.monitor_window: Optional[tk.Toplevel] = None
        
    def send_metrics(self, metrics: Metrics) -> None:
        """Send metrics to the server (from main program)"""
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass  # Drop if queue is full (shouldn't happen with maxsize=2)
    
    def get_recommendation(self, timeout: float = 0.5) -> Optional[Recommendation]:
        """Get recommendation from server (from main program)"""
        try:
            return self.recommendation_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self) -> None:
        """Signal server to stop"""
        self._stop_event.set()
    
    def run(self) -> None:
        """
        Main server loop (runs in separate thread).
        
        Every 5 seconds:
          1. Receive metrics
          2. Compute features
          3. Choose action with LinUCB
          4. Compute reward
          5. Update model
          6. Send recommendation
          7. Update GUI
        """
        # Optional: create GUI
        if self.show_gui:
            self._setup_gui()
        
        last_action_idx = -1
        
        while not self._stop_event.is_set():
            try:
                # Wait for metrics with timeout
                try:
                    metrics = self.metrics_queue.get(timeout=1.0)
                except queue.Empty:
                    # No metrics received, keep waiting
                    continue
                
                self.current_metrics = metrics
                
                # Compute features
                x = metrics.to_features(max_workers=self.max_workers)
                
                # Choose action
                action_idx, score = self.bandit.choose_action(x)
                action = self.bandit.actions[action_idx]
                
                # Update previous action with reward
                if last_action_idx >= 0:
                    reward = self._compute_reward(metrics)
                    self.bandit.update(reward)
                    self.history["reward"].append(reward)
                else:
                    reward = 0.0
                
                last_action_idx = action_idx
                
                # Guard rails: check physical constraints
                d_workers, chunk_mult = action
                new_workers = max(1, min(self.max_workers, metrics.num_workers + d_workers))
                new_chunk = max(self.min_chunk_size, min(self.max_chunk_size, int(metrics.chunk_size * chunk_mult)))
                
                # RAM protection
                if metrics.ram_percent >= self.max_ram_percent:
                    new_workers = 1
                    chunk_mult = 0.75
                    reason = f"GUARD: RAM {metrics.ram_percent:.1f}% >= {self.max_ram_percent}% → cap=1, chunk×=0.75"
                elif metrics.ram_percent >= (self.max_ram_percent - 3):
                    new_workers = min(new_workers, 2)
                    chunk_mult = min(chunk_mult, 0.95)
                    reason = f"GUARD: RAM {metrics.ram_percent:.1f}% near ceiling → cap≤2, chunk×≤0.95"
                else:
                    reason = f"AI: Δworkers={d_workers:+d}, chunk×={chunk_mult:.2f} | reward={reward:+.3f}"
                
                # Create recommendation
                rec = Recommendation(
                    d_workers=new_workers - metrics.num_workers,
                    chunk_mult=chunk_mult,
                    reason=reason,
                    score=float(score),
                    timestamp=metrics.timestamp,
                )
                self.last_recommendation = rec
                
                # Send to main program
                try:
                    self.recommendation_queue.put_nowait(rec)
                except queue.Full:
                    pass
                
                # Update history
                self.history["t"].append(metrics.timestamp)
                self.history["throughput"].append(metrics.throughput)
                self.history["workers"].append(new_workers)
                self.history["chunk_mult"].append(chunk_mult)
                self.history["ram"].append(metrics.ram_percent)
                self.history["cpu"].append(metrics.cpu_percent)
                
                # Keep only last 240 points (~20 minutes at 5s interval)
                for key in self.history:
                    if len(self.history[key]) > 240:
                        self.history[key] = self.history[key][-240:]
                
                # Update GUI if running
                if self.root and self.monitor_window:
                    self.root.after(0, self._update_gui)
                
            except Exception as e:
                print(f"[AIServer] Error: {e}")
                time.sleep(0.5)
    
    def _compute_reward(self, metrics: Metrics) -> float:
        """Compute reward based on metrics"""
        # Reward: increase in throughput, penalty for high RAM
        throughput_gain = (metrics.throughput - self._prev_throughput) / max(self._prev_throughput + 1.0, 1.0)
        ram_penalty = 0.0
        if metrics.ram_percent >= self.max_ram_percent:
            ram_penalty = -2.0
        elif metrics.ram_percent >= (self.max_ram_percent - 3):
            ram_penalty = -0.7
        
        reward = throughput_gain + ram_penalty
        self._prev_throughput = metrics.throughput
        return float(reward)
    
    def _setup_gui(self) -> None:
        """Create monitoring GUI"""
        self.root = tk.Tk()
        self.root.title("AI Optimizer - Real-time Monitor")
        self.root.geometry("900x700")
        
        # Top panel: current metrics
        top = ttk.LabelFrame(self.root, text="Current Metrics", padding=10)
        top.pack(fill="x", padx=10, pady=10)
        
        self.label_metrics = ttk.Label(top, text="Waiting for metrics...", font=("Courier", 9))
        self.label_metrics.pack(anchor="w")
        
        self.label_rec = ttk.Label(top, text="Waiting for recommendation...", font=("Courier", 9), foreground="blue")
        self.label_rec.pack(anchor="w", pady=(5, 0))
        
        # Main frame: two plots side by side
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: throughput graph
        left = ttk.LabelFrame(main, text="Throughput (rows/s)", padding=5)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.canvas_throughput = tk.Canvas(left, bg="#f0f0f0", highlightthickness=1, highlightbackground="#ccc")
        self.canvas_throughput.pack(fill="both", expand=True)
        
        # Right: parameters
        right = ttk.LabelFrame(main, text="Parameters", padding=5)
        right.pack(side="right", fill="both", expand=True)
        
        right1 = ttk.LabelFrame(right, text="Workers", padding=5)
        right1.pack(fill="x", pady=(0, 5))
        self.canvas_workers = tk.Canvas(right1, height=120, bg="#f0f0f0", highlightthickness=1, highlightbackground="#ccc")
        self.canvas_workers.pack(fill="both", expand=True)
        
        right2 = ttk.LabelFrame(right, text="Chunk Multiplier", padding=5)
        right2.pack(fill="x", pady=(0, 5))
        self.canvas_chunk = tk.Canvas(right2, height=120, bg="#f0f0f0", highlightthickness=1, highlightbackground="#ccc")
        self.canvas_chunk.pack(fill="both", expand=True)
        
        right3 = ttk.LabelFrame(right, text="RAM / CPU", padding=5)
        right3.pack(fill="x")
        self.canvas_resources = tk.Canvas(right3, height=120, bg="#f0f0f0", highlightthickness=1, highlightbackground="#ccc")
        self.canvas_resources.pack(fill="both", expand=True)
        
        # Bottom: log
        bottom = ttk.LabelFrame(self.root, text="Activity Log", padding=5, height=80)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        
        self.text_log = tk.Text(bottom, height=5, font=("Courier", 8), bg="#1e1e1e", fg="#00ff00")
        self.text_log.pack(fill="both", expand=True)
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
    
    def _update_gui(self) -> None:
        """Update GUI with current data"""
        if not self.root or not self.monitor_window:
            return
        
        # Update metrics label
        if self.current_metrics:
            m = self.current_metrics
            text = (
                f"[{datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S')}] "
                f"Workers={m.num_workers} | Chunk={m.chunk_size:,} | "
                f"Rows/s={m.throughput:,.0f} | RAM={m.ram_percent:.1f}% | CPU={m.cpu_percent:.1f}%"
            )
            self.label_metrics.config(text=text)
        
        if self.last_recommendation:
            rec = self.last_recommendation
            text = (
                f"→ Δworkers={rec.d_workers:+d} | chunk×={rec.chunk_mult:.2f} | "
                f"{rec.reason}"
            )
            self.label_rec.config(text=text)
        
        # Draw graphs
        self._draw_graphs()
        
        # Update log
        self._update_log()
    
    def _draw_graphs(self) -> None:
        """Draw time series graphs"""
        if not self.history["t"] or len(self.history["t"]) < 2:
            return
        
        # Helper: draw line graph
        def draw_timeseries(canvas, values, title_suffix="", color="#2563eb", y_min=None, y_max=None):
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w <= 1 or h <= 1:
                return
            
            # Margins
            mx_l, mx_r, my_t, my_b = 40, 10, 20, 30
            gx0, gx1 = mx_l, w - mx_r
            gy0, gy1 = my_t, h - my_b
            
            if not values or len(values) < 1:
                canvas.create_text(w/2, h/2, text="(no data)", fill="#999")
                return
            
            v_min = y_min if y_min is not None else min(values)
            v_max = y_max if y_max is not None else max(values)
            if v_max - v_min < 1e-6:
                v_max = v_min + 1.0
            
            # Plot
            points = []
            for i, v in enumerate(values):
                x = gx0 + (i / max(len(values) - 1, 1)) * (gx1 - gx0)
                y = gy1 - ((v - v_min) / (v_max - v_min)) * (gy1 - gy0)
                points.extend([x, y])
            
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2)
            
            # Axes
            canvas.create_line(gx0, gy1, gx1, gy1, fill="#333")
            canvas.create_line(gx0, gy0, gx0, gy1, fill="#333")
            canvas.create_text(gx0 - 5, gy1 + 10, anchor="e", text="0", font=("Arial", 8))
            canvas.create_text(gx0 - 5, gy0 - 5, anchor="e", text=f"{v_max:.0f}", font=("Arial", 8))
            canvas.create_text(gx1, gy1 + 10, anchor="e", text=f"Now", font=("Arial", 8))
            
            # Current value marker
            if points:
                last_x, last_y = points[-2], points[-1]
                canvas.create_oval(last_x - 3, last_y - 3, last_x + 3, last_y + 3, fill=color)
                canvas.create_text(last_x + 5, last_y, anchor="w", text=f"{values[-1]:.0f}", font=("Arial", 8, "bold"), fill=color)
        
        # Draw throughput
        draw_timeseries(
            self.canvas_throughput,
            self.history["throughput"],
            color="#22c55e",
            y_min=0,
            y_max=None,
        )
        
        # Draw workers
        draw_timeseries(
            self.canvas_workers,
            self.history["workers"],
            color="#38bdf8",
            y_min=1,
            y_max=self.max_workers,
        )
        
        # Draw chunk mult
        draw_timeseries(
            self.canvas_chunk,
            self.history["chunk_mult"],
            color="#a78bfa",
            y_min=0.6,
            y_max=1.4,
        )
        
        # Draw RAM/CPU
        canvas = self.canvas_resources
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        
        mx_l, mx_r, my_t, my_b = 40, 10, 20, 30
        gx0, gx1 = mx_l, w - mx_r
        gy0, gy1 = my_t, h - my_b
        
        # RAM as line
        if self.history["ram"]:
            points_ram = []
            for i, v in enumerate(self.history["ram"]):
                x = gx0 + (i / max(len(self.history["ram"]) - 1, 1)) * (gx1 - gx0)
                y = gy1 - (v / 100.0) * (gy1 - gy0)
                points_ram.extend([x, y])
            if len(points_ram) >= 4:
                canvas.create_line(*points_ram, fill="#ef4444", width=2)
        
        # CPU as dashed line
        if self.history["cpu"]:
            points_cpu = []
            for i, v in enumerate(self.history["cpu"]):
                x = gx0 + (i / max(len(self.history["cpu"]) - 1, 1)) * (gx1 - gx0)
                y = gy1 - (v / 100.0) * (gy1 - gy0)
                points_cpu.extend([x, y])
            if len(points_cpu) >= 4:
                canvas.create_line(*points_cpu, fill="#f59e0b", width=2, dash=(4, 2))
        
        # Axes + labels
        canvas.create_line(gx0, gy1, gx1, gy1, fill="#333")
        canvas.create_line(gx0, gy0, gx0, gy1, fill="#333")
        canvas.create_text(gx0 - 5, gy1 + 10, anchor="e", text="0%", font=("Arial", 8))
        canvas.create_text(gx0 - 5, gy0 - 5, anchor="e", text="100%", font=("Arial", 8))
        
        # Legend
        canvas.create_line(gx0 + 5, gy0 + 10, gx0 + 20, gy0 + 10, fill="#ef4444", width=2)
        canvas.create_text(gx0 + 25, gy0 + 10, anchor="w", text="RAM", font=("Arial", 8), fill="#ef4444")
        canvas.create_line(gx0 + 80, gy0 + 10, gx0 + 95, gy0 + 10, fill="#f59e0b", width=2, dash=(4, 2))
        canvas.create_text(gx0 + 100, gy0 + 10, anchor="w", text="CPU", font=("Arial", 8), fill="#f59e0b")
    
    def _update_log(self) -> None:
        """Add entry to activity log"""
        if self.last_recommendation:
            rec = self.last_recommendation
            ts = datetime.fromtimestamp(rec.timestamp).strftime("%H:%M:%S")
            msg = f"[{ts}] Δworkers={rec.d_workers:+d} chunk×={rec.chunk_mult:.2f} | {rec.reason}\n"
            
            self.text_log.insert("end", msg)
            self.text_log.see("end")
            
            # Keep max 100 lines
            lines = int(self.text_log.index("end-1c").split(".")[0])
            if lines > 100:
                self.text_log.delete("1.0", "2.0")