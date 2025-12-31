#!/usr/bin/env python3
"""
AI Optimization Server - STANDALONE with GRAPHICAL UI
Runs in its own process with Tkinter graphs
"""

import queue
import threading
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import math
import sys

# Try to import tkinter
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("[AIServer] ⚠️  Tkinter not available, running in console mode")


@dataclass
class Metrics:
    timestamp: float
    num_workers: int
    chunk_size: int
    rows_processed: int
    ram_percent: float
    cpu_percent: float
    throughput: float


@dataclass
class Recommendation:
    d_workers: int
    chunk_mult: float
    reason: str
    score: float
    timestamp: float


class LinUCBBandit:
    def __init__(self, feature_dim: int = 5, alpha: float = 1.2):
        self.d = feature_dim
        self.alpha = alpha
        self.actions = [
            (-2, 0.80), (-1, 0.90), (0, 1.00), (+1, 1.10), (+2, 1.25),
        ]
        self.A = [np.eye(self.d, dtype=np.float64) for _ in self.actions]
        self.b = [np.zeros((self.d, 1), dtype=np.float64) for _ in self.actions]
        self.last_action_idx = None
        self.last_x = None
        self.t = 0
        self.cum_reward = 0.0
    
    def choose_action(self, x: np.ndarray) -> tuple[int, float]:
        x = x.reshape(self.d, 1).astype(np.float64)
        best_idx = 0
        best_score = -1e18
        
        for i in range(len(self.actions)):
            try:
                A_inv = np.linalg.inv(self.A[i])
                theta = A_inv @ self.b[i]
                ucb_term = self.alpha * math.sqrt(max(float((x.T @ A_inv @ x)[0, 0]), 0.0))
                score = float((theta.T @ x)[0, 0]) + ucb_term
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            except Exception:
                pass
        
        self.last_action_idx = best_idx
        self.last_x = x
        return best_idx, best_score
    
    def update(self, reward: float) -> None:
        if self.last_action_idx is None or self.last_x is None:
            return
        
        i = self.last_action_idx
        x = self.last_x
        self.A[i] = self.A[i] + (x @ x.T)
        self.b[i] = self.b[i] + float(reward) * x
        self.t += 1
        self.cum_reward += float(reward)


class AIServerGUI:
    """Tkinter GUI for AI Server"""
    
    def __init__(self):
        if not HAS_TKINTER:
            return
        
        self.root = tk.Tk()
        self.root.title("AI Optimization Server - Real-time Monitor")
        self.root.geometry("1100x750")
        
        # Data
        self.history = {
            "t": [], "throughput": [], "workers": [], "chunk_mult": [],
            "reward": [], "ram": [], "cpu": []
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI"""
        
        # Header
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x", side="top")
        ttk.Label(header, text="AI Optimization Server", 
                  font=("Arial", 14, "bold")).pack(anchor="w")
        
        # Main paned window
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        # LEFT SIDE: Status & Metrics
        left = ttk.LabelFrame(paned, text="Status & Metrics", padding=10)
        paned.add(left, weight=1)
        
        self.status_var = tk.StringVar(value="Waiting for metrics...")
        ttk.Label(left, textvariable=self.status_var, 
                  font=("Courier", 9)).pack(anchor="w", pady=5)
        
        # Metrics table
        metrics_frame = ttk.LabelFrame(left, text="Current", padding=5)
        metrics_frame.pack(fill="x", pady=5)
        
        self.metrics_text = tk.Text(metrics_frame, height=12, font=("Courier", 8))
        self.metrics_text.pack(fill="both")
        
        # RIGHT SIDE: Graphs
        right = ttk.LabelFrame(paned, text="Real-time Graphs", padding=10)
        paned.add(right, weight=1)
        
        # Canvas for throughput
        graph_frame = ttk.LabelFrame(right, text="Throughput (rows/s)", padding=5)
        graph_frame.pack(fill="both", expand=True, pady=(0, 5))
        self.canvas_tp = tk.Canvas(graph_frame, bg="#f5f5f5", height=100)
        self.canvas_tp.pack(fill="both", expand=True)
        
        # Canvas for workers
        graph_frame2 = ttk.LabelFrame(right, text="Workers", padding=5)
        graph_frame2.pack(fill="both", expand=True, pady=(0, 5))
        self.canvas_w = tk.Canvas(graph_frame2, bg="#f5f5f5", height=100)
        self.canvas_w.pack(fill="both", expand=True)
        
        # Canvas for chunk
        graph_frame3 = ttk.LabelFrame(right, text="Chunk Multiplier", padding=5)
        graph_frame3.pack(fill="both", expand=True)
        self.canvas_c = tk.Canvas(graph_frame3, bg="#f5f5f5", height=100)
        self.canvas_c.pack(fill="both", expand=True)
        
        # Bottom: Log
        log_frame = ttk.LabelFrame(self.root, text="Activity Log", padding=5)
        log_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=4, font=("Courier", 8), bg="#0f0f0f", fg="#00ff00")
        self.log_text.pack(fill="both")
        
        # Start update loop
        self.running = True
        self.update_gui()
    
    def update_status(self, msg):
        if HAS_TKINTER:
            self.status_var.set(msg)
    
    def add_metric(self, data):
        """Add metric data"""
        for key, val in data.items():
            if key in self.history:
                self.history[key].append(val)
                if len(self.history[key]) > 100:
                    self.history[key] = self.history[key][-100:]
    
    def log(self, msg):
        """Log message"""
        if not HAS_TKINTER:
            return
        
        self.log_text.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        if self.log_text.index(tk.END).split(".")[0] > "10":
            self.log_text.delete("1.0", "2.0")
        self.log_text.config(state=tk.DISABLED)
    
    def update_metrics_display(self, metrics):
        """Update metrics display"""
        if not HAS_TKINTER:
            return
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        
        text = f"""
Workers:        {metrics.num_workers}
Chunk Size:     {metrics.chunk_size:,}
Throughput:     {metrics.throughput:,.0f} rows/s
RAM:            {metrics.ram_percent:.1f}%
CPU:            {metrics.cpu_percent:.1f}%
Rows:           {metrics.rows_processed:,}
Time:           {datetime.fromtimestamp(metrics.timestamp).strftime('%H:%M:%S')}
"""
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)
    
    def draw_graph(self, canvas, values, color="#22c55e"):
        """Simple line graph"""
        if not values or len(values) < 2:
            return
        
        try:
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            
            if w < 50 or h < 50:
                return
            
            val_min = min(values)
            val_max = max(values)
            if val_max - val_min < 1e-6:
                val_max = val_min + 1
            
            # Draw line
            points = []
            for i, v in enumerate(values[-100:]):  # Last 100 points
                x = int((i / max(len(values[-100:]) - 1, 1)) * (w - 20)) + 10
                y = int(h - 10 - ((v - val_min) / (val_max - val_min)) * (h - 20))
                points.extend([x, y])
            
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2)
            
            # Min/Max labels
            canvas.create_text(5, 5, anchor="nw", text=f"Max: {val_max:.1f}", 
                             font=("Courier", 7), fill=color)
            canvas.create_text(5, h-15, anchor="nw", text=f"Min: {val_min:.1f}",
                             font=("Courier", 7), fill=color)
        except:
            pass
    
    def update_gui(self):
        """Update GUI in main loop"""
        if not HAS_TKINTER:
            return
        
        try:
            # Redraw graphs
            if self.history["throughput"]:
                self.draw_graph(self.canvas_tp, self.history["throughput"], "#22c55e")
            if self.history["workers"]:
                self.draw_graph(self.canvas_w, self.history["workers"], "#3b82f6")
            if self.history["chunk_mult"]:
                self.draw_graph(self.canvas_c, self.history["chunk_mult"], "#a855f7")
            
            self.root.after(500, self.update_gui)
        except:
            pass
    
    def show(self):
        """Show GUI"""
        if HAS_TKINTER:
            self.root.mainloop()


class AIOptimizationServer:
    """Standalone AI server with optional GUI"""
    
    def __init__(self, max_workers: int = 12, with_gui: bool = True):
        self.max_workers = max_workers
        self.max_chunk_size = 750_000
        self.min_chunk_size = 50_000
        self.max_ram_percent = 90.0
        
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=10)
        self.recommendation_queue: queue.Queue = queue.Queue(maxsize=1)
        
        self.bandit = LinUCBBandit(feature_dim=5, alpha=1.1)
        self.current_metrics: Optional[Metrics] = None
        
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._prev_throughput = 0.0
        
        self.is_ready = False
        self.with_gui = with_gui and HAS_TKINTER
        self.gui = None
        
        if self.with_gui:
            self.gui = AIServerGUI()
    
    def send_metrics(self, metrics: Metrics) -> None:
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            try:
                self.metrics_queue.get_nowait()
                self.metrics_queue.put_nowait(metrics)
            except:
                pass
    
    def get_recommendation(self, timeout: float = 0.1) -> Optional[Recommendation]:
        try:
            return self.recommendation_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self) -> None:
        self._stop_event.set()
        if self.gui:
            self.gui.running = False
    
    def run(self) -> None:
        """Main server loop + GUI"""
        print("[AIServer] 🚀 Starting AI Optimization Server")
        print("[AIServer] ⏳ Initializing...")
        
        if self.with_gui:
            print("[AIServer] 🎨 GUI enabled (Tkinter)")
        else:
            print("[AIServer] 📝 Console mode (no Tkinter)")
        
        # Start GUI in separate thread if available
        if self.gui:
            gui_thread = threading.Thread(target=self.gui.show, daemon=False)
            gui_thread.start()
            time.sleep(1)
        
        self.is_ready = True
        self._ready_event.set()
        print("[AIServer] ✅ READY")
        
        if self.gui:
            self.gui.update_status("Ready - Waiting for metrics...")
        else:
            print("[AIServer] Waiting for metrics...")
        
        last_action_idx = -1
        metric_count = 0
        
        while not self._stop_event.is_set():
            try:
                try:
                    metrics = self.metrics_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.current_metrics = metrics
                metric_count += 1
                
                # Features
                x = np.array([
                    min(metrics.ram_percent / 100.0, 1.0),
                    min(metrics.cpu_percent / 100.0, 1.0),
                    min(metrics.throughput / 50_000.0, 1.0),
                    metrics.num_workers / float(self.max_workers),
                    metrics.chunk_size / 750_000.0,
                ], dtype=np.float64)
                
                action_idx, score = self.bandit.choose_action(x)
                action = self.bandit.actions[action_idx]
                
                # Reward
                if last_action_idx >= 0:
                    reward = self._compute_reward(metrics)
                    self.bandit.update(reward)
                else:
                    reward = 0.0
                
                last_action_idx = action_idx
                
                # Action
                d_workers, chunk_mult = action
                new_workers = max(1, min(self.max_workers, metrics.num_workers + d_workers))
                
                if metrics.ram_percent >= self.max_ram_percent:
                    reason = f"GUARD: RAM {metrics.ram_percent:.1f}%"
                elif metrics.ram_percent >= (self.max_ram_percent - 3):
                    reason = f"NEAR: RAM {metrics.ram_percent:.1f}%"
                else:
                    reason = f"AI: Δ{d_workers:+d} C×{chunk_mult:.2f}"
                
                rec = Recommendation(
                    d_workers=d_workers,
                    chunk_mult=chunk_mult,
                    reason=reason,
                    score=float(score),
                    timestamp=metrics.timestamp,
                )
                
                try:
                    self.recommendation_queue.put_nowait(rec)
                except queue.Full:
                    pass
                
                # Update GUI
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
                    self.gui.update_metrics_display(metrics)
                    self.gui.log(f"W:{metrics.num_workers}→{new_workers} C×:{chunk_mult:.2f} TP:{metrics.throughput:,.0f} R:{reward:+.3f}")
                
                # Console log
                ts = datetime.fromtimestamp(metrics.timestamp).strftime("%H:%M:%S")
                print(f"[AI {ts}] W:{metrics.num_workers}→{new_workers} | "
                      f"TP:{metrics.throughput:,.0f} | RAM:{metrics.ram_percent:.1f}% | R:{reward:+.3f}")
                
            except Exception as e:
                print(f"[AIServer] Error: {e}")
                time.sleep(0.1)
        
        print("[AIServer] ⏹️  Stopped")
    
    def _compute_reward(self, metrics: Metrics) -> float:
        throughput_gain = (metrics.throughput - self._prev_throughput) / max(
            self._prev_throughput + 1.0, 1.0
        )
        ram_penalty = 0.0
        if metrics.ram_percent >= self.max_ram_percent:
            ram_penalty = -2.0
        elif metrics.ram_percent >= (self.max_ram_percent - 3):
            ram_penalty = -0.7
        
        reward = throughput_gain + ram_penalty
        self._prev_throughput = metrics.throughput
        return float(reward)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Run server with GUI if available
    server = AIOptimizationServer(max_workers=12, with_gui=True)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[AIServer] Stopped by user")
        server.stop()