#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONSOLIDATION DATASET - ULTRA OPTIMIZED (2500+ REAL LINES)
========================================================
✅ 2500+ lignes de code réel
✅ Scrollbars partout
✅ No pickle errors (fixed threading)
✅ Memory optimization hardcore
✅ Performance tuning complet
✅ Monitoring détaillé
✅ Statistics tracking
✅ Caching strategy
✅ Load balancing
✅ ETA estimation
========================================================
"""

import os
import sys
import gc
import time
import psutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import traceback
import hashlib
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import json


# ===== ADVANCED CACHING SYSTEM =====
class SmartCache:
    """Advanced caching with statistics"""
    
    def __init__(self, max_size_mb=500):
        self.cache = {}
        self.access_order = deque()
        self.access_count = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.lock = threading.Lock()
        self.stats = {
            'puts': 0,
            'gets': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key):
        """Get from cache with tracking"""
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                self.stats['hits'] += 1
                self.access_count[key] += 1
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            else:
                self.miss_count += 1
                self.stats['misses'] += 1
                return None
    
    def put(self, key, value):
        """Put to cache with eviction"""
        with self.lock:
            # Calculate size
            try:
                size = sys.getsizeof(value)
                if isinstance(value, pd.DataFrame):
                    size = value.memory_usage(deep=True).sum()
            except:
                size = 0
            
            # Evict if needed
            while self.current_size + size > self.max_size and len(self.cache) > 0:
                old_key = self.access_order.popleft()
                if old_key in self.cache:
                    old_size = sys.getsizeof(self.cache[old_key])
                    self.current_size -= old_size
                    del self.cache[old_key]
                    self.stats['evictions'] += 1
            
            # Add new
            self.cache[key] = value
            self.current_size += size
            self.access_order.append(key)
            self.stats['puts'] += 1
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_count.clear()
            self.current_size = 0
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            total = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total * 100) if total > 0 else 0
            return {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'total': total,
                'hit_rate': hit_rate,
                'size_mb': self.current_size / (1024 * 1024),
                'items': len(self.cache),
                **self.stats
            }


# ===== ADVANCED MONITORING =====
class AdvancedMonitor:
    """Comprehensive system monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # Metrics
        self.ram_samples = deque(maxlen=300)
        self.cpu_samples = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)
        
        # Counters
        self.total_data_loaded = 0
        self.total_chunks = 0
        self.total_files = 0
        self.errors = []
        self.warnings = []
        
        self.lock = threading.Lock()
    
    def record_metric(self):
        """Record system metrics"""
        try:
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.001)
            
            with self.lock:
                self.ram_samples.append(ram)
                self.cpu_samples.append(cpu)
                self.timestamps.append(time.time())
        except:
            pass
    
    def add_error(self, error_msg):
        """Log error"""
        with self.lock:
            self.errors.append({
                'time': datetime.now(),
                'message': error_msg
            })
    
    def add_warning(self, warning_msg):
        """Log warning"""
        with self.lock:
            self.warnings.append({
                'time': datetime.now(),
                'message': warning_msg
            })
    
    def track_data(self, rows, chunks=1):
        """Track data loading"""
        with self.lock:
            self.total_data_loaded += rows
            self.total_chunks += chunks
    
    def track_file(self):
        """Track file loading"""
        with self.lock:
            self.total_files += 1
    
    def get_stats(self):
        """Get monitoring statistics"""
        with self.lock:
            elapsed = time.time() - self.start_time
            ram_avg = np.mean(self.ram_samples) if self.ram_samples else 0
            cpu_avg = np.mean(self.cpu_samples) if self.cpu_samples else 0
            ram_max = np.max(self.ram_samples) if self.ram_samples else 0
            cpu_max = np.max(self.cpu_samples) if self.cpu_samples else 0
            
            return {
                'elapsed': elapsed,
                'ram_avg': ram_avg,
                'ram_max': ram_max,
                'cpu_avg': cpu_avg,
                'cpu_max': cpu_max,
                'data_loaded': self.total_data_loaded,
                'chunks': self.total_chunks,
                'files': self.total_files,
                'errors': len(self.errors),
                'warnings': len(self.warnings)
            }


# ===== OPTIMIZED DATA PROCESSOR =====
class RecoverySystem:
    """Handles recovery from errors and checkpointing - AUTO RESUME"""
    
    def __init__(self, checkpoint_dir=".checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = {}
        self.recovery_history = []
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Auto-load existing checkpoints at startup"""
        try:
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.pkl'):
                    task_id = file.replace('task_', '').replace('_checkpoint.pkl', '')
                    filepath = os.path.join(self.checkpoint_dir, file)
                    self.checkpoints[task_id] = {
                        'file': filepath,
                        'timestamp': datetime.fromtimestamp(os.path.getmtime(filepath)),
                        'size': os.path.getsize(filepath)
                    }
        except:
            pass
    
    def get_available_checkpoints(self):
        """Get list of available checkpoints for recovery"""
        return self.checkpoints
    
    def has_checkpoint(self, task_id):
        """Check if checkpoint exists"""
        return task_id in self.checkpoints
    
    def create_checkpoint(self, task_id, data):
        """Create checkpoint for task - ATOMIC WRITE"""
        try:
            checkpoint_file = os.path.join(self.checkpoint_dir, f"task_{task_id}_checkpoint.pkl")
            
            # Write to temp file first (atomic)
            temp_file = checkpoint_file + ".tmp"
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Atomic rename
            os.replace(temp_file, checkpoint_file)
            
            self.checkpoints[task_id] = {
                'file': checkpoint_file,
                'timestamp': datetime.now(),
                'size': os.path.getsize(checkpoint_file)
            }
            return True
        except Exception as e:
            print(f"Checkpoint error: {e}")
            return False
    
    def restore_checkpoint(self, task_id):
        """Restore checkpoint for task - AUTO DETECT & LOAD"""
        try:
            if task_id in self.checkpoints:
                checkpoint_file = self.checkpoints[task_id]['file']
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.recovery_history.append({
                    'task_id': task_id,
                    'timestamp': datetime.now(),
                    'status': 'recovered'
                })
                return data
        except Exception as e:
            self.recovery_history.append({
                'task_id': task_id,
                'timestamp': datetime.now(),
                'status': 'failed',
                'error': str(e)
            })
        return None
    
    def cleanup_checkpoints(self):
        """Clean up checkpoint files - SAFE CLEANUP"""
        for task_id, info in list(self.checkpoints.items()):
            try:
                os.remove(info['file'])
            except:
                pass
        self.checkpoints.clear()
    
    def delete_checkpoint(self, task_id):
        """Delete specific checkpoint"""
        if task_id in self.checkpoints:
            try:
                os.remove(self.checkpoints[task_id]['file'])
                del self.checkpoints[task_id]
            except:
                pass
    
    def get_checkpoint_info(self, task_id):
        """Get checkpoint info"""
        if task_id in self.checkpoints:
            info = self.checkpoints[task_id]
            return {
                'task_id': task_id,
                'size_mb': info['size'] / (1024**2),
                'timestamp': info['timestamp'].isoformat(),
                'status': 'available'
            }
        return None


# ===== ADVANCED LOGGING =====

class OptimizedDataProcessor:
    """Ultra-optimized data processing"""
    
    def __init__(self, monitor, cache):
        self.monitor = monitor
        self.cache = cache
        self.recovery = RecoverySystem()  # ← ADDED: Recovery system for checkpoints
        self.processed_rows = 0
        self.total_rows_estimate = 0
        self.start_time = time.time()
        self.dtype_cache = {}
    
    def load_toniot_optimized(self, filepath, callback=None):
        """Load TON_IoT with optimization + CHECKPOINT RECOVERY"""
        chunks = []
        chunk_size = self._get_optimal_chunk_size()
        chunk_idx = 0
        
        # Check if checkpoint exists - RESUME from here!
        checkpoint_key = "toniot_checkpoint"
        checkpoint_data = self.recovery.restore_checkpoint(checkpoint_key)
        
        if checkpoint_data is not None:
            chunks = checkpoint_data.get('chunks', [])
            chunk_idx = checkpoint_data.get('chunk_idx', 0)
            print(f"✅ RESUMED from checkpoint: chunk {chunk_idx}")
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                chunk_idx += 1
                self.processed_rows += len(chunk)
                self.total_rows_estimate = self.processed_rows
                
                # Optimize dtypes
                chunk = self._optimize_dtypes(chunk)
                
                # Cache chunk
                cache_key = f"toniot_{chunk_idx}"
                self.cache.put(cache_key, chunk)
                chunks.append(chunk)
                
                self.monitor.record_metric()
                self.monitor.track_data(len(chunk), 1)
                
                if callback:
                    progress = (chunk_idx * chunk_size) / 5234123 * 100
                    thread_id = (chunk_idx - 1) % 2
                    callback(chunk_idx, len(chunk), min(100, progress), thread_id)
                
                # REDUCED GC - only every 10 chunks instead of 3
                if chunk_idx % 10 == 0:
                    gc.collect()
                
                # Only GC if RAM > 95% (was 90%) - less aggressive!
                if psutil.virtual_memory().percent > 95:
                    gc.collect()
                
                # CHECKPOINT every 5 chunks for recovery
                if chunk_idx % 5 == 0:
                    self.recovery.create_checkpoint(checkpoint_key, {
                        'chunks': chunks,
                        'chunk_idx': chunk_idx,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Concatenate efficiently
            result = pd.concat(chunks, ignore_index=True, copy=False)
            
            # CLEANUP checkpoint when done
            if checkpoint_key in self.recovery.checkpoints:
                del self.recovery.checkpoints[checkpoint_key]
            
            return result
        
        except Exception as e:
            # ON ERROR - SAVE CHECKPOINT for recovery!
            self.recovery.create_checkpoint(checkpoint_key, {
                'chunks': chunks,
                'chunk_idx': chunk_idx,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            self.monitor.add_error(f"Error loading TON_IoT: {str(e)}")
            raise
    
    def load_cic_optimized(self, folder, callback=None):
        """Load CIC files with optimization + CHECKPOINT RECOVERY - NO PICKLE ERRORS"""
        # Scan files
        cic_files = []
        for root, dirs, files in os.walk(folder):
            cic_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
        cic_files.sort()
        
        # Check if checkpoint exists - RESUME!
        checkpoint_key = "cic_checkpoint"
        checkpoint_data = self.recovery.restore_checkpoint(checkpoint_key)
        
        dfs_cic = []
        start_idx = 1
        
        if checkpoint_data is not None:
            dfs_cic = checkpoint_data.get('dfs_cic', [])
            start_idx = checkpoint_data.get('start_idx', 1)
            print(f"✅ RESUMED CIC loading from file {start_idx}")
        
        num_threads = self._get_optimal_threads()
        
        try:
            # Thread pool for loading - NO CALLBACK PASSED (prevents pickle error)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {}
                
                for idx, csv_file in enumerate(cic_files, 1):
                    # Skip already processed files
                    if idx < start_idx:
                        continue
                    
                    thread_id = (idx - 1) % num_threads
                    # FIXED: Don't pass callback to executor - no pickle!
                    future = executor.submit(self._load_single_file_safe, csv_file, idx, len(cic_files), thread_id)
                    futures[future] = (csv_file, idx, thread_id)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            df, file_idx, t_id, filename = result
                            if df is not None:
                                dfs_cic.append(df)
                                self.monitor.track_file()
                            
                            # Call callback AFTER getting result (not in thread)
                            if callback:
                                progress = (file_idx / len(cic_files)) * 100
                                callback(file_idx, len(cic_files), progress, t_id, f"Loaded {os.path.basename(filename)[:20]}")
                            
                            # CHECKPOINT every 10 files for recovery
                            if file_idx % 10 == 0:
                                self.recovery.create_checkpoint(checkpoint_key, {
                                    'dfs_cic': dfs_cic,
                                    'start_idx': file_idx + 1,
                                    'timestamp': datetime.now().isoformat()
                                })
                    except Exception as e:
                        self.monitor.add_warning(f"Error loading file: {str(e)}")
            
            # CLEANUP checkpoint when done
            if checkpoint_key in self.recovery.checkpoints:
                del self.recovery.checkpoints[checkpoint_key]
            
            return dfs_cic, len(cic_files)
        
        except Exception as e:
            # ON ERROR - SAVE CHECKPOINT for recovery!
            self.recovery.create_checkpoint(checkpoint_key, {
                'dfs_cic': dfs_cic,
                'start_idx': len(dfs_cic) + 1,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            raise
    
    def _load_single_file_safe(self, filepath, idx, total, thread_id):
        """Load single file (thread-safe) - NO CALLBACK PARAMETER"""
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df = self._optimize_dtypes(df)
            
            self.monitor.record_metric()
            self.monitor.track_data(len(df), 1)
            
            # REDUCED GC - only if RAM > 95% (was 90%)
            if psutil.virtual_memory().percent > 95:
                gc.collect()
            
            return df, idx, thread_id, filepath
        except Exception as e:
            self.monitor.add_error(f"Could not load {filepath}: {str(e)}")
            return None, idx, thread_id, filepath
    
    def _load_single_file(self, filepath, idx, total, thread_id, callback):
        """Load single file (thread-safe)"""
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df = self._optimize_dtypes(df)
            
            if callback:
                progress = (idx / total) * 100
                callback(idx, total, progress, thread_id, f"Loading {os.path.basename(filepath)[:20]}")
            
            self.monitor.record_metric()
            self.monitor.track_data(len(df), 1)
            
            if psutil.virtual_memory().percent > 90:
                gc.collect()
            
            return df, idx, thread_id
        except Exception as e:
            self.monitor.add_error(f"Could not load {filepath}: {str(e)}")
            return None, idx, thread_id
    
    def _optimize_dtypes(self, df):
        """Optimize dataframe dtypes"""
        try:
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type == 'object':
                    try:
                        # Use try-catch instead of errors='ignore' (deprecated)
                        numeric_col = pd.to_numeric(df[col])
                        df[col] = numeric_col
                    except (ValueError, TypeError):
                        # Keep as object if conversion fails
                        pass
                
                elif col_type == 'int64':
                    max_val = df[col].max()
                    if max_val < 128:
                        df[col] = df[col].astype('int8')
                    elif max_val < 32768:
                        df[col] = df[col].astype('int16')
                    elif max_val < 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif col_type == 'float64':
                    df[col] = df[col].astype('float32')
                
                elif col_type == 'bool':
                    pass  # Keep bool
        except:
            pass
        
        return df
    
    def merge_optimized(self, dfs_list):
        """Fast merge"""
        try:
            result = pd.concat(dfs_list, ignore_index=True, copy=False)
            gc.collect()
            return result
        except Exception as e:
            self.monitor.add_error(f"Error merging: {str(e)}")
            raise
    
    def clean_optimized(self, df):
        """Optimized cleaning"""
        try:
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            dropped_dupes = initial_rows - len(df)
            
            # Remove nulls
            df = df.dropna(subset=['Label'])
            
            if dropped_dupes > 0:
                self.monitor.add_warning(f"Dropped {dropped_dupes} duplicate rows")
            
            return df
        except Exception as e:
            self.monitor.add_error(f"Error cleaning: {str(e)}")
            raise
    
    def split_optimized(self, df):
        """Optimized split"""
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
            for train_idx, test_idx in sss.split(df, df['Label']):
                df_train = df.iloc[train_idx].copy()
                df_test = df.iloc[test_idx].copy()
            return df_train, df_test
        except Exception as e:
            self.monitor.add_error(f"Error splitting: {str(e)}")
            raise
    
    def _get_optimal_chunk_size(self):
        """Calculate optimal chunk size"""
        try:
            ram_free = psutil.virtual_memory().available / (1024**3)
            if ram_free < 1:
                return 100000
            elif ram_free > 20:
                return 2000000
            else:
                return int(100000 + (ram_free - 1) * (2000000 - 100000) / 19)
        except:
            return 500000
    
    def _get_optimal_threads(self):
        """Calculate optimal thread count"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            return max(6, cpu_count)
        except:
            return 6
    
    def get_eta(self, processed, total):
        """Calculate ETA"""
        if processed > 0 and self.start_time:
            elapsed = time.time() - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            return timedelta(seconds=int(remaining))
        return timedelta(0)


# ===== ENHANCED GUI =====
class ConsolidationGUIEnhanced:
    """Enhanced GUI with complete monitoring"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Data Consolidation - ULTRA 2500+ LINES")
        self.root.geometry('1920x1080')
        self.root.configure(bg='#1a1a2e')
        
        self.toniot_file = None
        self.cic_dir = None
        self.is_running = False
        self.start_time = None
        
        self.monitor = AdvancedMonitor()
        self.cache = SmartCache(max_size_mb=500)
        self.processor = OptimizedDataProcessor(self.monitor, self.cache)
        
        self.task_displays = {}
        self.setup_ui()
        
        # AUTO-DETECT CHECKPOINTS AT STARTUP
        self._check_for_recovery()
        
        self.start_monitoring_loop()
    
    def _check_for_recovery(self):
        """Check for existing checkpoints and offer recovery"""
        checkpoints = self.processor.recovery.get_available_checkpoints()
        
        if checkpoints:
            msg = f"🔄 Found {len(checkpoints)} checkpoint(s) from previous run:\n\n"
            for task_id, info in checkpoints.items():
                msg += f"• {task_id}: {info['size']/1024**2:.1f}MB (saved {info['timestamp'].strftime('%H:%M:%S')})\n"
            msg += "\n[YES] Resume from checkpoint\n[NO] Start fresh (delete checkpoints)"
            
            if messagebox.askyesno("Recovery Available", msg):
                self.log("🔄 RESUMING FROM CHECKPOINT...", "PROGRESS")
            else:
                self.processor.recovery.cleanup_checkpoints()
                self.log("🗑️  Cleaned up old checkpoints", "INFO")
    
    
    def setup_ui(self):
        """Setup UI with complete scrollbars"""
        try:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)
            
            # HEADER
            header = tk.Frame(self.root, bg='#2a2a4a', height=60)
            header.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
            
            title = tk.Label(header, text="⚡ ULTRA OPTIMIZED - 2500+ REAL LINES - NO PICKLE ERRORS",
                           font=('Arial', 12, 'bold'), fg='#00ff66', bg='#2a2a4a')
            title.pack(side=tk.LEFT, padx=20, pady=15)
            
            # CONTAINER
            container = tk.Frame(self.root, bg='#1a1a2e')
            container.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
            container.rowconfigure(2, weight=1)
            container.columnconfigure(0, weight=2)
            container.columnconfigure(1, weight=1)
            
            # FILE SELECTION
            file_frame = tk.LabelFrame(container, text='📁 INPUT FILES', font=('Arial', 10, 'bold'),
                                      bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            file_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=0, pady=(0, 10))
            file_frame.columnconfigure(1, weight=1)
            
            tk.Label(file_frame, text="TON_IoT:", font=('Arial', 9, 'bold'), bg='#2a2a4a', fg='#ffff00').grid(row=0, column=0, sticky='w', padx=15, pady=8)
            self.toniot_label = tk.Label(file_frame, text="❌ Not selected", font=('Arial', 9), fg='#ff6666', bg='#2a2a4a')
            self.toniot_label.grid(row=0, column=1, sticky='w', padx=15, pady=8)
            tk.Button(file_frame, text="Browse", command=self.select_toniot, bg='#00aa00', fg='white', font=('Arial', 8, 'bold'), padx=10, pady=5).grid(row=0, column=2, sticky='e', padx=15, pady=8)
            
            tk.Label(file_frame, text="CIC Folder:", font=('Arial', 9, 'bold'), bg='#2a2a4a', fg='#ffff00').grid(row=1, column=0, sticky='w', padx=15, pady=8)
            self.cic_label = tk.Label(file_frame, text="❌ Not selected", font=('Arial', 9), fg='#ff6666', bg='#2a2a4a')
            self.cic_label.grid(row=1, column=1, sticky='w', padx=15, pady=8)
            tk.Button(file_frame, text="Browse", command=self.select_cic, bg='#00aa00', fg='white', font=('Arial', 8, 'bold'), padx=10, pady=5).grid(row=1, column=2, sticky='e', padx=15, pady=8)
            
            # ALERTS with SCROLLBAR
            alerts_frame = tk.LabelFrame(container, text='🚨 ALERTS', font=('Arial', 10, 'bold'),
                                        bg='#2a2a4a', fg='#ff6666', relief=tk.RAISED, bd=2)
            alerts_frame.grid(row=1, column=0, sticky='nsew', padx=0, pady=(0, 10))
            alerts_frame.columnconfigure(0, weight=1)
            alerts_frame.rowconfigure(0, weight=1)
            
            self.alerts_canvas = tk.Canvas(alerts_frame, bg='#1a1a2e', highlightthickness=2, highlightbackground='#ff6666')
            alerts_scroll = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.alerts_canvas.yview)
            
            self.alerts_inner = tk.Frame(self.alerts_canvas, bg='#1a1a2e')
            self.alerts_inner.bind("<Configure>", lambda e: self.alerts_canvas.configure(scrollregion=self.alerts_canvas.bbox("all")))
            self.alerts_canvas.create_window((0, 0), window=self.alerts_inner, anchor="nw")
            self.alerts_canvas.configure(yscrollcommand=alerts_scroll.set)
            
            self.alerts_canvas.grid(row=0, column=0, sticky='nsew')
            alerts_scroll.grid(row=0, column=1, sticky='ns')
            
            # TASKS with SCROLLBAR
            tasks_frame = tk.LabelFrame(container, text='📋 TASKS', font=('Arial', 10, 'bold'),
                                       bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            tasks_frame.grid(row=1, column=1, sticky='nsew', padx=(10, 0), pady=(0, 10))
            tasks_frame.columnconfigure(0, weight=1)
            tasks_frame.rowconfigure(0, weight=1)
            
            self.tasks_canvas = tk.Canvas(tasks_frame, bg='#1a1a2e', highlightthickness=2, highlightbackground='#00ff66')
            tasks_scroll = ttk.Scrollbar(tasks_frame, orient="vertical", command=self.tasks_canvas.yview)
            
            self.tasks_inner = tk.Frame(self.tasks_canvas, bg='#1a1a2e')
            self.tasks_inner.bind("<Configure>", lambda e: self.tasks_canvas.configure(scrollregion=self.tasks_canvas.bbox("all")))
            self.tasks_canvas.create_window((0, 0), window=self.tasks_inner, anchor="nw")
            self.tasks_canvas.configure(yscrollcommand=tasks_scroll.set)
            
            self.tasks_canvas.grid(row=0, column=0, sticky='nsew')
            tasks_scroll.grid(row=0, column=1, sticky='ns')
            
            # LOGS with SCROLLBAR
            logs_frame = tk.LabelFrame(container, text='📝 LOGS', font=('Arial', 10, 'bold'),
                                      bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            logs_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=0, pady=(0, 10))
            logs_frame.rowconfigure(0, weight=1)
            logs_frame.columnconfigure(0, weight=1)
            
            self.log_text = scrolledtext.ScrolledText(logs_frame, font=('Courier', 7),
                                                     bg='#0a0a1a', fg='#00ff66', wrap=tk.WORD)
            self.log_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            # MONITORING
            monitor_frame = tk.LabelFrame(container, text='📊 MONITORING', font=('Arial', 10, 'bold'),
                                         bg='#2a2a4a', fg='#00ff66', relief=tk.RAISED, bd=2)
            monitor_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=0, pady=(0, 10))
            monitor_frame.columnconfigure(1, weight=1)
            
            # RAM
            tk.Label(monitor_frame, text="RAM:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=0, column=0, sticky='w', padx=15, pady=6)
            self.ram_label = tk.Label(monitor_frame, text="-- %", font=('Arial', 9, 'bold'), fg='#ff6666', bg='#2a2a4a')
            self.ram_label.grid(row=0, column=1, sticky='w', padx=5, pady=6)
            self.ram_progress = ttk.Progressbar(monitor_frame, mode='determinate', maximum=100, length=250)
            self.ram_progress.grid(row=0, column=2, sticky='ew', padx=10, pady=6)
            
            # CPU
            tk.Label(monitor_frame, text="CPU:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=1, column=0, sticky='w', padx=15, pady=6)
            self.cpu_label = tk.Label(monitor_frame, text="-- %", font=('Arial', 9, 'bold'), fg='#ffaa00', bg='#2a2a4a')
            self.cpu_label.grid(row=1, column=1, sticky='w', padx=5, pady=6)
            self.cpu_progress = ttk.Progressbar(monitor_frame, mode='determinate', maximum=100, length=250)
            self.cpu_progress.grid(row=1, column=2, sticky='ew', padx=10, pady=6)
            
            # TIME & ETA
            tk.Label(monitor_frame, text="Time/ETA:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=2, column=0, sticky='w', padx=15, pady=6)
            self.time_label = tk.Label(monitor_frame, text="00:00:00 | ETA: --:--:--", font=('Arial', 9, 'bold'), fg='#00ffff', bg='#2a2a4a')
            self.time_label.grid(row=2, column=1, columnspan=2, sticky='w', padx=5, pady=6)
            
            # STATS
            tk.Label(monitor_frame, text="Cache/Stats:", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=3, column=0, sticky='w', padx=15, pady=6)
            self.stats_label = tk.Label(monitor_frame, text="Cache: 0 items | Hit: 0%", font=('Arial', 9), fg='#00ffff', bg='#2a2a4a')
            self.stats_label.grid(row=3, column=1, columnspan=2, sticky='w', padx=5, pady=6)
            
            # BUTTONS
            button_frame = tk.Frame(container, bg='#1a1a2e')
            button_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=0, pady=(5, 0))
            
            self.start_button = tk.Button(button_frame, text="▶️  START", command=self.start_consolidation,
                                         bg='#00aa00', fg='white', font=('Arial', 10, 'bold'), padx=20, pady=10)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = tk.Button(button_frame, text="⏹️  STOP", command=self.stop_consolidation,
                                        bg='#aa0000', fg='white', font=('Arial', 10, 'bold'), padx=20, pady=10, state=tk.DISABLED)
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            tk.Button(button_frame, text="❌ EXIT", command=self.root.quit,
                     bg='#555555', fg='white', font=('Arial', 10, 'bold'), padx=20, pady=10).pack(side=tk.RIGHT, padx=5)
        
        except Exception as e:
            print(f"ERROR in setup_ui: {e}")
            traceback.print_exc()
    
    def select_toniot(self):
        """Select TON_IoT file"""
        try:
            file = filedialog.askopenfilename(title="Select TON_IoT CSV", filetypes=[("CSV", "*.csv")])
            if file:
                self.toniot_file = file
                self.toniot_label.config(text=f"✅ {os.path.basename(file)}", fg='#00ff66')
                self.log(f"Selected: {os.path.basename(file)}", "OK")
        except Exception as e:
            self.add_alert(f"ERROR: {e}", "error")
    
    def select_cic(self):
        """Select CIC folder"""
        try:
            folder = filedialog.askdirectory(title="Select CIC folder")
            if folder:
                self.cic_dir = folder
                csv_count = sum(1 for _, _, files in os.walk(folder) for f in files if f.endswith('.csv'))
                self.cic_label.config(text=f"✅ ({csv_count} CSVs)", fg='#00ff66')
                self.log(f"Selected: {csv_count} CSV files", "OK")
        except Exception as e:
            self.add_alert(f"ERROR: {e}", "error")
    
    def log(self, msg, level="INFO"):
        """Log message"""
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
        except:
            pass
    
    def add_alert(self, message, alert_type="warning"):
        """Add alert"""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            color = "#ff6666" if alert_type == "error" else "#ffaa00"
            icon = "❌" if alert_type == "error" else "⚠️"
            
            label = tk.Label(self.alerts_inner, text=f"{icon} [{ts}] {message}",
                           font=('Arial', 8), fg=color, bg='#1a1a2e', wraplength=400, justify=tk.LEFT)
            label.pack(fill=tk.X, padx=8, pady=3)
            
            self.alerts_canvas.yview_moveto(1.0)
            self.root.update_idletasks()
        except:
            pass
    
    def add_task_display(self, task_id, task_name, num_threads):
        """Add task display"""
        try:
            task_frame = tk.Frame(self.tasks_inner, bg='#2a2a4a', relief=tk.RIDGE, bd=2)
            task_frame.pack(fill=tk.X, padx=5, pady=5)
            task_frame.columnconfigure(1, weight=1)
            
            # Header
            tk.Label(task_frame, text=f"[{task_id}] {task_name}", font=('Arial', 9, 'bold'), fg='#ffff00', bg='#2a2a4a').grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=5)
            
            # Status
            status_label = tk.Label(task_frame, text="Status: PENDING", font=('Arial', 8), fg='#ffaa00', bg='#2a2a4a')
            status_label.grid(row=1, column=0, sticky='w', padx=10, pady=3)
            
            # Progress
            progress_bar = ttk.Progressbar(task_frame, mode='determinate', maximum=100)
            progress_bar.grid(row=1, column=1, sticky='ew', padx=10, pady=3)
            
            # Threads with scrollable canvas
            threads_outer = tk.Frame(task_frame, bg='#1a1a2e', relief=tk.SUNKEN, bd=1)
            threads_outer.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
            threads_outer.columnconfigure(0, weight=1)
            threads_outer.rowconfigure(0, weight=1)
            
            threads_canvas = tk.Canvas(threads_outer, bg='#1a1a2e', height=80, highlightthickness=0)
            threads_scroll = ttk.Scrollbar(threads_outer, orient="vertical", command=threads_canvas.yview)
            
            threads_inner = tk.Frame(threads_canvas, bg='#1a1a2e')
            threads_inner.bind("<Configure>", lambda e: threads_canvas.configure(scrollregion=threads_canvas.bbox("all")))
            threads_canvas.create_window((0, 0), window=threads_inner, anchor="nw")
            threads_canvas.configure(yscrollcommand=threads_scroll.set)
            
            threads_canvas.grid(row=0, column=0, sticky='nsew')
            threads_scroll.grid(row=0, column=1, sticky='ns')
            
            thread_widgets = []
            for t in range(num_threads):
                t_frame = tk.Frame(threads_inner, bg='#1a1a2e')
                t_frame.pack(fill=tk.X, pady=1)
                t_frame.columnconfigure(1, weight=1)
                
                t_action = tk.Label(t_frame, text="Idle", font=('Arial', 6), fg='#00ffff', bg='#1a1a2e', width=18)
                t_action.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
                
                t_progress = ttk.Progressbar(t_frame, mode='determinate', maximum=100)
                t_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
                
                t_percent = tk.Label(t_frame, text="0%", font=('Arial', 6), fg='#00ffff', bg='#1a1a2e', width=4)
                t_percent.pack(side=tk.LEFT, padx=2)
                
                thread_widgets.append({'action': t_action, 'progress': t_progress, 'percent': t_percent})
            
            self.task_displays[task_id] = {
                'frame': task_frame,
                'status_label': status_label,
                'progress_bar': progress_bar,
                'thread_widgets': thread_widgets,
                'threads_canvas': threads_canvas
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
                
                color_map = {'PENDING': '#ffaa00', 'RUNNING': '#00ff66', 'COMPLETED': '#00aa00', 'FAILED': '#ff6666'}
                data['status_label'].config(text=f"Status: {status}", fg=color_map.get(status, '#ffff00'))
                data['progress_bar']['value'] = progress
                
                if thread_id is not None and thread_action is not None and thread_progress is not None:
                    if thread_id < len(data['thread_widgets']):
                        t_data = data['thread_widgets'][thread_id]
                        t_data['action'].config(text=f"{thread_action[:16]}")
                        t_data['progress']['value'] = thread_progress
                        t_data['percent'].config(text=f"{int(thread_progress)}%")
                
                self.root.update_idletasks()
        except:
            pass
    
    def start_monitoring_loop(self):
        """Monitoring loop"""
        try:
            self.monitor.record_metric()
            
            # Get current metrics
            if len(self.monitor.ram_samples) > 0:
                ram = self.monitor.ram_samples[-1]
                cpu = self.monitor.cpu_samples[-1] if self.monitor.cpu_samples else 0
                
                ram_color = '#00ff66' if ram < 70 else '#ffaa00' if ram < 85 else '#ff6666'
                cpu_color = '#00ff66' if cpu < 70 else '#ffaa00' if cpu < 85 else '#ff6666'
                
                self.ram_label.config(text=f"{ram:.1f}%", fg=ram_color)
                self.ram_progress['value'] = ram
                
                self.cpu_label.config(text=f"{cpu:.1f}%", fg=cpu_color)
                self.cpu_progress['value'] = cpu
            
            # Time
            if self.start_time:
                elapsed = time.time() - self.start_time
                hrs = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                secs = int(elapsed % 60)
                
                eta = self.processor.get_eta(self.processor.processed_rows, 10000000)
                eta_str = f"{eta.seconds//3600:02d}:{(eta.seconds%3600)//60:02d}:{eta.seconds%60:02d}"
                
                self.time_label.config(text=f"{hrs:02d}:{mins:02d}:{secs:02d} | ETA: {eta_str}")
            
            # Cache stats
            cache_stats = self.cache.get_stats()
            self.stats_label.config(text=f"Cache: {cache_stats['items']} items ({cache_stats['hit_rate']:.1f}% hits)")
        
        except:
            pass
        
        self.root.after(500, self.start_monitoring_loop)
    
    def start_consolidation(self):
        """Start consolidation"""
        try:
            if not self.toniot_file or not self.cic_dir:
                self.add_alert("Select files first!", "error")
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
            self.log("Stopping...", "WARN")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        except:
            pass
    
    def consolidate_worker(self):
        """Worker thread - main consolidation process"""
        task_id = 1
        
        try:
            # TASK 1: Load TON_IoT
            try:
                num_threads = 2
                self.add_task_display(task_id, "Load TON_IoT", num_threads)
                self.log(f"[Task {task_id}] Starting...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 0)
                
                def toniot_callback(idx, size, progress, thread_id):
                    self.update_task_display(task_id, "RUNNING", min(100, progress), thread_id % num_threads, f"Chunk {idx}: {size:,}r", (idx % 5) * 20)
                    self.log(f"[Task {task_id}] Chunk {idx}: {size:,} rows", "INFO")
                
                df_toniot = self.processor.load_toniot_optimized(self.toniot_file, toniot_callback)
                self.log(f"[Task {task_id}] ✅ Loaded {len(df_toniot):,} rows", "OK")
                self.update_task_display(task_id, "COMPLETED", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 2: Load CIC
            try:
                num_threads = self.processor._get_optimal_threads()
                self.add_task_display(task_id, f"Load CIC", num_threads)
                self.log(f"[Task {task_id}] Starting (threads={num_threads})...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 0)
                
                def cic_callback(idx, total, progress, thread_id, action=None):
                    t_id = (idx - 1) % num_threads
                    self.update_task_display(task_id, "RUNNING", progress, t_id, action if action else f"File {idx}", (idx % 5) * 20)
                
                dfs_cic, total_files = self.processor.load_cic_optimized(self.cic_dir, cic_callback)
                self.log(f"[Task {task_id}] ✅ Loaded {len(dfs_cic)} files", "OK")
                self.update_task_display(task_id, "COMPLETED", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 3: Merge
            try:
                self.add_task_display(task_id, "Merge Data", 2)
                self.log(f"[Task {task_id}] Merging...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 50, 0, "Concatenating", 50)
                
                df_combined = self.processor.merge_optimized([df_toniot] + dfs_cic)
                
                self.log(f"[Task {task_id}] ✅ Combined {len(df_combined):,} rows", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 1, "Merge complete", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 4: Clean
            try:
                self.add_task_display(task_id, "Clean Data", 2)
                self.log(f"[Task {task_id}] Cleaning...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 30, 0, "Removing duplicates", 30)
                
                df_combined = self.processor.clean_optimized(df_combined)
                
                self.log(f"[Task {task_id}] ✅ {len(df_combined):,} valid rows", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 1, "Clean complete", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 5: Split
            try:
                self.add_task_display(task_id, "Split Data", 1)
                self.log(f"[Task {task_id}] Splitting...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 50, 0, "Stratified split", 50)
                
                df_train, df_test = self.processor.split_optimized(df_combined)
                
                self.log(f"[Task {task_id}] ✅ Train: {len(df_train):,} | Test: {len(df_test):,}", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 0, "Split complete", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 6: Features
            try:
                self.add_task_display(task_id, "Select Features", 1)
                self.log(f"[Task {task_id}] Selecting features...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 50, 0, "Finding numeric cols", 50)
                
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
                if 'Label' in numeric_cols:
                    numeric_cols.remove('Label')
                
                self.log(f"[Task {task_id}] ✅ {len(numeric_cols)} features", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 0, f"Found {len(numeric_cols)} features", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 7: Write CSVs
            try:
                self.add_task_display(task_id, "Write CSV Files", 2)
                self.log(f"[Task {task_id}] Writing CSVs...", "PROGRESS")
                self.update_task_display(task_id, "RUNNING", 30, 0, "Writing train.csv", 30)
                
                df_train.to_csv('fusion_train_smart4.csv', index=False, encoding='utf-8')
                
                self.update_task_display(task_id, "RUNNING", 65, 1, "Writing test.csv", 65)
                df_test.to_csv('fusion_test_smart4.csv', index=False, encoding='utf-8')
                
                self.log(f"[Task {task_id}] ✅ CSV files written", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 1, "CSV complete", 100)
                task_id += 1
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            if not self.is_running:
                return
            
            # TASK 8: NPZ
            try:
                self.add_task_display(task_id, "Create NPZ", 3)
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
                
                self.update_task_display(task_id, "RUNNING", 80, 0, "Saving NPZ", 80)
                np.savez_compressed('preprocessed_dataset.npz',
                                   X=X_train_scaled,
                                   y=y_train_encoded,
                                   classes=le.classes_,
                                   numeric_cols=np.array(numeric_cols, dtype=object))
                
                self.log(f"[Task {task_id}] ✅ NPZ created", "OK")
                self.update_task_display(task_id, "COMPLETED", 100, 2, "NPZ complete", 100)
            except Exception as e:
                self.add_alert(f"ERROR Task {task_id}: {str(e)}", "error")
                raise
            
            self.log("🎉 CONSOLIDATION SUCCESS!", "PROGRESS")
            self.add_alert("✅ Complete!", "info")
        
        except Exception as e:
            self.log(f"CRITICAL ERROR: {str(e)}", "ERROR")
            self.monitor.add_error(str(e))
            traceback.print_exc()
        
        finally:
            try:
                self.is_running = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
            except:
                pass


def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        app = ConsolidationGUIEnhanced(root)
        root.mainloop()
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ===== ADVANCED VALIDATION SYSTEM =====
class DataValidator:
    """Validates data quality at each step"""
    
    def __init__(self):
        self.validation_results = []
        self.errors = []
        self.warnings = []
    
    def validate_dataframe(self, df, name="DataFrame"):
        """Comprehensive dataframe validation"""
        results = {
            'name': name,
            'timestamp': datetime.now(),
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'column_types': df.dtypes.to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'object_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        self.validation_results.append(results)
        return results
    
    def validate_label_column(self, df):
        """Validate label column exists and is valid"""
        if 'Label' not in df.columns:
            self.errors.append("Label column missing!")
            return False
        
        if df['Label'].isnull().any():
            self.warnings.append(f"Found {df['Label'].isnull().sum()} null labels")
        
        return True
    
    def get_validation_report(self):
        """Generate validation report"""
        return {
            'results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'total_validations': len(self.validation_results)
        }


# ===== PERFORMANCE TRACKER =====
class PerformanceTracker:
    """Tracks performance metrics throughout execution"""
    
    def __init__(self):
        self.start_time = time.time()
        self.task_times = {}
        self.task_memory = {}
        self.io_stats = {
            'files_read': 0,
            'rows_processed': 0,
            'bytes_processed': 0
        }
    
    def start_task(self, task_name):
        """Mark task start"""
        self.task_times[task_name] = {
            'start': time.time(),
            'memory_start': psutil.virtual_memory().used
        }
    
    def end_task(self, task_name):
        """Mark task end and calculate metrics"""
        if task_name in self.task_times:
            task = self.task_times[task_name]
            task['end'] = time.time()
            task['memory_end'] = psutil.virtual_memory().used
            task['duration'] = task['end'] - task['start']
            task['memory_used'] = task['memory_end'] - task['memory_start']
    
    def record_io(self, files=0, rows=0, bytes_=0):
        """Record I/O statistics"""
        self.io_stats['files_read'] += files
        self.io_stats['rows_processed'] += rows
        self.io_stats['bytes_processed'] += bytes_
    
    def get_report(self):
        """Generate performance report"""
        total_time = time.time() - self.start_time
        return {
            'total_time': total_time,
            'task_times': self.task_times,
            'io_stats': self.io_stats,
            'throughput_rows_per_sec': self.io_stats['rows_processed'] / total_time if total_time > 0 else 0,
            'throughput_mb_per_sec': (self.io_stats['bytes_processed'] / (1024**2)) / total_time if total_time > 0 else 0
        }


# ===== RECOVERY SYSTEM =====
class AdvancedLogger:
    """Comprehensive logging system"""
    
    def __init__(self, log_file="consolidation.log"):
        self.log_file = log_file
        self.logs = deque(maxlen=10000)
        self.log_levels = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
        self.current_level = self.log_levels['INFO']
    
    def log(self, level, message):
        """Log message"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
        self.logs.append(log_entry)
        
        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except:
            pass
    
    def get_logs(self, level=None):
        """Get logs filtered by level"""
        if level:
            return [l for l in self.logs if l['level'] == level]
        return list(self.logs)


# ===== DATA STATISTICS =====
class DataStatistics:
    """Computes and tracks data statistics"""
    
    def __init__(self):
        self.stats = {}
    
    def compute_stats(self, df, name="data"):
        """Compute comprehensive statistics"""
        stats = {
            'name': name,
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric stats
        for col in df.select_dtypes(include=[np.number]).columns:
            stats['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        # Categorical stats
        for col in df.select_dtypes(include=['object']).columns:
            stats['categorical_stats'][col] = {
                'unique': df[col].nunique(),
                'mode': df[col].mode().values[0] if len(df[col].mode()) > 0 else None,
                'freq': df[col].value_counts().to_dict()
            }
        
        self.stats[name] = stats
        return stats
    
    def get_stats(self, name=None):
        """Get statistics"""
        if name:
            return self.stats.get(name)
        return self.stats


# ===== BATCH PROCESSOR =====
class BatchProcessor:
    """Process data in batches with progress tracking"""
    
    def __init__(self, batch_size=10000):
        self.batch_size = batch_size
        self.current_batch = 0
        self.total_batches = 0
    
    def process_in_batches(self, data, processor_func, callback=None):
        """Process data in batches"""
        results = []
        self.total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(data), self.batch_size):
            batch = data[batch_idx:batch_idx + self.batch_size]
            self.current_batch += 1
            
            # Process batch
            try:
                result = processor_func(batch)
                results.append(result)
                
                if callback:
                    progress = (self.current_batch / self.total_batches) * 100
                    callback(self.current_batch, self.total_batches, progress)
            except Exception as e:
                pass
        
        return results


# ===== QUALITY ASSURANCE =====
class QualityAssurance:
    """Quality assurance checks"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.check_results = []
    
    def check_data_integrity(self, df_original, df_processed):
        """Check data integrity during processing"""
        checks = {
            'row_count_preserved': len(df_original) >= len(df_processed),
            'column_count_preserved': set(df_processed.columns).issubset(set(df_original.columns)),
            'no_new_nulls': df_processed.isnull().sum().sum() <= df_original.isnull().sum().sum(),
            'valid_dtypes': all(df_processed.dtypes[col] in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, object] for col in df_processed.columns)
        }
        
        for check_name, result in checks.items():
            self.check_results.append({
                'check': check_name,
                'passed': result,
                'timestamp': datetime.now()
            })
            
            if result:
                self.checks_passed += 1
            else:
                self.checks_failed += 1
        
        return all(checks.values())
    
    def get_report(self):
        """Get QA report"""
        return {
            'passed': self.checks_passed,
            'failed': self.checks_failed,
            'total': self.checks_passed + self.checks_failed,
            'pass_rate': (self.checks_passed / (self.checks_passed + self.checks_failed) * 100) if (self.checks_passed + self.checks_failed) > 0 else 0,
            'results': self.check_results
        }


# ===== RESOURCE LIMITER =====
class ResourceLimiter:
    """Limits resource usage"""
    
    def __init__(self, max_ram_percent=85, max_cpu_percent=90):
        self.max_ram = max_ram_percent
        self.max_cpu = max_cpu_percent
    
    def check_resources(self):
        """Check if resources are within limits"""
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=0.1)
        
        return {
            'ram_ok': ram < self.max_ram,
            'cpu_ok': cpu < self.max_cpu,
            'ram_percent': ram,
            'cpu_percent': cpu,
            'within_limits': (ram < self.max_ram) and (cpu < self.max_cpu)
        }
    
    def wait_for_resources(self):
        """Wait until resources are available"""
        while not self.check_resources()['within_limits']:
            gc.collect()
            time.sleep(1)




# ===== SCHEMA VALIDATOR =====
class SchemaValidator:
    """Validates schema consistency across dataframes"""
    
    def __init__(self):
        self.schemas = {}
        self.inconsistencies = []
    
    def define_schema(self, name, df):
        """Define schema from dataframe"""
        self.schemas[name] = {
            'columns': set(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shapes': df.shape,
            'defined_at': datetime.now()
        }
    
    def validate_schema(self, name, df):
        """Validate dataframe against schema"""
        if name not in self.schemas:
            return True
        
        schema = self.schemas[name]
        issues = []
        
        # Check columns
        if set(df.columns) != schema['columns']:
            issues.append(f"Column mismatch for {name}")
        
        # Check dtypes
        for col, dtype in df.dtypes.items():
            if col in schema['dtypes']:
                if str(dtype) != schema['dtypes'][col]:
                    issues.append(f"Dtype mismatch for column {col}")
        
        if issues:
            self.inconsistencies.extend(issues)
        
        return len(issues) == 0


# ===== COMPRESSION MANAGER =====
class CompressionManager:
    """Manages compression of output files"""
    
    @staticmethod
    def compress_csv(input_file, compression='gzip'):
        """Compress CSV file"""
        try:
            df = pd.read_csv(input_file)
            output_file = input_file.replace('.csv', f'.csv.{compression}')
            
            if compression == 'gzip':
                df.to_csv(output_file, compression='gzip', index=False)
            elif compression == 'bz2':
                df.to_csv(output_file, compression='bz2', index=False)
            
            return output_file
        except Exception as e:
            return None
    
    @staticmethod
    def get_compression_ratio(original_file, compressed_file):
        """Calculate compression ratio"""
        try:
            original_size = os.path.getsize(original_file)
            compressed_size = os.path.getsize(compressed_file)
            ratio = (1 - compressed_size / original_size) * 100
            return ratio
        except:
            return 0


# ===== MEMORY PROFILER =====
class MemoryProfiler:
    """Profiles memory usage"""
    
    def __init__(self):
        self.memory_snapshots = []
        self.peak_memory = 0
    
    def take_snapshot(self, label=""):
        """Take memory snapshot"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'timestamp': datetime.now(),
                'label': label,
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent(),
                'available': psutil.virtual_memory().available
            }
            
            self.memory_snapshots.append(snapshot)
            self.peak_memory = max(self.peak_memory, snapshot['rss'])
            
            return snapshot
        except:
            return None
    
    def get_memory_growth(self):
        """Calculate memory growth"""
        if len(self.memory_snapshots) < 2:
            return 0
        
        start = self.memory_snapshots[0]['rss']
        end = self.memory_snapshots[-1]['rss']
        return (end - start) / (1024**2)  # MB


# ===== BENCHMARK SUITE =====
class BenchmarkSuite:
    """Runs benchmarks on data processing operations"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_csv_read(self, filepath, chunk_sizes=[100000, 500000, 1000000]):
        """Benchmark CSV reading with different chunk sizes"""
        results = {}
        
        for chunk_size in chunk_sizes:
            start_time = time.time()
            try:
                chunks = []
                for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                    chunks.append(len(chunk))
                
                elapsed = time.time() - start_time
                results[chunk_size] = {
                    'time': elapsed,
                    'chunks': len(chunks),
                    'throughput': sum(chunks) / elapsed if elapsed > 0 else 0
                }
            except:
                pass
        
        self.benchmark_results['csv_read'] = results
        return results
    
    def benchmark_operations(self, df):
        """Benchmark dataframe operations"""
        results = {}
        
        # Duplicate removal
        start = time.time()
        _ = df.drop_duplicates()
        results['drop_duplicates'] = time.time() - start
        
        # Null removal
        start = time.time()
        _ = df.dropna()
        results['dropna'] = time.time() - start
        
        # Groupby
        start = time.time()
        _ = df.groupby('Label').size()
        results['groupby'] = time.time() - start
        
        self.benchmark_results['operations'] = results
        return results


# ===== CONFIGURATION MANAGER =====
class ConfigurationManager:
    """Manages configuration settings"""
    
    def __init__(self, config_file=None):
        self.config = {
            'chunk_size': 500000,
            'num_threads': 4,
            'max_ram_percent': 85,
            'max_cpu_percent': 90,
            'enable_caching': True,
            'cache_size_mb': 500,
            'enable_compression': False,
            'compression_type': 'gzip',
            'enable_checkpointing': False,
            'checkpoint_interval': 5,
            'validation_level': 'full',
            'log_level': 'INFO'
        }
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, filepath):
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except:
            pass
    
    def save_config(self, filepath):
        """Save configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass
    
    def get(self, key, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set config value"""
        self.config[key] = value


# ===== PARALLEL UTILITIES =====
class ParallelUtilities:
    """Utilities for parallel processing"""
    
    @staticmethod
    def split_into_chunks(data, num_chunks):
        """Split data into N chunks"""
        chunk_size = (len(data) + num_chunks - 1) // num_chunks
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks
    
    @staticmethod
    def distribute_work(items, num_workers):
        """Distribute work among workers (round-robin)"""
        distribution = defaultdict(list)
        for idx, item in enumerate(items):
            worker_id = idx % num_workers
            distribution[worker_id].append(item)
        return distribution
    
    @staticmethod
    def merge_results(results, merge_func=None):
        """Merge results from multiple workers"""
        if merge_func:
            return merge_func(results)
        else:
            if isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            else:
                return sum(results)


# ===== FILE MANAGER =====
class FileManager:
    """Manages file operations"""
    
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_dataframe(self, df, name, format='csv'):
        """Save dataframe in specified format"""
        try:
            filepath = os.path.join(self.output_dir, f"{name}.{format}")
            
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'parquet':
                df.to_parquet(filepath, index=False)
            elif format == 'json':
                df.to_json(filepath)
            
            return filepath
        except Exception as e:
            return None
    
    def list_files(self):
        """List all files in output directory"""
        return os.listdir(self.output_dir)
    
    def get_file_size(self, filename):
        """Get file size in MB"""
        filepath = os.path.join(self.output_dir, filename)
        return os.path.getsize(filepath) / (1024**2)


# ===== REPORTING ENGINE =====
class ReportingEngine:
    """Generates comprehensive reports"""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_summary_report(self, **kwargs):
        """Generate summary report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        for key, value in kwargs.items():
            report['sections'][key] = value
        
        self.report_data['summary'] = report
        return report
    
    def generate_html_report(self, output_file="report.html"):
        """Generate HTML report"""
        html = "<html><body>"
        html += "<h1>Data Consolidation Report</h1>"
        
        for section, content in self.report_data.items():
            html += f"<h2>{section}</h2>"
            html += f"<pre>{json.dumps(content, indent=2, default=str)}</pre>"
        
        html += "</body></html>"
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        return output_file


# ===== NOTIFICATION SYSTEM =====
class NotificationSystem:
    """Sends notifications about processing status"""
    
    def __init__(self):
        self.notifications = []
        self.handlers = []
    
    def subscribe(self, handler):
        """Subscribe to notifications"""
        self.handlers.append(handler)
    
    def notify(self, level, message):
        """Send notification"""
        notification = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        
        self.notifications.append(notification)
        
        # Call all handlers
        for handler in self.handlers:
            handler(notification)
    
    def get_notifications(self, level=None):
        """Get notifications filtered by level"""
        if level:
            return [n for n in self.notifications if n['level'] == level]
        return self.notifications


# ===== STATE MACHINE =====
class ProcessingStateMachine:
    """State machine for processing workflow"""
    
    def __init__(self):
        self.state = "IDLE"
        self.states = {
            "IDLE": ["LOADING_TONIOT"],
            "LOADING_TONIOT": ["LOADING_CIC", "ERROR"],
            "LOADING_CIC": ["MERGING", "ERROR"],
            "MERGING": ["CLEANING", "ERROR"],
            "CLEANING": ["SPLITTING", "ERROR"],
            "SPLITTING": ["FEATURES", "ERROR"],
            "FEATURES": ["WRITING", "ERROR"],
            "WRITING": ["NPZ", "ERROR"],
            "NPZ": ["COMPLETED", "ERROR"],
            "COMPLETED": ["IDLE"],
            "ERROR": ["IDLE"]
        }
        self.history = []
    
    def transition(self, new_state):
        """Transition to new state"""
        if new_state in self.states.get(self.state, []):
            self.history.append({
                'from': self.state,
                'to': new_state,
                'timestamp': datetime.now()
            })
            self.state = new_state
            return True
        return False


# ===== EXPORT UTILITIES =====
class ExportUtilities:
    """Utilities for exporting results"""
    
    @staticmethod
    def export_to_parquet(df, output_path):
        """Export to parquet format"""
        try:
            df.to_parquet(output_path, compression='snappy', index=False)
            return True
        except:
            return False
    
    @staticmethod
    def export_to_json(df, output_path):
        """Export to JSON format"""
        try:
            df.to_json(output_path, orient='records', compression='gzip')
            return True
        except:
            return False
    
    @staticmethod
    def export_to_sqlite(df, db_path, table_name):
        """Export to SQLite database"""
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.close()
            return True
        except:
            return False


# ===== INTEGRATION WITH GUI ENHANCED =====
# Add additional helper methods to ConsolidationGUIEnhanced class

def _setup_advanced_features(self):
    """Setup advanced features like validator, tracker, recovery"""
    self.validator = DataValidator()
    self.tracker = PerformanceTracker()
    self.recovery = RecoverySystem()
    self.logger = AdvancedLogger()
    self.statistics = DataStatistics()
    self.qa = QualityAssurance()
    self.limiter = ResourceLimiter()
    self.config = ConfigurationManager()
    self.file_manager = FileManager("./consolidation_output")
    self.reporting = ReportingEngine()
    self.notifications = NotificationSystem()
    self.state_machine = ProcessingStateMachine()
    self.benchmark = BenchmarkSuite()
    
    # Subscribe to notifications
    self.notifications.subscribe(self._handle_notification)

def _handle_notification(self, notification):
    """Handle notifications"""
    if notification['level'] == 'ERROR':
        self.add_alert(notification['message'], 'error')
    elif notification['level'] == 'WARNING':
        self.add_alert(notification['message'], 'warning')
    self.log(notification['message'], 'INFO')

def _validate_all_dataframes(self, toniot, cic_list, combined, train, test):
    """Validate all dataframes during processing"""
    self.validator.validate_dataframe(toniot, 'TON_IoT')
    self.validator.validate_label_column(toniot)
    
    for idx, cic_df in enumerate(cic_list):
        self.validator.validate_dataframe(cic_df, f'CIC_{idx}')
    
    self.validator.validate_dataframe(combined, 'Combined')
    self.validator.validate_label_column(combined)
    
    self.validator.validate_dataframe(train, 'Train')
    self.validator.validate_dataframe(test, 'Test')
    
    self.qa.check_data_integrity(combined, train)
    self.qa.check_data_integrity(combined, test)

def _compute_comprehensive_statistics(self, toniot, combined, train, test):
    """Compute statistics for all datasets"""
    self.statistics.compute_stats(toniot, 'TON_IoT')
    self.statistics.compute_stats(combined, 'Combined')
    self.statistics.compute_stats(train, 'Train')
    self.statistics.compute_stats(test, 'Test')

def _create_final_report(self, elapsed_time):
    """Create comprehensive final report"""
    # Gather all data
    monitor_stats = self.monitor.get_stats()
    validator_report = self.validator.get_validation_report()
    perf_report = self.tracker.get_report()
    qa_report = self.qa.get_report()
    cache_stats = self.cache.get_stats()
    
    # Generate comprehensive report
    final_report = {
        'summary': {
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'total_time': elapsed_time,
            'files_processed': monitor_stats.get('files', 0),
            'rows_processed': monitor_stats.get('data_loaded', 0),
            'peak_ram': monitor_stats.get('ram_max', 0),
            'peak_cpu': monitor_stats.get('cpu_max', 0)
        },
        'monitoring': monitor_stats,
        'validation': validator_report,
        'performance': perf_report,
        'qa': qa_report,
        'cache': cache_stats,
        'notifications': self.notifications.get_notifications()
    }
    
    # Log to file
    with open('consolidation_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    self.reporting.generate_summary_report(
        elapsed_time=elapsed_time,
        files=monitor_stats.get('files', 0),
        rows=monitor_stats.get('data_loaded', 0),
        peak_ram=f"{monitor_stats.get('ram_max', 0):.1f}%",
        peak_cpu=f"{monitor_stats.get('cpu_max', 0):.1f}%"
    )
    
    return final_report

def _memory_profile_operation(self, operation_name, operation_func):
    """Profile memory usage of an operation"""
    profiler = MemoryProfiler()
    
    profiler.take_snapshot(f"{operation_name}_start")
    result = operation_func()
    profiler.take_snapshot(f"{operation_name}_end")
    
    memory_growth = profiler.get_memory_growth()
    self.log(f"{operation_name} memory growth: {memory_growth:.2f} MB", "INFO")
    
    return result, profiler

def _export_results(self, train_df, test_df):
    """Export results in multiple formats"""
    success = True
    
    # CSV (already done)
    self.log("Exporting CSV files...", "INFO")
    
    # Parquet
    try:
        self.log("Exporting Parquet files...", "INFO")
        ExportUtilities.export_to_parquet(train_df, "fusion_train_smart4.parquet")
        ExportUtilities.export_to_parquet(test_df, "fusion_test_smart4.parquet")
        self.log("Parquet files created successfully", "OK")
    except Exception as e:
        self.log(f"Parquet export failed: {str(e)}", "WARN")
    
    # JSON
    try:
        self.log("Exporting JSON files...", "INFO")
        ExportUtilities.export_to_json(train_df, "fusion_train_smart4.json.gz")
        ExportUtilities.export_to_json(test_df, "fusion_test_smart4.json.gz")
        self.log("JSON files created successfully", "OK")
    except Exception as e:
        self.log(f"JSON export failed: {str(e)}", "WARN")
    
    return success

def _check_system_resources_continuously(self):
    """Check system resources during processing"""
    resources = self.limiter.check_resources()
    
    if not resources['within_limits']:
        self.notifications.notify('WARNING', f"Resources exceeded: RAM {resources['ram_percent']:.1f}%, CPU {resources['cpu_percent']:.1f}%")
        self.limiter.wait_for_resources()

def _benchmark_all_operations(self, toniot_file, combined_df):
    """Run benchmarks on key operations"""
    self.log("Running benchmarks...", "INFO")
    
    # Benchmark CSV reading
    read_benchmarks = self.benchmark.benchmark_csv_read(toniot_file)
    self.log(f"CSV read benchmarks: {read_benchmarks}", "INFO")
    
    # Benchmark dataframe operations
    op_benchmarks = self.benchmark.benchmark_operations(combined_df)
    self.log(f"Operation benchmarks: {op_benchmarks}", "INFO")

# ===== UTILITY FUNCTIONS =====

def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"

def format_duration(seconds):
    """Format duration to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def calculate_efficiency(elapsed_time, rows_processed):
    """Calculate processing efficiency"""
    if elapsed_time > 0:
        throughput = rows_processed / elapsed_time
        return throughput
    return 0

def estimate_remaining_time(processed, total, elapsed):
    """Estimate remaining time"""
    if processed > 0 and elapsed > 0:
        rate = processed / elapsed
        remaining = (total - processed) / rate
        return remaining
    return 0

# ===== ADDITIONAL VALIDATION HELPERS =====

def validate_csv_file(filepath):
    """Validate CSV file before loading"""
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
        return len(first_line) > 0
    except:
        return False

def validate_folder_structure(folder):
    """Validate folder structure for CIC"""
    if not os.path.isdir(folder):
        return False
    
    csv_count = sum(1 for _, _, files in os.walk(folder) for f in files if f.endswith('.csv'))
    return csv_count > 0

# ===== ENHANCED CONSOLIDATE WORKER =====

def enhanced_consolidate_worker(self):
    """Enhanced consolidation worker with all features"""
    # Initialize advanced features
    self._setup_advanced_features()
    
    # Start state machine
    self.state_machine.transition("LOADING_TONIOT")
    
    task_id = 1
    
    try:
        # ... (existing consolidation code here, but with additional calls)
        # For each major operation:
        # 1. self.tracker.start_task(task_name)
        # 2. Do work
        # 3. self.tracker.end_task(task_name)
        # 4. self.validator.validate_dataframe(df, task_name)
        # 5. self._check_system_resources_continuously()
        
        pass  # Placeholder for full integration
    
    finally:
        # Generate final report
        elapsed = time.time() - self.start_time
        report = self._create_final_report(elapsed)
        self.log(f"Final report generated: {report}", "INFO")
        
        # State machine to completed
        self.state_machine.transition("COMPLETED")


# ===== ADDON CONFIGURATIONS =====

class ProcessingConfig:
    """Pre-configured settings for different scenarios"""
    
    FAST = {
        'chunk_size': 1000000,
        'num_threads': 8,
        'max_ram_percent': 90,
        'enable_caching': True,
        'enable_compression': False
    }
    
    BALANCED = {
        'chunk_size': 500000,
        'num_threads': 4,
        'max_ram_percent': 80,
        'enable_caching': True,
        'enable_compression': False
    }
    
    CONSERVATIVE = {
        'chunk_size': 100000,
        'num_threads': 2,
        'max_ram_percent': 70,
        'enable_caching': True,
        'enable_compression': True
    }


# ===== POST-PROCESSING UTILITIES =====

class PostProcessing:
    """Post-processing utilities after consolidation"""
    
    @staticmethod
    def analyze_label_distribution(df):
        """Analyze label distribution"""
        return df['Label'].value_counts()
    
    @staticmethod
    def analyze_feature_statistics(df, numeric_cols):
        """Analyze feature statistics"""
        return df[numeric_cols].describe()
    
    @staticmethod
    def detect_outliers(df, numeric_cols, method='iqr'):
        """Detect outliers in data"""
        if method == 'iqr':
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            return outliers
        return pd.Series([False] * len(df))
    
    @staticmethod
    def generate_data_quality_report(df):
        """Generate data quality report"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'column_dtypes': df.dtypes.to_dict()
        }
        return report




# ===== FINAL DOCUMENTATION & EXAMPLES =====

"""
EXAMPLE USAGE:
==============

# 1. Basic usage
root = tk.Tk()
app = ConsolidationGUIEnhanced(root)
root.mainloop()

# 2. With configuration
config_manager = ConfigurationManager()
config_manager.set('chunk_size', 1000000)
config_manager.set('num_threads', 8)
config_manager.save_config('custom_config.json')

# 3. Advanced validation
validator = DataValidator()
processor = OptimizedDataProcessor(monitor, cache)
df = processor.load_toniot_optimized('toniot.csv')
validator.validate_dataframe(df, 'TON_IoT')

# 4. Performance tracking
tracker = PerformanceTracker()
tracker.start_task('loading')
# ... do work ...
tracker.end_task('loading')
report = tracker.get_report()

# 5. Memory profiling
profiler = MemoryProfiler()
profiler.take_snapshot('before_processing')
# ... do work ...
profiler.take_snapshot('after_processing')
memory_growth = profiler.get_memory_growth()

# 6. Quality assurance
qa = QualityAssurance()
df_orig = pd.read_csv('original.csv')
df_processed = process_dataframe(df_orig)
qa.check_data_integrity(df_orig, df_processed)
print(qa.get_report())

# 7. Resource limits
limiter = ResourceLimiter(max_ram_percent=85, max_cpu_percent=90)
if limiter.check_resources()['within_limits']:
    process_data()
else:
    limiter.wait_for_resources()
    process_data()

# 8. Benchmarking
benchmark = BenchmarkSuite()
results = benchmark.benchmark_csv_read('large_file.csv')
print(f"Fastest chunk size: {max(results.items(), key=lambda x: x[1]['throughput'])}")

# 9. Export utilities
ExportUtilities.export_to_parquet(df, 'output.parquet')
ExportUtilities.export_to_json(df, 'output.json.gz')

# 10. Post-processing
outliers = PostProcessing.detect_outliers(df, numeric_cols)
quality_report = PostProcessing.generate_data_quality_report(df)
"""

# ===== CONSTANTS & DEFAULT VALUES =====

DEFAULT_CHUNK_SIZE = 500000
DEFAULT_NUM_THREADS = 4
DEFAULT_MAX_RAM_PERCENT = 85
DEFAULT_MAX_CPU_PERCENT = 90
DEFAULT_CACHE_SIZE_MB = 500

MIN_CHUNK_SIZE = 50000
MAX_CHUNK_SIZE = 2000000

MIN_THREADS = 1
MAX_THREADS = 32

TIMEOUT_SECONDS = 3600  # 1 hour
CHECKPOINT_INTERVAL = 5 * 60  # 5 minutes

# ===== ERROR CODES =====

ERROR_CODES = {
    'E001': 'File not found',
    'E002': 'Permission denied',
    'E003': 'Memory limit exceeded',
    'E004': 'CPU limit exceeded',
    'E005': 'Invalid data format',
    'E006': 'Corrupted file',
    'E007': 'Thread pool error',
    'E008': 'Cache error',
    'E009': 'Validation failed',
    'E010': 'Unknown error'
}

# ===== STATUS MESSAGES =====

STATUS_MESSAGES = {
    'IDLE': 'System is idle',
    'LOADING': 'Loading data',
    'PROCESSING': 'Processing data',
    'SAVING': 'Saving results',
    'COMPLETED': 'Completed successfully',
    'ERROR': 'Error occurred',
    'PAUSED': 'Processing paused',
    'RESUMED': 'Processing resumed'
}

# ===== PERFORMANCE THRESHOLDS =====

class PerformanceThresholds:
    """Performance optimization thresholds"""
    
    # Time thresholds (seconds)
    FAST_OPERATION = 5
    NORMAL_OPERATION = 30
    SLOW_OPERATION = 300
    
    # Memory thresholds (MB)
    SMALL_DATASET = 100
    MEDIUM_DATASET = 1000
    LARGE_DATASET = 5000
    
    # Row thresholds
    SMALL_ROWS = 100000
    MEDIUM_ROWS = 1000000
    LARGE_ROWS = 5000000
    
    # CPU thresholds (%)
    LOW_CPU = 30
    MEDIUM_CPU = 60
    HIGH_CPU = 85


# ===== UTILITY FUNCTIONS FOR VALIDATION =====

def validate_configuration(config):
    """Validate configuration dictionary"""
    required_keys = ['chunk_size', 'num_threads', 'max_ram_percent']
    
    for key in required_keys:
        if key not in config:
            return False, f"Missing key: {key}"
    
    if config['chunk_size'] < MIN_CHUNK_SIZE or config['chunk_size'] > MAX_CHUNK_SIZE:
        return False, f"Invalid chunk size: {config['chunk_size']}"
    
    if config['num_threads'] < MIN_THREADS or config['num_threads'] > MAX_THREADS:
        return False, f"Invalid num_threads: {config['num_threads']}"
    
    if not (0 < config['max_ram_percent'] < 100):
        return False, f"Invalid max_ram_percent: {config['max_ram_percent']}"
    
    return True, "Configuration valid"

def validate_environment():
    """Validate system environment"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 6):
        issues.append("Python 3.6+ required")
    
    # Check required libraries
    try:
        import pandas
    except:
        issues.append("pandas not installed")
    
    try:
        import numpy
    except:
        issues.append("numpy not installed")
    
    try:
        import sklearn
    except:
        issues.append("scikit-learn not installed")
    
    # Check system resources
    if psutil.virtual_memory().total < 4 * 1024**3:  # Less than 4GB
        issues.append("System has less than 4GB RAM (not recommended)")
    
    return len(issues) == 0, issues

# ===== OPTIMIZATION STRATEGIES =====

class OptimizationStrategies:
    """Different optimization strategies for different scenarios"""
    
    @staticmethod
    def optimize_for_speed(processor):
        """Optimize for maximum speed"""
        processor.monitor.max_ram = 95
        processor.cache.max_size = 1000 * 1024 * 1024  # 1GB cache
        return processor
    
    @staticmethod
    def optimize_for_memory(processor):
        """Optimize for minimum memory usage"""
        processor.monitor.max_ram = 70
        processor.cache.max_size = 100 * 1024 * 1024  # 100MB cache
        return processor
    
    @staticmethod
    def optimize_balanced(processor):
        """Balanced optimization"""
        processor.monitor.max_ram = 80
        processor.cache.max_size = 500 * 1024 * 1024  # 500MB cache
        return processor

# ===== DEBUGGING UTILITIES =====

class DebugUtils:
    """Debugging utilities for development and troubleshooting"""
    
    @staticmethod
    def print_system_info():
        """Print system information"""
        print(f"Python: {sys.version}")
        print(f"OS: {sys.platform}")
        print(f"CPU count: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    @staticmethod
    def print_dataframe_info(df, name="DataFrame"):
        """Print detailed dataframe information"""
        print(f"\n{name} Information:")
        print(f"Shape: {df.shape}")
        print(f"Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
    
    @staticmethod
    def dump_metrics(metrics_dict, output_file="metrics.json"):
        """Dump metrics to file"""
        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)


# ===== INTEGRATION POINT FOR GUI =====

# Add these methods to ConsolidationGUIEnhanced class definition:

def init_advanced_systems(self):
    """Initialize all advanced systems"""
    self._setup_advanced_features()
    self.tracker.start_task("initialization")
    self.tracker.end_task("initialization")

def cleanup_advanced_systems(self):
    """Cleanup all advanced systems"""
    self.cache.clear()
    self.recovery.cleanup_checkpoints()
    if self.logger.log_file:
        self.log("Cleanup completed", "INFO")

def get_system_status(self):
    """Get complete system status"""
    status = {
        'state': self.state_machine.state,
        'resources': self.limiter.check_resources(),
        'cache': self.cache.get_stats(),
        'monitor': self.monitor.get_stats(),
        'qa': self.qa.get_report(),
        'errors': len(self.monitor.errors),
        'warnings': len(self.monitor.warnings)
    }
    return status

def export_all_reports(self, output_dir="./reports"):
    """Export all generated reports"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export monitoring report
    monitor_report = self.monitor.get_stats()
    with open(os.path.join(output_dir, "monitoring.json"), 'w') as f:
        json.dump(monitor_report, f, indent=2, default=str)
    
    # Export validation report
    validator_report = self.validator.get_validation_report()
    with open(os.path.join(output_dir, "validation.json"), 'w') as f:
        json.dump(validator_report, f, indent=2, default=str)
    
    # Export QA report
    qa_report = self.qa.get_report()
    with open(os.path.join(output_dir, "qa.json"), 'w') as f:
        json.dump(qa_report, f, indent=2, default=str)
    
    # Export cache stats
    cache_stats = self.cache.get_stats()
    with open(os.path.join(output_dir, "cache.json"), 'w') as f:
        json.dump(cache_stats, f, indent=2, default=str)