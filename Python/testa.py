"""
Optimiseur IA pour Processus Parallèles
========================================

Système d'optimisation en temps réel pour:
- a: nombre de processus (workers)
- b: taille du chunk par processus
- Score: lignes traitées par seconde (throughput)
- CPU/Mémoire: utilisation par processus

Objectif: Maximiser le score en 5 secondes
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque
import time
from datetime import datetime


# ============================================================
# 1. STRUCTURE DE DONNÉES
# ============================================================

@dataclass
class ProcessMetrics:
    process_id: int
    chunk_size: int
    memory_mb: float
    cpu_percent: float
    lines_processed: int = 0       # ✅ DÉFAUT AJOUTÉ!
    score: float =0.0
    
    def __post_init__(self):
        if self.lines_processed >= 0:
            self.score = self.lines_processed / 5.0  # Par 5 secondes


@dataclass
class SystemSnapshot:
    """Snapshot du système à un moment T"""
    timestamp: float
    num_workers: int  # a
    processes: List[ProcessMetrics]
    total_memory_mb: float  # Total utilisé par tous les processus
    total_cpu_percent: float  # Total CPU de tous les processus
    avg_score: float  # Score moyen (lines/s moyen)
    
    def __post_init__(self):
        if self.processes:
            scores = [p.score for p in self.processes]
            self.avg_score = np.mean(scores)
            self.total_memory_mb = sum(p.memory_mb for p in self.processes)
            self.total_cpu_percent = sum(p.cpu_percent for p in self.processes)


# ============================================================
# 2. MONITEUR (Capture toutes les 5 secondes)
# ============================================================

class ProcessMonitor:
    """Capture les métriques toutes les 5 secondes"""
    
    def __init__(self, capture_interval: float = 5.0):
        self.capture_interval = capture_interval
        self.snapshots: deque = deque(maxlen=100)  # Garder derniers 100 snapshots
        self.previous_lines = {}  # Tracker les lignes précédentes par processus
    
    def capture(self, processes_data: Dict) -> SystemSnapshot:
        """
        Capture l'état actuel du système
        
        processes_data: {
            'num_workers': int,
            'processes': [
                {
                    'id': int,
                    'chunk_size': int,
                    'memory_mb': float,
                    'cpu_percent': float,
                    'total_lines_processed': int  # Cumulé depuis le début
                },
                ...
            ]
        }
        """
        timestamp = time.time()
        num_workers = processes_data['num_workers']
        
        metrics_list = []
        
        for proc_data in processes_data['processes']:
            proc_id = proc_data['id']
            total_lines = proc_data['total_lines_processed']
            
            # Calculer les lignes traitées dans ce snapshot (depuis le dernier)
            if proc_id not in self.previous_lines:
                lines_in_interval = 0
            else:
                lines_in_interval = total_lines - self.previous_lines[proc_id]
            
            # Mettre à jour le tracker
            self.previous_lines[proc_id] = total_lines
            
            metric = ProcessMetrics(
                process_id=proc_id,
                chunk_size=proc_data['chunk_size'],
                memory_mb=proc_data['memory_mb'],
                cpu_percent=proc_data['cpu_percent'],
                lines_processed=max(0, lines_in_interval)  # Pas de négatif
            )
            metrics_list.append(metric)
        
        snapshot = SystemSnapshot(
            timestamp=timestamp,
            num_workers=num_workers,
            processes=metrics_list,
            total_memory_mb=0,  # Calculé dans __post_init__
            total_cpu_percent=0,  # Calculé dans __post_init__
            avg_score=0
        )
        
        self.snapshots.append(snapshot)
        return snapshot


# ============================================================
# 3. OPTIMISEUR IA (Contextual Bandit - LinUCB)
# ============================================================

class AIOptimizer:
    """
    Optimiseur IA basé sur Contextual Bandit (LinUCB)
    
    Choisit les meilleures valeurs pour:
    - Nombre de workers (a)
    - Taille du chunk (b)
    
    pour maximiser le score en continu
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 12,
        min_chunk_size: int = 50_000,
        max_chunk_size: int = 750_000,
        max_ram_percent: float = 90.0,
        alpha: float = 1.5  # Exploration vs exploitation
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_ram_percent = max_ram_percent
        self.alpha = alpha
        
        # Actions discrètes: (delta_workers, chunk_multiplier)
        # delta_workers: +2, +1, 0, -1, -2
        # chunk_multiplier: 0.7, 0.85, 1.0, 1.15, 1.3
        self.actions = [
            (2, 0.7),    # Augmenter workers, diminuer chunk
            (1, 0.85),   # Peu plus de workers, chunk réduit
            (0, 1.0),    # Même nombre workers, même chunk
            (-1, 1.15),  # Moins de workers, chunk augmenté
            (-2, 1.3),   # Beaucoup moins workers, chunk fort augmenté
        ]
        
        # LinUCB: A[a] et b[a] pour chaque action
        feature_dim = 5  # Features: RAM%, CPU%, Score%, Variance, Temps
        self.A = [np.eye(feature_dim, dtype=np.float64) for _ in self.actions]
        self.b = [np.zeros((feature_dim, 1), dtype=np.float64) for _ in self.actions]
        
        self.last_choice = None
        self.last_x = None
        self.history = []
        self.episode = 0
    
    def extract_features(self, snapshot: SystemSnapshot, max_ram_mb: float) -> np.ndarray:
        """
        Extraire features normalisées (0-1) du snapshot
        
        Features:
        1. RAM usage % (0-1)
        2. CPU usage % (0-1)
        3. Score variation (coefficient de variation)
        4. Score absolue (normalisée)
        5. Nombre de workers normalisé
        """
        if not snapshot.processes:
            return np.zeros(5)
        
        # Feature 1: RAM %
        ram_percent = min(snapshot.total_memory_mb / max_ram_mb, 1.0)
        
        # Feature 2: CPU %
        cpu_percent = min(snapshot.total_cpu_percent / 100.0, 1.0)
        
        # Feature 3: Variation du score (stabilité)
        scores = [p.score for p in snapshot.processes]
        score_variation = np.std(scores) / (np.mean(scores) + 1e-6)
        score_variation = min(score_variation, 1.0)
        
        # Feature 4: Score absolu (normalisé, supposons max 10k lines/s)
        avg_score_norm = min(snapshot.avg_score / 10_000, 1.0)
        
        # Feature 5: Workers normalisé
        workers_norm = snapshot.num_workers / self.max_workers
        
        features = np.array([
            ram_percent,
            cpu_percent,
            1.0 - score_variation,  # 1 - variation = stabilité
            avg_score_norm,
            workers_norm
        ], dtype=np.float64)
        
        return features
    
    def choose_action(
        self,
        snapshot: SystemSnapshot,
        current_workers: int,
        current_chunk_size: int,
        max_ram_mb: float
    ) -> Tuple[int, int, str]:
        """
        Choisir les meilleures valeurs pour workers et chunk_size
        
        Returns:
            (new_workers, new_chunk_size, reason)
        """
        # Extraire features
        x = self.extract_features(snapshot, max_ram_mb)
        x = x.reshape(-1, 1).astype(np.float64)
        
        # Choisir action avec LinUCB
        best_action_idx = 0
        best_ucb = -1e18
        
        for i in range(len(self.actions)):
            A_inv = np.linalg.inv(self.A[i] + 1e-6 * np.eye(5))
            theta = A_inv @ self.b[i]
            
            # UCB = theta^T x + alpha * sqrt(x^T A_inv x)
            exploitation = float((theta.T @ x)[0, 0])
            exploration = self.alpha * np.sqrt(max(float((x.T @ A_inv @ x)[0, 0]), 0))
            ucb = exploitation + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action_idx = i
        
        self.last_choice = best_action_idx
        self.last_x = x
        
        delta_workers, chunk_mult = self.actions[best_action_idx]
        
        # Appliquer l'action avec contraintes
        new_workers = current_workers + delta_workers
        new_workers = max(self.min_workers, min(self.max_workers, new_workers))
        
        new_chunk_size = int(current_chunk_size * chunk_mult)
        new_chunk_size = max(self.min_chunk_size, min(self.max_chunk_size, new_chunk_size))
        
        reason = f"Action[{best_action_idx}]: Δworkers={delta_workers:+d}, chunk×={chunk_mult:.2f}"
        
        return new_workers, new_chunk_size, reason
    
    def update(self, reward: float):
        """
        Update LinUCB avec la récompense observée
        
        reward = score (lignes/s) ou combinaison de scores
        """
        if self.last_choice is None or self.last_x is None:
            return
        
        i = self.last_choice
        x = self.last_x
        
        # Update: A[i] = A[i] + x @ x.T
        self.A[i] = self.A[i] + (x @ x.T)
        # Update: b[i] = b[i] + reward * x
        self.b[i] = self.b[i] + float(reward) * x
        
        self.episode += 1
        self.history.append({
            'episode': self.episode,
            'action_idx': i,
            'reward': reward,
            'ucb': best_ucb if 'best_ucb' in locals() else 0
        })


# ============================================================
# 4. MANAGER D'OPTIMISATION
# ============================================================

class OptimizationManager:
    """Manager qui orchestre tout"""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 12,
        min_chunk_size: int = 50_000,
        max_chunk_size: int = 750_000,
        max_ram_mb: float = 32_000,  # 32GB
        capture_interval: float = 5.0
    ):
        self.monitor = ProcessMonitor(capture_interval)
        self.optimizer = AIOptimizer(
            min_workers=min_workers,
            max_workers=max_workers,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            max_ram_percent=90.0
        )
        self.max_ram_mb = max_ram_mb
        self.capture_interval = capture_interval
        
        self.current_workers = 1
        self.current_chunk_size = min_chunk_size
        self.last_snapshot = None
    
    def optimize_cycle(self, processes_data: Dict) -> Dict:
        """
        Exécuter un cycle d'optimisation (toutes les 5 secondes)
        
        Returns:
            {
                'recommended_workers': int,
                'recommended_chunk_size': int,
                'reason': str,
                'score': float,
                'metrics': {...}
            }
        """
        # Capturer l'état actuel
        snapshot = self.monitor.capture(processes_data)
        self.last_snapshot = snapshot
        
        # Choisir la prochaine action
        new_workers, new_chunk_size, reason = self.optimizer.choose_action(
            snapshot,
            self.current_workers,
            self.current_chunk_size,
            self.max_ram_mb
        )
        
        # Calculer la récompense
        # Récompense = score moyen normalisé
        reward = snapshot.avg_score / 10_000  # Normalisé par max expected
        
        # Appliquer le reward au modèle
        self.optimizer.update(reward)
        
        # Mettre à jour l'état
        self.current_workers = new_workers
        self.current_chunk_size = new_chunk_size
        
        return {
            'recommended_workers': new_workers,
            'recommended_chunk_size': new_chunk_size,
            'reason': reason,
            'score': snapshot.avg_score,
            'metrics': {
                'num_processes': len(snapshot.processes),
                'total_memory_mb': snapshot.total_memory_mb,
                'total_cpu_percent': snapshot.total_cpu_percent,
                'timestamp': datetime.fromtimestamp(snapshot.timestamp)
            }
        }
    
    def print_status(self, optimization_result: Dict):
        """Afficher l'état d'optimisation"""
        print(f"\n{'='*60}")
        print(f"⏱️  [{optimization_result['metrics']['timestamp']}]")
        print(f"{'='*60}")
        print(f"📊 SCORE: {optimization_result['score']:.1f} lines/s")
        print(f"👷 Workers: {optimization_result['recommended_workers']}")
        print(f"📦 Chunk Size: {optimization_result['recommended_chunk_size']:,}")
        print(f"💾 Memory: {optimization_result['metrics']['total_memory_mb']:.1f} MB")
        print(f"⚙️  CPU: {optimization_result['metrics']['total_cpu_percent']:.1f}%")
        print(f"🤖 {optimization_result['reason']}")
        print(f"{'='*60}\n")


# ============================================================
# 5. EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    import random
    
    # Initialiser le manager
    manager = OptimizationManager(
        min_workers=1,
        max_workers=12,
        min_chunk_size=50_000,
        max_chunk_size=750_000,
        max_ram_mb=32_000
    )
    
    print("🚀 Démarrage du système d'optimisation IA")
    print(f"   Intervalle de capture: 5 secondes")
    print(f"   Max workers: {manager.optimizer.max_workers}")
    print(f"   Max chunk: {manager.optimizer.max_chunk_size:,}")
    print()
    
    # Simuler 10 cycles d'optimisation (50 secondes)
    for cycle in range(10):
        print(f"\n📍 Cycle {cycle + 1}/10")
        
        # Simuler des données de processus
        num_workers = manager.current_workers
        processes_data = {
            'num_workers': num_workers,
            'processes': []
        }
        
        for i in range(num_workers):
            # Simuler: si chunk_size augmente, performance augmente mais avec coût
            base_lines = 1000 + (manager.current_chunk_size / 100_000) * 2000
            lines_noise = random.gauss(0, base_lines * 0.1)  # 10% de bruit
            
            processes_data['processes'].append({
                'id': i,
                'chunk_size': manager.current_chunk_size,
                'memory_mb': (manager.current_chunk_size / 100_000) * 50,  # ~50MB per 100k chunk
                'cpu_percent': 20 + (manager.current_chunk_size / 100_000) * 15,
                'total_lines_processed': int(base_lines + lines_noise)  # Cumulé
            })
        
        # Lancer l'optimisation
        result = manager.optimize_cycle(processes_data)
        
        # Afficher le résultat
        manager.print_status(result)
        
        # Simuler le délai de 5 secondes
        # time.sleep(0.1)  # Réduit pour la démo
    
    print("\n✅ Optimisation complétée!")
    print(f"\n📈 Historique des décisions:")
    for i, entry in enumerate(manager.optimizer.history[-5:], 1):
        print(f"   Episode {entry['episode']}: Action {entry['action_idx']}, Reward: {entry['reward']:.3f}")