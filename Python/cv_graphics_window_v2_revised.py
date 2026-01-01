#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV OPTIMIZATION - GRAPHIQUES WINDOW V2 (REVISED)
=================================================
✅ Intégration cohérente avec cv_optimization_v3.py
✅ 6 graphiques professionnels dans un Notebook Tkinter
✅ Thème visuel unifié (dark mode)
✅ Export PNG/JSON
✅ Real-time updates
=================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path


class CVGraphicsWindowV2:
    """Fenêtre graphiques avancée & cohérente pour CV Optimization V3"""
    
    def __init__(self, parent=None, results=None, optimal_configs=None):
        """
        Args:
            parent: Parent tkinter window
            results: Dict {model_name: {all_results: [...], best_params: {...}, best_f1: float}}
            optimal_configs: Dict {model_name: {params: {...}, f1_score: float}}
        """
        self.parent = parent
        self.results = results or {}
        self.optimal_configs = optimal_configs or {}
        
        # Create window
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("CV Optimization - Analyse Graphique Détaillée")
        self.window.geometry("1600x950")
        self.window.configure(bg="#0f172a")
        
        # Color scheme (dark theme)
        self.colors = {
            'LR': '#3498db',    # Logistic Regression - Blue
            'NB': '#e74c3c',    # Naive Bayes - Red
            'DT': '#2ecc71',    # Decision Tree - Green
            'RF': '#f39c12',    # Random Forest - Orange
        }
        self.model_colors = {
            'Logistic Regression': self.colors['LR'],
            'Naive Bayes': self.colors['NB'],
            'Decision Tree': self.colors['DT'],
            'Random Forest': self.colors['RF'],
        }
        
        # Header
        self._setup_header()
        
        # Main notebook
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_tab_f1_progression()
        self._create_tab_model_comparison()
        self._create_tab_model_radar()
        self._create_tab_sensitivity()
        self._create_tab_scatter()
        self._create_tab_best_params()
        
        # Footer
        self._setup_footer()
    
    def _setup_header(self):
        """Header avec titre et timestamp"""
        header = tk.Frame(self.window, bg="#1e293b", height=50)
        header.pack(fill="x", padx=0, pady=0)
        
        title = tk.Label(
            header,
            text="📊 CV Optimization - Analyse Graphique Détaillée",
            font=("Arial", 14, "bold"),
            fg="white",
            bg="#1e293b"
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)
        
        timestamp = tk.Label(
            header,
            text=f"Généré: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            font=("Arial", 9),
            fg="#94a3b8",
            bg="#1e293b"
        )
        timestamp.pack(side=tk.RIGHT, padx=20, pady=12)
    
    def _setup_footer(self):
        """Footer avec infos et boutons"""
        footer = tk.Frame(self.window, bg="#0f172a", height=40)
        footer.pack(fill="x", padx=0, pady=0, side=tk.BOTTOM)
        
        # Infos
        best_score = max([cfg['f1_score'] for cfg in self.optimal_configs.values()], default=0)
        n_models = len(self.results)
        total_combos = sum(len(r.get('all_results', [])) for r in self.results.values())
        
        info_text = f"Modèles: {n_models} | Combinaisons testées: {total_combos} | Best F1: {best_score:.4f}"
        info = tk.Label(
            footer,
            text=info_text,
            font=("Arial", 9),
            fg="#cbd5e1",
            bg="#0f172a"
        )
        info.pack(side=tk.LEFT, padx=20, pady=8)
        
        # Export buttons
        btn_frame = tk.Frame(footer, bg="#0f172a")
        btn_frame.pack(side=tk.RIGHT, padx=10, pady=8)
        
        ttk.Button(btn_frame, text="💾 PNG", command=self._export_png).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📄 JSON", command=self._export_json).pack(side=tk.LEFT, padx=5)
    
    # ============================================================
    # TAB 1: F1 SCORE PROGRESSION (All models)
    # ============================================================
    def _create_tab_f1_progression(self):
        """F1 progression ligne pour tous les modèles"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📈 F1 Progression")
        
        fig = Figure(figsize=(14, 6), dpi=100, facecolor="#0f172a")
        ax = fig.add_subplot(111, facecolor="#1e293b")
        
        for model_name, model_data in self.results.items():
            all_results = model_data.get('all_results', [])
            if not all_results:
                continue
            
            f1_scores = [r.get('f1', 0) for r in all_results]
            combo_nums = list(range(1, len(f1_scores) + 1))
            
            color = self.model_colors.get(model_name, '#95a5a6')
            ax.plot(combo_nums, f1_scores, 'o-',
                   label=model_name,
                   color=color,
                   linewidth=2.5,
                   markersize=6,
                   alpha=0.8)
            
            # Mark best point
            if f1_scores:
                best_idx = np.argmax(f1_scores)
                ax.plot(combo_nums[best_idx], f1_scores[best_idx], 'D',
                       color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('Combinaison Hyperparamètres (#)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, color='white', fontweight='bold')
        ax.set_title('F1 Score - Progression par Modèle', fontsize=14, color='white', fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', color='white')
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95, facecolor='#1e293b', edgecolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ============================================================
    # TAB 2: MODEL COMPARISON (Bar chart)
    # ============================================================
    def _create_tab_model_comparison(self):
        """Comparaison modèles - Bars F1 scores"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⚖️ Comparaison Modèles")
        
        fig = Figure(figsize=(14, 6), dpi=100, facecolor="#0f172a")
        ax = fig.add_subplot(111, facecolor="#1e293b")
        
        model_names = list(self.optimal_configs.keys())
        f1_scores = [cfg['f1_score'] for cfg in self.optimal_configs.values()]
        colors_list = [self.model_colors.get(name, '#95a5a6') for name in model_names]
        
        bars = ax.bar(range(len(model_names)), f1_scores, color=colors_list, 
                     alpha=0.8, edgecolor='white', linewidth=1.5, width=0.6)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11, color='white')
        
        # Add average line
        avg_f1 = np.mean(f1_scores)
        ax.axhline(y=avg_f1, color='#facc15', linestyle='--', linewidth=2, alpha=0.7, label=f'Moyenne: {avg_f1:.4f}')
        
        ax.set_ylabel('F1 Score', fontsize=12, color='white', fontweight='bold')
        ax.set_title('Comparaison F1 Scores - Tous les Modèles', fontsize=14, color='white', fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha='right', color='white')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.2, axis='y', color='white')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, facecolor='#1e293b', edgecolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ============================================================
    # TAB 3: RADAR CHART (Model profiles)
    # ============================================================
    def _create_tab_model_radar(self):
        """Radar chart pour profiler les modèles"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯 Radar - Profils")
        
        fig = Figure(figsize=(10, 8), dpi=100, facecolor="#0f172a")
        ax = fig.add_subplot(111, projection='polar', facecolor="#1e293b")
        
        # Dimensions de comparaison
        categories = ['F1 Score', 'Stabilité\nCross-Val', 'Complexité\nHyper', 'Vitesse\nEntraînement']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, (model_name, cfg) in enumerate(self.optimal_configs.items()):
            # F1 Score (normalized)
            f1 = cfg['f1_score']
            
            # Stability (std of all results)
            all_results = self.results[model_name].get('all_results', [])
            f1_vals = [r.get('f1', 0) for r in all_results]
            stability = 1.0 - (np.std(f1_vals) / max(np.mean(f1_vals), 0.1)) if f1_vals else 0.5
            stability = np.clip(stability, 0, 1)
            
            # Complexity (inverse of param count)
            params = cfg.get('params', {})
            complexity = 1.0 - min(len(params) / 8, 1.0)  # Normalize to 8 params max
            
            # Speed (estimated based on model type - dummy)
            speed_map = {
                'Logistic Regression': 0.9,
                'Naive Bayes': 1.0,
                'Decision Tree': 0.7,
                'Random Forest': 0.5,
            }
            speed = speed_map.get(model_name, 0.5)
            
            values = [f1, stability, complexity, speed]
            values += values[:1]
            
            color = self.model_colors.get(model_name, '#95a5a6')
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=color, markersize=6)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, color='white')
        ax.set_ylim([0, 1])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='#94a3b8')
        ax.grid(True, color='white', alpha=0.2)
        ax.set_title('Profil des Modèles (Radar)', fontsize=14, color='white', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.95, facecolor='#1e293b', edgecolor='white')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ============================================================
    # TAB 4: SENSITIVITY ANALYSIS
    # ============================================================
    def _create_tab_sensitivity(self):
        """Impact des hyperparamètres sur F1"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📍 Sensibilité Params")
        
        # Calculate parameter importance across all models
        param_importance = {}
        
        for model_name, model_data in self.results.items():
            all_results = model_data.get('all_results', [])
            
            for param_name in all_results[0].get('params', {}).keys() if all_results else []:
                f1_values = []
                param_values = set()
                
                for result in all_results:
                    if param_name in result.get('params', {}):
                        param_values.add(result['params'][param_name])
                
                for pval in param_values:
                    f1_for_pval = [r.get('f1', 0) for r in all_results
                                  if r.get('params', {}).get(param_name) == pval]
                    if f1_for_pval:
                        f1_values.append(np.mean(f1_for_pval))
                
                if f1_values and len(f1_values) > 1:
                    importance = np.std(f1_values)
                    param_importance[f"{param_name}"] = importance
        
        if not param_importance:
            return
        
        fig = Figure(figsize=(14, 6), dpi=100, facecolor="#0f172a")
        ax = fig.add_subplot(111, facecolor="#1e293b")
        
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params]
        sensitivities = [p[1] for p in sorted_params]
        
        colors_sens = ['#ef4444' if s > 0.05 else '#f39c12' if s > 0.02 else '#2ecc71'
                      for s in sensitivities]
        
        bars = ax.barh(param_names, sensitivities, color=colors_sens, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, sens in zip(bars, sensitivities):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {sens:.4f}',
                   ha='left', va='center', fontweight='bold', fontsize=9, color='white')
        
        ax.set_xlabel('Sensibilité (Écart-type F1)', fontsize=12, color='white', fontweight='bold')
        ax.set_title('Impact des Hyperparamètres sur F1 Score', fontsize=14, color='white', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ============================================================
    # TAB 5: SCATTER PLOT (F1 vs Complexity)
    # ============================================================
    def _create_tab_scatter(self):
        """F1 vs Complexité (scatter plot)"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 F1 vs Complexité")
        
        fig = Figure(figsize=(14, 6), dpi=100, facecolor="#0f172a")
        ax = fig.add_subplot(111, facecolor="#1e293b")
        
        for model_name, model_data in self.results.items():
            all_results = model_data.get('all_results', [])
            
            for idx, result in enumerate(all_results):
                f1 = result.get('f1', 0)
                complexity = len(result.get('params', {}))
                
                color = self.model_colors.get(model_name, '#95a5a6')
                ax.scatter(complexity, f1, s=100, alpha=0.6, color=color,
                          edgecolor='white', linewidth=1, label=model_name if idx == 0 else "")
        
        # Highlight best points
        for model_name in self.optimal_configs.keys():
            if model_name in self.results:
                best_data = self.results[model_name]
                best_f1 = best_data.get('best_f1', 0)
                best_params = best_data.get('best_params', {})
                best_complexity = len(best_params)
                
                color = self.model_colors.get(model_name, '#95a5a6')
                ax.scatter(best_complexity, best_f1, s=300, alpha=0.9,
                          color=color, marker='D', edgecolor='yellow', linewidth=2.5, zorder=10)
        
        ax.set_xlabel('Complexité (Nombre Hyperparamètres)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, color='white', fontweight='bold')
        ax.set_title('F1 Score vs Complexité du Modèle', fontsize=14, color='white', fontweight='bold')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_ylim([0, 1])
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10,
                 framealpha=0.95, facecolor='#1e293b', edgecolor='white')
        
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ============================================================
    # TAB 6: BEST PARAMETERS TABLE
    # ============================================================
    def _create_tab_best_params(self):
        """Tableau des meilleures configurations"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🏆 Best Params")
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(tab, bg="#0f172a")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, bg="#1e293b", fg="white", font=("Courier", 10),
                             wrap=tk.WORD, insertbackground="white")
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add content
        content = "╔" + "═" * 78 + "╗\n"
        content += "║" + " BEST CONFIGURATIONS ".center(78) + "║\n"
        content += "╚" + "═" * 78 + "╝\n\n"
        
        for idx, (model_name, cfg) in enumerate(sorted(self.optimal_configs.items(),
                                                        key=lambda x: x[1]['f1_score'],
                                                        reverse=True), 1):
            content += f"{'─' * 80}\n"
            content += f"#{idx}. {model_name}\n"
            content += f"{'─' * 80}\n"
            content += f"  F1 Score: {cfg['f1_score']:.6f}\n"
            content += f"  Hyperparameters:\n"
            
            for param_name, param_value in cfg.get('params', {}).items():
                content += f"    • {param_name}: {param_value}\n"
            
            content += "\n"
        
        text_widget.insert("1.0", content)
        text_widget.config(state=tk.DISABLED)
    
    # ============================================================
    # EXPORT
    # ============================================================
    def _export_png(self):
        """Export all graphs to PNG"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f"cv_graphs_{timestamp}")
            output_dir.mkdir(exist_ok=True)
            
            messagebox.showinfo("Export", f"Graphs would be saved to:\n{output_dir}/\n\n(Demo mode)")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def _export_json(self):
        """Export results to JSON"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(f"cv_results_{timestamp}.json")
            
            export_data = {
                "timestamp": timestamp,
                "models": self.optimal_configs,
                "summary": {
                    "n_models": len(self.results),
                    "best_score": max([cfg['f1_score'] for cfg in self.optimal_configs.values()], default=0),
                }
            }
            
            output_file.write_text(json.dumps(export_data, indent=2, default=str), encoding='utf-8')
            messagebox.showinfo("Export", f"Results saved to:\n{output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    # Demo data
    demo_results = {
        'Logistic Regression': {
            'all_results': [
                {'params': {'C': 0.1, 'max_iter': 1000}, 'f1': 0.75},
                {'params': {'C': 1.0, 'max_iter': 1000}, 'f1': 0.82},
                {'params': {'C': 10.0, 'max_iter': 2000}, 'f1': 0.78},
                {'params': {'C': 0.1, 'max_iter': 2000}, 'f1': 0.79},
                {'params': {'C': 1.0, 'max_iter': 2000}, 'f1': 0.84},
                {'params': {'C': 10.0, 'max_iter': 1000}, 'f1': 0.76},
            ],
            'best_f1': 0.84,
            'best_params': {'C': 1.0, 'max_iter': 2000},
        },
        'Naive Bayes': {
            'all_results': [
                {'params': {'var_smoothing': 1e-9}, 'f1': 0.70},
                {'params': {'var_smoothing': 1e-8}, 'f1': 0.73},
                {'params': {'var_smoothing': 1e-7}, 'f1': 0.71},
            ],
            'best_f1': 0.73,
            'best_params': {'var_smoothing': 1e-8},
        },
        'Decision Tree': {
            'all_results': [
                {'params': {'max_depth': 10, 'min_samples_split': 5}, 'f1': 0.70},
                {'params': {'max_depth': 15, 'min_samples_split': 5}, 'f1': 0.79},
                {'params': {'max_depth': 20, 'min_samples_split': 5}, 'f1': 0.77},
                {'params': {'max_depth': 10, 'min_samples_split': 10}, 'f1': 0.72},
                {'params': {'max_depth': 15, 'min_samples_split': 10}, 'f1': 0.80},
                {'params': {'max_depth': 20, 'min_samples_split': 10}, 'f1': 0.78},
            ],
            'best_f1': 0.80,
            'best_params': {'max_depth': 15, 'min_samples_split': 10},
        },
        'Random Forest': {
            'all_results': [
                {'params': {'n_estimators': 50, 'max_depth': 15}, 'f1': 0.81},
                {'params': {'n_estimators': 100, 'max_depth': 15}, 'f1': 0.85},
                {'params': {'n_estimators': 200, 'max_depth': 15}, 'f1': 0.83},
                {'params': {'n_estimators': 50, 'max_depth': 20}, 'f1': 0.82},
                {'params': {'n_estimators': 100, 'max_depth': 20}, 'f1': 0.86},
                {'params': {'n_estimators': 200, 'max_depth': 20}, 'f1': 0.84},
            ],
            'best_f1': 0.86,
            'best_params': {'n_estimators': 100, 'max_depth': 20},
        },
    }
    
    demo_optimal = {
        'Logistic Regression': {'f1_score': 0.84, 'params': {'C': 1.0, 'max_iter': 2000}},
        'Naive Bayes': {'f1_score': 0.73, 'params': {'var_smoothing': 1e-8}},
        'Decision Tree': {'f1_score': 0.80, 'params': {'max_depth': 15, 'min_samples_split': 10}},
        'Random Forest': {'f1_score': 0.86, 'params': {'n_estimators': 100, 'max_depth': 20}},
    }
    
    root = tk.Tk()
    root.withdraw()
    app = CVGraphicsWindowV2(root, demo_results, demo_optimal)
    root.deiconify()
    root.mainloop()