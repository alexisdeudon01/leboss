#!/usr/bin/env python3
"""
Test rapide des splits Decision Tree pour vérifier l'overfitting.
Charge preprocessed_dataset.npz/tensor_data.npz et évalue F1 sur plusieurs tailles de test.
Amélioration: multiple runs + std + graphique + comparaison CV
"""
import numpy as np
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_npz():
    """Charger NPZ avec fallback"""
    npz_file = "preprocessed_dataset.npz" if os.path.exists("preprocessed_dataset.npz") else "tensor_data.npz"
    if not os.path.exists(npz_file):
        print(f"❌ {npz_file} non trouvé!")
        return None, None
    print(f"✅ Chargement {npz_file}...")
    try:
        data = np.load(npz_file, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        print(f"✅ Données chargées: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None, None

def load_cv_splits():
    """Charger les splits optimaux du CV"""
    fname = "cv_optimal_splits_kfold.json" if os.path.exists("cv_optimal_splits_kfold.json") else "cv_optimal_splits.json"
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        dt_config = splits.get('Decision Tree', {})
        print(f"✅ CV splits chargés")
        if dt_config:
            print(f"   Decision Tree CV: {dt_config['train_size']*100:.0f}% train, F1={dt_config['f1_score']:.4f} ± {dt_config.get('f1_std', 0):.4f}")
        return dt_config
    except FileNotFoundError:
        print(f"⚠️  {fname} non trouvé")
        return {}
    except Exception as e:
        print(f"⚠️  Erreur chargement CV splits: {e}")
        return {}

def main():
    print("="*70)
    print("TEST DECISION TREE OVERFITTING DETECTION")
    print("="*70)
    print()
    
    # Charger données
    X, y = load_npz()
    if X is None:
        return
    
    print()
    
    # Charger CV results
    cv_dt_config = load_cv_splits()
    cv_f1 = cv_dt_config.get('f1_score', None)
    cv_std = cv_dt_config.get('f1_std', None)
    
    print()
    print("="*70)
    print("TEST SUR DIFFÉRENTS TEST_SIZE (5 runs chacun)")
    print("="*70)
    print()
    
    test_sizes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
    results = {
        'test_size': [],
        'train_size': [],
        'f1_mean': [],
        'f1_std': [],
        'recall_mean': [],
        'precision_mean': [],
        'avt_ms_mean': [],
    }
    
    print(f"{'test_size':>12} | {'train_size':>12} | {'F1 (mean±std)':>18} | {'Recall':>8} | {'Precision':>10}")
    print("-"*85)
    
    for test_size in test_sizes:
        train_size = 1 - test_size
        f1_runs = []
        recall_runs = []
        precision_runs = []
        avt_runs = []
        
        for run in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42+run, stratify=y
            )
            
            model = DecisionTreeClassifier(random_state=42+run)
            
            start = time.time()
            model.fit(X_train, y_train)
            
            start_pred = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start_pred
            
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            avt_ms = (pred_time / len(X_test) * 1000) if len(X_test) > 0 else 0
            
            f1_runs.append(f1)
            recall_runs.append(recall)
            precision_runs.append(precision)
            avt_runs.append(avt_ms)
        
        f1_mean = np.mean(f1_runs)
        f1_std = np.std(f1_runs)
        recall_mean = np.mean(recall_runs)
        precision_mean = np.mean(precision_runs)
        avt_mean = np.mean(avt_runs)
        
        results['test_size'].append(test_size)
        results['train_size'].append(train_size)
        results['f1_mean'].append(f1_mean)
        results['f1_std'].append(f1_std)
        results['recall_mean'].append(recall_mean)
        results['precision_mean'].append(precision_mean)
        results['avt_ms_mean'].append(avt_mean)
        
        print(f"{test_size:>12.2f} | {train_size:>12.2f} | {f1_mean:>7.4f}±{f1_std:>6.4f}  | {recall_mean:>8.4f} | {precision_mean:>10.4f}")
    
    print()
    print("="*70)
    print("ANALYSE OVERFITTING")
    print("="*70)
    print()
    
    # Détection overfitting
    f1_array = np.array(results['f1_mean'])
    min_idx = np.argmin(f1_array)
    max_idx = np.argmax(f1_array)
    f1_min = f1_array[min_idx]
    f1_max = f1_array[max_idx]
    f1_drop = f1_max - f1_min
    
    print(f"F1 min:     {f1_min:.4f} (test_size={results['test_size'][min_idx]:.2f}, {results['train_size'][min_idx]*100:.0f}% train)")
    print(f"F1 max:     {f1_max:.4f} (test_size={results['test_size'][max_idx]:.2f}, {results['train_size'][max_idx]*100:.0f}% train)")
    print(f"F1 drop:    {f1_drop:.4f} ({f1_drop/f1_min*100:.1f}% relative)")
    print()
    
    # Comparaison avec CV si disponible
    if cv_f1 is not None:
        print("COMPARAISON AVEC CV RESULTS")
        print("-"*70)
        print(f"CV F1:      {cv_f1:.4f} (±{cv_std:.4f})")
        print(f"Test F1:    {f1_mean:.4f} (meilleur)")
        print()
        
        if f1_drop > 0.10:
            print("🚨 OVERFITTING DÉTECTÉ!")
            print(f"   - F1 varie significativement ({f1_drop:.4f})")
            print(f"   - F1 augmente quand train% augmente (petit test set)")
            print(f"   - Recommandation: REJETER Decision Tree ou limiter à test_size >= {results['test_size'][min_idx]:.2f}")
        elif f1_drop > 0.05:
            print("⚠️  POSSIBLE OVERFITTING")
            print(f"   - F1 varie modérément ({f1_drop:.4f})")
            print(f"   - Recommandation: Utiliser avec précaution")
        else:
            print("✅ BON GENERALIZATION")
            print(f"   - F1 varie peu ({f1_drop:.4f})")
            print(f"   - Recommandation: Decision Tree sûr")
    
    print()
    print("="*70)
    print("GRAPHIQUE")
    print("="*70)
    print()
    
    # Créer graphique
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Decision Tree Overfitting Detection", fontsize=14, fontweight='bold')
    
    # F1 vs test_size
    ax = axes[0]
    ts_pct = [t*100 for t in results['test_size']]
    ax.errorbar(ts_pct, results['f1_mean'], yerr=results['f1_std'], 
                marker='o', markersize=8, linewidth=2, capsize=5, color='#3498db')
    if cv_f1 is not None:
        ax.axhline(cv_f1, color='#e74c3c', linestyle='--', linewidth=2, label=f"CV F1={cv_f1:.4f}")
        ax.fill_between(ts_pct, cv_f1-cv_std, cv_f1+cv_std, alpha=0.2, color='#e74c3c')
    ax.set_xlabel("Test Size (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Score", fontsize=11, fontweight='bold')
    ax.set_title("F1 vs Test Size", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if cv_f1 is not None:
        ax.legend()
    ax.set_ylim([0, 1])
    
    # Recall vs test_size
    ax = axes[1]
    ax.plot(ts_pct, results['recall_mean'], marker='s', markersize=8, linewidth=2, color='#e74c3c')
    ax.set_xlabel("Test Size (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Recall Score", fontsize=11, fontweight='bold')
    ax.set_title("Recall vs Test Size", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Precision vs test_size
    ax = axes[2]
    ax.plot(ts_pct, results['precision_mean'], marker='^', markersize=8, linewidth=2, color='#f39c12')
    ax.set_xlabel("Test Size (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Precision Score", fontsize=11, fontweight='bold')
    ax.set_title("Precision vs Test Size", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig("test_dt_splits.png", dpi=150, bbox_inches='tight')
    print("✅ Graphique sauvegardé: test_dt_splits.png")
    plt.close()
    
    print()
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    
    if f1_drop > 0.10:
        print("❌ DECISION TREE EST OVERFIITTED")
        print(f"   F1 chute de {f1_drop:.4f} quand test_size augmente")
        print(f"   Avec petit test set (5%): F1={f1_array[0]:.4f}")
        print(f"   Avec gros test set (50%): F1={f1_array[-1]:.4f}")
        print()
        print("   Raison: DT apprend parfaitement le training set")
        print("           mais génère mal sur données nouvelles")
        print()
        print("   Recommandation: REJETER DT ou LIMITER train_size <= 80%")
    else:
        print("✅ DECISION TREE EST OK")
        print(f"   F1 varie seulement de {f1_drop:.4f}")
        print("   Généralisation acceptable")

if __name__ == "__main__":
    main()