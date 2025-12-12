"""
Plotting utilities for evaluation results.
"""

import matplotlib.pyplot as plt
from typing import Dict


def plot_precision_recall_vs_k(results_dict: Dict[str, Dict[int, Dict[str, float]]],
                               save_path: str = None, show_plot: bool = True):
    """
    Plot precision and recall vs k for different recommendation methods.
    
    Parameters:
    -----------
    results_dict : Dict[str, Dict[int, Dict[str, float]]]
        Dictionary mapping method names to k-value results
        Format: {'method_name': {k: {'precision': float, 'recall': float}}}
    save_path : str, optional
        Path to save the plot (default: None, don't save)
    show_plot : bool
        Whether to display the plot (default: True)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_values = sorted(list(results_dict[list(results_dict.keys())[0]].keys()))
    
    # Plot precision
    for method_name, results in results_dict.items():
        precisions = [results[k]['precision'] for k in k_values]
        ax1.plot(k_values, precisions, marker='o', label=method_name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Recommendations (k)', fontsize=12)
    ax1.set_ylabel('Precision@k', fontsize=12)
    ax1.set_title('Precision vs Number of Recommendations', fontsize=14, fontweight='bold')
    ax1.set_xticks(k_values)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(bottom=0)
    
    # Plot recall
    for method_name, results in results_dict.items():
        recalls = [results[k]['recall'] for k in k_values]
        ax2.plot(k_values, recalls, marker='s', label=method_name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Recommendations (k)', fontsize=12)
    ax2.set_ylabel('Recall@k', fontsize=12)
    ax2.set_title('Recall vs Number of Recommendations', fontsize=14, fontweight='bold')
    ax2.set_xticks(k_values)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

