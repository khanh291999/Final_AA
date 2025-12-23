"""
Demo: Graph Sub-sampling for Divide-and-Conquer Community Detection
Based on: "Graph sub-sampling for divide-and-conquer algorithms in large networks"
by Eric Yanchenko (2025)

This demo compares 7 graph sub-sampling algorithms on community detection task.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import pandas as pd

# Import our modules
from graph_sampling import (
    RandomNodeSampling, RandomEdgeSampling, RandomWalkSampling,
    BFSSampling, DFSSampling, RandomNodeNeighborSampling,
    DegreeBasedSampling, get_sampler
)
from community_detection import (
    SpectralCommunityDetection, LouvainCommunityDetection,
    DivideAndConquerCommunityDetection, evaluate_communities,
    generate_stochastic_block_model
)


def visualize_graph_with_communities(G: nx.Graph, communities: Dict[int, int],
                                     title: str = "Graph with Communities",
                                     save_path: str = None):
    """
    Visualize graph with community structure.
    
    Args:
        G: NetworkX graph
        communities: Dictionary mapping node -> community
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Get community colors
    num_communities = len(set(communities.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
    
    node_colors = [colors[communities.get(node, 0)] for node in G.nodes()]
    
    # Draw graph
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    nx.draw(G, pos, node_color=node_colors, node_size=100,
            edge_color='gray', alpha=0.6, with_labels=False)
    
    # Add legend
    for i in range(num_communities):
        plt.scatter([], [], c=[colors[i]], label=f'Community {i}', s=100)
    plt.legend(loc='upper right')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_sampling_methods(G: nx.Graph, true_labels: Dict[int, int],
                             num_communities: int, sample_ratio: float = 0.3,
                             num_subgraphs: int = 10) -> pd.DataFrame:
    """
    Compare different sampling methods for community detection.
    
    Args:
        G: NetworkX graph
        true_labels: Ground truth community labels
        num_communities: Number of communities
        sample_ratio: Ratio of nodes to sample
        num_subgraphs: Number of sub-graphs in divide-and-conquer
        
    Returns:
        DataFrame with comparison results
    """
    # Define sampling methods to compare (7 methods from the paper)
    sampling_methods = {
        'Random Node (RN)': RandomNodeSampling(),
        'Degree Node (DN)': DegreeBasedSampling(),
        'Random Edge (RE)': RandomEdgeSampling(),
        'BFS': BFSSampling(),
        'DFS': DFSSampling(),
        'Random Node-Neighbor (RNN)': RandomNodeNeighborSampling(),
        'Random Walk (RW)': RandomWalkSampling()
    }
    
    # Base detector
    base_detector = SpectralCommunityDetection()
    
    results = []
    
    # Baseline: Apply base algorithm to full graph
    print("Running baseline (full graph)...")
    start_time = time.time()
    baseline_communities = base_detector.detect(G, num_communities)
    baseline_time = time.time() - start_time
    baseline_metrics = evaluate_communities(true_labels, baseline_communities)
    
    results.append({
        'Method': 'Baseline (Full Graph)',
        'NMI': baseline_metrics['nmi'],
        'ARI': baseline_metrics['ari'],
        'Accuracy': baseline_metrics['accuracy'],
        'Time (s)': baseline_time
    })
    
    print(f"Baseline - NMI: {baseline_metrics['nmi']:.3f}, Time: {baseline_time:.3f}s")
    
    # Test each sampling method with divide-and-conquer
    for method_name, sampler in sampling_methods.items():
        print(f"\nRunning {method_name}...")
        
        try:
            # Create divide-and-conquer detector
            dc_detector = DivideAndConquerCommunityDetection(
                base_detector=base_detector,
                sampler=sampler,
                num_subgraphs=num_subgraphs,
                sample_ratio=sample_ratio,
                beta=2
            )
            
            # Run detection
            start_time = time.time()
            pred_communities = dc_detector.detect(G, num_communities)
            elapsed_time = time.time() - start_time
            
            # Evaluate
            metrics = evaluate_communities(true_labels, pred_communities)
            
            results.append({
                'Method': method_name,
                'NMI': metrics['nmi'],
                'ARI': metrics['ari'],
                'Accuracy': metrics['accuracy'],
                'Time (s)': elapsed_time
            })
            
            print(f"{method_name} - NMI: {metrics['nmi']:.3f}, ARI: {metrics['ari']:.3f}, Time: {elapsed_time:.3f}s")
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results.append({
                'Method': method_name,
                'NMI': 0.0,
                'ARI': 0.0,
                'Accuracy': 0.0,
                'Time (s)': 0.0
            })
    
    return pd.DataFrame(results)


def plot_comparison_results(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot comparison results.
    
    Args:
        results_df: DataFrame with comparison results
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = results_df['Method']
    
    # NMI comparison
    axes[0, 0].bar(range(len(methods)), results_df['NMI'], color='steelblue')
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].set_ylabel('NMI Score', fontsize=12)
    axes[0, 0].set_title('Normalized Mutual Information', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # ARI comparison
    axes[0, 1].bar(range(len(methods)), results_df['ARI'], color='coral')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].set_ylabel('ARI Score', fontsize=12)
    axes[0, 1].set_title('Adjusted Rand Index', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Accuracy comparison
    axes[1, 0].bar(range(len(methods)), results_df['Accuracy'], color='mediumseagreen')
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('Classification Accuracy', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Time comparison
    axes[1, 1].bar(range(len(methods)), results_df['Time (s)'], color='mediumpurple')
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Computation Time', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main demo function."""
    print("=" * 70)
    print("Graph Sub-sampling for Divide-and-Conquer Community Detection")
    print("Based on: Eric Yanchenko (2025)")
    print("=" * 70)
    
    # Parameters
    n_nodes = 200  # Number of nodes
    num_communities = 4  # Number of communities
    p_in = 0.3  # Probability of edge within community
    p_out = 0.02  # Probability of edge between communities
    sample_ratio = 0.3  # Sample 30% of nodes
    num_subgraphs = 15  # Number of sub-graphs for divide-and-conquer
    
    print(f"\nGenerating Stochastic Block Model graph...")
    print(f"  - Nodes: {n_nodes}")
    print(f"  - Communities: {num_communities}")
    print(f"  - p_in (within community): {p_in}")
    print(f"  - p_out (between communities): {p_out}")
    
    # Generate graph
    G, true_labels = generate_stochastic_block_model(
        n=n_nodes,
        num_communities=num_communities,
        p_in=p_in,
        p_out=p_out,
        seed=42
    )
    
    print(f"\nGraph statistics:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print(f"  - Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    # Visualize original graph
    print("\nVisualizing original graph with true communities...")
    visualize_graph_with_communities(
        G, true_labels,
        title="Original Graph with True Communities",
        save_path="original_graph.png"
    )
    
    # Compare sampling methods
    print("\n" + "=" * 70)
    print("Comparing Sampling Methods for Divide-and-Conquer")
    print("=" * 70)
    print(f"  - Sample ratio: {sample_ratio}")
    print(f"  - Number of sub-graphs: {num_subgraphs}")
    
    results_df = compare_sampling_methods(
        G, true_labels, num_communities,
        sample_ratio=sample_ratio,
        num_subgraphs=num_subgraphs
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv('sampling_comparison_results.csv', index=False)
    print("\nResults saved to: sampling_comparison_results.csv")
    
    # Plot results
    print("\nGenerating comparison plots...")
    plot_comparison_results(results_df, save_path="sampling_comparison.png")
    
    # Visualize best method
    best_method_idx = results_df['NMI'].idxmax()
    best_method = results_df.iloc[best_method_idx]['Method']
    
    print(f"\n" + "=" * 70)
    print(f"Best Method: {best_method}")
    print(f"  - NMI: {results_df.iloc[best_method_idx]['NMI']:.3f}")
    print(f"  - ARI: {results_df.iloc[best_method_idx]['ARI']:.3f}")
    print(f"  - Accuracy: {results_df.iloc[best_method_idx]['Accuracy']:.3f}")
    print(f"  - Time: {results_df.iloc[best_method_idx]['Time (s)']:.3f}s")
    print("=" * 70)
    
    # Key findings from the paper
    print("\n" + "=" * 70)
    print("KEY FINDINGS FROM THE PAPER:")
    print("=" * 70)
    print("1. Random node sampling (uniform) often yields best performance")
    print("2. Sometimes baseline (full graph) is better in terms of both")
    print("   identification and computational time")
    print("3. Different sampling methods work better for different tasks")
    print("4. Careful selection of sub-sampling routine is important")
    print("=" * 70)


if __name__ == "__main__":
    main()
