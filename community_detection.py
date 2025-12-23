"""
Community Detection using Divide-and-Conquer with Graph Sub-sampling
Based on: "Graph sub-sampling for divide-and-conquer algorithms in large networks"

Implements the divide-and-conquer algorithm for community detection with various
sub-sampling schemes.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class CommunityDetector:
    """Base class for community detection algorithms."""
    
    def detect(self, G: nx.Graph, num_communities: int) -> Dict[int, int]:
        """
        Detect communities in graph G.
        
        Args:
            G: NetworkX graph
            num_communities: Number of communities to detect
            
        Returns:
            Dictionary mapping node_id -> community_id
        """
        raise NotImplementedError


class SpectralCommunityDetection(CommunityDetector):
    """
    Spectral clustering-based community detection.
    Uses the graph Laplacian for clustering.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
    
    def detect(self, G: nx.Graph, num_communities: int) -> Dict[int, int]:
        """Detect communities using spectral clustering."""
        if len(G.nodes()) == 0:
            return {}
        
        if len(G.nodes()) < num_communities:
            # If graph is too small, assign each node to its own community
            return {node: i for i, node in enumerate(G.nodes())}
        
        # Get adjacency matrix
        nodes = list(G.nodes())
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
        
        # Apply spectral clustering
        try:
            clustering = SpectralClustering(
                n_clusters=num_communities,
                affinity='precomputed',
                random_state=self.seed,
                n_init=10
            )
            labels = clustering.fit_predict(adj_matrix)
            
            # Create node -> community mapping
            community_map = {nodes[i]: labels[i] for i in range(len(nodes))}
            return community_map
        except:
            # Fallback: assign nodes randomly
            labels = np.random.randint(0, num_communities, len(nodes))
            return {nodes[i]: labels[i] for i in range(len(nodes))}


class LouvainCommunityDetection(CommunityDetector):
    """
    Louvain method for community detection.
    Uses modularity optimization.
    """
    
    def detect(self, G: nx.Graph, num_communities: int = None) -> Dict[int, int]:
        """Detect communities using Louvain method."""
        if len(G.nodes()) == 0:
            return {}
        
        # Use greedy modularity communities as a simple alternative
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Create node -> community mapping
        community_map = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                community_map[node] = comm_id
        
        return community_map


class DivideAndConquerCommunityDetection:
    """
    Divide-and-Conquer algorithm for community detection.
    
    Algorithm from the paper:
    1. Sample B sub-graphs from the original graph
    2. Detect communities in each sub-graph
    3. Aggregate results using a clustering matrix
    4. Final clustering based on aggregated results
    """
    
    def __init__(self, base_detector: CommunityDetector, sampler, 
                 num_subgraphs: int = 10, sample_ratio: float = 0.3,
                 beta: int = 2, seed: int = 42):
        """
        Initialize Divide-and-Conquer community detection.
        
        Args:
            base_detector: Base community detection algorithm
            sampler: Graph sampling algorithm
            num_subgraphs: Number of sub-graphs to sample (B in the paper)
            sample_ratio: Ratio of nodes to sample in each sub-graph
            beta: Minimum number of co-occurrences for clustering matrix (smoothing)
            seed: Random seed
        """
        self.base_detector = base_detector
        self.sampler = sampler
        self.num_subgraphs = num_subgraphs
        self.sample_ratio = sample_ratio
        self.beta = beta
        self.seed = seed
        np.random.seed(seed)
    
    def detect(self, G: nx.Graph, num_communities: int) -> Dict[int, int]:
        """
        Detect communities using divide-and-conquer approach.
        
        Args:
            G: NetworkX graph
            num_communities: Number of communities to detect
            
        Returns:
            Dictionary mapping node_id -> community_id
        """
        nodes = list(G.nodes())
        n = len(nodes)
        sample_size = max(10, int(n * self.sample_ratio))
        
        # Track co-occurrence of nodes in sub-graphs and their cluster assignments
        N_ij = defaultdict(int)  # Number of times nodes i,j sampled together
        same_cluster = defaultdict(int)  # Number of times i,j in same cluster
        
        # Step 1 & 2: Sample sub-graphs and detect communities
        for b in range(self.num_subgraphs):
            # Sample sub-graph
            subgraph = self.sampler.sample(G, sample_size)
            
            if len(subgraph.nodes()) < 2:
                continue
            
            # Detect communities in sub-graph
            communities = self.base_detector.detect(subgraph, num_communities)
            
            # Update co-occurrence and clustering matrices
            subgraph_nodes = list(subgraph.nodes())
            for i in range(len(subgraph_nodes)):
                for j in range(i + 1, len(subgraph_nodes)):
                    node_i, node_j = subgraph_nodes[i], subgraph_nodes[j]
                    pair = tuple(sorted([node_i, node_j]))
                    
                    N_ij[pair] += 1
                    
                    # Check if in same community
                    if communities[node_i] == communities[node_j]:
                        same_cluster[pair] += 1
        
        # Step 3: Build clustering matrix C
        # C_ij = 1 if nodes i,j should be in same cluster, 0 otherwise
        C = np.zeros((n, n))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    C[i, j] = 1
                else:
                    pair = tuple(sorted([node_i, node_j]))
                    if N_ij[pair] >= self.beta:
                        # Estimate: prob that i,j in same cluster
                        C[i, j] = same_cluster[pair] / N_ij[pair]
        
        # Step 4: Final clustering based on C
        # Use spectral clustering on the clustering matrix
        try:
            final_clustering = SpectralClustering(
                n_clusters=num_communities,
                affinity='precomputed',
                random_state=self.seed,
                n_init=10
            )
            labels = final_clustering.fit_predict(C)
            
            community_map = {nodes[i]: labels[i] for i in range(n)}
        except:
            # Fallback: use most frequent assignment from sub-graphs
            community_map = self._fallback_clustering(G, nodes, num_communities)
        
        return community_map
    
    def _fallback_clustering(self, G: nx.Graph, nodes: List, 
                            num_communities: int) -> Dict[int, int]:
        """Fallback clustering if spectral clustering fails."""
        # Just apply base detector to full graph
        return self.base_detector.detect(G, num_communities)


def evaluate_communities(true_labels: Dict[int, int], 
                         pred_labels: Dict[int, int]) -> Dict[str, float]:
    """
    Evaluate community detection results.
    
    Args:
        true_labels: Ground truth community assignments
        pred_labels: Predicted community assignments
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    # Get common nodes
    common_nodes = set(true_labels.keys()) & set(pred_labels.keys())
    
    if not common_nodes:
        return {'nmi': 0.0, 'ari': 0.0, 'accuracy': 0.0}
    
    # Extract labels for common nodes
    true_arr = [true_labels[node] for node in common_nodes]
    pred_arr = [pred_labels[node] for node in common_nodes]
    
    # Calculate metrics
    nmi = normalized_mutual_info_score(true_arr, pred_arr)
    ari = adjusted_rand_score(true_arr, pred_arr)
    
    # Simple accuracy (best matching)
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_arr, pred_arr)
    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / len(common_nodes)
    
    return {
        'nmi': nmi,
        'ari': ari,
        'accuracy': accuracy
    }


def generate_stochastic_block_model(n: int, num_communities: int, 
                                    p_in: float, p_out: float,
                                    seed: int = 42) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a graph using Stochastic Block Model (SBM).
    
    Args:
        n: Number of nodes
        num_communities: Number of communities
        p_in: Probability of edge within community
        p_out: Probability of edge between communities
        seed: Random seed
        
    Returns:
        Tuple of (graph, true_community_labels)
    """
    np.random.seed(seed)
    
    # Assign nodes to communities
    nodes_per_community = n // num_communities
    true_labels = {}
    
    for node in range(n):
        community = node // nodes_per_community
        if community >= num_communities:
            community = num_communities - 1
        true_labels[node] = community
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add edges based on SBM
    for i in range(n):
        for j in range(i + 1, n):
            if true_labels[i] == true_labels[j]:
                # Same community
                if np.random.random() < p_in:
                    G.add_edge(i, j)
            else:
                # Different communities
                if np.random.random() < p_out:
                    G.add_edge(i, j)
    
    return G, true_labels
