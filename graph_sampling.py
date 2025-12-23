"""
Graph Sub-sampling Algorithms
Based on: "Graph sub-sampling for divide-and-conquer algorithms in large networks"
Author: Eric Yanchenko (2025)

This module implements seven graph sub-sampling algorithms for network analysis.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Set
import random


class GraphSampler:
    """Base class for graph sampling algorithms."""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Sample a subgraph from G with approximately sample_size nodes."""
        raise NotImplementedError


class RandomNodeSampling(GraphSampler):
    """
    Random Node Sampling (RNS): Sample nodes uniformly at random.
    This is the simplest sampling method.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Sample nodes uniformly at random."""
        nodes = list(G.nodes())
        sample_size = min(sample_size, len(nodes))
        sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False)
        return G.subgraph(sampled_nodes).copy()


class RandomEdgeSampling(GraphSampler):
    """
    Random Edge Sampling (RES): Sample edges uniformly at random.
    The subgraph consists of all nodes incident to the sampled edges.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Sample edges uniformly at random."""
        edges = list(G.edges())
        # Estimate number of edges needed to get approximately sample_size nodes
        num_edges = min(len(edges), sample_size // 2)
        sampled_edges = random.sample(edges, num_edges)
        
        # Create subgraph with sampled edges
        subgraph = nx.Graph()
        subgraph.add_edges_from(sampled_edges)
        return subgraph


class RandomWalkSampling(GraphSampler):
    """
    Random Walk Sampling (RWS): Perform a random walk on the graph.
    The subgraph consists of all nodes visited during the walk.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Perform random walk sampling."""
        nodes = list(G.nodes())
        if not nodes:
            return nx.Graph()
        
        # Start from a random node
        current_node = random.choice(nodes)
        visited = {current_node}
        walk_path = [current_node]
        
        # Perform random walk until we have enough nodes
        while len(visited) < sample_size and len(visited) < len(nodes):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                # If stuck, jump to a random unvisited node
                unvisited = set(nodes) - visited
                if unvisited:
                    current_node = random.choice(list(unvisited))
                else:
                    break
            else:
                current_node = random.choice(neighbors)
            
            visited.add(current_node)
            walk_path.append(current_node)
        
        return G.subgraph(visited).copy()


class ForestFireSampling(GraphSampler):
    """
    Forest Fire Sampling (FFS): Start from a random node and "burn" neighbors
    with forward burning probability p.
    """
    
    def __init__(self, p=0.7, seed=42):
        """
        Initialize Forest Fire Sampling.
        
        Args:
            p: Forward burning probability (default 0.7)
            seed: Random seed
        """
        super().__init__(seed)
        self.p = p
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Perform forest fire sampling."""
        nodes = list(G.nodes())
        if not nodes:
            return nx.Graph()
        
        # Start from a random node
        seed_node = random.choice(nodes)
        sampled_nodes = {seed_node}
        queue = [seed_node]
        
        while queue and len(sampled_nodes) < sample_size:
            current = queue.pop(0)
            neighbors = list(G.neighbors(current))
            
            # Randomly select neighbors to "burn"
            num_to_burn = np.random.binomial(len(neighbors), self.p)
            to_burn = random.sample(neighbors, min(num_to_burn, len(neighbors)))
            
            for neighbor in to_burn:
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    queue.append(neighbor)
                    
                    if len(sampled_nodes) >= sample_size:
                        break
        
        return G.subgraph(sampled_nodes).copy()


class DegreeBasedSampling(GraphSampler):
    """
    Degree-Based Sampling (DBS): Sample nodes with probability proportional to their degree.
    High-degree nodes are more likely to be sampled.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Sample nodes based on their degree."""
        nodes = list(G.nodes())
        degrees = np.array([G.degree(node) for node in nodes])
        
        # Normalize degrees to get probabilities
        if degrees.sum() == 0:
            probabilities = np.ones(len(nodes)) / len(nodes)
        else:
            probabilities = degrees / degrees.sum()
        
        sample_size = min(sample_size, len(nodes))
        sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False, p=probabilities)
        return G.subgraph(sampled_nodes).copy()


class PageRankSampling(GraphSampler):
    """
    PageRank-Based Sampling (PBS): Sample nodes with probability proportional to their PageRank.
    Important nodes in the network structure are more likely to be sampled.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Sample nodes based on their PageRank."""
        nodes = list(G.nodes())
        pagerank = nx.pagerank(G)
        
        # Extract PageRank values in node order
        pr_values = np.array([pagerank[node] for node in nodes])
        
        # Normalize to get probabilities
        probabilities = pr_values / pr_values.sum()
        
        sample_size = min(sample_size, len(nodes))
        sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False, p=probabilities)
        return G.subgraph(sampled_nodes).copy()


class BFSSampling(GraphSampler):
    """
    Breadth-First Search Sampling (BFS): Sample nodes using breadth-first traversal.
    Tìm kiếm theo chiều rộng - như vết dầu loang, mở rộng đều ra mọi hướng.
    
    Đặc điểm:
    - Bắt đầu từ một nút gốc ngẫu nhiên
    - Lấy tất cả bạn bè (hàng xóm) trực tiếp của nút đó
    - Sau khi lấy hết bạn bè trực tiếp, mới tiếp tục lấy "bạn của bạn"
    - Trong bài báo: BFS thường cho kết quả kém trong Community Detection
      vì nó gom cục bộ quá mức, không đại diện được toàn bộ mạng lưới.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Perform BFS sampling."""
        nodes = list(G.nodes())
        if not nodes:
            return nx.Graph()
        
        # Start from a random node
        seed_node = random.choice(nodes)
        sampled_nodes = {seed_node}
        queue = [seed_node]
        
        # BFS traversal
        while queue and len(sampled_nodes) < sample_size:
            current = queue.pop(0)
            neighbors = list(G.neighbors(current))
            
            for neighbor in neighbors:
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    queue.append(neighbor)
                    
                    if len(sampled_nodes) >= sample_size:
                        break
        
        return G.subgraph(sampled_nodes).copy()


class DFSSampling(GraphSampler):
    """
    Depth-First Search Sampling (DFS): Sample nodes using depth-first traversal.
    Tìm kiếm theo chiều sâu - như đi thám hiểm mê cung, đi sâu một mạch.
    
    Đặc điểm:
    - Bắt đầu từ một nút gốc ngẫu nhiên
    - Chọn một người bạn để đi tới, từ người bạn đó lại chọn tiếp một người khác
    - Đi một mạch sâu nhất có thể ("đâm lao phải theo lao") cho đến khi tắc đường
    - Khi tắc đường, quay lùi lại (backtrack) đến ngã rẽ gần nhất
    - Trong bài báo: DFS rất tệ cho Community Detection vì mẫu lấy ra có cấu trúc
      rất khác so với đồ thị gốc, nhưng lại khá ổn khi dùng để tìm Core-Periphery.
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Perform DFS sampling."""
        nodes = list(G.nodes())
        if not nodes:
            return nx.Graph()
        
        # Start from a random node
        seed_node = random.choice(nodes)
        sampled_nodes = {seed_node}
        stack = [seed_node]
        
        # DFS traversal
        while stack and len(sampled_nodes) < sample_size:
            current = stack.pop()  # LIFO for DFS
            neighbors = list(G.neighbors(current))
            random.shuffle(neighbors)  # Randomize order
            
            for neighbor in neighbors:
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    stack.append(neighbor)
                    
                    if len(sampled_nodes) >= sample_size:
                        break
        
        return G.subgraph(sampled_nodes).copy()


class RandomNodeNeighborSampling(GraphSampler):
    """
    Random Node-Neighbor Sampling (RNN): Sample random nodes with all their neighbors.
    Lấy nút và hàng xóm - chọn ngẫu nhiên một nút, sau đó lấy luôn tất cả bạn bè.
    
    Đặc điểm:
    - Chọn ngẫu nhiên một nút
    - Lấy luôn tất cả bạn bè (hàng xóm) của nút đó vào mẫu
    - Tiếp tục lặp lại với nút khác cho đến khi đủ số lượng
    - Bảo toàn được cấu trúc cục bộ tốt hơn Random Node
    """
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Perform random node-neighbor sampling."""
        nodes = list(G.nodes())
        if not nodes:
            return nx.Graph()
        
        sampled_nodes = set()
        remaining_nodes = set(nodes)
        
        while len(sampled_nodes) < sample_size and remaining_nodes:
            # Pick a random node from remaining nodes
            seed_node = random.choice(list(remaining_nodes))
            
            # Add the node and all its neighbors
            sampled_nodes.add(seed_node)
            neighbors = set(G.neighbors(seed_node))
            sampled_nodes.update(neighbors)
            
            # Remove sampled nodes from remaining
            remaining_nodes -= sampled_nodes
            
            if len(sampled_nodes) >= sample_size:
                # If we exceeded, randomly trim to exact size
                sampled_nodes = set(random.sample(list(sampled_nodes), 
                                                 min(sample_size, len(sampled_nodes))))
                break
        
        return G.subgraph(sampled_nodes).copy()


class SnowballSampling(GraphSampler):
    """
    Snowball Sampling: Start from seed nodes and expand by k-hops.
    Sample all neighbors within k hops from the seed nodes.
    (Giữ lại để tương thích với code cũ, nhưng không phải 7 phương pháp chính)
    """
    
    def __init__(self, k=2, seed=42):
        """
        Initialize Snowball Sampling.
        
        Args:
            k: Number of hops (default 2)
            seed: Random seed
        """
        super().__init__(seed)
        self.k = k
    
    def sample(self, G: nx.Graph, sample_size: int) -> nx.Graph:
        """Perform snowball sampling."""
        nodes = list(G.nodes())
        if not nodes:
            return nx.Graph()
        
        # Estimate number of seed nodes needed
        num_seeds = max(1, sample_size // (2 ** self.k))
        seed_nodes = random.sample(nodes, min(num_seeds, len(nodes)))
        
        sampled_nodes = set(seed_nodes)
        
        # Expand by k hops
        current_layer = set(seed_nodes)
        for _ in range(self.k):
            next_layer = set()
            for node in current_layer:
                neighbors = set(G.neighbors(node))
                next_layer.update(neighbors)
            
            sampled_nodes.update(next_layer)
            current_layer = next_layer
            
            if len(sampled_nodes) >= sample_size:
                # Randomly sample from collected nodes if we exceeded the limit
                sampled_nodes = set(random.sample(list(sampled_nodes), sample_size))
                break
        
        return G.subgraph(sampled_nodes).copy()


def get_sampler(method: str, **kwargs) -> GraphSampler:
    """
    Factory function to get a sampler by name.
    
    7 Phương pháp chính từ bài báo:
        'random_node' (RN): Random Node - Lấy nút ngẫu nhiên
        'degree_node' (DN): Degree Node - Lấy nút theo bậc (KOL)
        'random_edge' (RE): Random Edge - Lấy cạnh ngẫu nhiên
        'bfs': Breadth-First Search - Tìm kiếm theo chiều rộng
        'dfs': Depth-First Search - Tìm kiếm theo chiều sâu
        'random_node_neighbor' (RNN): Random Node-Neighbor - Lấy nút và hàng xóm
        'random_walk' (RW): Random Walk - Bước đi ngẫu nhiên
    
    Phương pháp khác (tương thích code cũ):
        'forest_fire': Forest Fire Sampling
        'pagerank': PageRank-Based Sampling
        'snowball': Snowball Sampling
    
    Args:
        method: Name of sampling method
        **kwargs: Additional arguments for specific samplers
    
    Returns:
        GraphSampler instance
    """
    samplers = {
        # 7 phương pháp chính từ bài báo
        'random_node': RandomNodeSampling,
        'degree_node': DegreeBasedSampling,  # DN - Degree Node
        'random_edge': RandomEdgeSampling,
        'bfs': BFSSampling,
        'dfs': DFSSampling,
        'random_node_neighbor': RandomNodeNeighborSampling,  # RNN
        'random_walk': RandomWalkSampling,
        
        # Alias cho tương thích
        'degree_based': DegreeBasedSampling,  # alias for degree_node
        'rnn': RandomNodeNeighborSampling,    # alias
        
        # Các phương pháp khác (không trong 7 phương pháp chính)
        'forest_fire': ForestFireSampling,
        'pagerank': PageRankSampling,
        'snowball': SnowballSampling
    }
    
    if method not in samplers:
        raise ValueError(f"Unknown sampling method: {method}. Available: {list(samplers.keys())}")
    
    return samplers[method](**kwargs)
