"""
Demo: Minh họa sự khác biệt giữa BFS và DFS
Visualize the difference between Breadth-First Search and Depth-First Search sampling

Giải thích chi tiết:
- BFS (Breadth-First Search): Như vết dầu loang, mở rộng đều ra mọi hướng
- DFS (Depth-First Search): Như đi thám hiểm mê cung, đi sâu một mạch
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph_sampling import BFSSampling, DFSSampling, RandomNodeSampling


def create_grid_graph(rows=10, cols=10):
    """Tạo đồ thị dạng lưới để dễ quan sát BFS/DFS."""
    G = nx.grid_2d_graph(rows, cols)
    # Convert to simple node IDs
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G, rows, cols


def get_node_positions(G, rows, cols):
    """Get positions for grid layout."""
    pos = {}
    for node in G.nodes():
        row = node // cols
        col = node % cols
        pos[node] = (col, -row)
    return pos


def visualize_sampling_comparison():
    """
    So sánh trực quan giữa BFS, DFS và Random Node sampling.
    """
    print("=" * 80)
    print("Minh họa sự khác biệt giữa BFS, DFS và Random Node Sampling")
    print("=" * 80)
    
    # Tạo đồ thị dạng lưới
    rows, cols = 10, 10
    G, rows, cols = create_grid_graph(rows, cols)
    pos = get_node_positions(G, rows, cols)
    
    print(f"\nĐồ thị gốc: Lưới {rows}x{cols} = {G.number_of_nodes()} nodes")
    print(f"Số cạnh: {G.number_of_edges()}")
    
    # Sample size
    sample_size = 30
    
    # Initialize samplers
    samplers = {
        'Random Node (RN)': RandomNodeSampling(seed=42),
        'BFS - Tìm kiếm theo chiều rộng': BFSSampling(seed=42),
        'DFS - Tìm kiếm theo chiều sâu': DFSSampling(seed=42)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    # Plot original graph
    ax = axes[0]
    nx.draw(G, pos, ax=ax, node_size=50, node_color='lightgray', 
            edge_color='gray', alpha=0.3, with_labels=False)
    ax.set_title(f'Đồ thị gốc\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-1, cols)
    ax.set_ylim(-rows, 1)
    
    # Sample and visualize each method
    for idx, (method_name, sampler) in enumerate(samplers.items(), start=1):
        subgraph = sampler.sample(G, sample_size)
        sampled_nodes = set(subgraph.nodes())
        
        ax = axes[idx]
        
        # Color nodes: sampled vs not sampled
        node_colors = ['red' if node in sampled_nodes else 'lightgray' 
                       for node in G.nodes()]
        node_sizes = [200 if node in sampled_nodes else 30 
                     for node in G.nodes()]
        
        # Draw all nodes
        nx.draw(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                edge_color='gray', alpha=0.2, with_labels=False)
        
        # Highlight sampled edges
        sampled_edges = list(subgraph.edges())
        nx.draw_networkx_edges(G, pos, edgelist=sampled_edges, ax=ax,
                              edge_color='red', width=2, alpha=0.6)
        
        # Add statistics
        stats_text = (
            f'Sampled: {len(sampled_nodes)} nodes, {len(sampled_edges)} edges\n'
            f'Density: {nx.density(subgraph):.3f}'
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Title with explanation
        if 'BFS' in method_name:
            explanation = 'Mở rộng đều như vết dầu loang'
            ax.set_title(f'{method_name}\n{explanation}',
                        fontsize=13, fontweight='bold', color='darkblue')
        elif 'DFS' in method_name:
            explanation = 'Đi sâu một mạch như thám hiểm mê cung'
            ax.set_title(f'{method_name}\n{explanation}',
                        fontsize=13, fontweight='bold', color='darkgreen')
        else:
            ax.set_title(f'{method_name}\nLấy ngẫu nhiên, phân tán',
                        fontsize=13, fontweight='bold', color='darkred')
        
        ax.set_xlim(-1, cols)
        ax.set_ylim(-rows, 1)
    
    plt.tight_layout()
    plt.savefig('bfs_dfs_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Đã lưu hình minh họa: bfs_dfs_comparison.png")
    plt.show()
    
    # Print detailed explanations
    print("\n" + "=" * 80)
    print("GIẢI THÍCH CHI TIẾT")
    print("=" * 80)
    
    print("\n1️⃣  RANDOM NODE (RN) - Lấy nút ngẫu nhiên:")
    print("   • Chọn ngẫu nhiên các nút trong mạng lưới")
    print("   • Nếu hai nút được chọn có kết nối, kết nối đó được giữ lại")
    print("   • Kết quả: Các nút phân tán, ít kết nối với nhau")
    
    print("\n2️⃣  BFS - Breadth-First Search (Tìm kiếm theo chiều rộng):")
    print("   • Bắt đầu từ một nút gốc ngẫu nhiên")
    print("   • Lấy TẤT CẢ bạn bè trực tiếp của nút đó")
    print("   • Sau đó mới tiếp tục lấy 'bạn của bạn' (lớp thứ 2)")
    print("   • Đặc điểm: Mở rộng ĐỀU ra mọi hướng xung quanh điểm xuất phát")
    print("   • Ví dụ: Như vết dầu loang, hoặc bóc vỏ hành tây từng lớp")
    print("   ⚠️  Trong bài báo: BFS THƯỜNG CHO KẾT QUẢ KÉM trong Community Detection")
    print("       vì nó gom cục bộ quá mức, không đại diện được toàn bộ mạng lưới")
    
    print("\n3️⃣  DFS - Depth-First Search (Tìm kiếm theo chiều sâu):")
    print("   • Bắt đầu từ một nút gốc ngẫu nhiên")
    print("   • Thay vì lấy hết bạn bè, chỉ chọn MỘT người bạn để đi tới")
    print("   • Từ người bạn đó, lại chọn tiếp MỘT người bạn khác chưa gặp")
    print("   • Đi một mạch sâu nhất có thể cho đến khi tắc đường (ngõ cụt)")
    print("   • Khi tắc đường, QUAY LÙI (backtrack) đến ngã rẽ gần nhất")
    print("   • Đặc điểm: Tạo các nhánh DÀI và NGOẰN NGOÈO, đi xa khỏi điểm xuất phát")
    print("   • Ví dụ: Như đi thám hiểm mê cung - 'đâm lao phải theo lao'")
    print("   ⚠️  Trong bài báo: DFS RẤT TỆ cho Community Detection vì mẫu lấy ra")
    print("       có cấu trúc rất khác so với đồ thị gốc, NHƯNG khá ổn cho")
    print("       việc tìm cấu trúc Lõi-Vành đai (Core-Periphery)")
    
    print("\n" + "=" * 80)
    print("KẾT LUẬN:")
    print("=" * 80)
    print("• BFS: Tạo mẫu CỤC BỘ, GẮN KẾT CAO, nhưng KHÔNG ĐẠI DIỆN toàn mạng")
    print("• DFS: Tạo mẫu NGOẰN NGOÈO, ĐI XA, cấu trúc KHÁC với đồ thị gốc")
    print("• Random Node: Tạo mẫu PHÂN TÁN, ít kết nối, nhưng ĐẠI DIỆN hơn")
    print("=" * 80)


def test_on_social_network():
    """
    Test BFS/DFS trên mạng xã hội thực tế (Karate Club).
    """
    print("\n\n" + "=" * 80)
    print("TEST TRÊN MẠNG XÃ HỘI THỰC TẾ (Karate Club)")
    print("=" * 80)
    
    # Load Karate Club graph (famous social network)
    G = nx.karate_club_graph()
    print(f"\nKarate Club Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    sample_size = 15
    
    # Initialize samplers
    samplers = {
        'Random Node': RandomNodeSampling(seed=42),
        'BFS': BFSSampling(seed=42),
        'DFS': DFSSampling(seed=42)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get layout
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Plot original
    ax = axes[0]
    nx.draw(G, pos, ax=ax, node_size=300, node_color='lightblue',
            edge_color='gray', alpha=0.6, with_labels=True, font_size=8)
    ax.set_title('Original Karate Club Network', fontsize=14, fontweight='bold')
    
    # Sample and visualize
    for idx, (method_name, sampler) in enumerate(samplers.items(), start=1):
        subgraph = sampler.sample(G, sample_size)
        sampled_nodes = set(subgraph.nodes())
        
        ax = axes[idx]
        
        # Colors
        node_colors = ['red' if node in sampled_nodes else 'lightgray' 
                       for node in G.nodes()]
        node_sizes = [400 if node in sampled_nodes else 100 
                     for node in G.nodes()]
        
        # Draw
        nx.draw(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                edge_color='lightgray', alpha=0.3, with_labels=True, font_size=8)
        
        # Highlight sampled edges
        nx.draw_networkx_edges(G, pos, edgelist=list(subgraph.edges()), ax=ax,
                              edge_color='red', width=2, alpha=0.8)
        
        # Calculate clustering coefficient
        clustering = nx.average_clustering(subgraph) if len(subgraph) > 0 else 0
        
        title = f'{method_name}\n{len(sampled_nodes)} nodes, {len(subgraph.edges())} edges\nClustering: {clustering:.3f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('bfs_dfs_karate.png', dpi=150, bbox_inches='tight')
    print("\n✓ Đã lưu hình minh họa: bfs_dfs_karate.png")
    plt.show()
    
    # Compare metrics
    print("\n" + "=" * 80)
    print("SO SÁNH CÁC CHỈ SỐ:")
    print("=" * 80)
    print(f"{'Method':<15} {'Nodes':<8} {'Edges':<8} {'Density':<10} {'Clustering':<12}")
    print("-" * 80)
    
    for method_name, sampler in samplers.items():
        subgraph = sampler.sample(G, sample_size)
        density = nx.density(subgraph)
        clustering = nx.average_clustering(subgraph) if len(subgraph) > 0 else 0
        
        print(f"{method_name:<15} {len(subgraph.nodes()):<8} {len(subgraph.edges()):<8} "
              f"{density:<10.3f} {clustering:<12.3f}")
    
    print("\nNHẬN XÉT:")
    print("• BFS thường có CLUSTERING CAO hơn (gom cục bộ)")
    print("• DFS thường có cấu trúc DẠNG ĐƯỜNG (path-like)")
    print("• Random Node có mẫu PHÂN TÁN hơn")


if __name__ == '__main__':
    # Run visualizations
    visualize_sampling_comparison()
    test_on_social_network()
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH! Đã tạo 2 hình minh họa:")
    print("  1. bfs_dfs_comparison.png - So sánh trên đồ thị lưới")
    print("  2. bfs_dfs_karate.png - So sánh trên mạng xã hội thực")
    print("=" * 80)
