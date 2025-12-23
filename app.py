"""
Graph Sampling Web Application
Ứng dụng web để chọn và minh họa các phương pháp lấy mẫu đồ thị
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from graph_sampling import (
    RandomNodeSampling, DegreeBasedSampling, RandomEdgeSampling,
    BFSSampling, DFSSampling, RandomNodeNeighborSampling, RandomWalkSampling
)
from community_detection import (
    SpectralCommunityDetection, DivideAndConquerCommunityDetection,
    generate_stochastic_block_model, evaluate_communities
)
import time

# Cấu hình trang
st.set_page_config(
    page_title="Graph Sampling Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Graph Sub-sampling</h1>', unsafe_allow_html=True)
st.markdown("### Triển khai 7 phương pháp lấy mẫu từ bài báo khoa học")
st.markdown("---")

# Tab chính
tab1, tab2, tab3 = st.tabs([
    "🎯 So sánh phương pháp", 
    "🔍 Minh họa chi tiết",
    "📈 Phân tích Community Detection"
])

# TAB 1: So sánh các phương pháp
with tab1:
    st.header("So sánh 7 phương pháp lấy mẫu")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Chọn đồ thị")
        graph_type = st.selectbox(
            "Loại đồ thị",
            ["Karate Club", "Stochastic Block Model", "Random Graph", "Grid Graph"]
        )
        
        if graph_type == "Stochastic Block Model":
            n_nodes = st.slider("Số nodes", 50, 300, 100)
            n_communities = st.slider("Số communities", 2, 6, 3)
        elif graph_type == "Random Graph":
            n_nodes = st.slider("Số nodes", 20, 100, 50)
            edge_prob = st.slider("Xác suất cạnh", 0.1, 0.5, 0.2)
        elif graph_type == "Grid Graph":
            grid_size = st.slider("Kích thước lưới", 5, 15, 10)
    
    with col2:
        st.subheader("Tham số lấy mẫu")
        sample_size = st.slider("Số nodes lấy mẫu", 10, 100, 20)
        seed = st.number_input("Random seed", 0, 999, 42)
        
        selected_methods = st.multiselect(
            "Chọn phương pháp (chọn nhiều)",
            ["Random Node (RN)", "Degree Node (DN)", "Random Edge (RE)", 
             "BFS", "DFS", "Random Node-Neighbor (RNN)", "Random Walk (RW)"],
            default=["Random Node (RN)", "BFS", "DFS"]
        )
    
    if st.button("🚀 Chạy so sánh", type="primary"):
        # Tạo đồ thị
        with st.spinner("Đang tạo đồ thị..."):
            if graph_type == "Karate Club":
                G = nx.karate_club_graph()
            elif graph_type == "Stochastic Block Model":
                G, _ = generate_stochastic_block_model(n_nodes, n_communities, 0.3, 0.02)
            elif graph_type == "Random Graph":
                G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
            elif graph_type == "Grid Graph":
                G = nx.grid_2d_graph(grid_size, grid_size)
                G = nx.convert_node_labels_to_integers(G)
        
        st.success(f"✅ Đồ thị: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Mapping tên -> sampler
        sampler_map = {
            "Random Node (RN)": RandomNodeSampling(seed=seed),
            "Degree Node (DN)": DegreeBasedSampling(seed=seed),
            "Random Edge (RE)": RandomEdgeSampling(seed=seed),
            "BFS": BFSSampling(seed=seed),
            "DFS": DFSSampling(seed=seed),
            "Random Node-Neighbor (RNN)": RandomNodeNeighborSampling(seed=seed),
            "Random Walk (RW)": RandomWalkSampling(seed=seed)
        }
        
        # Chạy sampling
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, method_name in enumerate(selected_methods):
            status_text.text(f"Đang chạy {method_name}...")
            sampler = sampler_map[method_name]
            
            start_time = time.time()
            subgraph = sampler.sample(G, sample_size)
            elapsed = time.time() - start_time
            
            density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0
            avg_degree = sum(dict(subgraph.degree()).values()) / max(subgraph.number_of_nodes(), 1)
            clustering = nx.average_clustering(subgraph) if subgraph.number_of_nodes() > 0 else 0
            
            results.append({
                'Phương pháp': method_name,
                'Nodes': subgraph.number_of_nodes(),
                'Edges': subgraph.number_of_edges(),
                'Density': density,
                'Avg Degree': avg_degree,
                'Clustering': clustering,
                'Time (ms)': elapsed * 1000
            })
            
            progress_bar.progress((idx + 1) / len(selected_methods))
        
        status_text.text("✅ Hoàn thành!")
        
        # Hiển thị kết quả
        st.subheader("📊 Kết quả so sánh")
        df = pd.DataFrame(results)
        st.dataframe(df.style.format({
            'Density': '{:.3f}',
            'Avg Degree': '{:.2f}',
            'Clustering': '{:.3f}',
            'Time (ms)': '{:.2f}'
        }), width='stretch')
        
        # Biểu đồ so sánh
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(df, x='Phương pháp', y=['Nodes', 'Edges'], 
                         title="Số lượng Nodes và Edges",
                         barmode='group')
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            fig2 = px.bar(df, x='Phương pháp', y='Density',
                         title="Mật độ đồ thị (Density)",
                         color='Density', color_continuous_scale='viridis')
            st.plotly_chart(fig2, width='stretch')
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3 = px.bar(df, x='Phương pháp', y='Clustering',
                         title="Hệ số Clustering",
                         color='Clustering', color_continuous_scale='blues')
            st.plotly_chart(fig3, width='stretch')
        
        with col4:
            fig4 = px.bar(df, x='Phương pháp', y='Time (ms)',
                         title="Thời gian thực hiện (ms)",
                         color='Time (ms)', color_continuous_scale='reds')
            st.plotly_chart(fig4, width='stretch')

# TAB 2: Minh họa chi tiết
with tab2:
    st.header("🔍 Minh họa chi tiết BFS vs DFS")
    
    st.markdown("""
    ### So sánh trực quan giữa Breadth-First Search và Depth-First Search
    
    - **BFS**: Mở rộng đều ra mọi hướng như vết dầu loang
    - **DFS**: Đi sâu một mạch như thám hiểm mê cung
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        viz_sample_size = st.slider("Số nodes để visualize", 10, 30, 15, key="viz_sample")
        viz_seed = st.number_input("Random seed", 0, 999, 42, key="viz_seed")
    
    if st.button("🎨 Tạo visualization", type="primary"):
        # Tạo đồ thị Karate Club
        G = nx.karate_club_graph()
        
        samplers = {
            'Random Node': RandomNodeSampling(seed=viz_seed),
            'BFS': BFSSampling(seed=viz_seed),
            'DFS': DFSSampling(seed=viz_seed)
        }
        
        # Tạo layout
        pos = nx.spring_layout(G, seed=viz_seed, k=0.5, iterations=50)
        
        cols = st.columns(3)
        
        for idx, (name, sampler) in enumerate(samplers.items()):
            with cols[idx]:
                st.subheader(name)
                
                subgraph = sampler.sample(G, viz_sample_size)
                sampled_nodes = set(subgraph.nodes())
                
                # Tạo figure
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Vẽ tất cả nodes
                node_colors = ['red' if node in sampled_nodes else 'lightgray' 
                              for node in G.nodes()]
                node_sizes = [300 if node in sampled_nodes else 50 
                             for node in G.nodes()]
                
                nx.draw(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                       edge_color='lightgray', alpha=0.3, with_labels=False)
                
                # Highlight sampled edges
                sampled_edges = list(subgraph.edges())
                nx.draw_networkx_edges(G, pos, edgelist=sampled_edges, ax=ax,
                                     edge_color='red', width=2, alpha=0.8)
                
                ax.set_title(f'{name}\n{len(sampled_nodes)} nodes, {len(sampled_edges)} edges',
                           fontsize=12, fontweight='bold')
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close()
                
                # Metrics
                density = nx.density(subgraph) if len(subgraph) > 1 else 0
                st.metric("Density", f"{density:.3f}")

# TAB 3: Community Detection
with tab3:
    st.header("📈 Phân tích Community Detection")
    
    st.markdown("""
    ### So sánh hiệu suất phát hiện cộng đồng với các phương pháp lấy mẫu khác nhau
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Tham số đồ thị")
        cd_n_nodes = st.slider("Số nodes", 100, 500, 200, key="cd_nodes")
        cd_n_communities = st.slider("Số communities", 2, 8, 4, key="cd_comm")
        cd_p_in = st.slider("p_in (trong cộng đồng)", 0.1, 0.8, 0.3, key="cd_pin")
        cd_p_out = st.slider("p_out (giữa cộng đồng)", 0.01, 0.2, 0.02, key="cd_pout")
    
    with col2:
        st.subheader("Tham số Divide-and-Conquer")
        cd_sample_ratio = st.slider("Tỷ lệ sampling", 0.2, 0.8, 0.3, key="cd_ratio")
        cd_num_subgraphs = st.slider("Số sub-graphs", 5, 30, 15, key="cd_subgraphs")
        cd_beta = st.slider("Beta (ngưỡng)", 1, 5, 2, key="cd_beta")
    
    if st.button("🔬 Chạy phân tích", type="primary"):
        with st.spinner("Đang tạo đồ thị và chạy phân tích..."):
            # Tạo đồ thị
            G, true_labels = generate_stochastic_block_model(
                cd_n_nodes, cd_n_communities, cd_p_in, cd_p_out
            )
            
            st.success(f"✅ Đã tạo đồ thị: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Base detector
            base_detector = SpectralCommunityDetection()
            
            # Baseline
            st.info("Đang chạy baseline (full graph)...")
            start_time = time.time()
            baseline_communities = base_detector.detect(G, cd_n_communities)
            baseline_time = time.time() - start_time
            baseline_metrics = evaluate_communities(true_labels, baseline_communities)
            
            # Test với các phương pháp sampling
            methods = {
                "Random Node": RandomNodeSampling(seed=42),
                "Degree Node": DegreeBasedSampling(seed=42),
                "BFS": BFSSampling(seed=42),
                "DFS": DFSSampling(seed=42),
                "Random Node-Neighbor": RandomNodeNeighborSampling(seed=42),
                "Random Walk": RandomWalkSampling(seed=42)
            }
            
            results = [{
                'Method': 'Baseline (Full Graph)',
                'NMI': baseline_metrics['nmi'],
                'ARI': baseline_metrics['ari'],
                'Accuracy': baseline_metrics['accuracy'],
                'Time (s)': baseline_time
            }]
            
            progress_bar = st.progress(0)
            
            for idx, (name, sampler) in enumerate(methods.items()):
                st.info(f"Đang chạy {name}...")
                
                dc_detector = DivideAndConquerCommunityDetection(
                    base_detector=base_detector,
                    sampler=sampler,
                    num_subgraphs=cd_num_subgraphs,
                    sample_ratio=cd_sample_ratio,
                    beta=cd_beta
                )
                
                start_time = time.time()
                pred_communities = dc_detector.detect(G, cd_n_communities)
                elapsed = time.time() - start_time
                
                metrics = evaluate_communities(true_labels, pred_communities)
                
                results.append({
                    'Method': name,
                    'NMI': metrics['nmi'],
                    'ARI': metrics['ari'],
                    'Accuracy': metrics['accuracy'],
                    'Time (s)': elapsed
                })
                
                progress_bar.progress((idx + 1) / len(methods))
            
            st.success("✅ Hoàn thành!")
            
            # Hiển thị kết quả
            st.subheader("📊 Kết quả Community Detection")
            df_cd = pd.DataFrame(results)
            st.dataframe(df_cd.style.format({
                'NMI': '{:.3f}',
                'ARI': '{:.3f}',
                'Accuracy': '{:.3f}',
                'Time (s)': '{:.2f}'
            }).highlight_max(subset=['NMI', 'ARI', 'Accuracy'], color='lightgreen')
            .highlight_min(subset=['Time (s)'], color='lightblue'),
            width='stretch')
            
            # Biểu đồ
            col1, col2 = st.columns(2)
            
            with col1:
                fig_nmi = px.bar(df_cd, x='Method', y='NMI',
                                title="Normalized Mutual Information (NMI)",
                                color='NMI', color_continuous_scale='greens')
                fig_nmi.add_hline(y=0.8, line_dash="dash", line_color="red",
                                 annotation_text="Good threshold")
                st.plotly_chart(fig_nmi, width='stretch')
            
            with col2:
                fig_time = px.scatter(df_cd, x='Time (s)', y='NMI', 
                                    text='Method', size='ARI',
                                    title="NMI vs Thời gian thực hiện",
                                    color='Accuracy', color_continuous_scale='viridis')
                fig_time.update_traces(textposition='top center')
                st.plotly_chart(fig_time, width='stretch')
            
            # Tìm phương pháp tốt nhất
            best_idx = df_cd['NMI'].idxmax()
            best_method = df_cd.iloc[best_idx]
            
            st.success(f"""
            ### 🏆 Phương pháp tốt nhất: **{best_method['Method']}**
            - **NMI**: {best_method['NMI']:.3f}
            - **ARI**: {best_method['ARI']:.3f}
            - **Accuracy**: {best_method['Accuracy']:.3f}
            - **Time**: {best_method['Time (s)']:.2f}s
            """)

# Footer
st.markdown("---")
st.markdown("""
""", unsafe_allow_html=True)
