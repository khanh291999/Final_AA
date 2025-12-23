# Graph Sub-sampling Web App 📊

Ứng dụng web để so sánh và minh họa **7 phương pháp lấy mẫu đồ thị** từ bài báo khoa học.

## 🎯 Tổng quan

Triển khai 7 phương pháp lấy mẫu đồ thị (Graph Sub-sampling) từ bài báo:  
**"Graph sub-sampling for divide-and-conquer algorithms in large networks"** - Eric Yanchenko (2025)

### 7 Phương pháp:
1. **Random Node (RN)** - Lấy nút ngẫu nhiên
2. **Degree Node (DN)** - Lấy nút theo bậc (ưu tiên KOL)
3. **Random Edge (RE)** - Lấy cạnh ngẫu nhiên
4. **BFS** - Tìm kiếm theo chiều rộng
5. **DFS** - Tìm kiếm theo chiều sâu
6. **Random Node-Neighbor (RNN)** - Lấy nút và hàng xóm
7. **Random Walk (RW)** - Bước đi ngẫu nhiên

## 🚀 Cài đặt môi trường

### Yêu cầu:
- Python 3.11 trở lên (đã test với Python 3.14)
- pip (đi kèm với Python)

### Bước 1: Tạo virtual environment (nếu chưa có)

```powershell
# Tạo virtual environment
python -m venv venv
```

### Bước 2: Kích hoạt virtual environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
.\venv\Scripts\activate.bat
```

### Bước 3: Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

**Thư viện sẽ được cài:**
- networkx==3.2.1
- streamlit
- plotly
- matplotlib
- scikit-learn
- scipy
- pandas
- numpy

## ▶️ Chạy ứng dụng

### Cách 1: Dùng streamlit command (Đơn giản)

```powershell
streamlit run app.py
```

### Cách 2: Dùng đường dẫn đầy đủ

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

### Cách 3: Chỉ định port khác (nếu port 8501 bận)

```powershell
streamlit run app.py --server.port 8502
```

**App sẽ tự động mở tại:** http://localhost:8501

### Dừng ứng dụng

Nhấn `Ctrl + C` trong terminal để dừng app

## 🎨 Tính năng

### Tab 1: So sánh phương pháp
- Chọn loại đồ thị (Karate Club, Stochastic Block Model, Random Graph, Grid)
- Điều chỉnh tham số (số nodes, sample size, seed)
- So sánh nhiều phương pháp cùng lúc
- Xem biểu đồ: Nodes/Edges, Density, Clustering, Time

### Tab 2: Minh họa chi tiết
- Visualize trực quan: Random Node vs BFS vs DFS
- Xem cách mỗi phương pháp chọn nodes
- So sánh density và cấu trúc

### Tab 3: Community Detection
- Tạo đồ thị với communities
- So sánh hiệu suất phát hiện cộng đồng
- Metrics: NMI, ARI, Accuracy
- Tìm phương pháp tốt nhất

## 📁 Cấu trúc project

```
finalcode/
├── app.py                       # Web UI chính
├── graph_sampling.py            # 7 phương pháp lấy mẫu
├── community_detection.py       # Thuật toán phát hiện cộng đồng
├── demo.py                      # Demo command line
├── bfs_dfs_visualization.py     # Visualization BFS vs DFS
├── requirements.txt             # Dependencies
├── README.md                    # File này
└── 25B_AA_5.pdf                # Bài báo gốc
```

## 📦 Thư viện sử dụng

- **networkx** - Xử lý đồ thị
- **streamlit** - Web framework
- **plotly** - Biểu đồ tương tác
- **matplotlib** - Visualization
- **scikit-learn** - Machine learning
- **scipy** - Scientific computing
- **pandas** - Data analysis
- **numpy** - Numerical computing

## 💻 Yêu cầu hệ thống

- **Python**: 3.11 trở lên (đã test với Python 3.14)
- **RAM**: 4GB+
- **Browser**: Chrome, Firefox, Edge (phiên bản mới)

## 🔧 Các lệnh khác

### Chạy demo command line

```powershell
python demo.py
```

### Tạo visualization BFS vs DFS

```powershell
python bfs_dfs_visualization.py
```

## 📊 Kết quả mẫu

Testing trên Stochastic Block Model (200 nodes, 4 communities):

| Phương pháp | NMI | ARI | Thời gian |
|------------|-----|-----|-----------|
| RNN | 0.965 | 0.973 | Nhanh ⚡ |
| DFS | 0.883 | 0.893 | Nhanh ⚡ |
| Random Walk | 0.858 | 0.853 | Trung bình |
| Random Node | 0.820 | 0.829 | Rất nhanh ⚡⚡ |
| BFS | 0.632 | 0.491 | Nhanh ⚡ |
| Random Edge | 0.028 | 0.007 | Nhanh ⚡ |

**→ Random Node-Neighbor (RNN) cho kết quả tốt nhất!**

## 📚 Tài liệu tham khảo

Yanchenko, E. (2025). Graph sub-sampling for divide-and-conquer algorithms in large networks. April 3, 2025.

---

