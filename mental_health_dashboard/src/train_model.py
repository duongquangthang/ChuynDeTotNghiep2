import os
import time
import joblib
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np

# FEATURE ĐƯỢC SỬ DỤNG
FEATURES = [
    "cgpa",
    "sleep_duration",
    "study_hours",
    "social_media_hours",
    "physical_activity",
    "stress_level"
]

# KHAI BÁO ĐƯỜNG DẪN FILE
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "student_lifestyle_5000.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mutual_knn_leiden_model.pkl")
GRAPH_PATH = os.path.join(BASE_DIR, "models", "student_graph.gexf")

K = 20

# TẢI DỮ LIỆU
def load_data():
    """
    Đọc dữ liệu sinh viên từ file Excel.

    Returns
    -------
    pd.DataFrame
        DataFrame chứa toàn bộ dữ liệu sinh viên.
    """
    return pd.read_excel(DATASET_PATH)

# CHUẨN HÓA DỮ LIỆU
def scale_data(df):
    """
    Chuẩn hóa dữ liệu theo:
    1. MinMaxScaler:
       - Scale dữ liệu về khoảng [0, 1]
       - Dùng để xây graph KNN
    2. StandardScaler:
       - Chuẩn hóa Z-score
       - Dùng để tính risk score

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu sinh viên.

    Returns
    -------
    tuple (X_minmax, X_z, minmax_scaler, standard_scaler)
        X_minmax : ndarray
            Dữ liệu sau khi MinMax scaling.
        X_z : ndarray
            Dữ liệu sau khi Z-score scaling.
        minmax_scaler : MinMaxScaler
            Object scaler đã fit.
        standard_scaler : StandardScaler
            Object scaler đã fit.
    """
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Dữ liệu dùng cho graph
    X_minmax = minmax_scaler.fit_transform(df[FEATURES])
    # Dữ liệu dùng cho risk score
    X_z = standard_scaler.fit_transform(df[FEATURES])

    return X_minmax, X_z, minmax_scaler, standard_scaler

# XÂY DỰNG ĐỒ THỊ
def build_graph(X):
    """
    Xây dựng đồ thị Mutual KNN từ dữ liệu sinh viên.

    Parameters
    ----------
    X : ndarray
        Ma trận dữ liệu đã được scale.

    Returns
    -------
    networkx.Graph
        Đồ thị student graph.
    """

    A = kneighbors_graph(X, n_neighbors=K, mode="distance", metric="cosine", include_self=False).tocsr()

     # Convert cosine distance -> similarity
    A.data = 1 - A.data

    mutual_A = A.multiply(A.T)
    return nx.from_scipy_sparse_array(mutual_A)

# PHÁT HIỆN CỘNG ĐỒNG BẰNG LEIDEN
def leiden_detection(G):
    """
    Phát hiện community bằng thuật toán Leiden.

    Parameters
    ----------
    G : networkx.Graph
        Đồ thị sinh viên.

    Returns
    -------
    tuple(communities, modularity, runtime)
        communities : list[set]
            Danh sách community.
        modularity : float
            Điểm modularity của graph.
        runtime : float
            Thời gian chạy (giây).
    """

    start = time.time()

    # Convert networkx -> igraph
    g = ig.Graph.from_networkx(G)
    # Chạy Leiden algorithm
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    communities = [set(c) for c in partition]
    # Tính modularity
    modularity = nx.community.modularity(G, communities)

    runtime = time.time() - start
    return communities, modularity, runtime

# TÍNH ĐIỂM RỦI RO
def compute_risk_score(df, X_z):
    """
    Tính risk score cho từng sinh viên.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame gốc.
    X_z : ndarray
        Dữ liệu đã Z-score.

    Returns
    -------
    tuple(df, risk_scaler)
        df : pd.DataFrame
            DataFrame đã thêm cột risk_score.
        risk_scaler : MinMaxScaler
            Scaler dùng normalize risk score.
    """

    # Convert ndarray -> DataFrame
    z = pd.DataFrame(X_z, columns=FEATURES)

    # Công thức risk score
    df["risk_score_raw"] = (
        0.30 * z["stress_level"]
        + 0.20 * (1 - z["sleep_duration"])
        + 0.10 * z["study_hours"]
        + 0.15 * z["social_media_hours"]
        + 0.15 * (1 - z["physical_activity"])
        + 0.10 * (1 - z["cgpa"])
    )

    risk_scaler = MinMaxScaler()
    df["risk_score"] = risk_scaler.fit_transform(df[["risk_score_raw"]])
    return df, risk_scaler

# PHÂN LOẠI MỨC ĐỘ RỦI RO
def risk_level(score):
    """
    Chuyển risk score -> level text.

    Parameters
    ----------
    score : float
        Điểm risk trong khoảng [0, 1].

    Returns
    -------
    str
        Nhãn mức độ rủi ro.
    """

    if score >= 0.6:
        return "🔴 Rất cao"
    elif score >= 0.5:
        return "🟠 Cao"
    elif score >= 0.4:
        return "🟡 Trung bình"
    return "🟢 Thấp"

# PHÂN TÍCH CỘNG ĐỒNG
def analyze_communities(df):
    """
    Phân tích mức độ rủi ro của từng community.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa risk score và community.

    Returns
    -------
    pd.DataFrame
        Bảng thống kê community.
    """

    # Tính risk trung bình theo community
    risk_by_comm = (df.groupby("community")["risk_score"].agg(["mean", "count"]).reset_index())
    # Rename column
    risk_by_comm.columns = ["community_id", "avg_risk", "size"]
    # Convert score -> level
    risk_by_comm["level"] = (risk_by_comm["avg_risk"].apply(risk_level))
    # Convert sang %
    risk_by_comm["avg_risk"] = (risk_by_comm["avg_risk"] * 100).round(2)
    # Sort giảm dần
    risk_by_comm = risk_by_comm.sort_values("avg_risk", ascending=False).reset_index(drop=True)

    return risk_by_comm[["community_id", "level", "avg_risk", "size"]]


# TÍNH INFLUENCE SCORE
def compute_influence_score(G, risk_series):
    """
    Tính influence score cho từng node.

    Parameters
    ----------
    G : networkx.Graph
        Student graph.
    risk_series : pd.Series
        Risk score của sinh viên.

    Returns
    -------
    pd.DataFrame
        DataFrame chứa influence score.
    """

    # Betweenness centrality
    between = nx.betweenness_centrality(G)
    # Degree của node
    degree = dict(G.degree())
    # Danh sách node
    nodes = list(G.nodes())
    # Tạo DataFrame
    df_inf = pd.DataFrame({
        "node": nodes,
        "betweenness": [between[n] for n in nodes],
        "degree": [degree[n] for n in nodes],
        "risk": risk_series[nodes].values
    })

    scaler = MinMaxScaler()
    df_inf[["bet_norm", "deg_norm"]] = scaler.fit_transform(
        df_inf[["betweenness", "degree"]]
    )

    # Công thức influence score
    df_inf["influence_score"] = (0.6 * df_inf["bet_norm"] * df_inf["risk"] + 0.4 * df_inf["deg_norm"])

    # Sort giảm dần
    return df_inf.sort_values("influence_score", ascending=False)

# TÌM SUPER SPREADER TRONG MỖI COMMUNITY
def get_super_spreader_per_comm(df, influence_df):
    """
    Tìm node có influence score cao nhất trong mỗi community.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sinh viên.
    influence_df : pd.DataFrame
        DataFrame influence score.

    Returns
    -------
    pd.DataFrame
        Bảng super spreader theo community.
    """

    df2 = df.copy()
    # Map influence score vào dataframe
    df2["influence_score"] = df2.index.map(influence_df.set_index("node")["influence_score"])

    # Xóa node bị thiếu influence
    df2 = df2.dropna(subset=["influence_score"])

    # Lấy node có influence cao nhất mỗi community
    idx = df2.groupby("community")["influence_score"].idxmax()
    result = df2.loc[idx, ["community"]].copy()
    result["node"] = df2.loc[idx].index
    result["influence_score"] = df2.loc[idx, "influence_score"].values
    # Rename columns
    result = result.rename(columns={"community": "community_id", "node": "super_spreader"})

    return result

# HUẤN LUYỆN MÔ HÌNH
def train_model():
    """
    Pipeline huấn luyện toàn bộ hệ thống.

    Các bước:
    ----------
    1. Load dữ liệu
    2. Scale dữ liệu
    3. Xây dựng graph
    4. Community detection
    5. Tính risk score
    6. Tính influence score
    7. Phân tích community
    8. Lưu model
    """

    # LOAD DATA
    print("LOAD DATA...")
    df = load_data()

    # SCALE DATA
    print("SCALE...")
    X_minmax, X_z, minmax_scaler, standard_scaler = scale_data(df)

    # BUILD GRAPH
    print("BUILD GRAPH...")
    G = build_graph(X_minmax)

    # LEIDEN COMMUNITY DETECTION
    print("LEIDEN...")
    communities, modularity, runtime = leiden_detection(G)
    print(f"Communities: {len(communities)}")
    print(f"Modularity: {modularity:.4f}")
    print(f"Runtime: {runtime:.2f}s")

    # MAP NODE -> COMMUNITY
    node2comm = {node: comm_id for comm_id, community in enumerate(communities) for node in community}
    df["community"] = (df.index.map(node2comm).fillna(-1).astype(int))

    # RISK SCORE
    print("RISK SCORE...")
    df, risk_scaler = compute_risk_score(df, X_z)

    # INFLUENCE SCORE
    risk_series = df["risk_score"]
    influence_df = compute_influence_score(G, risk_series)
    # Dictionary node -> influence score
    node_influence = influence_df.set_index("node")["influence_score"].to_dict()

    # SUPER SPREADER
    super_spreader_df = get_super_spreader_per_comm(df, influence_df)

    # COMMUNITY ANALYSIS
    risk_by_comm = analyze_communities(df)
    # Merge super spreader vào bảng community
    risk_by_comm = risk_by_comm.merge(super_spreader_df, on="community_id", how="left")

    # TẠO THƯ MỤC MODEL
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # ADJACENCY LIST
    adj_list = {node: list(G.neighbors(node)) for node in G.nodes()}

    # MODEL DATA
    MODEL_DATA = {
        "graph": G,
        "features": FEATURES,
        "df": df,
        "X_train": X_minmax,
        "node2comm": node2comm,
        "risk_by_comm": risk_by_comm,
        "minmax_scaler": minmax_scaler,
        "standard_scaler": standard_scaler,
        "risk_scaler": risk_scaler,
        "k": K,
        "adj_list": adj_list,
        "node_influence": node_influence,
        "influence_df": influence_df
    }

    # SAVE MODEL
    joblib.dump(MODEL_DATA, MODEL_PATH)

    # Save graph
    nx.write_gexf(G, GRAPH_PATH)
    print("MODEL SAVED SUCCESSFULLY")

# MAIN
if __name__ == "__main__":
    train_model()