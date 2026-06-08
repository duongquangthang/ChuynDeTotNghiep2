import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# ĐƯỜNG DẪN FILE MODEL
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mutual_knn_leiden_model.pkl")
model = None
FEATURES = None

# LOAD MODEL
def load_model():
    """
    Load model từ file .pkl.

    Sau khi load:
    - Gán vào biến global model
    - Lấy danh sách FEATURES

    Returns
    -------
    dict
        Dictionary chứa toàn bộ dữ liệu model.
    """

    global model, FEATURES
    # Load model
    model = joblib.load(MODEL_PATH)
    # Lấy feature list
    FEATURES = model["features"]

    return model

# GET MODEL
def get_model():
    """
    Lấy model hiện tại. Nếu model chưa load -> tự động load.

    Returns
    -------
    dict
        Model dictionary.
    """

    if model is None:
        load_model()

    return model

# PHÂN LOẠI MỨC ĐỘ RỦI RO
def risk_level(score):
    """
    Chuyển risk score -> nhãn tiếng Việt.

    Parameters
    ----------
    score : float
        Risk score trong khoảng [0, 1].

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

# LẤY THÔNG TIN NODE AN TOÀN
def get_node_info(node, df):
    """
    Lấy thông tin node an toàn.

    Parameters
    ----------
    node : int
        ID node cần lấy thông tin.
    df : pd.DataFrame
        DataFrame sinh viên.

    Returns
    -------
    dict | None
        Dictionary thông tin node.
        Nếu node không tồn tại:
        -> trả về None
    """

    # Node không tồn tại
    if node not in df.index:
        return None

    # Lấy dòng dữ liệu
    row = df.loc[node]

    return {
        "node": int(node),
        "risk": float(row.get("risk_score", 0)),
        "community": int(row.get("community", -1)),
        "age": row.get("age", None),
        "major": row.get("major", None),
        "gender": row.get("gender", None)
    }

# SUPER SPREADER DETECTION
def detect_super_spreader(sim_scores, top_k_idx, node_influence, threshold=0.75):
    """
    Phát hiện super spreader.

    Parameters
    ----------
    sim_scores : ndarray
        Mảng similarity scores.
    top_k_idx : ndarray
        Danh sách K node gần nhất.
    node_influence : dict
        Dictionary node -> influence score.
    threshold : float, optional
        Ngưỡng xác định super spreader.

    Returns
    -------
    dict{"score": float, "is_super_spreader": bool}
    """

    # Influence trung bình
    influence = np.mean([node_influence.get(i, 0) for i in top_k_idx])
    # Similarity trung bình
    sim_strength = np.mean(sim_scores[top_k_idx])
    # Score tổng hợp
    score = 0.6 * influence + 0.4 * sim_strength
    # Clamp về [0, 1]
    score = float(np.clip(score, 0, 1))

    return {
        "score": score,
        "is_super_spreader": score > threshold
    }   


# PHÂN TÍCH HÀNG XÓM (NEIGHBORS)
def analyze_neighbors(node_list, df, node_influence):
    """
    Phân tích danh sách neighbor nodes.

    Parameters
    ----------
    node_list : list
        Danh sách node cần phân tích.
    df : pd.DataFrame
        DataFrame sinh viên.
    node_influence : dict
        Dictionary influence score.

    Returns
    -------
    pd.DataFrame
        Bảng phân tích neighbors.
    """

    rows = []

    for n in node_list:
        # Bỏ qua node không tồn tại
        if n not in df.index:
            continue

        row = df.loc[n]
        # Risk score
        risk = float(row.get("risk_score", 0))

        if risk > 0.6:
            level = "🔴 Rất cao"
        elif risk > 0.5:
            level = "🟠 Cao"
        elif risk > 0.4:
            level = "🟡 Trung bình"
        else:
            level = "🟢 Thấp"

        rows.append({
            "node": int(n),
            "community": int(row.get("community", -1)),
            "risk": round(risk * 100, 2),
            "level_vi": level,
            "influence": float(node_influence.get(n, 0)),
            "age": row.get("age", None),
            "major": row.get("major", None),
            "gender": row.get("gender", None)
        })

    # Sort risk giảm dần
    return pd.DataFrame(rows).sort_values("risk", ascending=False)

# KIỂM TRA NODE CÓ NẰM TRONG PATH NGUY HIỂM
def check_path_involvement(node, dangerous_paths):
    """
    Kiểm tra node có nằm trong đường lan truyền nguy hiểm không.

    Parameters
    ----------
    node : int
        Node cần kiểm tra.
    dangerous_paths : list
        Danh sách dangerous paths.

    Returns
    -------
    list
        Các path chứa node.
    """

    result = []

    for p in dangerous_paths:
        # Node nằm trong path
        if node in p["path"]:
            result.append(p)

    return result

# DỰ ĐOÁN SINH VIÊN
def predict_student(student_data):
    """
    Dự đoán:
    - Community
    - Risk score
    - Risk level
    - Super spreader
    - Neighbor nodes

    Parameters
    ----------
    student_data : dict
        Dữ liệu sinh viên mới.

    Returns
    -------
    dict
        Kết quả dự đoán.
    """

    # Load model
    model = get_model()
    # Lấy dữ liệu từ model
    X_train = model["X_train"]

    node2comm = model["node2comm"]

    minmax_scaler = model["minmax_scaler"]

    standard_scaler = model["standard_scaler"]

    risk_scaler = model["risk_scaler"]

    k = model["k"]

    risk_by_comm = model["risk_by_comm"]

    adj_list = model.get("adj_list", {})

    node_influence = model.get("node_influence", {})

    x = pd.DataFrame([student_data])

    # DỰ ĐOÁN CỘNG ĐỒNG
    x_scaled = minmax_scaler.transform(x[FEATURES])

    # Tính cosine similarity
    sims = cosine_similarity(x_scaled, X_train)[0]

    # Lấy top-k node giống nhất
    top_k_idx = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_idx]

    # Voting theo similarity
    community_scores = {}
    for idx, sim in zip(top_k_idx, top_k_sims):
        comm = node2comm[idx]
        community_scores[comm] = (community_scores.get(comm, 0) + sim)

    # Community có score cao nhất
    pred_comm = max(community_scores, key=community_scores.get)
    # Level community
    community_level = risk_by_comm[risk_by_comm["community_id"] == pred_comm]["level"].values[0]

    # RISK SCORE
    z = pd.DataFrame(standard_scaler.transform(x[FEATURES]), columns=FEATURES)

    risk_raw = (
        0.30 * z["stress_level"].iloc[0]
        + 0.20 * (1 - z["sleep_duration"].iloc[0])
        + 0.10 * z["study_hours"].iloc[0]
        + 0.15 * z["social_media_hours"].iloc[0]
        + 0.15 * (1 - z["physical_activity"].iloc[0])
        + 0.10 * (1 - z["cgpa"].iloc[0])
    )

    student_risk = risk_scaler.transform([[risk_raw]])[0][0]
    student_risk = np.clip(student_risk, 0, 1)

    # CONFIDENCE SCORE
    confidence = np.mean(top_k_sims)
    # SUPER SPREADER SCORE
    super_spreader_score = np.mean([node_influence.get(i, 0) for i in top_k_idx])
    is_super_spreader = super_spreader_score > 0.75

    # LẤY NEIGHBORS
    neighbors = []
    for idx in top_k_idx:
        neighbors.extend(adj_list.get(idx, []))

    # Remove duplicate
    neighbors = list(set(neighbors))[:10]

    return {
        "community": int(pred_comm),
        "community_level": community_level,
        "risk": round(student_risk * 100, 2),
        "risk_level": risk_level(student_risk),
        "confidence": round(confidence * 100, 2),
        "anomaly": confidence < 0.40,
        "super_spreader": {
            "score": float(super_spreader_score),
            "is_super_spreader": bool(is_super_spreader)
        },

        "neighbors": neighbors
    }


# TÌM ĐƯỜNG LAN TRUYỀN NGUY HIỂM
def find_dangerous_paths(community_id, top_n_paths=5, max_depth=6):
    """
    Tìm các đường lan truyền nguy hiểm trong community.
    Bắt đầu từ super spreader và mở rộng theo node có influence cao nhất.

    Parameters
    ----------
    community_id : int
        ID community.
    top_n_paths : int
        Số path cần lấy.
    max_depth : int
        Độ sâu tối đa của mỗi path.

    Returns
    -------
    dict | None
        Thông tin dangerous paths.
    """

    model = get_model()

    G = model["graph"]
    df = model["df"]
    node_influence = model["node_influence"]

    # LẤY SUPER SPREADER
    risk_by_comm = model["risk_by_comm"]
    row = risk_by_comm[risk_by_comm["community_id"] == community_id]
    if row.empty:
        return None

    source = row["super_spreader"].values[0]
    if pd.isna(source):
        return None
    source = int(source)

    # SUBGRAPH CỦA COMMUNITY
    community_nodes = df[ df["community"] == community_id].index
    G_sub = G.subgraph(community_nodes).copy()

    if source not in G_sub.nodes():
        return None

    # TÌM NEIGHBORS QUAN TRỌNG
    top_neighbors = []
    for n in G_sub.neighbors(source):
        influence = node_influence.get(n, 0)
        weight = G_sub.get_edge_data(source, n).get("weight", 1.0)
        top_neighbors.append((n, influence, weight))

    # Sort giảm dần
    top_neighbors.sort( key=lambda x: x[1], reverse=True)

    # XÂY DỰNG PATHS
    dangerous_paths = []
    for branch, (start_node, start_score, start_weight) in enumerate(top_neighbors[:top_n_paths], 1):
        # Path khởi tạo
        path = [source, start_node]
        scores = [node_influence.get(source, 0), start_score]
        current = start_node

        # Mở rộng path
        for _ in range(2, max_depth):
            candidates = []
            # Không lặp node
            for neigh in G_sub.neighbors(current):
                if neigh not in path:
                    influence = node_influence.get(neigh, 0)
                    weight = G_sub.get_edge_data(current, neigh).get("weight", 1.0)
                    candidates.append((neigh, influence, weight))

            # Không còn neighbor
            if not candidates:
                break

            # Chọn node nguy hiểm nhất
            candidates.sort(key=lambda x: x[1], reverse=True)
            next_node, next_score, _ = candidates[0]
            path.append(next_node)
            scores.append(next_score)
            current = next_node

        # TÍNH PATH SCORE
        avg_score = np.mean(scores)
        min_score = min(scores)
        path_score = (avg_score * (1 + 0.5 * (avg_score - min_score)))

        dangerous_paths.append({
            "branch": branch,
            "path": path,
            "scores": scores,
            "avg_score": float(avg_score),
            "path_score": float(path_score)
        })

    # Sort path mạnh nhất
    dangerous_paths = sorted(dangerous_paths, key=lambda x: x["path_score"], reverse=True)

    return {
        "community": community_id,
        "source": source,
        "paths": dangerous_paths
    }

# KIỂM TRA STUDENT CÓ THUỘC ĐƯỜNG LAN TRUYỀN KHÔNG=
def student_in_dangerous_paths(neighbors, dangerous_paths):
    """
    Kiểm tra neighbor của student có nằm trong dangerous paths không.

    Parameters
    ----------
    neighbors : list
        Danh sách neighbor nodes.
    dangerous_paths : list
        Danh sách dangerous paths.

    Returns
    -------
    list
        Danh sách node liên quan.
    """

    involved = []
    for path_info in dangerous_paths:
        path_nodes = path_info["path"]
        for n in neighbors:
            if n in path_nodes:
                involved.append({
                    "node": n,
                    "branch": path_info["branch"],
                    "path": path_nodes
                })
    return involved

# VẼ ĐỒ THỊ LAN TRUYỀN NGUY HIỂM
def draw_dangerous_paths(path_result):
    """
    Visualize dangerous paths bằng graph.

    Parameters
    ----------
    path_result : dict
        Kết quả từ find_dangerous_paths()

    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib.
    """

    model = get_model()
    G = model["graph"]
    df = model["df"]

    source = path_result["source"]
    paths = path_result["paths"]

    G_viz = nx.MultiDiGraph()

    # THÊM NODE + EDGE CHO TỪNG PATH
    for idx, p in enumerate(paths):
        path = p["path"]
        # Add nodes
        for node in path:
            if node not in G_viz:
                G_viz.add_node(node)

        # Add edges
        for k in range(len(path) - 1):
            u = path[k]
            v = path[k + 1]
            G_viz.add_edge(u, v, path_id=idx)

    fig = plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G_viz, seed=42, k=1.0, iterations=300)

    # MÀU CHO TỪNG PATH
    path_colors = [
        "#E03131",
        "#1971C2",
        "#2F9E44",
        "#7048E8",
        "#F76707",
        "#C2255C",
        "#1098AD", 
        "#343A40"
    ]

    # TÍNH ĐỘ CONG RIÊNG CHO TỪNG PATH
    total_paths = len(paths)
    if total_paths == 1:
        path_rads = [0]
    else:
        path_rads = np.linspace(-0.55, 0.55, total_paths)

    # VẼ TỪNG PATH ĐỘC LẬP
    for idx, p in enumerate(paths):
        path = p["path"]
        color = path_colors[idx % len(path_colors)]
        rad = path_rads[idx]
        for k in range(len(path) - 1):
            u = path[k]
            v = path[k + 1]
            nx.draw_networkx_edges(
                G_viz,
                pos,
                edgelist=[(u, v)],
                edge_color=color,
                width=1.5,
                alpha=0.85,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=4,
                connectionstyle=f'arc3,rad={rad}',
                min_source_margin=10,
                min_target_margin=10
            )

    # MAP RISK -> COLOR
    def risk_to_color(risk):
        """
        Convert risk score -> color.
        """
        if risk < 0.40:
            if risk < 0.25:
                return "#C3F0CA"
            elif risk < 0.30:
                return "#8CE99A"
            elif risk < 0.35:
                return "#69DB7C"
            return "#40C057"

        elif risk < 0.50:
            if risk < 0.45:
                return "#FFF3BF"
            return "#FFD43B"

        elif risk < 0.60:
            if risk < 0.55:
                return "#FFC078"
            return "#FF922B"

        else:
            if risk < 0.70:
                return "#FFA8A8"
            elif risk < 0.80:
                return "#FF6B6B"
            return "#E03131"

    # NODE COLORS
    all_nodes = list(G_viz.nodes())
    node_colors = [risk_to_color(df.loc[n, "risk_score"]) for n in all_nodes]

    # VẼ NODE
    nx.draw_networkx_nodes(
        G_viz,
        pos,
        nodelist=all_nodes,
        node_color=node_colors,
        node_size=520,
        edgecolors="#333333",
        linewidths=1.2,
        alpha=0.95
    )

    # SUPER SPREADER
    source_risk = df.loc[source, "risk_score"]
    source_color = risk_to_color(source_risk)
    nx.draw_networkx_nodes(
        G_viz,
        pos,
        nodelist=[source],
        node_color=source_color,
        node_size=950,
        edgecolors="black",
        linewidths=3
    )

    # LABELS
    labels = {n: f"{n}\n{df.loc[n, 'risk_score']:.2f}" for n in all_nodes}

    nx.draw_networkx_labels(
        G_viz,
        pos,
        labels,
        font_size=5.8,
        font_weight="bold",
        font_color="#222222"
    )

    plt.axis("off")
    plt.tight_layout()
    return fig
