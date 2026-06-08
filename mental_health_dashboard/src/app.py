import os
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from utils import (get_model, predict_student, find_dangerous_paths, student_in_dangerous_paths, draw_dangerous_paths)
import pandas as pd

# LOAD MODEL
model = get_model()
if model is None:
    st.error("Model chưa được load")
    st.stop()

# CẤU HÌNH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mutual_knn_leiden_model.pkl")

st.set_page_config(
    page_title="Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

# LẤY DỮ LIỆU TỪ MODEL
# DataFrame sinh viên
df = model["df"]
# Student graph
G = model["graph"]
# Bảng risk cộng đồng
risk_by_comm = model["risk_by_comm"]

# TIÊU ĐỀ
st.title("HỆ THỐNG DỰ ĐOÁN NGUY CƠ SỨC KHỎE TÂM THẦN")

# SIDEBAR
st.sidebar.header("Nhập thông tin sinh viên")

cgpa = st.sidebar.slider("🎓 Điểm trung bình", 0.0, 4.0, 3.0, 0.1)
sleep_duration = st.sidebar.slider("😴 Thời gian ngủ", 0.0, 12.0, 8.0, 0.5)
study_hours = st.sidebar.slider("📚 Thời gian học tập", 0.0, 13.0, 5.0, 0.5)
social_media_hours = st.sidebar.slider("📱 Thời gian sử dụng MXH", 0.0, 10.0, 4.0, 0.5)
physical_activity = st.sidebar.slider("🏃 Thời gian vận động", 0, 150, 50, 5)
stress_level = st.sidebar.slider("😵 Mức độ căng thẳng", 0, 10, 5, 1)

# DỰ ĐOÁN
if st.sidebar.button("🔍 Dự đoán"):
    # TẠO INPUT DATA
    student_data = {
        "cgpa": cgpa,
        "sleep_duration": sleep_duration,
        "study_hours": study_hours,
        "social_media_hours": social_media_hours,
        "physical_activity": physical_activity,
        "stress_level": stress_level
    }

    # PREDICT
    result = predict_student(student_data)

    st.session_state["result"] = result

    # KẾT QUẢ DỰ ĐOÁN
    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mã cộng đồng", result["community"])
    col2.metric("Điểm rủi ro", f"{result['risk']}/100")
    col3.metric("Độ tin cậy", f"{result['confidence']}%")

    progress_value = max(0.0, min(result["risk"] / 100, 1.0))
    st.progress(progress_value)

    # MỨC ĐỘ RỦI RO
    if result["risk_level"] == "🔴 Rất cao":
        st.markdown("""
        <div style="
            background-color:#ff4b4b;
            padding:15px;
            border-radius:10px;
            color:white;
            font-size:18px;
            font-weight:bold;">
            Điểm rủi ro ở mức rất cao
        </div>
        """, unsafe_allow_html=True)

    elif result["risk_level"] == "🟠 Cao":
        st.markdown("""
        <div style="
            background-color:#ff8c00;
            padding:15px;
            border-radius:10px;
            color:white;
            font-size:18px;
            font-weight:bold;">
            Điểm rủi ro ở mức cao
        </div>
        """, unsafe_allow_html=True)

    elif result["risk_level"] == "🟡 Trung bình":
        st.markdown("""
        <div style="
            background-color:#ffd700;
            padding:15px;
            border-radius:10px;
            color:black;
            font-size:18px;
            font-weight:bold;">
            Điểm rủi ro ở mức trung bình
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background-color:#28a745;
            padding:15px;
            border-radius:10px;
            color:white;
            font-size:18px;
            font-weight:bold;">
            Điểm rủi ro ở mức thấp
        </div>
        """, unsafe_allow_html=True)

    if result["anomaly"]:
        st.error("⚠️ Sinh viên có dấu hiệu bất thường")

    st.markdown(f"""
        ### 📌 Giải thích kết quả
        - Sinh viên xếp vào cộng đồng có mã: **{result['community']}**
        - Sinh viên thuộc cộng đồng có nguy cơ: **{result['community_level']}**
        - Điểm rủi ro của sinh viên: **{result['risk']}/100**
        - Mức nguy cơ của sinh viên: **{result['risk_level']}**
        - Độ tin cậy của mô hình: **{result['confidence']}%**

        Độ tin cậy càng cao → sinh viên càng giống với nhóm sinh viên tương đồng trong cộng đồng.
    """)

    # ĐƯỜNG LAN TRUYỀN NGUY HIỂM
    path_result = find_dangerous_paths(result["community"])
    # lấy super spreader từ path_result
    super_spreader = (path_result["source"] if path_result is not None else "N/A")
    st.subheader(f"🔥 ĐƯỜNG LAN TRUYỀN NGUY HIỂM TỪ SUPER SPREADER {super_spreader}")

    path_result = find_dangerous_paths(result["community"])

    if path_result is not None:
        fig = draw_dangerous_paths(path_result)
        st.pyplot(fig)

        st.markdown("### 📌 Các tuyến lan truyền")
        shown_paths = set()
        displayed_count = 0

        for p in path_result["paths"]:
            path_tuple = tuple(p["path"])
            if path_tuple in shown_paths: 
                continue
            shown_paths.add(path_tuple)

            pretty_path = " → ".join(map(str, p["path"]))
            st.write(f"#{p['branch']} | {pretty_path}")

            displayed_count += 1
            if displayed_count >= 5:
                break

        # KIỂM TRA STUDENT CÓ LIÊN QUAN
        involved = student_in_dangerous_paths(result["neighbors"], path_result["paths"])

        # PHÂN LOẠI MỨC ĐỘ NGUY CƠ
        low_case = (result["risk"] < 40 and result["community_level"] == "🟢 Thấp")
        medium_case = ((40 <= result["risk"] < 50) or ( result["community_level"] == "🟡 Trung bình"))
        high_case = not low_case and not medium_case

        # TRƯỜNG HỢP NGUY HIỂM
        if involved and high_case:
            dangerous_neighbors = sorted(list(set(item["node"] for item in involved)))

            dangerous_neighbors = ", ".join(map(str, dangerous_neighbors))

            unique_paths = []
            seen_paths = set()

            for item in involved:
                path_tuple = tuple(item["path"])
                if path_tuple in seen_paths:
                    continue

                seen_paths.add(path_tuple)
                pretty_path = " → ".join(map(str, item["path"]))
                unique_paths.append(f"• Tuyến #{item['branch']}: {pretty_path}")

                if len(unique_paths) >= 5:
                    break

            all_paths_text = "<br>".join(unique_paths)

            st.markdown(f"""
            <div style="
                background:#fdecea;
                padding:20px;
                border-radius:12px;
                border-left:7px solid #dc3545;
                margin-bottom:15px;
                color:#721c24;
            ">

            <h3>🔥 Phát hiện nguy cơ lan truyền tâm lý</h3>

            <p>
                Hệ thống phát hiện sinh viên có mức độ tương đồng cao với 
                <b>sinh viên có mã là {dangerous_neighbors}</b> 
                đang xuất hiện trong những tuyến lan truyền nguy hiểm
                của cộng đồng.
            </p>

            <p>
                Điều này cho thấy sinh viên có khả năng chịu ảnh hưởng từ các
                nhóm có mức độ rủi ro cao và có liên kết mạnh trong mạng lưới cộng đồng.
            </p>

            <p>
            Các dấu hiệu có thể liên quan gồm:
            </p>

            <ul>
                <li>Căng thẳng học tập</li>
                <li>Áp lực tâm lý</li>
                <li>Ảnh hưởng từ xã hội</li>
                <li>Thiếu ngủ kéo dài</li>
                <li>Ít vận động</li>
            </ul>

            <h4>📌 Các tuyến lan truyền nguy hiểm</h4>

            <div style="
                background:white;
                padding:15px;
                border-radius:10px;
                margin-top:10px;
                line-height:1.8;
                font-weight:bold;
                color:#d32f2f;
            ">
                {all_paths_text}
            </div>

            <p style="margin-top:15px;">
            🧠 Hệ thống khuyến nghị theo dõi thêm tình trạng sức khỏe tinh thần
            và mức độ căng thẳng của sinh viên.
            </p>

            </div>
            """, unsafe_allow_html=True)

        elif involved and medium_case:
            unique_paths = []
            seen_paths = set()

            for item in involved:
                path_tuple = tuple(item["path"])
                if path_tuple in seen_paths:
                    continue

                seen_paths.add(path_tuple)
                pretty_path = " → ".join(map(str, item["path"]))

                unique_paths.append(f"• Tuyến #{item['branch']}: {pretty_path}")
                if len(unique_paths) >= 5:
                    break

            all_paths_text = "<br>".join(unique_paths)

            st.markdown(f"""
            <div style="
                background:#fff3cd;
                padding:20px;
                border-radius:12px;
                border-left:7px solid #ffc107;
                margin-bottom:15px;
                color:#856404;
            ">

            <h3>🟡 Có dấu hiệu cần theo dõi</h3>

            <p>
                Hệ thống phát hiện sinh viên có liên quan đến một số tuyến
                lan truyền trong cộng đồng.
            </p>

            <p>
                Tuy nhiên mức độ rủi ro hiện tại chỉ ở mức trung bình.
            </p>

            <ul>
                <li>Cộng đồng có mức nguy cơ trung bình hoặc ổn định</li>
                <li>Sinh viên chưa xuất hiện dấu hiệu nguy hiểm cao</li>
                <li>Cần theo dõi thêm trạng thái tâm lý và mức độ căng thẳng</li>
            </ul>

            <h4>📌 Các tuyến liên quan</h4>

            <div style="
                background:white;
                padding:15px;
                border-radius:10px;
                margin-top:10px;
                line-height:1.8;
                font-weight:bold;
                color:#856404;
            ">
                {all_paths_text}
            </div>

            <p style="margin-top:15px;">
            🟡 Khuyến nghị duy trì nghỉ ngơi hợp lý,
            giảm áp lực học tập và theo dõi thêm trong thời gian tới.
            </p>

            </div>
            """, unsafe_allow_html=True)

        elif involved and low_case:
            st.markdown(f"""
            <div style="
                background:#d4edda;
                padding:20px;
                border-radius:12px;
                border-left:7px solid #28a745;
                margin-bottom:15px;
                color:#155724;
            ">

            <h3>✅ Phát hiện liên kết nhưng mức độ an toàn</h3>

            <p>
                Hệ thống phát hiện sinh viên có liên quan tới một số tuyến
                lan truyền trong cộng đồng.
            </p>

            <p>
                Tuy nhiên:
            </p>

            <ul>
                <li>Cộng đồng <b>{result['community']}</b> có mức rủi ro thấp</li>
                <li>Điểm rủi ro của sinh viên đang ở mức thấp</li>
                <li>Không có dấu hiệu nguy hiểm đáng lo ngại</li>
            </ul>

            <p>
                🟢 Hiện tại sinh viên không cần quá lo lắng.
                Hệ thống chỉ khuyến nghị duy trì thói quen sinh hoạt lành mạnh
                và theo dõi sức khỏe tinh thần định kỳ.
            </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.success(
                "✅ Không phát hiện sinh viên liên quan đến tuyến lan truyền nguy hiểm"
            )

st.subheader("👥 Danh sách sinh viên tương đồng")

result = st.session_state.get("result", None)

if result is None:
    st.info("⚠️ Vui lòng bấm 'Dự đoán' để xem phân tích hàng xóm")
else:
    neighbors = result.get("neighbors", [])
    if not neighbors:
        st.info("Không có hàng xóm trực tiếp")
    else:
        h1, h2, h3, h4, h5 = st.columns([2, 2, 2, 2, 4])
        h1.write("**Mã sinh viên**")
        h2.write("**Mã cộng đồng**")
        h3.write("**Điểm rủi ro (%)**")
        h4.write("**Mức độ**")
        h5.write("**Chi tiết**")

        for n in neighbors[:10]:
            neighbor_risk = df.loc[n, "risk_score"]
            neighbor_comm = df.loc[n, "community"]

            # mapping mức độ tiếng Việt
            if neighbor_risk > 0.6:
                danger_vi = "🔴 Rất cao"
            elif neighbor_risk > 0.5:
                danger_vi = "🟠 Cao"
            elif neighbor_risk > 0.4:
                danger_vi = "🟡 Trung bình"
            else:
                danger_vi = "🟢 Thấp"

            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 4])

            col1.write(f"**{n}**")
            col2.write(f"{neighbor_comm}")
            col3.write(f"{round(neighbor_risk * 100, 2)}")
            col4.write(danger_vi)

            with col5:
                with st.expander("🔎 Xem chi tiết"):
                    if n in df.index:
                        row = df.loc[n]
                        gender_map = {0: "Nữ", 1: "Nam"}
                        depression_map = {0: "Không", 1: "Có"}

                        st.markdown(f"""
                            **Mã sinh viên:** {n}  
                            **Tuổi:** {row.get('age','N/A')}  
                            **Giới tính:** {gender_map.get(row.get('gender'), 'N/A')}  
                            **Khoa/Viện:** {row.get('department','N/A')}  
                            **Điểm trung bình:** {row.get('cgpa','N/A')}  
                            **Thời gian ngủ (giờ):** {row.get('sleep_duration','N/A')}  
                            **Thời gian học tập (giờ):** {row.get('study_hours','N/A')}  
                            **Thời gian sử dụng MXH (giờ):** {row.get('social_media_hours','N/A')}  
                            **Thời gian vận động (phút):** {row.get('physical_activity','N/A')}  
                            **Mức độ căng thẳng:** {row.get('stress_level','N/A')}  
                            **Trầm cảm:** {depression_map.get(row.get('depression'), 'N/A')}  
                            **Điểm rủi ro:** {round(neighbor_risk * 100, 2)}
                        """)

# BẢNG CỘNG ĐỒNG
st.subheader("📋 BẢNG XẾP HẠNG RỦI RO CỘNG ĐỒNG")
st.markdown("**Super Spreader**: Sinh viên có ảnh hưởng mạnh nhất trong cộng đồng",unsafe_allow_html=True)

display_df = risk_by_comm.copy()

h1, h2, h3, h4, h5, h6 = st.columns([2, 2, 2, 2, 2, 5])

h1.write("**Mã cộng đồng**")
h2.write("**Mức độ**")
h3.write("**Điểm rủi ro TB (%)**")
h4.write("**Số lượng sinh viên**")
h5.write("**Super Spreader**")
h6.write("**Chi tiết Super Spreader**")

for _, row in display_df.iterrows():

    comm_id = row["community_id"]
    level = row["level"]
    avg_risk = row["avg_risk"]
    size = row["size"]

    super_spreader = row.get("super_spreader", None)
    super_spreader = int(super_spreader) if pd.notna(super_spreader) else None

    c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 2, 2, 5])

    c1.write(f"**{comm_id}**")
    c2.write(f"{level}")
    c3.write(f"{avg_risk}")
    c4.write(f"{size}")

    if super_spreader is not None:
        c5.write(f"🔥 {super_spreader}")
    else:
        c5.write("—")

    if super_spreader is not None:
        with c6.expander("🔎 Chi tiết"):
            if super_spreader in df.index:
                row_s = df.loc[super_spreader]
                gender_map_s = {0: "Nữ", 1: "Nam"}
                depression_map_s = {0: "Không", 1: "Có"}
                
                st.markdown(f"""
                            **Mã sinh viên:** {super_spreader}  
                            **Tuổi:** {row_s.get('age','N/A')}  
                            **Giới tính:** {gender_map_s.get(row_s.get('gender'), 'N/A')}  
                            **Khoa/Viện:** {row_s.get('department','N/A')}  
                            **Điểm trung bình:** {row_s.get('cgpa','N/A')}  
                            **Thời gian ngủ (giờ):** {row_s.get('sleep_duration','N/A')}  
                            **Thời gian học tập (giờ):** {row_s.get('study_hours','N/A')}  
                            **Thời gian sử dụng MXH (giờ):** {row_s.get('social_media_hours','N/A')}  
                            **Thời gian vận động (phút):** {row_s.get('physical_activity','N/A')}  
                            **Mức độ căng thẳng:** {row_s.get('stress_level','N/A')}  
                            **Trầm cảm:** {depression_map_s.get(row_s.get('depression'), 'N/A')}  
                            **Điểm rủi ro:** {round(row_s.get('risk_score', 0)*100, 2)}
                        """)
            else:
                st.write("—")
    else:
        c6.write("—")

# PHÂN BỐ ĐIỂM RỦI RO
st.subheader("📈 PHÂN BỐ ĐIỂM RỦI RO")
plot_df = df.copy()
plot_df["risk_percent"] = (plot_df["risk_score"] * 100).round(2)
fig = px.histogram(plot_df, x="risk_percent", nbins=30, labels={"risk_percent": "Điểm rủi ro", "count": "Số lượng"})
fig.update_traces(hovertemplate="Điểm rủi ro: %{x}<br>Số lượng: %{y}<extra></extra>")
fig.update_layout(yaxis_title="Số lượng")
st.plotly_chart(fig, use_container_width=True)

# ĐỒ THỊ MỐI LIÊN KẾT CỘNG ĐỒNG
st.subheader("🌐 MỐI LIÊN KẾT GIỮA CÁC CỘNG ĐỒNG")
community_graph = nx.Graph()

# Thêm node
for _, row in risk_by_comm.iterrows():
    comm_id = row["community_id"]
    risk = row["avg_risk"]
    size = row["size"]
    community_graph.add_node(comm_id, risk=risk, size=size)

# Xây dựng cạnh
edge_weights = {}
for u, v in G.edges():
    cu = df.loc[u, "community"]
    cv = df.loc[v, "community"]
    if cu != cv:
        edge = tuple(sorted((cu, cv)))
        edge_weights[edge] = (edge_weights.get(edge, 0) + 1)

# Lọc cạnh mạnh
MIN_EDGE_WEIGHT = 15
for (cu, cv), weight in edge_weights.items():
    if weight >= MIN_EDGE_WEIGHT:
        community_graph.add_edge(cu, cv, weight=weight)

# Hàm chuyển risk sang màu
def risk_to_color(risk):
    if risk < 40:
        if risk < 25:
            return "#003300"
        elif risk < 30:
            return "#004d00"
        elif risk < 35:
            return "#008000"
        return "#00aa00"

    elif risk < 50:
        if risk < 45:
            return "#f6ff00" 
        return "#d0d302"

    elif risk < 60:
        if risk < 55:
            return "#FFA54F"
        return "#FF7F24"
    
    else:
        if risk < 65:
            return "#FF4D4D"
        elif risk < 70:
            return "#FF0000"
        elif risk < 75:
            return "#DD0000"
        return "#990000"
    
# Vẽ đồ thị
node_colors = [risk_to_color(community_graph.nodes[n]["risk"]) for n in community_graph.nodes()]
node_sizes = [community_graph.nodes[n]["size"] * 20 for n in community_graph.nodes()]
edge_widths = [community_graph[u][v]["weight"] * 0.02 for u, v in community_graph.edges()]

fig_comm = plt.figure(figsize=(14, 12))
if len(community_graph.nodes()) > 0:
    pos = nx.spring_layout(community_graph, seed=42, k=1.5)
else:
    st.warning("Graph rỗng - không thể hiển thị")
    st.stop()

nx.draw_networkx_nodes(community_graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.95)
nx.draw_networkx_edges(community_graph, pos, width=edge_widths, alpha=0.15, edge_color="gray")
nx.draw_networkx_labels(community_graph, pos, font_size=11, font_weight="bold", font_color="white")

plt.axis("off")
st.pyplot(fig_comm)