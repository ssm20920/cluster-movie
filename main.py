import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 (앱 제목 및 브라우저 탭 아이콘/제목 설정)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

raw_df = load_data()
total_raw_count = len(raw_df)

# 필수 속성 결측치 및 첫 주 관객 0 이하 제외
req_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
df_valid = raw_df.dropna(subset=req_cols).copy()
df_valid = df_valid[df_valid['first_week_audi'] > 0].copy()

# 속성 전처리
# 1) 스크린 수(first_scrn): 상용로그
# 2) 누적 관객(total_audi): 상용로그
# 3) 10위권 일수(days_in_top10): 그대로
# 4) 롱런 지수: (total_audi / first_week_audi), 최대 20으로 클리핑
df_valid['log_first_scrn'] = np.log10(df_valid['first_scrn'])
df_valid['log_total_audi'] = np.log10(df_valid['total_audi'])
df_valid['days_in_top10_feat'] = df_valid['days_in_top10']
df_valid['long_run_feat'] = (df_valid['total_audi'] / df_valid['first_week_audi']).clip(upper=20)

feature_map = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10_feat",
    "롱런 지수": "long_run_feat"
}

# 전체 편수와 묶은 편수를 화면 한 줄로 출력
used_count = len(df_valid)
st.write(f"**전체 영화 편수:** {total_raw_count}편 | **군집화에 사용된 영화 편수:** {used_count}편")

st.markdown("---")

# 3. 군집화에 사용할 속성 선택 (화면 위쪽, 기본 4개 선택)
all_feature_names = list(feature_map.keys())
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (둘 이상 선택):",
    options=all_feature_names,
    default=all_feature_names
)

if len(selected_feature_names) < 2:
    st.warning("군집화를 진행하려면 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_cols = [feature_map[name] for name in selected_feature_names]

# 4. 표준화(StandardScaler) 및 K-Means 군집화 (K=3, 난수 고정 random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_valid[selected_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df_valid['raw_cluster'] = raw_clusters

# 누적 관객(total_audi) 평균이 큰 순서대로 ㉮, ㉯, ㉰ 매핑
cluster_audi_mean = df_valid.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
label_map_dict = {}
labels = ['㉮', '㉯', '㉰']
for rank, old_cluster_id in enumerate(cluster_audi_mean.index):
    label_map_dict[old_cluster_id] = labels[rank]

df_valid['cluster'] = df_valid['raw_cluster'].map(label_map_dict)

# 그래프 색상 및 카테고리 순서 정의
color_discrete_map = {'㉮': '#1f77b4', '㉯': '#ff7f0e', '㉰': '#2ca02c'}
category_orders = {'cluster': ['㉮', '㉯', '㉰']}

st.markdown("---")

# 5. 2차원 산점도 (Plotly)
st.header("📊 2차원 산점도")

col_x1, col_y1 = st.columns(2)
with col_x1:
    x_axis_2d = st.selectbox("2차원 가로축(X) 속성 선택:", options=selected_feature_names, index=0)
with col_y1:
    y_axis_2d = st.selectbox("2차원 세로축(Y) 속성 선택:", options=selected_feature_names, index=min(1, len(selected_feature_names)-1))

fig_2d = px.scatter(
    df_valid,
    x=feature_map[x_axis_2d],
    y=feature_map[y_axis_2d],
    color='cluster',
    hover_name='movieNm',
    labels={feature_map[name]: name for name in selected_feature_names},
    title=f"2차원 산점도 ({x_axis_2d} vs {y_axis_2d})",
    color_discrete_map=color_discrete_map,
    category_orders=category_orders
)
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("---")

# 6. 3차원 산점도 (Plotly)
st.header("🧊 3차원 산점도")

if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 그리려면 군집화에 사용할 속성을 3개 이상 선택해야 합니다.")
else:
    col_x2, col_y2, col_z2 = st.columns(3)
    with col_x2:
        x_axis_3d = st.selectbox("3차원 X축 속성 선택:", options=selected_feature_names, index=0, key="3d_x")
    with col_y2:
        y_axis_3d = st.selectbox("3차원 Y축 속성 선택:", options=selected_feature_names, index=1, key="3d_y")
    with col_z2:
        z_axis_3d = st.selectbox("3차원 Z축 속성 선택:", options=selected_feature_names, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df_valid,
        x=feature_map[x_axis_3d],
