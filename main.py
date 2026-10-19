import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 페이지 설정 (앱 제목 및 브라우저 탭 아이콘/제목)
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

# 필수 속성 결측치가 있거나 첫 주 관객(first_week_audi)이 0 이하인 데이터 제거
req_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
df_valid = raw_df.dropna(subset=req_cols).copy()
df_valid = df_valid[df_valid['first_week_audi'] > 0].copy()

# 속성 파생
# - 스크린 수(first_scrn): 상용로그(log10)
# - 누적 관객(total_audi): 상용로그(log10)
# - 10위권 일수(days_in_top10): 그대로
# - 롱런 지수: (누적 관객 / 첫 주 관객), 최대 20으로 클리핑
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

# 전체 편수와 묶은 편수를 한 줄로 표시
used_count = len(df_valid)
st.write(f"**전체 영화 편수:** {total_raw_count}편 | **군집화에 사용된 영화 편수:** {used_count}편")

st.markdown("---")

# 3. 군집화에 사용할 속성 및 묶음 수(K) 선택
all_feature_names = list(feature_map.keys())

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_feature_names = st.multiselect(
        "군집화에 사용할 속성을 선택하세요 (둘 이상 선택):",
        options=all_feature_names,
        default=all_feature_names
    )

with col_sel2:
    n_clusters = st.slider("묶음 수 (K) 선택:", min_value=2, max_value=7, value=3, step=1)

if len(selected_feature_names) < 2:
    st.warning("군집화를 진행하려면 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_cols = [feature_map[name] for name in selected_feature_names]

# 4. 표준화(StandardScaler) 및 K-Means 군집화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_valid[selected_cols])

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df_valid['raw_cluster'] = raw_clusters

# 묶음 기호 리스트 (㉮ ~ ㉴)
all_labels = ['㉮', '㉯', '㉰', '㉱', '㉲', '㉳', '㉴']
current_labels = all_labels[:n_clusters]

# 누적 관객(total_audi) 평균이 큰 순서대로 매핑 (㉮부터 순서대로)
cluster_audi_mean = df_valid.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
label_map_dict = {}
for rank, old_cluster_id in enumerate(cluster_audi_mean.index):
    label_map_dict[old_cluster_id] = current_labels[rank]

df_valid['cluster'] = df_valid['raw_cluster'].map(label_map_dict)

# 그래프 색상 팔레트 및 범례 순서 설정
colors = px.colors.qualitative.Plotly
color_discrete_map = {label: colors[i % len(colors)] for i, label in enumerate(current_labels)}
category_orders = {'cluster': current_labels}

# 고른 묶음 수의 실루엣 점수 계산 및 표시
sil_score = silhouette_score(X_scaled, df_valid['raw_cluster'])
st.info(f"💡 현재 선택된 묶음 수({n_clusters}개)의 **실루엣 점수**: `{sil_score:.4f}` (점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷하다는 뜻입니다.)")

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
    st.info("💡 3차원 산점도를 그리려면 군집화 속성을 3개 이상 선택해야 합니다.")
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
        y=feature_map[y_axis_3d],
        z=feature_map[z_axis_3d],
        color='cluster',
        hover_name='movieNm',
        labels={feature_map[name]: name for name in selected_feature_names},
        title="3차원 산점도",
        color_discrete_map=color_discrete_map,
        category_orders=category_orders
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 7. 묶음별 편수 및 네 속성 평균 표 (원래 단위)
st.header("📋 묶음별 편수 및 속성 평균 (원래 단위)")

df_valid['long_run_orig'] = (df_valid['total_audi'] / df_valid['first_week_audi']).clip(upper=20)

summary_df = df_valid.groupby('cluster').agg(
    편수=('movieNm', 'count'),
    스크린수_평균=('first_scrn', 'mean'),
    누적관객_평균=('total_audi', 'mean'),
    top10일수_평균=('days_in_top10', 'mean'),
    롱런지수_평균=('long_run_orig', 'mean')
).reindex(current_labels).reset_index()

summary_df.columns = ['묶음', '편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']

st.dataframe(summary_df.style.format({
    '편수': '{:,.0f}',
    '스크린 수 평균': '{:,.1f}',
    '누적 관객 평균': '{:,.1f}',
    '10위권 일수 평균': '{:,.1f}',
    '롱런 지수 평균': '{:,.2f}'
}), use_container_width=True)

st.markdown("---")

# 8. 묶음별 누적 관객 상위 5개 영화 목록
st.header("🏆 묶음별 누적 관객 상위 5개 영화")

cols_top5 = st.columns(min(n_clusters, 4))
for idx, cluster_label in enumerate(current_labels):
    with cols_top5[idx % min(n_clusters, 4)]:
        st.subheader(f"묶음 {cluster_label}")
        top5_movies = (
            df_valid[df_valid['cluster'] == cluster_label]
            .sort_values(by='total_audi', ascending=False)
            .head(5)[['movieNm', 'total_audi']]
        )
        top5_movies['total_audi'] = top5_movies['total_audi'].apply(lambda x: f"{x:,.0f}명")
        top5_movies.columns = ['영화 제목', '누적 관객']
        st.table(top5_movies.reset_index(drop=True))

st.markdown("---")

# 9. 묶음 수(K=1~7)별 관내 제곱합(Inertia) 계산 및 분석
st.header("📈 묶음 수(K) 선택에 따른 군집 응집도 분석")

k_values = list(range(1, 8))
inertias = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# 꺾은선 그래프
fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(
    x=k_values,
    y=inertias,
    mode='lines+markers',
    name='관내 제곱합 (Inertia)',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=8)
))

# 현재 선택된 K값 자리에 세로선 그리기
fig_elbow.add_vline(
    x=n_clusters,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택된 묶음 수 (K={n_clusters})",
    annotation_position="top right"
)

fig_elbow.update_layout(
    title="묶음 수(K=1~7)에 따른 관내 제곱합(Inertia) 변화",
    xaxis_title="묶음 수 (K)",
    yaxis_title="관내 제곱합 (Inertia)",
    xaxis=dict(dtick=1)
)

st.plotly_chart(fig_elbow, use_container_width=True)

# 묶음 수별 관내 제곱합 및 전 단계 대비 감소량 계산 표
diffs = [None]  # K=1일 때는 비교 대상이 없으므로 빈칸(None)
for i in range(1, len(inertias)):
    diffs.append(inertias[i-1] - inertias[i])

elbow_df = pd.DataFrame({
    '묶음 수 (K)': k_values,
    '관내 제곱합 (Inertia)': inertias,
    '전 단계 대비 감소량': diffs
})

formatted_elbow_df = elbow_df.copy()
formatted_elbow_df['관내 제곱합 (Inertia)'] = formatted_elbow_df['관내 제곱합 (Inertia)'].apply(lambda x: f"{x:,.2f}")
formatted_elbow_df['전 단계 대비 감소량'] = formatted_elbow_df['전 단계 대비 감소량'].apply(
    lambda x: "" if pd.isna(x) else f"{x:,.2f}"
)

st.dataframe(formatted_elbow_df, use_container_width=True)
