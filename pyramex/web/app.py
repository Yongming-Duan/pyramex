"""
PyRamEx Web界面
基于Streamlit的交互式界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# 页面配置
st.set_page_config(
    page_title="PyRamEx - 拉曼光谱分析系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API配置
API_URL = "http://pyramex-app:8000"

# 标题
st.title("🔬 PyRamEx - GPU加速的拉曼光谱分析系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置选项")

    # 分析类型
    analysis_type = st.selectbox(
        "分析类型",
        ["预处理", "质控分析", "ML分析", "AI报告生成"]
    )

    # GPU加速
    enable_gpu = st.checkbox("启用GPU加速", value=True)

    # 模型选择
    if analysis_type == "AI报告生成":
        llm_model = st.selectbox(
            "LLM模型",
            ["qwen:7b", "deepseek-coder", "llama3:8b"]
        )
    else:
        llm_model = None

    st.markdown("---")

    # 系统状态
    st.subheader("📊 系统状态")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API服务正常")
        else:
            st.error("❌ API服务异常")
    except:
        st.error("❌ 无法连接API服务")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 数据上传")

    # 文件上传
    uploaded_files = st.file_uploader(
        "选择拉曼光谱文件",
        type=["csv", "txt", "xlsx"],
        accept_multiple_files=True,
        help="支持CSV、TXT、Excel格式"
    )

    if uploaded_files:
        st.info(f"已上传 {len(uploaded_files)} 个文件")

        # 显示文件列表
        for file in uploaded_files:
            st.write(f"📄 {file.name}")

with col2:
    st.header("📈 示例数据")

    # 生成示例数据
    if st.button("生成示例数据"):
        with st.spinner("生成中..."):
            wavenumber = np.linspace(400, 4000, 1000)
            intensity = np.random.randn(1000) * 0.1 + np.sin(wavenumber / 100)

            # 绘制光谱
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wavenumber,
                y=intensity,
                mode='lines',
                name='示例光谱',
                line=dict(color='blue', width=2)
            ))

            fig.update_layout(
                title="示例拉曼光谱",
                xaxis_title="波数 (cm⁻¹)",
                yaxis_title="强度",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

# 分析按钮
st.markdown("---")
st.header("🚀 开始分析")

if st.button("开始分析", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("⚠️ 请先上传数据文件")
    else:
        with st.spinner("分析中..."):
            try:
                # TODO: 调用API进行分析
                st.success(f"✅ 分析完成！处理了 {len(uploaded_files)} 个文件")
            except Exception as e:
                st.error(f"❌ 分析失败: {e}")

# 结果展示
if "analysis_results" in st.session_state:
    st.markdown("---")
    st.header("📊 分析结果")

    # 显示结果
    results = st.session_state.analysis_results

    # 绘图
    # 表格
    # 统计信息

# 底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>PyRamEx v2.0.0 | GPU加速 | AI原生 | Docker部署</p>
    <p>© 2026 小龙虾1号 🦞</p>
    </div>
    """,
    unsafe_allow_html=True
)
