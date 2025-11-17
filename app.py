# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:40:21 2025

@author: Lenovo
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="PP改性数字工厂 (Digital Factory)",
    page_icon="🏭",
    layout="wide"
)

# 自定义CSS让界面更像工业仪表盘
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 后台核心：数据生成与模型训练 (缓存)
# ==========================================
@st.cache_resource
def build_digital_twin_model():
    """
    构建虚拟的数字孪生模型。
    在实际应用中，这里应该替换为您真实的实验室数据读取逻辑。
    """
    np.random.seed(42)
    n_samples = 500
    
    # 模拟历史数据
    data = {
        'rPP_MFI': np.random.uniform(5, 25, n_samples),      # 原料熔指
        'rPP_Ash': np.random.uniform(0.5, 5.0, n_samples),   # 杂质含量
        'POE_Ratio': np.random.uniform(0, 30, n_samples),    # 增韧剂
        'Talc_Ratio': np.random.uniform(0, 40, n_samples),   # 填充剂
        'Screw_RPM': np.random.uniform(200, 600, n_samples), # 螺杆转速
        'Barrel_Temp': np.random.uniform(180, 230, n_samples)# 筒体温度
    }
    df = pd.DataFrame(data)
    
    # 模拟物理规律 (用于训练AI)
    # 冲击强度：POE提升显著，滑石粉略降，杂质降低
    df['Impact'] = (3 + 1.8 * df['POE_Ratio'] - 0.1 * df['Talc_Ratio'] 
                    - 0.6 * df['rPP_Ash'] + np.random.normal(0, 1.5, n_samples))
    
    # 拉伸强度：滑石粉提升，POE降低
    df['Tensile'] = (22 - 0.4 * df['POE_Ratio'] + 0.5 * df['Talc_Ratio'] 
                     - 0.5 * df['rPP_Ash'] + np.random.normal(0, 1.5, n_samples))
    
    # 熔体流动速率 (MFI)：原料MFI影响大，滑石粉降低流动性，降解(高温/高转速)提高MFI
    df['Final_MFI'] = (df['rPP_MFI'] * 0.8 - 0.2 * df['Talc_Ratio'] 
                       + 0.01 * (df['Barrel_Temp'] - 180) 
                       + np.random.normal(0, 1, n_samples))

    # 训练模型
    features = ['rPP_MFI', 'rPP_Ash', 'POE_Ratio', 'Talc_Ratio', 'Screw_RPM', 'Barrel_Temp']
    
    model_dict = {
        'Impact': RandomForestRegressor(n_estimators=100).fit(df[features], df['Impact']),
        'Tensile': RandomForestRegressor(n_estimators=100).fit(df[features], df['Tensile']),
        'Final_MFI': RandomForestRegressor(n_estimators=100).fit(df[features], df['Final_MFI'])
    }
    
    return model_dict

# 加载模型
models = build_digital_twin_model()

# ==========================================
# 3. 侧边栏：中央控制室 (输入参数)
# ==========================================
st.sidebar.header("🎛️ 中央控制室")

st.sidebar.subheader("1. 原料属性 (Recycled PP)")
input_rpp_mfi = st.sidebar.slider("rPP 熔指 (g/10min)", 5.0, 30.0, 12.0)
input_rpp_ash = st.sidebar.slider("rPP 灰分/杂质 (%)", 0.0, 5.0, 1.5)

st.sidebar.subheader("2. 配方设计 (Formulation)")
input_poe = st.sidebar.slider("增韧剂 POE (%)", 0, 40, 15)
input_talc = st.sidebar.slider("滑石粉 Talc (%)", 0, 50, 10)

st.sidebar.subheader("3. 工艺参数 (Process)")
input_rpm = st.sidebar.slider("螺杆转速 (RPM)", 100, 800, 350)
input_temp = st.sidebar.slider("挤出温度 (°C)", 180, 240, 210)

# 简单的成本计算逻辑 (元/吨)
cost_rpp = 6000
cost_poe = 14000
cost_talc = 2000
cost_process = 800 # 加工费

total_weight = 100
pp_weight = 100 - input_poe - input_talc
material_cost = (pp_weight/100 * cost_rpp) + (input_poe/100 * cost_poe) + (input_talc/100 * cost_talc) + cost_process

# ==========================================
# 4. 主界面：数字工厂仪表盘
# ==========================================

st.title("🏭 PP回收改性·数字孪生工厂")
st.markdown("基于机器学习算法，实时模拟配方与工艺对最终性能的影响。")

# --- A. 实时预测 ---
st.subheader("📊 实时性能预测")

# 构造输入数据
input_data = pd.DataFrame({
    'rPP_MFI': [input_rpp_mfi],
    'rPP_Ash': [input_rpp_ash],
    'POE_Ratio': [input_poe],
    'Talc_Ratio': [input_talc],
    'Screw_RPM': [input_rpm],
    'Barrel_Temp': [input_temp]
})

# 执行预测
pred_impact = models['Impact'].predict(input_data)[0]
pred_tensile = models['Tensile'].predict(input_data)[0]
pred_mfi = models['Final_MFI'].predict(input_data)[0]

# 显示 KPI 指标卡
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("预估成本", f"¥{material_cost:.0f}/吨", delta="-50 (同比)" if material_cost < 7500 else "+120")
with col2:
    st.metric("缺口冲击强度", f"{pred_impact:.1f} kJ/m²", delta="达标" if pred_impact > 15 else "未达标", delta_color="normal")
with col3:
    st.metric("拉伸强度", f"{pred_tensile:.1f} MPa", delta="达标" if pred_tensile > 20 else "偏低")
with col4:
    st.metric("成品 MFI", f"{pred_mfi:.1f} g/10min")

st.markdown("---")

# --- B. 可视化分析 ---
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.subheader("🎯 综合性能雷达图")
    # 数据归一化处理以便于绘图 (假设最大值为参考)
    categories = ['冲击强度', '拉伸强度', '流动性', '成本优势']
    
    # 成本优势：成本越低分越高
    cost_score = max(0, (10000 - material_cost) / 50) 
    
    values = [
        min(pred_impact * 4, 100),   # 放大便于显示
        min(pred_tensile * 3, 100), 
        min(pred_mfi * 5, 100), 
        cost_score
    ]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='当前配方'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_right:
    st.subheader("📈 智能模拟：POE含量敏感度分析")
    st.info("模拟保持其他条件不变，仅改变 POE 含量 (0-40%) 时的性能变化趋势。")
    
    # 创建模拟数据序列
    sim_poe_range = np.linspace(0, 40, 50)
    sim_data = pd.DataFrame({
        'rPP_MFI': input_rpp_mfi,
        'rPP_Ash': input_rpp_ash,
        'POE_Ratio': sim_poe_range,
        'Talc_Ratio': input_talc,
        'Screw_RPM': input_rpm,
        'Barrel_Temp': input_temp
    })
    
    # 批量预测
    sim_impact = models['Impact'].predict(sim_data)
    sim_tensile = models['Tensile'].predict(sim_data)
    
    # 绘制趋势图
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=sim_poe_range, y=sim_impact, name='冲击强度 (kJ/m²)', line=dict(color='orange', width=4)))
    fig_line.add_trace(go.Scatter(x=sim_poe_range, y=sim_tensile, name='拉伸强度 (MPa)', line=dict(color='blue', width=4), yaxis='y2'))
    
    fig_line.update_layout(
        xaxis_title='POE 含量 (%)',
        yaxis=dict(title='冲击强度'),
        yaxis2=dict(title='拉伸强度', overlaying='y', side='right'),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- C. 配方优化建议 ---
with st.expander("💡 查看 AI 配方优化建议"):
    if pred_impact < 10:
        st.warning("⚠️ 当前冲击强度较低。建议：\n1. 提高 POE 含量至 20% 以上。\n2. 检查原料灰分是否过高。\n3. 适当提高螺杆转速以加强分散。")
    elif pred_tensile < 20:
        st.warning("⚠️ 当前拉伸强度偏低。建议：\n1. 适当增加滑石粉含量。\n2. 减少 POE 用量。\n3. 确认是否可以使用 MFI 较低的基料。")
    else:
        st.success("✅ 当前配方性能均衡，建议进行小试验证。")