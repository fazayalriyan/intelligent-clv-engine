import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(
    page_title="RetailRev | CLV Intelligence",
    page_icon="📊",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    model_dir = Path(__file__).resolve().parents[1] / 'models'
    reg = joblib.load(model_dir / 'xgboost_clv.pkl')
    km = joblib.load(model_dir / 'kmeans_segment.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    return reg, km, scaler

@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'final_data_with_segments.csv'
    return pd.read_csv(data_path)

regressor, kmeans, scaler = load_models()
df = load_data()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("RetailRev AI Engine")
page = st.sidebar.radio("Navigate", ["Business Intelligence", "Prediction Engine", "Model Insights"])

# --- PAGE 1: BUSINESS INTELLIGENCE ---
if page == "Business Intelligence":
    st.title("📊 Customer Segmentation Overview")
    st.markdown("""
    This dashboard segments customers based on RFM (Recency, Frequency, Monetary) analysis 
    using K-Means Clustering.
    """)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Avg CLV (90d)", f"${df['CLV_90'].mean():,.2f}")
    with col3:
        loyal_count = len(df[df['Segment'] == 'Loyal Customers'])
        st.metric("Loyal Customers", loyal_count)
    with col4:
        st.metric("High Value Segments", len(df[df['Monetary'] > df['Monetary'].quantile(0.75)]))

    st.divider()
    
    # Visualizations
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Customer Distribution by Segment")
        seg_counts = df['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig_pie = px.pie(seg_counts, values='Count', names='Segment', color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        st.subheader("Monetary Value by Segment")
        fig_bar = px.bar(df, x='Segment', y='Monetary', color='Segment', color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3D Scatter Plot
    st.subheader("3D Cluster View (Recency, Frequency, Monetary)")
    fig_3d = px.scatter_3d(df.sample(500), x='Recency', y='Frequency', z='Monetary', 
                           color='Segment', symbol='Segment', opacity=0.7)
    st.plotly_chart(fig_3d, use_container_width=True)

# --- PAGE 2: PREDICTION ENGINE ---
elif page == "Prediction Engine":
    st.title("🔮 CLV Prediction Simulator")
    st.markdown("Input customer metrics to predict their future value over the next 90 days.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recency = st.slider("Days Since Last Purchase (Recency)", 0, 365, 30)
        frequency = st.slider("Total Unique Invoices", 1, 200, 10)
        monetary = st.number_input("Total Lifetime Spend ($)", min_value=0.0, value=500.0)
        avg_basket = st.number_input("Avg Basket Size ($)", min_value=0.0, value=50.0)
        is_uk = st.selectbox("Located in UK?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        
    if st.button("Predict CLV", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'Monetary': [monetary],
            'AvgBasketSize': [avg_basket],
            'IsUK': [is_uk]
        })
        
        # Predict
        pred_log = regressor.predict(input_data)[0]
        pred_val = np.expm1(pred_log)
        
        st.success(f"**Predicted 90-Day CLV: ${pred_val:,.2f}**")
        
        # Determine Segment
        input_scaled = scaler.transform(input_data[['Recency', 'Frequency', 'Monetary']])
        cluster = kmeans.predict(input_scaled)[0]
        seg_map = {0: "New/Low", 1: "Loyal Customers", 2: "Big Spenders", 3: "At Risk"}
        segment = seg_map.get(cluster, "Unknown")
        
        st.info(f"Suggested Segment Strategy: **{segment}**")

# --- PAGE 3: MODEL INSIGHTS ---
elif page == "Model Insights":
    st.title("🤖 Model Performance & Data")
    
    st.subheader("Feature Importance (XGBoost)")
    st.info("The model relies heavily on 'Monetary' and 'Frequency' to predict future CLV.")
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Fazayal")