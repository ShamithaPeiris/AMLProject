import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from catboost import CatBoostRegressor
import shap
import datetime
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Market Price Predictor",
    page_icon="🥦",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3 { color: #00E676 !important; font-family: 'Segoe UI', sans-serif; }
    .stMetric { 
        background-color: #1e1e1e; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 5px solid #00E676;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00E676;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADER ---
@st.cache_data
def load_and_prep_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
        
        # Robust Column Renaming
        temp_col = next((c for c in df.columns if "Temp" in c), 'Temperature')
        
        df = df.rename(columns={
            temp_col: 'Temperature',
            'Rainfall (mm)': 'Rainfall',
            'Humidity (%)': 'Humidity',
            'fruit_Commodity': 'Fruit_Item',
            'fruit_Price per Unit (LKR/kg)': 'Fruit_Price',
            'vegitable_Commodity': 'Veg_Item',
            'vegitable_Price per Unit (LKR/kg)': 'Veg_Price'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month_Num'] = df['Date'].dt.month
        
        # Melt
        common = ['Date', 'Region', 'Temperature', 'Rainfall', 'Humidity', 'Month_Num']
        df_f = df[common + ['Fruit_Item', 'Fruit_Price']].rename(columns={'Fruit_Item': 'Item', 'Fruit_Price': 'Price'})
        df_f['Category'] = 'Fruit'
        df_v = df[common + ['Veg_Item', 'Veg_Price']].rename(columns={'Veg_Item': 'Item', 'Veg_Price': 'Price'})
        df_v['Category'] = 'Vegetable'
        
        final_df = pd.concat([df_f, df_v], ignore_index=True)
        final_df = final_df[final_df['Price'] > 0].dropna()
        
        return final_df.sort_values('Date')
        
    except Exception as e:
        return pd.DataFrame()

# --- 2. MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        model = CatBoostRegressor()
        model.load_model("market_price_model.cbm")
        return model
    except:
        return None

# Load Resources
FILE_PATH = "Vegetables_fruit_prices.csv"
model = load_model()
df_full = load_and_prep_data(FILE_PATH)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("AML Project")
    st.markdown("Name : M. S. L. Peiris")
    st.markdown("Index No : 258811B")
    st.markdown("---")
    
    if df_full.empty:
        st.error("⚠️ Dataset Failed to Load")
        sel_region, sel_item = None, None
    else:
        regions = sorted(df_full['Region'].unique())
        items = sorted(df_full['Item'].unique())
        
        st.subheader("📍 Prediction Settings")
        sel_region = st.selectbox("Select Region", regions)
        sel_item = st.selectbox("Select Commodity", items)
        
        item_row = df_full[df_full['Item'] == sel_item].iloc[0]
        sel_cat = item_row['Category']
        st.info(f"Category: **{sel_cat}**")

# --- 4. MAIN DASHBOARD ---
st.title("Intelligent Market Analysis")
st.markdown("Price Forecasting & Market Dynamics for Sri Lanka.")

if not model or df_full.empty:
    st.error("⚠️ System Check Failed: Missing Model or Data.")
else:
    # Filter Data
    item_data = df_full[(df_full['Item'] == sel_item) & (df_full['Region'] == sel_region)].copy()
    
    # --- KPI ROW (Restored at Top) ---
    c1, c2, c3, c4 = st.columns(4)
    if not item_data.empty:
        curr_price = item_data.sort_values('Date').iloc[-1]['Price']
        avg_price = item_data['Price'].mean()
        max_price = item_data['Price'].max()
        volatility = item_data['Price'].std()
        
        c1.metric("Current Price", f"Rs {curr_price:.0f}")
        c2.metric("Average Price", f"Rs {avg_price:.0f}")
        c3.metric("Peak Price", f"Rs {max_price:.0f}")
        c4.metric("Volatility (Std Dev)", f"±{volatility:.1f}")
    
    st.markdown("---")

    # --- TABS ---
    tab_pred, tab_xai, tab_adv, tab_analytics = st.tabs([
        "🤖 AI Forecast", "🧠 Explainability (XAI)", "📈 Advanced Charts", "📊 Deep Analytics"
    ])

    # === TAB 1: FORECAST ===
    with tab_pred:
        st.subheader(f"Predict Price: {sel_item}")
        
        col_in1, col_in2, col_in3, col_in4 = st.columns(4)
        in_rain = col_in1.slider("🌧️ Rainfall (mm)", 0, 400, 100)
        in_temp = col_in2.slider("🌡️ Temperature (°C)", 15, 40, 30)
        in_hum = col_in3.slider("💧 Humidity (%)", 40, 100, 75)
        in_month = col_in4.select_slider("📅 Month", options=range(1,13), value=datetime.datetime.now().month)

        if st.button("🚀 Generate Prediction", type="primary", width='stretch'):
            input_df = pd.DataFrame([[sel_region, sel_item, sel_cat, in_month, in_temp, in_rain, in_hum]], 
                                    columns=['Region', 'Item', 'Category', 'Month_Num', 'Temperature', 'Rainfall', 'Humidity'])
            
            pred = model.predict(input_df)[0]
            
            st.markdown("---")
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.metric("Predicted Price", f"Rs {pred:.2f}", delta=f"{pred - curr_price:.1f} vs Last Known")
            with c_res2:
                st.caption("Local SHAP Explanation (Why this prediction?)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                fig_water, ax = plt.subplots(figsize=(6, 2.5))
                shap.plots.waterfall(shap_values[0], max_display=6, show=False)
                st.pyplot(fig_water, width='stretch')

    # === TAB 2: EXPLAINABILITY (XAI) ===
    with tab_xai:
        st.subheader("Model Transparency")
        st.markdown("#### 🌍 Global Drivers (Swarm Plot)")
        
        shap_data = df_full.sample(500, random_state=42) if len(df_full) > 500 else df_full
        X_shap = shap_data[['Region', 'Item', 'Category', 'Month_Num', 'Temperature', 'Rainfall', 'Humidity']]
        
        explainer = shap.TreeExplainer(model)
        shap_values_global = explainer(X_shap)
        
        fig_swarm, ax = plt.subplots(figsize=(10, 5))
        shap.plots.beeswarm(shap_values_global, max_display=10, show=False)
        st.pyplot(fig_swarm, width='stretch')

    # === TAB 3: ADVANCED CHARTS (NEW!) ===
    with tab_adv:
        st.subheader("Advanced Market Indicators")
        
        # 1. BOLLINGER BANDS
        st.markdown("#### 📉 Price Volatility (Bollinger Bands)")
        
        if len(item_data) > 20:
            item_data['SMA_30'] = item_data['Price'].rolling(window=30).mean()
            item_data['STD_30'] = item_data['Price'].rolling(window=30).std()
            item_data['Upper'] = item_data['SMA_30'] + (item_data['STD_30'] * 2)
            item_data['Lower'] = item_data['SMA_30'] - (item_data['STD_30'] * 2)
            
            fig_boll = go.Figure()
            fig_boll.add_trace(go.Scatter(x=item_data['Date'], y=item_data['Upper'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig_boll.add_trace(go.Scatter(x=item_data['Date'], y=item_data['Lower'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(0, 230, 118, 0.2)', name='Volatility Zone'))
            fig_boll.add_trace(go.Scatter(x=item_data['Date'], y=item_data['Price'], mode='lines', name='Actual Price', line_color='#00E676'))
            fig_boll.add_trace(go.Scatter(x=item_data['Date'], y=item_data['SMA_30'], mode='lines', name='30-Day Trend', line=dict(dash='dash', color='white')))
            
            fig_boll.update_layout(template="plotly_dark", height=400, hovermode="x unified")
            st.plotly_chart(fig_boll, width='stretch')
        else:
            st.warning("Not enough data points for Bollinger Bands.")

        # 2. DUAL AXIS: PRICE vs RAINFALL
        st.markdown("#### ⛈️ Impact of Rain on Price")
        
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Bar(x=item_data['Date'], y=item_data['Rainfall'], name="Rainfall (mm)", marker_color='#4FC3F7', opacity=0.5), secondary_y=True)
        fig_dual.add_trace(go.Scatter(x=item_data['Date'], y=item_data['Price'], name="Price (LKR)", line_color='#00E676'), secondary_y=False)
        
        fig_dual.update_layout(template="plotly_dark", height=400, title_text="Price vs. Rainfall Correlation")
        st.plotly_chart(fig_dual, width='stretch')

    # === TAB 4: DEEP ANALYTICS ===
    with tab_analytics:
        st.subheader("Market Dynamics")
        
        c1, c2 = st.columns(2)
        
        # 1. REGIONAL RADAR CHART
        with c1:
            st.markdown("#### 🕸️ Regional Comparison")
            

            reg_stats = df_full[df_full['Item'] == sel_item].groupby('Region')['Price'].agg(['mean', 'max', 'std']).reset_index()
            if not reg_stats.empty:
                top_regs = reg_stats.head(5)
                fig_radar = go.Figure()
                for i, row in top_regs.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(r=[row['mean'], row['max'], row['std']], theta=['Avg Price', 'Max Price', 'Volatility'], fill='toself', name=row['Region']))
                fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig_radar, width='stretch')

        # 2. VIOLIN PLOT
        with c2:
            st.markdown("#### 🎻 Price Probability Density")
            
            all_region_data = df_full[df_full['Item'] == sel_item]
            fig_violin = px.violin(all_region_data, y="Price", x="Region", color="Region", box=True, points=False, template="plotly_dark")
            st.plotly_chart(fig_violin, width='stretch')
            
        # 3. Correlation Matrix
        st.markdown("#### 🌡️ Climate-Price Correlation Heatmap")
        corr_df = item_data[['Price', 'Temperature', 'Rainfall', 'Humidity']].corr()
        fig_heat = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_heat, width='stretch')