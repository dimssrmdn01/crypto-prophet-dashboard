import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Config
st.set_page_config(page_title="Crypto Prophet", layout="wide")

# Styling
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .metric-card { background-color: #1E2129; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #F7931A; }
    .metric-label { font-size: 1rem; color: #A0AEC0; }
    </style>
""", unsafe_allow_html=True)

# Fetch
@st.cache_data(ttl=3600)
def fetch_crypto_data(ticker, days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten cols
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    df['DayIndex'] = np.arange(len(df))
    return df

# Engine
def train_and_predict(df, forecast_days, model_type):
    X = df[['DayIndex']]
    y = df['Close'].values.ravel() 
    
    if model_type == "Lasso":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Lasso(alpha=0.1)
        model.fit(X_scaled, y)
    else:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X, y)
    
    # Future dates
    last_day = df['DayIndex'].max()
    future_X = pd.DataFrame({'DayIndex': np.arange(last_day + 1, last_day + 1 + forecast_days)})
    future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=forecast_days)
    
    if model_type == "Lasso":
        future_preds = model.predict(scaler.transform(future_X))
    else:
        future_preds = model.predict(future_X)
        
    return future_dates, future_preds

# Header
st.markdown("<h1 style='text-align: center;'>Crypto Prophet Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AEC0;'>AI-Powered Real-Time Cryptocurrency Forecasting</p>", unsafe_allow_html=True)

# Inputs
COINS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Dogecoin (DOGE)": "DOGE-USD"
}

col1, col2, col3 = st.columns(3)
with col1:
    selected_coin = st.selectbox("Select Asset", list(COINS.keys()))
with col2:
    selected_model = st.selectbox("Select ML Model", ["Lasso", "XGBoost"])
with col3:
    forecast_days = st.number_input("Forecast Period", min_value=1, max_value=30, value=7)

# Process
if st.button("Analyze Market", use_container_width=True):
    with st.spinner("Fetching data..."):
        ticker = COINS[selected_coin]
        df_hist = fetch_crypto_data(ticker, days=365)
        
        # Current price
        current_price = float(df_hist['Close'].iloc[-1].iloc[0]) if isinstance(df_hist['Close'].iloc[-1], pd.Series) else float(df_hist['Close'].iloc[-1])
        
        # Predict
        future_dates, future_preds = train_and_predict(df_hist, forecast_days, selected_model)
        predicted_price = future_preds[-1]
        
        price_change = predicted_price - current_price
        change_pct = (price_change / current_price) * 100
        
        # Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Predicted Price (Day {forecast_days})</div><div class='metric-value'>${predicted_price:,.2f}</div></div>", unsafe_allow_html=True)
        
        color = "#00FF88" if price_change > 0 else "#FF3366"
        sign = "+" if price_change > 0 else ""
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Estimated Change</div><div class='metric-value' style='color: {color};'>{sign}${price_change:,.2f} ({sign}{change_pct:.2f}%)</div></div>", unsafe_allow_html=True)
        
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Current Price</div><div class='metric-value' style='color: white;'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)
        
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>Model Active</div><div class='metric-value' style='color: #3399FF;'>{selected_model}</div></div>", unsafe_allow_html=True)

        # Chart
        st.markdown("<h3 style='text-align: center; margin-top: 30px;'>Market Trend & AI Forecast</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        hist_prices = df_hist['Close'].values.ravel()
        
        fig.add_trace(go.Scatter(
            x=df_hist['Date'][-90:], y=hist_prices[-90:], 
            mode='lines', name='Historical',
            line=dict(color='#888888', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds, 
            mode='lines+markers', name=f'Predicted ({selected_model})',
            line=dict(color='#F7931A' if price_change > 0 else '#FF3366', width=3, dash='dot'), marker=dict(size=6)
        ))
        
        fig.update_layout(
            plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333', tickprefix="$"),
            hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)