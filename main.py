from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Init
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema
class PredictRequest(BaseModel):
    asset: str
    model_type: str
    days: int

# Fetch
def get_data(ticker, days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    df['DayIndex'] = np.arange(len(df))
    return df

# API
@app.post("/predict")
def predict_crypto(req: PredictRequest):
    df = get_data(req.asset)
    
    # Check data
    if df.empty:
        return {"error": "Data ditarik kosong."}
        
    current_price = float(df['Close'].iloc[-1])
    
    X = df[['DayIndex']]
    y = df['Close'].values.ravel()
    
    # Train
    if req.model_type == "Lasso":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Lasso(alpha=0.1)
        model.fit(X_scaled, y)
    else:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X, y)
        
    # Predict
    last_day = df['DayIndex'].max()
    future_X = pd.DataFrame({'DayIndex': np.arange(last_day + 1, last_day + 1 + req.days)})
    
    if req.model_type == "Lasso":
        preds = model.predict(scaler.transform(future_X))
    else:
        preds = model.predict(future_X)
        
    predicted_price = float(preds[-1])
    
    # Format
    hist_prices = df['Close'].tail(30).tolist()
    future_prices = preds.tolist()
    
    # Return
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "change": predicted_price - current_price,
        "change_pct": ((predicted_price - current_price) / current_price) * 100,
        "history": hist_prices,
        "forecast": future_prices
    }