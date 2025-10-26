# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os

app = FastAPI(
    title="House Price Prediction API",
    description="AI-powered house price prediction using machine learning",
    version="2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
try:
    model = joblib.load('models/house_price_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    metrics = joblib.load('models/model_metrics.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    feature_importance = pd.read_csv('models/feature_importance.csv')
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run model_training.py first!")

# Request model
class HousePredictionRequest(BaseModel):
    LotArea: int = Field(default=8450, ge=1000, le=100000)
    OverallQual: int = Field(default=7, ge=1, le=10)
    OverallCond: int = Field(default=5, ge=1, le=10)
    YearBuilt: int = Field(default=2003, ge=1800, le=2025)
    YearRemodAdd: int = Field(default=2003, ge=1800, le=2025)
    TotalBsmtSF: int = Field(default=856, ge=0, le=6000)
    FirstFlrSF: int = Field(default=856, ge=300, le=5000, alias='1stFlrSF')
    SecondFlrSF: int = Field(default=854, ge=0, le=3000, alias='2ndFlrSF')
    GrLivArea: int = Field(default=1710, ge=300, le=10000)
    FullBath: int = Field(default=2, ge=0, le=5)
    HalfBath: int = Field(default=1, ge=0, le=3)
    BedroomAbvGr: int = Field(default=3, ge=0, le=10)
    KitchenAbvGr: int = Field(default=1, ge=1, le=3)
    TotRmsAbvGrd: int = Field(default=8, ge=2, le=20)
    Fireplaces: int = Field(default=0, ge=0, le=5)
    GarageCars: int = Field(default=2, ge=0, le=5)
    GarageArea: int = Field(default=548, ge=0, le=1500)
    WoodDeckSF: int = Field(default=0, ge=0, le=1000)
    OpenPorchSF: int = Field(default=61, ge=0, le=500)
    MSSubClass: int = Field(default=60)
    YrSold: int = Field(default=2008, ge=2006, le=2025)
    
    # Categorical
    MSZoning: str = Field(default="RL")
    Neighborhood: str = Field(default="CollgCr")
    HouseStyle: str = Field(default="2Story")
    ExterQual: str = Field(default="Gd")
    KitchenQual: str = Field(default="Gd")
    HeatingQC: str = Field(default="Ex")
    CentralAir: str = Field(default="Y")
    GarageType: str = Field(default="Attchd")
    
    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: dict
    feature_contributions: dict
    model_metrics: dict

def engineer_input_features(data: dict) -> pd.DataFrame:
    """Apply same feature engineering as training"""
    df = pd.DataFrame([data])
    
    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    
    # Area features
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
    
    # Quality × Area
    df['QualityArea'] = df['OverallQual'] * df['GrLivArea']
    df['GarageScore'] = df['GarageCars'] * df['GarageArea']
    
    # Boolean features
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    return df

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the HTML frontend"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Frontend not found. Please create static/index.html</h1>"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "test_r2": metrics.get('test_r2', 0)
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get model performance metrics"""
    return {
        "metrics": metrics,
        "top_features": feature_importance.head(10).to_dict('records')
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(request: HousePredictionRequest):
    """Predict house price"""
    try:
        print(f"\n=== Prediction Request Received ===")
        
        # Convert to dict and engineer features
        input_data = request.model_dump(by_alias=True)
        print(f"Input data keys: {list(input_data.keys())}")
        
        df = engineer_input_features(input_data)
        print(f"After engineering - shape: {df.shape}, columns: {len(df.columns)}")
        
        # Prepare features in correct order
        X = df[feature_names].copy()
        print(f"Features prepared: {X.shape}")
        
        # Encode categorical
        for col, le in label_encoders.items():
            if col in X.columns:
                # Handle unknown categories
                if X[col].iloc[0] in le.classes_:
                    X[col] = le.transform(X[col])
                else:
                    print(f"Warning: Unknown category for {col}, using default")
                    X[col] = le.transform([le.classes_[0]])[0]
        
        # Scale numerical features
        num_features = feature_info['num_features']
        X[num_features] = scaler.transform(X[num_features])
        print(f"Features scaled")
        
        # Predict
        prediction = model.predict(X)[0]
        print(f"Raw prediction: ${prediction:,.2f}")
        
        # Calculate confidence interval (approximation)
        std_error = metrics['test_rmse']
        confidence_interval = {
            "lower": float(prediction - 1.96 * std_error),
            "upper": float(prediction + 1.96 * std_error)
        }
        print(f"Confidence interval: ${confidence_interval['lower']:,.2f} - ${confidence_interval['upper']:,.2f}")
        
        # Feature contributions (approximate using feature importance)
        top_features = feature_importance.head(5)
        feature_contributions = {
            row['feature']: float(row['importance'])
            for _, row in top_features.iterrows()
        }
        
        response_data = PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=confidence_interval,
            feature_contributions=feature_contributions,
            model_metrics={
                "r2_score": float(metrics['test_r2']),
                "mae": float(metrics['test_mae']),
                "rmse": float(metrics['test_rmse'])
            }
        )
        
        print(f"Response prepared successfully")
        print(f"=== Prediction Complete ===\n")
        
        return response_data
        
    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/defaults")
async def get_defaults():
    """Get default values for the form"""
    return {
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "GrLivArea": 1710,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 8,
        "Fireplaces": 0,
        "GarageCars": 2,
        "GarageArea": 548,
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "MSSubClass": 60,
        "YrSold": 2008,
        "MSZoning": "RL",
        "Neighborhood": "CollgCr",
        "HouseStyle": "2Story",
        "ExterQual": "Gd",
        "KitchenQual": "Gd",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "GarageType": "Attchd"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)