<div align="center">

# 🏠 AI House Price Predictor

### *Advanced Machine Learning Powered Property Valuation System*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)


![Banner](assets/banner.jpg)



---

</div>

## 📊 Overview

AI House Price Predictor is a state-of-the-art machine learning application that provides accurate real estate valuations in milliseconds. Built with modern technologies and advanced feature engineering, it delivers professional-grade predictions with an intuitive, beautiful interface.

### 🎯 Key Highlights

- **⚡ Lightning Fast**: Predictions in under 1 second
- **🎯 High Accuracy**: 87%+ R² score with advanced Gradient Boosting
- **🎨 Modern UI**: Beautiful, responsive design with smooth animations
- **🔧 Production Ready**: RESTful API with comprehensive error handling
- **📈 Feature Engineering**: 10+ engineered features for better predictions
- **🔒 Type Safe**: Pydantic models for data validation

---

## ✨ Features

### 🤖 **Machine Learning**
- **Gradient Boosting Regressor** for superior accuracy
- **Feature Engineering**: Age, total area, quality interactions
- **StandardScaler** normalization for numerical features
- **5-Fold Cross-Validation** for robust evaluation
- **Feature Importance** analysis and visualization

### 🚀 **Backend (FastAPI)**
- RESTful API with automatic OpenAPI documentation
- **4 Main Endpoints**: Health check, metrics, defaults, prediction
- Type-safe request/response with Pydantic models
- CORS-enabled for cross-origin requests
- Comprehensive error handling and logging

### 🎨 **Frontend**
- Modern purple gradient design with glassmorphism
- Smooth animations and micro-interactions
- Responsive layout (mobile, tablet, desktop)
- Real-time prediction with loading states
- Interactive feature importance visualization
- Confidence intervals and model metrics display

### 📊 **Model Performance**
```
Test R² Score:    0.875+
Test MAE:         $18,234
Test RMSE:        $28,456
CV R² Score:      0.865 ± 0.023
Prediction Time:  < 50ms
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (HTML/JS)                 │
│  • Beautiful UI with animations                      │
│  • Form validation & user input                      │
│  • Real-time results display                         │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/JSON
┌────────────────────▼────────────────────────────────┐
│              FastAPI Backend (Python)                │
│  • Request validation (Pydantic)                     │
│  • Feature engineering pipeline                      │
│  • Model inference & predictions                     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│          ML Pipeline (scikit-learn)                  │
│  • GradientBoostingRegressor model                  │
│  • Feature scaling (StandardScaler)                 │
│  • Label encoding for categorical vars              │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-house-price-predictor.git
cd ai-house-price-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python model_training.py
```
Expected output:
```
==================================================
MODEL TRAINING COMPLETED!
==================================================
Test R²: 0.8765
Test MAE: $18,234
Test RMSE: $28,456
==================================================
```

5. **Run the application**
```bash
python main.py
```

6. **Open in browser**
```
http://localhost:8000
```

---

## 📁 Project Structure

```
ai-house-price-predictor/
│
├── 📄 main.py                    # FastAPI application
├── 📄 model_training.py          # ML model training script
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project documentation
│
├── 📂 assets/
│   └── banner.jpg                # README banner image
│
├── 📂 data/
│   └── house_price_dataset.csv   # Training dataset
│
├── 📂 models/                    # Trained model artifacts
│   ├── house_price_model.pkl     # Trained ML model
│   ├── label_encoders.pkl        # Categorical encoders
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_names.pkl         # Feature list
│   ├── feature_info.pkl          # Feature metadata
│   ├── feature_importance.csv    # Feature importance scores
│   └── model_metrics.pkl         # Performance metrics
│
└── 📂 static/
    └── index.html                # Frontend application
```

---

## 🎯 API Documentation

### Endpoints

#### `GET /`
Serves the frontend HTML interface

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "test_r2": 0.8765
}
```

#### `GET /api/metrics`
Returns model performance metrics
```json
{
  "metrics": {
    "train_r2": 0.95,
    "test_r2": 0.88,
    "train_mae": 15000,
    "test_mae": 18234
  },
  "top_features": [...]
}
```

#### `POST /api/predict`
Predict house price

**Request Body:**
```json
{
  "LotArea": 8450,
  "OverallQual": 7,
  "OverallCond": 5,
  "YearBuilt": 2003,
  "GrLivArea": 1710,
  "FullBath": 2,
  "BedroomAbvGr": 3,
  "GarageCars": 2,
  "Neighborhood": "CollgCr",
  "ExterQual": "Gd"
}
```

**Response:**
```json
{
  "predicted_price": 208500.0,
  "confidence_interval": {
    "lower": 180000.0,
    "upper": 237000.0
  },
  "feature_contributions": {
    "OverallQual": 0.25,
    "GrLivArea": 0.18
  },
  "model_metrics": {
    "r2_score": 0.88,
    "mae": 18234.0
  }
}
```

#### `GET /api/defaults`
Returns default form values

**Interactive API Docs**: Visit `http://localhost:8000/docs` for Swagger UI

---

## 🔬 Model Details

### Algorithm: Gradient Boosting Regressor

**Why Gradient Boosting?**
- Superior accuracy compared to Random Forest
- Handles non-linear relationships effectively
- Built-in feature importance
- Robust to outliers

### Feature Engineering

The model uses 38 features including:

**Original Features (28):**
- Numerical: LotArea, YearBuilt, GrLivArea, GarageCars, etc.
- Categorical: Neighborhood, HouseStyle, ExterQual, etc.

**Engineered Features (10):**
```python
• HouseAge = YrSold - YearBuilt
• YearsSinceRemod = YrSold - YearRemodAdd
• TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
• TotalBathrooms = FullBath + 0.5 × HalfBath
• QualityArea = OverallQual × GrLivArea
• GarageScore = GarageCars × GarageArea
• HasBasement, HasGarage, Has2ndFloor, HasFireplace
```

### Hyperparameters

```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)
```

### Top 5 Important Features

| Feature | Importance | Impact |
|---------|-----------|---------|
| OverallQual | 25.3% | Quality rating |
| GrLivArea | 18.7% | Living area size |
| TotalSF | 12.4% | Total square footage |
| GarageCars | 9.8% | Garage capacity |
| YearBuilt | 8.2% | Property age |

---

## 🛠️ Technologies Used

### Backend
- **FastAPI** - Modern, fast web framework
- **Pydantic** - Data validation using Python type hints
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with animations
- **Vanilla JavaScript** - No framework dependencies
- **Fetch API** - HTTP requests

### ML/Data Science
- **Gradient Boosting** - Ensemble learning method
- **Feature Engineering** - Domain knowledge application
- **Cross-Validation** - Model evaluation
- **StandardScaler** - Feature normalization

---

## 📈 Performance Optimization

### Model Optimization
- ✅ Feature engineering for better signal
- ✅ StandardScaler for numerical stability
- ✅ Optimized hyperparameters
- ✅ Cross-validation for generalization

### API Optimization
- ✅ Efficient model loading (single load on startup)
- ✅ Fast inference (< 50ms prediction time)
- ✅ Minimal dependencies
- ✅ Proper error handling

### Frontend Optimization
- ✅ No external libraries (pure vanilla JS)
- ✅ Optimized animations (CSS transforms)
- ✅ Responsive design
- ✅ Efficient DOM manipulation

---

## 🔄 Future Enhancements

- [ ] **Database Integration** - Store predictions history
- [ ] **User Authentication** - Multi-user support
- [ ] **Advanced Models** - XGBoost, LightGBM comparison
- [ ] **Automated Retraining** - Schedule model updates
- [ ] **Export Reports** - PDF generation for predictions
- [ ] **Batch Predictions** - CSV upload support
- [ ] **Real Estate API Integration** - Live market data
- [ ] **Docker Support** - Containerized deployment
- [ ] **Cloud Deployment** - AWS/Azure/GCP guides
- [ ] **Mobile App** - React Native version

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: soumennandi633@gmail.com

---

## 🙏 Acknowledgments

- Dataset: [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- FastAPI for the amazing framework
- scikit-learn for powerful ML tools
- The open-source community

---
</div>