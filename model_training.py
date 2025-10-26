# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Create additional features from existing ones"""
    df = df.copy()
    
    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    
    # Area features
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
    
    # Quality × Area interactions
    df['QualityArea'] = df['OverallQual'] * df['GrLivArea']
    df['GarageScore'] = df['GarageCars'] * df['GarageArea']
    
    # Boolean features
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    return df

def prepare_data(df):
    """Clean and prepare data with feature engineering"""
    # Apply feature engineering
    df = engineer_features(df)
    
    # Extended feature list
    features = [
        # Original numerical
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'MSSubClass', 'YrSold',
        # Engineered features
        'HouseAge', 'YearsSinceRemod', 'TotalSF', 'TotalBathrooms',
        'QualityArea', 'GarageScore', 'HasBasement', 'HasGarage',
        'Has2ndFloor', 'HasFireplace',
        # Categorical
        'MSZoning', 'Neighborhood', 'HouseStyle', 'ExterQual',
        'KitchenQual', 'HeatingQC', 'CentralAir', 'GarageType'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    y = df['SalePrice'].fillna(df['SalePrice'].median())
    
    # Handle missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna('None', inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Separate numerical and categorical
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    
    return X, y, label_encoders, scaler, available_features, num_features, cat_features

def train_model():
    """Train improved model with better hyperparameters"""
    print("Loading data...")
    df = pd.read_csv("data/house_price_dataset.csv")
    
    print("Preparing features...")
    X, y, label_encoders, scaler, feature_names, num_features, cat_features = prepare_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    # GradientBoosting often performs better than RandomForest
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }
    
    # Save artifacts
    print("Saving model artifacts...")
    joblib.dump(model, 'models/house_price_model.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(metrics, 'models/model_metrics.pkl')
    joblib.dump({
        'num_features': num_features,
        'cat_features': cat_features
    }, 'models/feature_info.pkl')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED!")
    print("="*50)
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Test MAE: ${metrics['test_mae']:,.0f}")
    print(f"Test RMSE: ${metrics['test_rmse']:,.0f}")
    print(f"CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print("="*50)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, label_encoders, scaler, feature_names, metrics

if __name__ == "__main__":
    train_model()