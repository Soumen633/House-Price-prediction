# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    """Clean and prepare the data for modeling"""
    # Select important features for the model
    features = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'MSSubClass', 'MSZoning', 'Neighborhood', 'HouseStyle', 'ExterQual',
        'KitchenQual', 'HeatingQC', 'CentralAir', 'GarageType'
    ]
    
    # Filter features that exist in the dataset
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    y = df['SalePrice'].fillna(df['SalePrice'].median())

    
    # Handle missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna('None', inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    return X, y, label_encoders, available_features

def train_model():
    """Train and save the machine learning model"""
    # Load data
    df = pd.read_csv("data\house_price_dataset.csv")
    
    # Prepare data
    X, y, label_encoders, feature_names = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calculate performance metrics
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }
    
    # Save model and encoders
    joblib.dump(model, 'models/house_price_model.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')  
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(metrics, 'models/model_metrics.pkl')
    
    
    # Save feature importance for visualization
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print("Model training completed!")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    print(f"Test MAE: ${metrics['test_mae']:,.0f}")
    
    return model, label_encoders, feature_names, metrics

if __name__ == "__main__":
    train_model()
