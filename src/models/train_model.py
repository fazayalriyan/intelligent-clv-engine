import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models():
    project_dir = Path(__file__).resolve().parents[2]
    features_path = project_dir / 'data' / 'processed' / 'features.csv'
    model_dir = project_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    df = pd.read_csv(features_path)
    
    # --- 1. REGRESSION (CLV PREDICTION) ---
    print("Training XGBoost Regressor...")
    
    X = df.drop('CLV_90', axis=1)
    y = df['CLV_90']
    
    # Log transform target to handle skew
    y_log = y.apply(lambda x: x if x > 0 else 1).apply(lambda x: np.log1p(x))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Evaluate
    preds = regressor.predict(X_test)
    preds_exp = np.expm1(preds)
    y_test_exp = np.expm1(y_test)
    
    rmse = mean_squared_error(y_test_exp, preds_exp, squared=False)
    r2 = r2_score(y_test_exp, preds_exp)
    
    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model R2: {r2:.2f}")
    
    # Save Regressor
    joblib.dump(regressor, model_dir / 'xgboost_clv.pkl')
    
    # --- 2. CLUSTERING (SEGMENTATION) ---
    print("Training K-Means Clustering...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_labels = {0: "New/Low", 1: "Loyal Customers", 2: "Big Spenders", 3: "At Risk"}
    df['Segment'] = df['Cluster'].map(cluster_labels)
    
    # Save Clusterer & Scaler
    joblib.dump(kmeans, model_dir / 'kmeans_segment.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    # Save data with labels for App
    df.to_csv(project_dir / 'data' / 'processed' / 'final_data_with_segments.csv', index=False)
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_models()