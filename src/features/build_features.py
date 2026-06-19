import pandas as pd
from pathlib import Path
import datetime as dt

def build_features():
    project_dir = Path(__file__).resolve().parents[2]
    cleaned_path = project_dir / 'data' / 'processed' / 'cleaned_retail.csv'
    features_path = project_dir / 'data' / 'processed' / 'features.csv'
    
    print("Loading cleaned data...")
    df = pd.read_csv(cleaned_path, parse_dates=['InvoiceDate'])
    
    # Reference date for analysis
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    print("Engineering RFM Features...")
    
    # Group by Customer
    customer_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique', # Frequency
        'TotalPrice': 'sum' # Monetary
    }).reset_index()
    
    customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Target Variable: CLV (Next 90 days spend)
    ninety_days_ago = reference_date - dt.timedelta(days=90)
    future_clv = df[df['InvoiceDate'] >= ninety_days_ago].groupby('CustomerID')['TotalPrice'].sum().reset_index()
    future_clv.columns = ['CustomerID', 'CLV_90']
    
    # Merge Target
    customer_df = pd.merge(customer_df, future_clv, on='CustomerID', how='left')
    customer_df['CLV_90'] = customer_df['CLV_90'].fillna(0)
    
    # Avg Basket Size
    avg_basket = df.groupby('CustomerID')['TotalPrice'].mean().reset_index()
    avg_basket.columns = ['CustomerID', 'AvgBasketSize']
    customer_df = pd.merge(customer_df, avg_basket, on='CustomerID')
    
    # Country (Mode)
    country = df.groupby('CustomerID')['Country'].agg(lambda x: x.mode()[0]).reset_index()
    customer_df = pd.merge(customer_df, country, on='CustomerID')
    
    # One-Hot Encode Country (UK vs Non-UK)
    customer_df['IsUK'] = (customer_df['Country'] == 'United Kingdom').astype(int)
    
    # Drop irrelevant columns
    final_df = customer_df.drop(['Country', 'CustomerID'], axis=1)
    
    print(f"Final Feature Shape: {final_df.shape}")
    
    # Save
    final_df.to_csv(features_path, index=False)
    print("Features saved.")

if __name__ == "__main__":
    build_features()