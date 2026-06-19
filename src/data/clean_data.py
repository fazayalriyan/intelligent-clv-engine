import pandas as pd
from pathlib import Path

def clean_data():
    project_dir = Path(__file__).resolve().parents[2]
    raw_path = project_dir / 'data' / 'raw' / 'online_retail.xlsx'
    processed_path = project_dir / 'data' / 'processed' / 'cleaned_retail.csv'
    
    print("Loading raw data...")
    df = pd.read_excel(raw_path)
    
    print(f"Original shape: {df.shape}")
    
    # 1. Remove missing CustomerIDs
    df = df.dropna(subset=['CustomerID'])
    
    # 2. Remove negative quantities (Returns)
    df = df[df['Quantity'] > 0]
    
    # 3. Remove negative UnitPrice
    df = df[df['UnitPrice'] > 0]
    
    # 4. Remove duplicates
    df = df.drop_duplicates()
    
    # 5. Convert CustomerID to string
    df['CustomerID'] = df['CustomerID'].astype(str).str.strip()
    
    # 6. Create TotalPrice feature
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # 7. Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print(f"Cleaned shape: {df.shape}")
    
    # Save processed data
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print("Cleaned data saved.")

if __name__ == "__main__":
    clean_data()