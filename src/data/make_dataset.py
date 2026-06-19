import pandas as pd
import os
from pathlib import Path

def fetch_data():
    """Fetches the Online Retail dataset from UCI ML Repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / 'data' / 'raw' / 'online_retail.xlsx'
    
    print(f"Downloading data to {raw_data_path}...")
    
    try:
        df = pd.read_excel(url)
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(raw_data_path, index=False)
        print("Data downloaded and saved successfully.")
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

if __name__ == "__main__":
    fetch_data()