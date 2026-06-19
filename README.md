🛒 Intelligent CLV Optimization Engine
PythonStreamlitScikit-LearnXGBoost

An end-to-end Machine Learning system designed to predict Customer Lifetime Value (CLV) and automate customer segmentation for E-commerce retail. Transforms raw transactional data into actionable business insights to optimize marketing ROI.

🚀 Live Demo
View Live Application [https://intelligent-clv-engine-gbykqqxvrgfujgbk7ivpoe.streamlit.app/]

🎯 Business Impact
Problem: Marketing budgets are wasted on low-value customers while high-value clients remain underserved.
Solution: A predictive engine that identifies high-value segments and forecasts future revenue.
Tech Stack: Python, Pandas, XGBoost, Scikit-Learn, Streamlit, Plotly.

📂 Project Architecture

intelligent-clv-engine/
├── data/ # Datasets (Raw & Processed)
├── models/ # Serialized ML Models (.pkl)
├── notebooks/ # Exploratory Analysis
├── src/ # Source Code (Modular Pipeline)
│ ├── data/ # Data Ingestion & Cleaning
│ ├── features/ # Feature Engineering (RFM)
│ ├── models/ # Model Training Logic
│ └── visualization/ # Plotting Utilities
├── app/ # Streamlit Dashboard
├── tests/ # Unit Tests
└── README.md


## 🛠️ Installation & Execution

1. **Clone the Repo**
   ```bash
   git clone https://github.com/fazayalriyan/intelligent-clv-engine.git
   cd intelligent-clv-engine

2. Setup Environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Run the Pipeline
   python src/data/make_dataset.py
   python src/data/clean_data.py
   python src/features/build_features.py
   python src/models/train_model.py

4. Launch App
    streamlit run app/streamlit_app.py

📊 Methodology
1. Data Cleaning: Handled missing Customer IDs, removed cancelled transactions, and deduplicated 500k+ records.
2. Feature Engineering: Calculated RFM (Recency, Frequency, Monetary) metrics and created a 90-day forward-looking target variable.
3. Modeling:
- Regression (XGBoost): Predicted CLV_90 (Log-transformed target to handle financial skewness).
- Clustering (K-Means): Segmented users into 'Loyal', 'At Risk', 'Big Spenders', etc.
4. Deployment: Packaged into an interactive Plotly dashboard for non-technical stakeholders.

✍️ Author

Mohammed Fazayal
 
