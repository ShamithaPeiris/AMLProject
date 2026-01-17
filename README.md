# AMLProject

LINK : https://amlproject-market-analysis.streamlit.app/

🥦 Market Price Forecasting System
THis project is a Machine Learning solution designed to predict the weekly market prices of vegetables and fruits in Sri Lanka. It combines historical price data with climate indicators (Rainfall, Temperature) to provide actionable insights for farmers, consumers, and policymakers.

🚀 Key Features

🔮 Price Forecasting: Predict future prices (LKR/kg) based on Region, Season (Month), and Weather conditions.

🧠 Explainable AI (XAI): Uses SHAP (SHapley Additive exPlanations) to reveal why the model predicts a specific price (e.g., "High Rainfall = Price Spike").

⛈️ Climate-Market Correlation: Dual-axis charts visualizing the direct impact of weather patterns on food prices.

📉 Volatility Analysis: Bollinger Bands and Radar Charts to identify market stability and regional disparities.

🕸️ Interactive Dashboard: A user-friendly web app built with Streamlit.

🛠️ Tech Stack

Language: Python
Machine Learning: CatBoost Regressor (Gradient Boosting on Decision Trees)
Explainability: SHAP (Shapley Values)
Visualization: Plotly, Matplotlib, Seaborn
Web Framework: Streamlit

📊 Dataset
Source: Consolidated weekly reports from HARTI (Hector Kobbekaduwa Agrarian Research and Training Institute) and the Department of Meteorology.

Size: ~130,000 Records (2020–2025).

Features:
Date, Month: Seasonal trends.
Region: Spatial price differences (e.g., Colombo vs. Rural).
Item: Commodity type (e.g., Carrot, Banana).
Rainfall, Temperature, Humidity: Climate impact indicators.

⚠️ Data Note: Some historical data points exhibit repetitive year-over-year patterns, suggesting imputation in the raw source. The model mitigates this by weighing climate features heavily to introduce variance.

⚙️ Installation & Usage

1. Clone the Repository

git clone https://github.com/yourusername/LankaAgri-AI.git
cd LankaAgri-AI

2. Install Dependencies

pip install pandas numpy scikit-learn catboost shap streamlit plotly matplotlib seaborn

3. Train the Model

Run the training script to generate the model file (market_price_model.cbm).

python train_model.py

You should see an output confirming the model saved successfully with R² and MAE scores.

4. Launch the Dashboard

Start the Streamlit app:

streamlit run app.py

The app will open automatically in your browser at http://localhost:8501.

🧠 Model Performance

Metric Score Description
R² Score 0.88 The model explains 88% of the price variance.
MAE ±24 LKR Average prediction error is within 24 Rupees.

Selected Algorithm: CatBoost was chosen over Random Forest for its superior handling of categorical features (Region, Item) and "Ordered Boosting" technique, which reduces overfitting on time-series data.

📜 License

This project is for educational purposes as part of an MSc in Artificial Intelligence. Data belongs to the respective government bodies.

Student Details
Name: M.S.L.Peiris
Course: MSc in Artificial Intelligence
Module: Applied Machine Learning
