📉 Customer Churn Prediction

Predicting customer churn (whether a customer will leave a company) is critical for subscription businesses like telecom, streaming, and e-commerce. In this project, I built an end-to-end machine learning solution that not only predicts churn but also deploys an interactive web app where users can test the model in real time.

🧐 Problem Statement

Every year, telecom providers lose millions in revenue due to customer churn. Identifying high-risk customers early allows companies to:

Personalize retention offers

Improve customer satisfaction

Reduce revenue loss

The goal of this project is to build a predictive model that can classify whether a customer is likely to churn, based on demographics, billing details, and service usage.

📊 Dataset

Source: Telco Customer Churn Dataset (Kaggle)

Rows: ~7,000 customers

Features:

Demographics (gender, senior citizen, dependents)

Account information (tenure, contract type, payment method)

Service usage (internet, phone, streaming services)

Charges (monthly charges, total charges)

Target Variable: Churn (Yes / No)

🔎 Approach

Exploratory Data Analysis (EDA)

Visualized churn rates by demographics and contract types

Detected correlations between churn and month-to-month contracts / high monthly charges

Data Preprocessing

Cleaned missing values

Encoded categorical variables (One-Hot Encoding)

Scaled numerical features

Modeling

Compared Logistic Regression and XGBoost

Tuned hyperparameters with cross-validation

Evaluated models on accuracy, precision, recall, F1, and ROC-AUC

Deployment

Built an interactive Streamlit web app

Users can input customer details → app predicts churn probability instantly

📈 Results
Model	Accuracy	ROC-AUC
Logistic Regression	0.81	0.84
XGBoost (final model)	0.86	0.89

✔️ Final chosen model: XGBoost (better ROC-AUC, stable performance).

🎥 Screenshots
<img width="1029" height="963" alt="Screenshot 2025-08-22 002621" src="https://github.com/user-attachments/assets/cf233169-8e97-43fc-aec6-44ce036a232a" />
<img width="1076" height="1013" alt="Screenshot 2025-08-22 002532" src="https://github.com/user-attachments/assets/e3385029-c295-483f-9476-84626ea13173" />
<img width="1003" height="975" alt="Screenshot 2025-08-22 002552" src="https://github.com/user-attachments/assets/1fbcf192-cbda-4797-92ce-54a3f12f17e0" />



(To be added: Run the Streamlit app, take snips, and save in visuals/ folder)

⚡ Instructions
🔧 1. Clone the repo
git clone https://github.com/Yusuf-Khovawale/churn-prediction.git
cd churn-prediction

🔧 2. Create environment & install dependencies
conda create -n ds-churn python=3.10 -y
conda activate ds-churn
pip install -r requirements.txt

🔧 3. Train the model (optional)
jupyter notebook notebooks/churn_starter.ipynb

🔧 4. Run the Streamlit App
streamlit run app.py

🚀 Future Work

Add SHAP explainability plots (why a prediction was made)

Deploy on Streamlit Cloud for live demo access

Integrate real-time data (simulating actual customer feeds)

👨‍💻 Author

Mohammed Yusuf Khovawale

MSc Data Science & AI, University of Liverpool

LinkedIn:www.linkedin.com/in/yusuf-khovawale
GitHub:https://github.com/Yusuf-Khovawale
