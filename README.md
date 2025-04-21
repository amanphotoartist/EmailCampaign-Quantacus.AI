# EmailCampaign-Quantacus.AI

📧 Email Campaign Click Prediction – QuantusAI
This project analyzes user interaction data from email marketing campaigns to predict whether a user will click a link in an email. It includes data preprocessing, feature engineering, handling class imbalance with SMOTE, model training using LightGBM, model interpretation using SHAP, and CTR optimization.

🧾 Problem Statement
QuantusAI wants to optimize its email marketing by targeting users who are most likely to click on email links. This notebook builds a machine learning model that predicts click behavior using user and campaign metadata.

📁 Datasets Used
email_table.csv – Base dataset containing information on every email sent.

email_opened_table.csv – Contains email_ids of emails that were opened.

link_clicked_table.csv – Contains email_ids of emails where a link was clicked.

🔧 Steps Performed
Data Loading
Loaded all three datasets using pandas.

Feature Engineering
Created two binary features:
– opened → 1 if email was opened
– clicked → 1 if email was clicked
Dropped irrelevant columns like email_id.
Filled missing values with "missing".
Label encoded all categorical columns.

Train-Test Split
Used an 80-20 train-test split with stratification on the target (clicked).

Class Imbalance Handling
Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.

Model Training
Used LightGBM Classifier with:
– 300 estimators
– Learning rate = 0.05
– num_leaves = 31

Model Evaluation
Set prediction threshold = 0.3 to favor recall.
Evaluated using:
– Classification Report
– ROC AUC Score
– Confusion Matrix (visualized)

Feature Importance
Used LightGBM’s built-in feature importance and visualized it using seaborn.

Model Explainability
Used SHAP to:
– Visualize top contributing features
– Understand model decision behavior

CTR Optimization
Focused on the top 5% users with the highest predicted click probabilities.
Calculated optimized Click Through Rate (CTR) for that group.

🧪 Libraries Used
This project uses several Python libraries to handle data preprocessing, modeling, evaluation, visualization, and interpretability:

pandas and numpy for data manipulation and numerical operations
scikit-learn (sklearn) for preprocessing, model evaluation, and train-test splitting
lightgbm for building a high-performance gradient boosting model
imbalanced-learn (imblearn) for handling class imbalance using SMOTE
matplotlib and seaborn for visualizing results and insights
SHAP for interpreting the model’s predictions and understanding feature importance
joblib for saving and loading the trained model efficiently

📈 Results Summary
Achieved a good balance between precision and recall with a custom threshold.
SHAP analysis shows strong influence from features like email_type, opened, etc.
Optimized CTR in the top 5% predicted users shows potential for campaign targeting.

🚀 How to Run
Clone the repository from GitHub

Upload the required CSVs into your environment (email_table.csv, email_opened_table.csv, link_clicked_table.csv)

Open the Jupyter notebook EmailCampaign_Quantusai.ipynb

Run all cells step-by-step to execute the complete pipeline

🎯 Business Impact
This model helps identify the most promising users to target, increasing overall click rates and campaign ROI while reducing marketing waste.
