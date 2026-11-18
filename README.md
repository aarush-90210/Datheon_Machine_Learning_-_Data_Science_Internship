# Datheon_Machine_Learning_-_Data_Science_Internship
# Datheon Machine Learning & Data Science Internship

Submission for the Datheon ML & Data Science internship, covering two machine learning tasks built in Python with scikit-learn.

## Projects

| Task | Topic | Key Result |
|------|-------|-----------|
| [Task 3](Task_3_Customer_Churn_Prediction) | Customer churn classifier (Random Forest & XGBoost, SMOTE) | Test ROC-AUC 0.842 (RF) / 0.843 (XGBoost) |
| [Task 4](Task_4_Sentiment_Analysis_TFIDF) | Sentiment classification with TF-IDF (Naive Bayes & Logistic Regression) | 85.3% test accuracy (LR), 79.9% (NB) |

## Task 3: Customer Churn Prediction
Telco Customer Churn dataset. TotalCharges cleaned, categorical features one-hot encoded, class imbalance handled with SMOTE inside an imblearn pipeline (no leakage), models compared with 5-fold CV. Includes ROC curves, confusion matrices and top-10 feature importances.

## Task 4: Sentiment Analysis
NLTK movie reviews plus supplemental product reviews (2,009 records). Text is lowercased, HTML/punctuation stripped, stopwords removed and lemmatized, then vectorized with TF-IDF (5,000 features, unigrams + bigrams). Includes feature interpretation, error analysis and a reusable `predict_sentiment()` function.

## Run it
1. `pip install pandas numpy scikit-learn imbalanced-learn xgboost nltk matplotlib seaborn`
2. Open the notebook in Jupyter or Google Colab and run all cells.

## Tech Stack
Python, scikit-learn, XGBoost, imbalanced-learn, NLTK, pandas, matplotlib, seaborn

**Author:** Aarush
