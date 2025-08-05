import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import joblib
#from xgboost import XGBClassifier

# 1. Load the data
try:
    df = pd.read_csv('cleaned_TREC_05.csv', on_bad_lines='skip')
    print("Dataset loaded successfully.")
    print("\nDataset shape:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: file not found.")
    exit()

# 2. Data preparation and feature engineering - combine 'sender', 'subject', 'body', and if there's missing values fill them with an empty string
df['text_feature'] = df['sender'].fillna('') + ' ' + df['body'].fillna('') + ' ' + df['subject'].fillna('')

# Use TF-IDF vectorization to convert the text feature into a matrix
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['text_feature'])
print("\nText data has been vectorized using TF-IDF.")

# Add the urls feature as a column to the text matrix
X_urls = df[['urls']].values
X = hstack((X_text, X_urls))
X = X.tocsr()

print("\nFeature vector shape:", X.shape)
# Label 1 is for phishing and label 0 is not phishing
y = df['label']

print("\nTarget label distribution:")
print(y.value_counts())

# Split the Train and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)




# # 3. Model training and evaluation
print("\nTraining and evaluating models using 10-fold stratified cross-validation.")

# Create a dictionary with the ML models to be compared
models = {
    "Logistic_Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Multinomial_NB": MultinomialNB(),
    #"SVM": SVC(kernel='linear', random_state=42),
    #"Extra_Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
}


results = {}

# Loop through the models dictionary
for model_name, model in models.items():
    print("\nTraining model: ", model_name)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # compute the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    #save the model
    model_filename = f"{model_name}_model.joblib"
    joblib.dump(model, model_filename)
    print("Model saved as: ", model_filename)
    
    # store results
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion matrix": cm
    }
   
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("\nTF-IDF vectorizer saved as 'tfidf_vectorizer.joblib'")
        
# 4. Display results
    
print("\n-----------Final Evaluation for all models -----------")

for model_name, metrics in results.items():
    print("\n-----------Results model: ", model_name)
    print("Accuracy:", metrics["Accuracy"])
    print("Precision:", metrics["Precision"])
    print("Recall: ", metrics["Recall"])
    print("F1-Score: ", metrics["F1-Score"])
    print("Confusion matrix:\n", metrics["Confusion matrix"])
   

# # ------ To load the model-------
# # Load the saved model
# model = joblib.load("XGBoost_model.joblib")

# # Use the model to make predictions
# predictions = model.predict(X_test)

















