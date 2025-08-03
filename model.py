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
#from xgboost import XGBClassifier

# 1. Load the data
try:
    df = pd.read_csv('cleansed_TREC-05.csv', on_bad_lines='skip')
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
# print("\nTraining and evaluating models using 10-fold stratified cross-validation.")

# Create a dictionary with the ML models to be compared
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Multinomial NB": MultinomialNB(),
    #"SVM": SVC(kernel='linear', probability=True, random_state=42),
    #"Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
}

# # The paper uses 10-fold stratified cross-validation.
# skf = StratifiedKFold(n_splits=10)

# results = {}

# # Loop through the models dictionary
# for model_name, model in models.items():
#     print("\nTraining model: ", model_name)
#     accuracies, precisions, recalls, f1_scores = [], [], [], []
#     final_cm = np.zeros((2,2), dtype=int)

#     # Loop through the training splits
#     for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
#         X_train_folds = X_train[train_index]
#         X_test_folds = X_train[test_index]
#         y_train_folds = y_train.iloc[train_index]
#         y_test_folds = y_train.iloc[test_index]

#         unique_labels, unique_counts = np.unique(y_train_folds, return_counts=True)
#         print(f"\nFold {i+1} label distribution:\nPhishing emails {unique_counts[1]}\nNon-phishing emails {unique_counts[0]}\n")

#         model.fit(X_train_folds, y_train_folds)
#         y_pred = model.predict(X_test_folds)

#         accuracies.append(accuracy_score(y_test_folds, y_pred))
#         precisions.append(precision_score(y_test_folds, y_pred))
#         recalls.append(recall_score(y_test_folds, y_pred))
#         f1_scores.append(f1_score(y_test_folds, y_pred))

#         final_cm += confusion_matrix(y_test_folds, y_pred, labels=unique_labels)

#     results[model_name] = {
#         "Accuracy": np.mean(accuracies),
#         "Precision": np.mean(precisions),
#         "Recall": np.mean(recalls),
#         "F1-Score": np.mean(f1_scores),
#         "Confusion matrix": final_cm
#     }
# # 4. Display results

# for model_name in results.keys():
#     print("\n-----------Results model: ", model_name)
#     print("Accuracy:", results[model_name]["Accuracy"])
#     print("Precision:", results[model_name]["Precision"])
#     print("Recall:", results[model_name]["Recall"])
#     print("F1-Score:", results[model_name]["F1-Score"])
#     print("Confusion matrix:\n", results[model_name]["Confusion matrix"])
    
print("\n-----------Final Evaluation -----------")

# Choose the best model based on metrics
best_model_name = "XGBoost"
best_model = models[best_model_name]

print(f"\nTraining the best model ({best_model_name}) on the full training set...")
best_model.fit(X_train, y_train)

print("Evaluating on the unseen test set...")
y_final_pred = best_model.predict(X_test)

# Calculate and print final metrics
print("\nFinal Test Set Performance:")
print("Accuracy:", accuracy_score(y_test, y_final_pred))
print("Precision:", precision_score(y_test, y_final_pred))
print("Recall:", recall_score(y_test, y_final_pred))
print("F1-Score:", f1_score(y_test, y_final_pred))
print("\nClassification Report:\n", classification_report(y_test, y_final_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_final_pred))


# # target_names = ['Real (0.0)', 'Phishing (1.0)']
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(results["Logistic Regression"]["Confusion matrix"], annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
# # plt.title('Confusion Matrix for Spam Detection')
# # plt.ylabel('Actual Label')
# # plt.xlabel('Predicted Label')
# # plt.show()

# print("\nEvaluation complete.")













