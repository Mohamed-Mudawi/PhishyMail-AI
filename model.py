import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    df = pd.read_csv('cleaned_TREC-05.csv', on_bad_lines='skip')
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

# 3. Model training and evaluation
print("\nTraining and evaluating models using 10-fold stratified cross-validation.")

# Create a dictionary with the ML models to be compared
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Multinomial NB": MultinomialNB(),
    #"SVM": SVC(kernel='linear', probability=True, random_state=42),
    #"Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
}

# The paper uses 10-fold stratified cross-validation.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}

# Loop through the models dictionary
for model_name, model in models.items():
    print("\nTraining model: ", model_name)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    final_cm = np.zeros((2,2), dtype=int)

    # Loop through the training splits
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        unique_labels, unique_counts = np.unique(y_train, return_counts=True)
        print(f"\nFold {i+1} label distribution:\nPhishing emails {unique_counts[1]}\nNon-phishing emails {unique_counts[0]}\n")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        final_cm += confusion_matrix(y_test, y_pred, labels=unique_labels)

    results[model_name] = {
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1-Score": np.mean(f1_scores),
        "Confusion matrix": final_cm
    }
# 4. Display results

for model_name in results.keys():
    print("\n-----------Results model: ", model_name)
    print("Accuracy:", results[model_name]["Accuracy"])
    print("Precision:", results[model_name]["Precision"])
    print("Recall:", results[model_name]["Recall"])
    print("F1-Score:", results[model_name]["F1-Score"])
    print("Confusion matrix:\n", results[model_name]["Confusion matrix"])


# target_names = ['Real (0.0)', 'Phishing (1.0)']
# plt.figure(figsize=(8, 6))
# sns.heatmap(results["Logistic Regression"]["Confusion matrix"], annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
# plt.title('Confusion Matrix for Spam Detection')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

print("\nEvaluation complete.")













