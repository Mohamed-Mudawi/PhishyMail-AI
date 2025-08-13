import pandas as pd
import numpy as np # Import numpy for calculations
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# 1. Load dataset
df = pd.read_csv('cleaned_TREC-05.csv')
# Ensure subject and body are strings to prevent errors
df['subject'] = df['subject'].fillna('').astype(str)
df['body'] = df['body'].fillna('').astype(str)


# 2. Preprocessing functions
def extract_url_count(df):
    return df['urls'].astype(float).values.reshape(-1, 1)

def combine_text(df):
    return (df['subject'] + ' ' + df['body']).values

def extract_body_len(df):
    return df['body'].str.len().values.reshape(-1, 1)

def extract_subject_len(df):
    return df['subject'].str.len().values.reshape(-1, 1)
    
def extract_caps_ratio(df):
    # Calculate the ratio of capital letters in the body
    caps_count = df['body'].str.findall(r'[A-Z]').str.len()
    total_len = df['body'].str.len().replace(0, 1) # Avoid division by zero
    ratio = caps_count / total_len
    return ratio.values.reshape(-1, 1)

# 3. Build ML pipeline
text_vectorizer = Pipeline([
    ('selector', FunctionTransformer(combine_text, validate=False)),
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1,2)))
])

url_count = Pipeline([
    ('selector', FunctionTransformer(extract_url_count, validate=False)),
    ('scaler', StandardScaler())
])

# --- PIPELINES FOR NEW FEATURES ---
body_len = Pipeline([
    ('selector', FunctionTransformer(extract_body_len, validate=False)),
    ('scaler', StandardScaler())
])

subject_len = Pipeline([
    ('selector', FunctionTransformer(extract_subject_len, validate=False)),
    ('scaler', StandardScaler())
])

caps_ratio = Pipeline([
    ('selector', FunctionTransformer(extract_caps_ratio, validate=False)),
    ('scaler', StandardScaler())
])


features = FeatureUnion([
    ('text', text_vectorizer),
    ('url_count', url_count),
    ('body_len', body_len), # Add new feature pipeline
    ('subject_len', subject_len), # Add new feature pipeline
    ('caps_ratio', caps_ratio) # Add new feature pipeline
])

pipeline = Pipeline([
    ('features', features),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced'))
])

# 4. Train/test split
y = df['label']
# We pass the entire DataFrame 'df' to X, as our functions inside the pipeline will select the columns they need.
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y) # Added stratify=y

# 5. Fit model
pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save model to file
dump(pipeline, 'LexiGuard.joblib')

