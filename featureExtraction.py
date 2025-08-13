import pandas as pd
import re
import csv
import numpy as np

# Increase max CSV field size to handle long text
csv.field_size_limit(2**25)

# Load CSV using Python engine and handle multi-line quoted fields
df = pd.read_csv('TREC-05.csv', engine='python', quotechar='"')

# Remove 'receiver' and 'date' columns
df = df.drop(['receiver', 'date'], axis=1)

# Remove duplicate rows
df = df.drop_duplicates()

# Drop rows with any missing values
df = df.dropna()

def extract_selected_features(df):
    """Extract only the high-importance features for optimal XGBoost performance"""
    
    print("Extracting optimized feature set...")
    
    # Initialize results dictionary
    results = {}
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        features = {}
        
        # Get text fields
        subject = str(row['subject'])
        body = str(row['body'])
        sender = str(row['sender'])
        
        # === SUBJECT FEATURES ===
        # subject_non_ascii_ratio
        features['subject_non_ascii_ratio'] = sum(1 for c in subject if ord(c) > 127) / max(len(subject), 1)
        
        # subject_colon_count
        features['subject_colon_count'] = subject.count(':')
        
        # subject_capital_ratio
        features['subject_capital_ratio'] = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
        
        # subject_non_ascii_count
        features['subject_non_ascii_count'] = sum(1 for c in subject if ord(c) > 127)
        
        # subject_digit_count
        features['subject_digit_count'] = sum(1 for c in subject if c.isdigit())
        
        # subject_all_caps_words
        features['subject_all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', subject))
        
        # subject_sentence_count
        features['subject_sentence_count'] = len(re.findall(r'[.!?]+', subject))
        
        # subject_money_mentions
        money_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
            r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd)\b',
            r'\b(?:free|win|won|prize|reward|money|cash|payment)\b'
        ]
        features['subject_money_mentions'] = sum(len(re.findall(pattern, subject, re.IGNORECASE)) for pattern in money_patterns)
        
        # subject_mixed_case_words
        features['subject_mixed_case_words'] = len(re.findall(r'\b[a-z]+[A-Z]+[a-zA-Z]*\b', subject))
        
        # === BODY FEATURES ===
        # body_url_char_ratio
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, body, re.IGNORECASE)
        features['body_url_char_ratio'] = sum(len(url) for url in urls) / max(len(body), 1)
        
        # body_hyphen_count
        features['body_hyphen_count'] = body.count('-')
        
        # body_phone_count
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',    # (XXX) XXX-XXXX
            r'\b\d{10}\b'                       # 10 digits
        ]
        features['body_phone_count'] = sum(len(re.findall(pattern, body)) for pattern in phone_patterns)
        
        # body_email_count
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        features['body_email_count'] = len(re.findall(email_pattern, body))
        
        # body_exclamation_count
        features['body_exclamation_count'] = body.count('!')
        
        # body_colon_count
        features['body_colon_count'] = body.count(':')
        
        # body_action_words
        action_words = [
            'click', 'verify', 'confirm', 'update', 'download', 'install',
            'login', 'sign in', 'submit', 'provide', 'enter', 'complete'
        ]
        body_lower = body.lower()
        features['body_action_words'] = sum(1 for word in action_words if word in body_lower)
        
        # body_money_mentions
        features['body_money_mentions'] = sum(len(re.findall(pattern, body, re.IGNORECASE)) for pattern in money_patterns)
        
        # body_all_caps_words
        features['body_all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', body))
        
        # body_non_ascii_ratio
        features['body_non_ascii_ratio'] = sum(1 for c in body if ord(c) > 127) / max(len(body), 1)
        
        # === SENDER FEATURES ===
        # sender_domain_length
        if '@' in sender:
            domain = sender.split('@', 1)[1]
            features['sender_domain_length'] = len(domain)
        else:
            features['sender_domain_length'] = 0
        
        # sender_length
        features['sender_length'] = len(sender)
        
        # === COMBINED FEATURES ===
        # total_urls (from subject and body)
        subject_urls = re.findall(url_pattern, subject, re.IGNORECASE)
        features['total_urls'] = len(subject_urls) + len(urls)
        
        results[idx] = features
    
    # Convert to DataFrame
    feature_df = pd.DataFrame.from_dict(results, orient='index')
    
    return feature_df

# Extract the selected features
print(f"Processing {len(df)} rows...")
selected_features_df = extract_selected_features(df)

# Combine with original data
final_df = pd.concat([df.reset_index(drop=True), selected_features_df.reset_index(drop=True)], axis=1)

# Keep only the selected features + label column
top_features = [
    'subject_non_ascii_ratio', 'sender_domain_length', 'subject_colon_count',
    'subject_capital_ratio', 'body_url_char_ratio', 'body_hyphen_count',
    'subject_non_ascii_count', 'sender_length', 'subject_digit_count',
    'total_urls', 'body_phone_count', 'body_email_count', 'subject_all_caps_words',
    'body_exclamation_count', 'subject_sentence_count', 'body_colon_count',
    'subject_money_mentions', 'subject_mixed_case_words', 'body_action_words',
    'body_money_mentions', 'body_all_caps_words', 'body_non_ascii_ratio'
]

# Create optimized dataset with only selected features + label
optimized_columns = top_features + ['label']
optimized_df = final_df[optimized_columns]

print(f"\n✅ Feature extraction complete!")
print(f"📊 Dataset shape: {optimized_df.shape}")
print(f"🎯 Selected features: {len(top_features)}")

# Display feature summary
print(f"\n📈 FEATURE SUMMARY")
print("=" * 40)
print("Subject features:", len([f for f in top_features if f.startswith('subject_')]))
print("Body features:   ", len([f for f in top_features if f.startswith('body_')]))
print("Sender features: ", len([f for f in top_features if f.startswith('sender_')]))
print("Combined features:", len([f for f in top_features if not any(f.startswith(p) for p in ['subject_', 'body_', 'sender_'])]))

# Preview the optimized dataset
print(f"\n🔍 DATASET PREVIEW")
print("=" * 40)
print(optimized_df.head())

print(f"\n📊 FEATURE STATISTICS")
print("=" * 40)
print(optimized_df[top_features].describe())

# Save the optimized dataset
print(f"\n💾 Saving optimized dataset...")
optimized_df.to_csv('optimized_phishing_dataset.csv', index=False)
print("✅ Dataset saved as 'optimized_phishing_dataset.csv'")

# Verify no missing values in selected features
missing_values = optimized_df[top_features].isnull().sum()
if missing_values.sum() > 0:
    print(f"\n⚠️  WARNING: Missing values detected:")
    print(missing_values[missing_values > 0])
else:
    print(f"\n✅ No missing values in selected features")

print(f"\n🚀 Dataset ready for XGBoost training!")
print(f"   Features: {len(top_features)}")
print(f"   Samples: {len(optimized_df)}")
print(f"   File: optimized_phishing_dataset.csv")

# === OPTIONAL: QUICK XGBOOST TRAINING VERIFICATION ===
def verify_with_xgboost():
    """Quick training to verify the optimized features work well"""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        print(f"\n🧪 VERIFICATION: Training XGBoost with optimized features...")
        print("=" * 55)
        
        # Prepare data
        X = optimized_df[top_features]
        y = optimized_df['label']
        
        # Convert target to binary if it's text
        if y.dtype == 'object':
            y = (y == 'phishing').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Accuracy with optimized features: {accuracy:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance of selected features
        feature_importance = pd.DataFrame({
            'feature': top_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 TOP 10 FEATURES (of your selected set):")
        print("-" * 45)
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        return True
        
    except ImportError:
        print(f"\n⚠️  XGBoost not installed - skipping verification")
        print("   Install with: pip install xgboost scikit-learn")
        return False
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        return False

# Run verification
verify_with_xgboost()