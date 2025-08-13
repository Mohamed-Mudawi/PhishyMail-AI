import pandas as pd
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
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

def extract_comprehensive_features(text, prefix=''):
    """Extract comprehensive features for XGBoost from raw text"""
    text = str(text)
    features = {}
    
    # === BASIC TEXT STATISTICS ===
    features[f'{prefix}char_count'] = len(text)
    features[f'{prefix}word_count'] = len(text.split())
    features[f'{prefix}sentence_count'] = len(re.findall(r'[.!?]+', text))
    features[f'{prefix}avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # === CHARACTER PATTERNS ===
    features[f'{prefix}capital_count'] = sum(1 for c in text if c.isupper())
    features[f'{prefix}capital_ratio'] = features[f'{prefix}capital_count'] / max(len(text), 1)
    features[f'{prefix}digit_count'] = sum(1 for c in text if c.isdigit())
    features[f'{prefix}digit_ratio'] = features[f'{prefix}digit_count'] / max(len(text), 1)
    features[f'{prefix}space_count'] = text.count(' ')
    features[f'{prefix}space_ratio'] = features[f'{prefix}space_count'] / max(len(text), 1)
    
    # === PUNCTUATION PATTERNS (CRUCIAL FOR PHISHING) ===
    features[f'{prefix}exclamation_count'] = text.count('!')
    features[f'{prefix}question_count'] = text.count('?')
    features[f'{prefix}period_count'] = text.count('.')
    features[f'{prefix}comma_count'] = text.count(',')
    features[f'{prefix}colon_count'] = text.count(':')
    features[f'{prefix}semicolon_count'] = text.count(';')
    features[f'{prefix}hyphen_count'] = text.count('-')
    features[f'{prefix}underscore_count'] = text.count('_')
    
    # Suspicious punctuation patterns
    features[f'{prefix}multiple_exclamation'] = int('!!' in text)
    features[f'{prefix}multiple_question'] = int('??' in text)
    features[f'{prefix}mixed_punctuation'] = int('!?' in text or '?!' in text)
    
    # === URL ANALYSIS ===
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    features[f'{prefix}url_count'] = len(urls)
    features[f'{prefix}has_url'] = int(len(urls) > 0)
    
    # URL characteristics
    if urls:
        features[f'{prefix}avg_url_length'] = np.mean([len(url) for url in urls])
        features[f'{prefix}max_url_length'] = max([len(url) for url in urls])
        features[f'{prefix}url_char_ratio'] = sum(len(url) for url in urls) / max(len(text), 1)
    else:
        features[f'{prefix}avg_url_length'] = 0
        features[f'{prefix}max_url_length'] = 0
        features[f'{prefix}url_char_ratio'] = 0
    
    # Suspicious URL patterns
    suspicious_domains = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'short.link', 'tiny.cc']
    features[f'{prefix}has_url_shortener'] = int(any(domain in text.lower() for domain in suspicious_domains))
    
    # IP addresses in URLs
    features[f'{prefix}has_ip_url'] = int(bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', text)))
    
    # === EMAIL ANALYSIS ===
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    features[f'{prefix}email_count'] = len(emails)
    features[f'{prefix}has_email'] = int(len(emails) > 0)
    
    # === PHONE NUMBER ANALYSIS ===
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
        r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',    # (XXX) XXX-XXXX
        r'\b\d{10}\b'                       # 10 digits
    ]
    phone_count = sum(len(re.findall(pattern, text)) for pattern in phone_patterns)
    features[f'{prefix}phone_count'] = phone_count
    features[f'{prefix}has_phone'] = int(phone_count > 0)
    
    # === FINANCIAL/MONETARY PATTERNS ===
    money_patterns = [
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
        r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd)\b',
        r'\b(?:free|win|won|prize|reward|money|cash|payment)\b'
    ]
    features[f'{prefix}money_mentions'] = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in money_patterns)
    features[f'{prefix}has_money_mention'] = int(features[f'{prefix}money_mentions'] > 0)
    
    # === URGENCY/THREAT INDICATORS ===
    urgency_words = [
        'urgent', 'immediate', 'asap', 'hurry', 'quick', 'fast', 'now', 'today',
        'expires', 'deadline', 'limited', 'act now', 'dont wait', 'last chance'
    ]
    threat_words = [
        'suspended', 'blocked', 'locked', 'frozen', 'closed', 'terminated',
        'violation', 'unauthorized', 'security', 'breach', 'compromised'
    ]
    action_words = [
        'click', 'verify', 'confirm', 'update', 'download', 'install',
        'login', 'sign in', 'submit', 'provide', 'enter', 'complete'
    ]
    
    text_lower = text.lower()
    features[f'{prefix}urgency_words'] = sum(1 for word in urgency_words if word in text_lower)
    features[f'{prefix}threat_words'] = sum(1 for word in threat_words if word in text_lower)
    features[f'{prefix}action_words'] = sum(1 for word in action_words if word in text_lower)
    
    # === GRAMMAR/LANGUAGE QUALITY ===
    # Simple indicators of poor grammar (common in phishing)
    features[f'{prefix}repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))  # aaa, !!!, ???
    features[f'{prefix}all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
    features[f'{prefix}mixed_case_words'] = len(re.findall(r'\b[a-z]+[A-Z]+[a-zA-Z]*\b', text))
    
    # === ENCODING/CHARACTER ISSUES ===
    features[f'{prefix}non_ascii_count'] = sum(1 for c in text if ord(c) > 127)
    features[f'{prefix}non_ascii_ratio'] = features[f'{prefix}non_ascii_count'] / max(len(text), 1)
    
    # === DOMAIN-SPECIFIC PATTERNS ===
    # Banking/financial terms
    banking_terms = ['bank', 'account', 'credit', 'debit', 'card', 'paypal', 'amazon', 'ebay']
    features[f'{prefix}banking_terms'] = sum(1 for term in banking_terms if term in text_lower)
    
    # Social engineering terms
    social_terms = ['congratulations', 'winner', 'selected', 'chosen', 'lucky', 'prize', 'reward']
    features[f'{prefix}social_terms'] = sum(1 for term in social_terms if term in text_lower)
    
    return features

def extract_sender_features(sender):
    """Extract features specific to sender/email address"""
    sender = str(sender)
    features = {}
    
    # Basic sender features
    features['sender_length'] = len(sender)
    features['sender_has_at'] = int('@' in sender)
    
    if '@' in sender:
        local, domain = sender.split('@', 1)
        features['sender_local_length'] = len(local)
        features['sender_domain_length'] = len(domain)
        features['sender_has_subdomain'] = int(domain.count('.') > 1)
        
        # Suspicious domain patterns
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        features['sender_suspicious_tld'] = int(any(sender.lower().endswith(tld) for tld in suspicious_tlds))
        
        # Common legitimate domains (inverse indicator)
        legit_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
        features['sender_is_common_domain'] = int(any(domain.lower().endswith(legit) for legit in legit_domains))
        
        # Numbers in domain (suspicious)
        features['sender_domain_has_numbers'] = int(any(c.isdigit() for c in domain))
        
        # Hyphens in domain (can be suspicious)
        features['sender_domain_hyphens'] = domain.count('-')
    else:
        # If no @ symbol, mark as suspicious
        features['sender_local_length'] = 0
        features['sender_domain_length'] = 0
        features['sender_has_subdomain'] = 0
        features['sender_suspicious_tld'] = 1
        features['sender_is_common_domain'] = 0
        features['sender_domain_has_numbers'] = 0
        features['sender_domain_hyphens'] = 0
    
    return features

# === MAIN PROCESSING ===
print("Extracting comprehensive features for XGBoost...")

# Extract features for each text column
print("Processing subject lines...")
subject_features = df['subject'].apply(lambda x: extract_comprehensive_features(x, 'subject_')).apply(pd.Series)

print("Processing email bodies...")
body_features = df['body'].apply(lambda x: extract_comprehensive_features(x, 'body_')).apply(pd.Series)

print("Processing senders...")
sender_features = df['sender'].apply(extract_sender_features).apply(pd.Series)

print("Processing URLs column...")
# If URLs column exists and has data
if 'urls' in df.columns:
    url_features = df['urls'].apply(lambda x: extract_comprehensive_features(str(x), 'urls_')).apply(pd.Series)
    all_features = pd.concat([df, subject_features, body_features, sender_features, url_features], axis=1)
else:
    all_features = pd.concat([df, subject_features, body_features, sender_features], axis=1)

# Add interaction features (ratios between different text parts)
all_features['subject_to_body_ratio'] = all_features['subject_char_count'] / np.maximum(all_features['body_char_count'], 1)
all_features['total_urls'] = all_features['subject_url_count'] + all_features['body_url_count']
all_features['total_urgency_words'] = all_features['subject_urgency_words'] + all_features['body_urgency_words']
all_features['total_threat_words'] = all_features['subject_threat_words'] + all_features['body_threat_words']

# Remove original text columns to save space (keep only if you need them)
text_columns = ['sender', 'subject', 'body']
if 'urls' in df.columns:
    text_columns.append('urls')

# Option 1: Keep original text columns
# final_df = all_features

# Option 2: Remove original text columns (recommended for XGBoost)
final_df = all_features.drop(columns=text_columns)

print(f"\nFeature engineering complete!")
print(f"Original columns: {len(df.columns)}")
print(f"Final feature count: {len(final_df.columns)}")
print(f"Rows: {len(final_df)}")

# Preview features
print("\nSample of extracted features:")
feature_cols = [col for col in final_df.columns if col != 'label'][:10]
print(final_df[feature_cols].head())

# Save the feature-engineered dataset
print("\nSaving feature-rich dataset...")
final_df.to_csv('xgboost_ready_TREC-05.csv', index=False)

print("Dataset ready for XGBoost training!")

# === OPTIONAL: TRAIN XGBOOST AND ANALYZE FEATURE IMPORTANCE ===
def train_and_analyze_features(df, target_column='label'):
    """Train XGBoost model and analyze feature importance"""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        import matplotlib.pyplot as plt
        
        print("\n" + "="*50)
        print("TRAINING XGBOOST AND ANALYZING FEATURE IMPORTANCE")
        print("="*50)
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target to binary if it's text
        if y.dtype == 'object':
            y = (y == 'phishing').astype(int)  # Adjust based on your labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(X.columns)}")
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance analysis
        feature_importance = model.feature_importances_
        feature_names = X.columns
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*50)
        print("TOP 20 MOST IMPORTANT FEATURES")
        print("="*50)
        for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Save detailed feature importance
        importance_df.to_csv('feature_importance_analysis.csv', index=False)
        print(f"\nFull feature importance saved to 'feature_importance_analysis.csv'")
        
        # Plot top features
        try:
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features for Phishing Detection')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Feature importance plot saved as 'feature_importance_plot.png'")
        except ImportError:
            print("Matplotlib not available - skipping plot generation")
        
        # Feature importance by category
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE BY CATEGORY")
        print("="*50)
        
        categories = {
            'Subject': importance_df[importance_df['feature'].str.startswith('subject_')]['importance'].sum(),
            'Body': importance_df[importance_df['feature'].str.startswith('body_')]['importance'].sum(),
            'Sender': importance_df[importance_df['feature'].str.startswith('sender_')]['importance'].sum(),
            'URLs': importance_df[importance_df['feature'].str.startswith('urls_')]['importance'].sum() if 'urls_' in str(importance_df['feature'].values) else 0,
            'Combined': importance_df[~importance_df['feature'].str.contains('_')]['importance'].sum()
        }
        
        for category, importance in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            if importance > 0:
                print(f"{category:<12} {importance:.4f}")
        
        # Identify potentially redundant features (very low importance)
        low_importance = importance_df[importance_df['importance'] < 0.001]
        if len(low_importance) > 0:
            print(f"\n{len(low_importance)} features have very low importance (<0.001)")
            print("Consider removing these features to reduce complexity:")
            print(low_importance['feature'].head(10).tolist())
        
        return model, importance_df
        
    except ImportError as e:
        print(f"\nSkipping feature importance analysis - missing library: {e}")
        print("Install with: pip install xgboost scikit-learn matplotlib")
        return None, None
    except Exception as e:
        print(f"\nError during feature importance analysis: {e}")
        return None, None

# Run feature importance analysis if target column exists
if 'label' in final_df.columns:
    model, importance_df = train_and_analyze_features(final_df)
else:
    print("\nNo 'label' column found - skipping feature importance analysis")
    print("Add your target column and call train_and_analyze_features(df) to analyze feature importance")