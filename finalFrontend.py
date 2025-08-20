import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
import os
import glob
import base64

# Page configuration with custom favicon
st.set_page_config(
    page_title="PhishyMail AI",
    page_icon="phishymail_ai_logo.png",  # This sets the favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations
MODEL_CONFIGS = {
    "XGBoost Model": {
        "pattern": "phishing_xgboost_model_*.joblib",
        "type": "xgboost",
        "description": "XGBoost-based detector with handcrafted features",
        "features": "22 engineered features including sender patterns, subject characteristics, and body content analysis",
        "strengths": "Fast inference, interpretable features, good baseline performance"
    },
    "LexiGuard Model": {
        "pattern": "LexiGuard.joblib",
        "type": "lexiguard",
        "description": "Advanced NLP-based detector using TF-IDF and linguistic features",
        "features": "TF-IDF vectorization with n-grams, URL count, body length, subject length, and capitalization ratio",
        "strengths": "Deep text analysis, captures semantic patterns, handles linguistic variations"
    },
    "Random Forest": {
        "pattern": "clf_phishing_pipeline.joblib",
        "type": "random_forest",
        "description": "Random Forest classifier with TF-IDF and URL features",
        "features": "TF-IDF text vectorization combined with URL count detection and text cleaning",
        "strengths": "Robust ensemble method, handles overfitting well, good with mixed feature types"
    }
}

def extract_xgboost_features(sender, subject, body):
    """
    Extract features for XGBoost model (original handcrafted features)
    """
    features = {}
    
    # Sender features
    features['sender_length'] = len(sender) if sender else 0
    features['sender_domain_length'] = len(sender.split('@')[-1]) if '@' in sender else 0
    
    # Subject features
    if subject:
        features['subject_non_ascii_ratio'] = sum(1 for c in subject if ord(c) > 127) / len(subject) if subject else 0
        features['subject_non_ascii_count'] = sum(1 for c in subject if ord(c) > 127)
        features['subject_colon_count'] = subject.count(':')
        features['subject_capital_ratio'] = sum(1 for c in subject if c.isupper()) / len(subject) if subject else 0
        features['subject_digit_count'] = sum(1 for c in subject if c.isdigit())
        features['subject_all_caps_words'] = len([word for word in subject.split() if word.isupper() and len(word) > 1])
        features['subject_sentence_count'] = len(re.findall(r'[.!?]+', subject))
        features['subject_money_mentions'] = len(re.findall(r'\$|money|cash|prize|win|lottery|reward', subject.lower()))
        features['subject_mixed_case_words'] = len([word for word in subject.split() if any(c.isupper() for c in word) and any(c.islower() for c in word)])
    else:
        features.update({
            'subject_non_ascii_ratio': 0, 'subject_non_ascii_count': 0, 'subject_colon_count': 0,
            'subject_capital_ratio': 0, 'subject_digit_count': 0, 'subject_all_caps_words': 0,
            'subject_sentence_count': 0, 'subject_money_mentions': 0, 'subject_mixed_case_words': 0
        })
    
    # Body features
    if body:
        features['body_url_char_ratio'] = len(re.findall(r'http[s]?://\S+|www\.\S+', body)) / len(body) if body else 0
        features['body_hyphen_count'] = body.count('-')
        features['total_urls'] = len(re.findall(r'http[s]?://\S+|www\.\S+', body))
        features['body_phone_count'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', body))
        features['body_email_count'] = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', body))
        features['body_exclamation_count'] = body.count('!')
        features['body_colon_count'] = body.count(':')
        features['body_action_words'] = len(re.findall(r'\b(click|call|act|buy|order|download|install|verify|confirm|update|login|sign)\b', body.lower()))
        features['body_money_mentions'] = len(re.findall(r'\$|money|cash|prize|win|lottery|reward|free|discount', body.lower()))
        features['body_all_caps_words'] = len([word for word in body.split() if word.isupper() and len(word) > 1])
        features['body_non_ascii_ratio'] = sum(1 for c in body if ord(c) > 127) / len(body) if body else 0
    else:
        features.update({
            'body_url_char_ratio': 0, 'body_hyphen_count': 0, 'total_urls': 0,
            'body_phone_count': 0, 'body_email_count': 0, 'body_exclamation_count': 0,
            'body_colon_count': 0, 'body_action_words': 0, 'body_money_mentions': 0,
            'body_all_caps_words': 0, 'body_non_ascii_ratio': 0
        })
    
    return features

def prepare_lexiguard_data(sender, subject, body):
    """
    Prepare data in the format expected by LexiGuard model
    """
    # Ensure we have strings, not None
    subject = subject if subject else ""
    body = body if body else ""
    sender = sender if sender else ""
    
    # Count URLs in the body (simplified URL detection)
    urls = len(re.findall(r'http[s]?://\S+|www\.\S+', body))
    
    # Create a DataFrame with the expected structure
    data = pd.DataFrame({
        'subject': [subject],
        'body': [body],
        'sender': [sender],  # Include sender even if not used directly
        'urls': [urls]
    })
    
    return data

def load_available_models():
    """
    Load all available models and return them with their metadata
    """
    available_models = {}
    model_info = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        try:
            if "*" in config["pattern"]:
                # Pattern with wildcard
                model_files = glob.glob(config["pattern"])
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    model = joblib.load(latest_model)
                    available_models[model_name] = model
                    model_info[model_name] = f"Loaded: {os.path.basename(latest_model)}"
            else:
                # Direct file name
                if os.path.exists(config["pattern"]):
                    model = joblib.load(config["pattern"])
                    available_models[model_name] = model
                    model_info[model_name] = f"Loaded: {config['pattern']}"
                else:
                    model_info[model_name] = f"File not found: {config['pattern']}"
                    
        except ModuleNotFoundError as e:
            if "LexiGuard" in str(e):
                model_info[model_name] = f"Missing LexiGuard.py - Please place the training script in the same directory"
            else:
                model_info[model_name] = f"Missing module: {str(e)}"
        except Exception as e:
            model_info[model_name] = f"Error loading: {str(e)}"
    
    return available_models, model_info

def predict_with_xgboost(model, sender, subject, body):
    """
    Make prediction with XGBoost model using handcrafted features
    """
    # Extract features
    features = extract_xgboost_features(sender, subject, body)
    
    # Define feature order (same as original)
    feature_order = [
        'subject_non_ascii_ratio', 'sender_domain_length', 'subject_colon_count',
        'subject_capital_ratio', 'body_url_char_ratio', 'body_hyphen_count',
        'subject_non_ascii_count', 'sender_length', 'subject_digit_count',
        'total_urls', 'body_phone_count', 'body_email_count', 'subject_all_caps_words',
        'body_exclamation_count', 'subject_sentence_count', 'body_colon_count',
        'subject_money_mentions', 'subject_mixed_case_words', 'body_action_words',
        'body_money_mentions', 'body_all_caps_words', 'body_non_ascii_ratio'
    ]
    
    # Create feature vector
    feature_vector = [features.get(feature_name, 0) for feature_name in feature_order]
    X = pd.DataFrame([feature_vector], columns=feature_order)
    
    # Make prediction
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]
    
    return prediction, prediction_proba, features

def predict_with_random_forest(model_pipeline, sender, subject, body):
    """
    Make prediction with Random Forest model using its pipeline
    """
    # Extract the components from the pipeline
    pipeline_model = model_pipeline['model']
    vectorizer = model_pipeline['vectorizer'] 
    scaler = model_pipeline['scaler']
    
    # Clean text function (from the original app.py)
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)               # Remove URLs
        text = re.sub(r"\S+@\S+", "", text)               # Remove emails
        text = re.sub(r"[^A-Za-z\s]", "", text)           # Remove symbols/numbers
        return text.lower().strip()                       # Lowercase & strip
    
    # Combine subject and body
    full_text = f"{subject} {body}" if subject and body else (subject or body or "")
    
    # Clean the text
    cleaned_text = clean_text(full_text)
    
    # Get TF-IDF vector
    tfidf_vector = vectorizer.transform([cleaned_text])
    
    # Count URLs in original text
    url_count = len(re.findall(r"http\S+", full_text))
    
    # Scale URL count
    scaled_url = scaler.transform([[url_count]])
    
    # Combine features using scipy.sparse
    import scipy.sparse
    final_vector = scipy.sparse.hstack([tfidf_vector, scaled_url])
    
    # Make prediction
    prediction = pipeline_model.predict(final_vector)[0]
    prediction_proba = pipeline_model.predict_proba(final_vector)[0]
    
    # Extract features for display
    features = {
        'cleaned_text_length': len(cleaned_text),
        'original_text_length': len(full_text),
        'url_count': url_count,
        'word_count': len(cleaned_text.split()) if cleaned_text else 0,
        'tfidf_features': tfidf_vector.shape[1],
        'removed_urls': len(re.findall(r"http\S+", full_text)),
        'removed_emails': len(re.findall(r"\S+@\S+", full_text)),
        'removed_symbols': len(full_text) - len(cleaned_text) - len(re.findall(r"http\S+", full_text)) * 10  # Approximate
    }
    
    return prediction, prediction_proba, features

def predict_with_lexiguard(model, sender, subject, body):
    """
    Make prediction with LexiGuard model using its pipeline
    """
    # Prepare data in the format expected by LexiGuard
    data = prepare_lexiguard_data(sender, subject, body)
    
    # Make prediction using the pipeline
    prediction = model.predict(data)[0]
    prediction_proba = model.predict_proba(data)[0]
    
    # Extract some basic features for display
    features = {
        'body_length': len(body) if body else 0,
        'subject_length': len(subject) if subject else 0,
        'url_count': len(re.findall(r'http[s]?://\S+|www\.\S+', body)) if body else 0,
        'caps_ratio': (sum(1 for c in body if c.isupper()) / len(body) if body and len(body) > 0 else 0),
        'combined_text_length': len((subject + ' ' + body).strip()),
        'word_count': len((subject + ' ' + body).split()) if (subject or body) else 0
    }
    
    return prediction, prediction_proba, features

def predict_phishing(model, model_name, sender, subject, body):
    """
    Route prediction to appropriate model-specific function
    """
    config = MODEL_CONFIGS[model_name]
    
    if config["type"] == "xgboost":
        return predict_with_xgboost(model, sender, subject, body)
    elif config["type"] == "lexiguard":
        return predict_with_lexiguard(model, sender, subject, body)
    elif config["type"] == "random_forest":
        return predict_with_random_forest(model, sender, subject, body)
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

def get_risk_factors_xgboost(features):
    """Get risk factors for XGBoost model"""
    risk_factors = []
    if features.get('total_urls', 0) > 2:
        risk_factors.append(f"Multiple URLs detected ({features['total_urls']})")
    if features.get('body_action_words', 0) > 3:
        risk_factors.append(f"High number of action words ({features['body_action_words']})")
    if features.get('subject_money_mentions', 0) > 0:
        risk_factors.append("Money-related keywords in subject")
    if features.get('body_money_mentions', 0) > 0:
        risk_factors.append("Money-related keywords in body")
    if features.get('subject_all_caps_words', 0) > 2:
        risk_factors.append("Excessive capitalization in subject")
    if features.get('body_exclamation_count', 0) > 3:
        risk_factors.append("Excessive exclamation marks")
    return risk_factors

def get_risk_factors_lexiguard(features):
    """Get risk factors for LexiGuard model"""
    risk_factors = []
    if features.get('url_count', 0) > 2:
        risk_factors.append(f"Multiple URLs detected ({features['url_count']})")
    if features.get('caps_ratio', 0) > 0.3:
        risk_factors.append(f"High capitalization ratio ({features['caps_ratio']:.2%})")
    if features.get('body_length', 0) < 50:
        risk_factors.append("Very short email body (suspicious brevity)")
    if features.get('body_length', 0) > 2000:
        risk_factors.append("Very long email body")
    if features.get('subject_length', 0) > 100:
        risk_factors.append("Very long subject line")
    return risk_factors

def get_risk_factors_random_forest(features):
    """Get risk factors for Random Forest model"""
    risk_factors = []
    if features.get('url_count', 0) > 2:
        risk_factors.append(f"Multiple URLs detected ({features['url_count']})")
    if features.get('removed_emails', 0) > 1:
        risk_factors.append(f"Multiple email addresses found ({features['removed_emails']})")
    if features.get('cleaned_text_length', 0) < 20:
        risk_factors.append("Very short email content (suspicious brevity)")
    if features.get('word_count', 0) < 5:
        risk_factors.append("Very few words in email")
    if features.get('original_text_length', 0) > features.get('cleaned_text_length', 0) * 2:
        risk_factors.append("High ratio of removed content (URLs, emails, symbols)")
    return risk_factors

# Streamlit App
def main():
    # Display logo in header if available
    logo_path = "phishymail_ai_logo.png"
    if os.path.exists(logo_path):
        # Center the logo and title
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}' width='300'>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown("<h1 style='text-align: center;'>PhishyMail AI</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Multi-Model Phishing Email Detector</h3>", unsafe_allow_html=True)
    else:
        # Fallback if logo is not found
        st.title("PhishyMail AI - Multi-Model Phishing Email Detector")
    
    st.markdown("---")
    st.markdown("""
    **Detect phishing emails using multiple machine learning approaches.**
    
    Compare predictions from different models: handcrafted features vs. advanced NLP techniques.
    """)
    
    # Load all available models
    available_models, model_info = load_available_models()
    
    if not available_models:
        st.error("No models found! Please ensure your model files are in the correct directory.")
        st.info("""
        **Expected model files:**
        - XGBoost models: `phishing_xgboost_model_*.joblib`
        - LexiGuard model: `LexiGuard.joblib`
        - Random Forest model: `clf_phishing_pipeline.joblib`
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Model Selection")
        
        # Model dropdown
        selected_model_name = st.selectbox(
            "Choose Model",
            options=list(available_models.keys()),
            help="Select which model to use for prediction"
        )
        
        # Display model loading status
        st.markdown("### Model Status")
        for model_name, status in model_info.items():
            if "Loaded:" in status:
                st.success(status)
            else:
                st.error(status)
        
        # Model description
        config = MODEL_CONFIGS[selected_model_name]
        st.markdown("---")
        st.header("Model Details")
        st.markdown(f"**{config['description']}**")
        st.markdown(f"**Features:** {config['features']}")
        st.markdown(f"**Strengths:** {config['strengths']}")
        
        st.markdown("---")
        st.header("How Models Differ")
        st.markdown("""
        **XGBoost Model:**
        - Uses 22 handcrafted features
        - Analyzes specific patterns (URLs, capitalization, keywords)
        - Fast and interpretable
        
        **LexiGuard Model:**  
        - Uses TF-IDF text vectorization with n-grams
        - Captures semantic meaning and context
        - Includes linguistic features (body/subject length, caps ratio)
        - Better at detecting novel phishing patterns
        
        **Random Forest:**
        - TF-IDF vectorization + URL count features
        - Robust ensemble method with text cleaning
        - Removes URLs/emails during preprocessing
        - Good balance of performance and interpretability
        """)
    
    # Main content
    selected_model = available_models.get(selected_model_name)
    
    if selected_model:
        # Create input form
        with st.form("email_form"):
            st.header("Enter Email Details")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                sender = st.text_input(
                    "Sender Email Address",
                    placeholder="sender@example.com",
                    help="The email address of the sender"
                )
                
                subject = st.text_input(
                    "Email Subject",
                    placeholder="Subject line of the email",
                    help="The subject line of the email"
                )
            
            with col2:
                body = st.text_area(
                    "Email Body",
                    placeholder="Enter the full email content here...",
                    height=200,
                    help="The main content/body of the email"
                )
            
            # Submit buttons
            col1, col2 = st.columns(2)
            with col1:
                submitted_single = st.form_submit_button(f"Analyze with {selected_model_name}", use_container_width=True)
            with col2:
                submitted_all = st.form_submit_button("Compare All Models", use_container_width=True)
        
        # Process predictions
        if submitted_single or submitted_all:
            if not any([sender, subject, body]):
                st.warning("Please provide at least one field (sender, subject, or body) to analyze.")
            else:
                # Single model analysis
                if submitted_single:
                    with st.spinner(f"Analyzing email with {selected_model_name}..."):
                        try:
                            prediction, probabilities, features = predict_phishing(
                                selected_model, selected_model_name, sender, subject, body
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.header(f"{selected_model_name} Analysis Results")
                            
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                if prediction == 1:
                                    st.error("**PHISHING DETECTED**")
                                    st.error(f"This email appears to be a phishing attempt with {probabilities[1]:.1%} confidence.")
                                else:
                                    st.success("**LEGITIMATE EMAIL**")
                                    st.success(f"This email appears to be legitimate with {probabilities[0]:.1%} confidence.")
                            
                            with col2:
                                st.metric("Phishing Probability", f"{probabilities[1]:.1%}")
                            
                            with col3:
                                st.metric("Legitimacy Probability", f"{probabilities[0]:.1%}")
                            
                            # Feature analysis
                            st.markdown("---")
                            st.header("Feature Analysis")
                            
                            # Show detected features
                            feature_df = pd.DataFrame({
                                'Feature': [k.replace('_', ' ').title() for k in features.keys()],
                                'Value': list(features.values())
                            })
                            
                            non_zero_features = feature_df[feature_df['Value'] > 0]
                            if len(non_zero_features) > 0:
                                st.subheader("Detected Patterns:")
                                st.dataframe(non_zero_features.sort_values('Value', ascending=False), hide_index=True)
                            else:
                                st.info("No notable patterns detected in this email.")
                            
                            # Risk assessment
                            st.markdown("---")
                            st.header("Risk Assessment")
                            
                            config = MODEL_CONFIGS[selected_model_name]
                            if config["type"] == "xgboost":
                                risk_factors = get_risk_factors_xgboost(features)
                            elif config["type"] == "lexiguard":
                                risk_factors = get_risk_factors_lexiguard(features)
                            elif config["type"] == "random_forest":
                                risk_factors = get_risk_factors_random_forest(features)
                            else:
                                risk_factors = []
                            
                            if risk_factors:
                                st.warning("**Identified Risk Factors:**")
                                for factor in risk_factors:
                                    st.write(f"• {factor}")
                            else:
                                st.success("No major risk factors identified.")
                                
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.error("Please check that the model file is compatible and properly formatted.")
                
                # Multi-model comparison
                if submitted_all:
                    with st.spinner("Running analysis with all available models..."):
                        st.markdown("---")
                        st.header("Model Comparison Results")
                        
                        comparison_results = {}
                        detailed_results = {}
                        
                        for model_name, model in available_models.items():
                            try:
                                pred, prob, feats = predict_phishing(model, model_name, sender, subject, body)
                                comparison_results[model_name] = {
                                    'Prediction': 'Phishing' if pred == 1 else 'Legitimate',
                                    'Confidence': f"{prob[pred]:.1%}",
                                    'Phishing Probability': f"{prob[1]:.1%}",
                                    'Legitimacy Probability': f"{prob[0]:.1%}"
                                }
                                detailed_results[model_name] = (pred, prob, feats)
                            except Exception as e:
                                comparison_results[model_name] = {
                                    'Prediction': 'Error',
                                    'Confidence': 'N/A',
                                    'Phishing Probability': 'N/A',
                                    'Legitimacy Probability': str(e)[:50] + "..."
                                }
                        
                        # Display comparison table
                        comparison_df = pd.DataFrame(comparison_results).T
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Agreement analysis
                        predictions = [detailed_results[name][0] for name in detailed_results.keys()]
                        if len(set(predictions)) == 1:
                            st.success("**All models agree** on the prediction!")
                        else:
                            st.warning("**Models disagree** - consider the confidence levels and context.")
        
        # Example emails section
        st.markdown("---")
        st.header("Try These Examples")
        
        tab1, tab2, tab3 = st.tabs(["Phishing Example", "Legitimate Example", "Advanced Phishing"])
        
        with tab1:
            st.code("""
Sender: noreply@bank-security.com
Subject: URGENT: Your Account Will Be Suspended - ACT NOW!
Body: Dear Customer,

Your account has been flagged for suspicious activity. 
Click here immediately to verify your identity: http://fake-bank-verify.com

If you don't act within 24 hours, your account will be permanently suspended!

CLICK HERE NOW TO VERIFY: http://malicious-link.com/verify

Best regards,
Security Team
            """)
        
        with tab2:
            st.code("""
Sender: newsletter@company.com
Subject: Weekly Newsletter - New Product Updates
Body: Hi there,

Hope you're having a great week! We wanted to share some exciting updates about our latest product features.

This week we've released:
- Improved user dashboard
- Better mobile experience
- Enhanced security features

You can read more about these updates on our blog.

Best regards,
The Team
            """)
        
        with tab3:
            st.code("""
Sender: support@paypal-security.net
Subject: Payment Issue - Please Review
Body: Hello,

We noticed an unusual payment on your account. Please review the transaction details and confirm if this was authorized by you.

If you did not authorize this payment, please click here to report it immediately.

Transaction ID: TXN123456789
Amount: $299.99
Date: Today

Thank you for your attention to this matter.
PayPal Security Team
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>PhishyMail AI - Multi-Model Phishing Email Detector</p>
        <p><em>Combining XGBoost, LexiGuard NLP, and Random Forest approaches</em></p>
        <p><em>Always verify suspicious emails through official channels</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 