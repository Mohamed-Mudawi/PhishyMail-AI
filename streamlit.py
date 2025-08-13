import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
import os
import glob

# Page configuration
st.set_page_config(
    page_title="🛡️ Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature extraction functions (based on your training pipeline)
def extract_features(sender, subject, body):
    """
    Extract features from email components matching your training pipeline
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

def load_model():
    """
    Load the most recent XGBoost model
    """
    try:
        # Look for model files in current directory
        model_files = glob.glob("phishing_xgboost_model_*.joblib")
        if not model_files:
            return None, "No trained model found. Please ensure your XGBoost model file is in the same directory."
        
        # Use the most recent model file
        latest_model = max(model_files, key=os.path.getctime)
        model = joblib.load(latest_model)
        return model, f"Model loaded: {latest_model}"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def predict_phishing(model, sender, subject, body):
    """
    Predict if email is phishing based on extracted features
    """
    # Extract features
    features = extract_features(sender, subject, body)
    
    # Define the exact feature order from your training pipeline
    feature_order = [
        'subject_non_ascii_ratio', 'sender_domain_length', 'subject_colon_count',
        'subject_capital_ratio', 'body_url_char_ratio', 'body_hyphen_count',
        'subject_non_ascii_count', 'sender_length', 'subject_digit_count',
        'total_urls', 'body_phone_count', 'body_email_count', 'subject_all_caps_words',
        'body_exclamation_count', 'subject_sentence_count', 'body_colon_count',
        'subject_money_mentions', 'subject_mixed_case_words', 'body_action_words',
        'body_money_mentions', 'body_all_caps_words', 'body_non_ascii_ratio'
    ]
    
    # Create feature vector in correct order
    feature_vector = []
    for feature_name in feature_order:
        feature_vector.append(features.get(feature_name, 0))
    
    # Convert to DataFrame for prediction
    X = pd.DataFrame([feature_vector], columns=feature_order)
    
    # Make prediction
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]
    
    return prediction, prediction_proba, features

# Streamlit App
def main():
    # Title and description
    st.title("🛡️ Phishing Email Detector")
    st.markdown("---")
    st.markdown("""
    **Detect phishing emails using machine learning!**
    
    Enter the email details below and our trained XGBoost model will analyze the content to determine 
    if it's likely to be a phishing attempt.
    """)
    
    # Load model
    model, model_status = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        if model:
            st.success("✅ Model loaded successfully!")
            st.info(model_status)
        else:
            st.error("❌ Model not found")
            st.error(model_status)
            st.markdown("""
            **To use this app:**
            1. Train your XGBoost model using the provided training script
            2. Place the generated `.joblib` model file in the same directory as this app
            3. Restart the app
            """)
        
        st.markdown("---")
        st.header("🔍 How it works")
        st.markdown("""
        The model analyzes:
        - **Sender patterns** (length, domain)
        - **Subject characteristics** (caps, symbols, keywords)
        - **Body content** (URLs, action words, formatting)
        
        Based on 22 different features extracted from your email.
        """)
    
    # Main content
    if model:
        # Create input form
        with st.form("email_form"):
            st.header("📧 Enter Email Details")
            
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
            
            # Submit button
            submitted = st.form_submit_button("🔍 Analyze Email", use_container_width=True)
        
        # Process prediction
        if submitted:
            if not any([sender, subject, body]):
                st.warning("⚠️ Please provide at least one field (sender, subject, or body) to analyze.")
            else:
                with st.spinner("Analyzing email..."):
                    try:
                        prediction, probabilities, features = predict_phishing(model, sender, subject, body)
                        
                        # Display results
                        st.markdown("---")
                        st.header("🎯 Analysis Results")
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            if prediction == 1:
                                st.error("🚨 **PHISHING DETECTED**")
                                st.error(f"This email appears to be a phishing attempt with {probabilities[1]:.1%} confidence.")
                            else:
                                st.success("✅ **LEGITIMATE EMAIL**")
                                st.success(f"This email appears to be legitimate with {probabilities[0]:.1%} confidence.")
                        
                        with col2:
                            st.metric("Phishing Probability", f"{probabilities[1]:.1%}")
                        
                        with col3:
                            st.metric("Legitimacy Probability", f"{probabilities[0]:.1%}")
                        
                        # Feature analysis
                        st.markdown("---")
                        st.header("📊 Feature Analysis")
                        
                        # Create feature importance visualization
                        feature_df = pd.DataFrame({
                            'Feature': list(features.keys()),
                            'Value': list(features.values())
                        })
                        
                        # Show top suspicious features
                        suspicious_features = feature_df[feature_df['Value'] > 0].sort_values('Value', ascending=False)
                        
                        if len(suspicious_features) > 0:
                            st.subheader("🔍 Detected Suspicious Patterns:")
                            for _, row in suspicious_features.head(10).iterrows():
                                feature_name = row['Feature'].replace('_', ' ').title()
                                st.write(f"• **{feature_name}**: {row['Value']}")
                        else:
                            st.info("No particularly suspicious patterns detected in this email.")
                        
                        # Risk factors explanation
                        st.markdown("---")
                        st.header("⚠️ Risk Assessment")
                        
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
                        
                        if risk_factors:
                            st.warning("**Identified Risk Factors:**")
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                        else:
                            st.success("No major risk factors identified.")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        # Example emails section
        st.markdown("---")
        st.header("📝 Try These Examples")
        
        tab1, tab2 = st.tabs(["🎣 Phishing Example", "✅ Legitimate Example"])
        
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🛡️ Phishing Email Detector | Built with Streamlit & XGBoost</p>
        <p><em>Always verify suspicious emails through official channels</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()