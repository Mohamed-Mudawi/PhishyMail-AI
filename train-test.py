import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

def complete_xgboost_training(dataset_path='optimized_phishing_dataset.csv'):
    """
    Complete XGBoost training pipeline for phishing detection
    """
    print("🚀 COMPLETE XGBOOST TRAINING PIPELINE")
    print("=" * 50)
    
    # Load optimized dataset
    print("\n📂 Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Define features
    top_features = [
        'subject_non_ascii_ratio', 'sender_domain_length', 'subject_colon_count',
        'subject_capital_ratio', 'body_url_char_ratio', 'body_hyphen_count',
        'subject_non_ascii_count', 'sender_length', 'subject_digit_count',
        'total_urls', 'body_phone_count', 'body_email_count', 'subject_all_caps_words',
        'body_exclamation_count', 'subject_sentence_count', 'body_colon_count',
        'subject_money_mentions', 'subject_mixed_case_words', 'body_action_words',
        'body_money_mentions', 'body_all_caps_words', 'body_non_ascii_ratio'
    ]
    
    # Prepare features and target
    X = df[top_features]
    y = df['label']
    
    # Convert target to binary if needed
    if y.dtype == 'object':
        y = (y == 'phishing').astype(int)
        print(f"   Converted labels: {y.value_counts().to_dict()}")
    
    print(f"   Features: {len(top_features)}")
    print(f"   Samples: {len(X)}")
    
    # Check for any data issues
    print(f"   Missing values: {X.isnull().sum().sum()}")
    print(f"   Label distribution: {y.value_counts().to_dict()}")
    
    # === STEP 1: INITIAL TRAIN/VALIDATION/TEST SPLIT ===
    print("\n🔀 SPLITTING DATA...")
    print("-" * 20)
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 75% train, 25% validation (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"   Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # === STEP 2: BASELINE MODEL ===
    print("\n🏁 TRAINING BASELINE MODEL...")
    print("-" * 30)
    
    baseline_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_val)
    baseline_accuracy = accuracy_score(y_val, baseline_pred)
    
    print(f"   Baseline accuracy: {baseline_accuracy:.4f}")
    
    # === STEP 3: HYPERPARAMETER TUNING ===
    print("\n🔧 HYPERPARAMETER TUNING...")
    print("-" * 27)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Grid search with cross-validation
    print("   Running grid search (this may take a while)...")
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,  # 3-fold CV to save time
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    # === STEP 4: TRAIN OPTIMIZED MODEL ===
    print("\n🏆 TRAINING OPTIMIZED MODEL...")
    print("-" * 31)
    
    best_model = grid_search.best_estimator_
    
    # Validate on validation set
    val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"   Validation accuracy: {val_accuracy:.4f}")
    print(f"   Improvement over baseline: {val_accuracy - baseline_accuracy:.4f}")
    
    # === STEP 5: FINAL EVALUATION ON TEST SET ===
    print("\n🎯 FINAL EVALUATION ON TEST SET...")
    print("-" * 35)
    
    # Predict on test set
    test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"   Test accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-" * 37)
    print(classification_report(y_test, test_pred, target_names=['Ham', 'Phishing']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n🔢 CONFUSION MATRIX:")
    print(f"   True Negatives (Ham as Ham): {cm[0,0]:,}")
    print(f"   False Positives (Ham as Phishing): {cm[0,1]:,}")
    print(f"   False Negatives (Phishing as Ham): {cm[1,0]:,}")
    print(f"   True Positives (Phishing as Phishing): {cm[1,1]:,}")
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # === STEP 6: FEATURE IMPORTANCE ANALYSIS ===
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS...")
    print("-" * 34)
    
    feature_importance = pd.DataFrame({
        'feature': top_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # === STEP 7: CROSS-VALIDATION FOR ROBUSTNESS ===
    print("\n🔄 CROSS-VALIDATION FOR ROBUSTNESS...")
    print("-" * 37)
    
    cv_scores = cross_val_score(best_model, X_temp, y_temp, cv=5, scoring='accuracy')
    print(f"   5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Individual folds: {[f'{score:.4f}' for score in cv_scores]}")
    
    # === STEP 8: SAVE MODEL AND RESULTS ===
    print("\n💾 SAVING MODEL AND RESULTS...")
    print("-" * 29)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the trained model
    model_filename = f'phishing_xgboost_model_{timestamp}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"   ✅ Model saved: {model_filename}")
    
    # Save feature importance
    importance_filename = f'feature_importance_{timestamp}.csv'
    feature_importance.to_csv(importance_filename, index=False)
    print(f"   ✅ Feature importance saved: {importance_filename}")
    
    # Save training results
    results = {
        'timestamp': timestamp,
        'dataset_shape': df.shape,
        'n_features': len(top_features),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'baseline_accuracy': float(baseline_accuracy),
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'val_accuracy': float(val_accuracy),
        'test_accuracy': float(test_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'features_used': top_features
    }
    
    results_filename = f'training_results_{timestamp}.json'
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ Results saved: {results_filename}")
    
    # === STEP 9: VISUALIZATION (if matplotlib available) ===
    try:
        print("\n📈 CREATING VISUALIZATIONS...")
        print("-" * 27)
        
        # Feature importance plot
        plt.figure(figsize=(10, 8))
        top_10_features = feature_importance.head(10)
        plt.barh(range(len(top_10_features)), top_10_features['importance'])
        plt.yticks(range(len(top_10_features)), top_10_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features for Phishing Detection')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_filename = f'feature_importance_plot_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Feature plot saved: {plot_filename}")
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Ham', 'Phishing'],
                   yticklabels=['Ham', 'Phishing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_filename = f'confusion_matrix_{timestamp}.png'
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Confusion matrix saved: {cm_filename}")
        
    except ImportError:
        print("   ⚠️  Matplotlib/Seaborn not available - skipping plots")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETE - FINAL SUMMARY")
    print("="*50)
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔄 Recall: {recall:.4f}")
    print(f"⚖️  F1-Score: {f1:.4f}")
    print(f"✅ Model saved as: {model_filename}")
    print(f"📁 All files saved with timestamp: {timestamp}")
    
    if test_accuracy > 0.95:
        print("🏆 EXCELLENT! Your model achieved >95% accuracy!")
    elif test_accuracy > 0.90:
        print("👍 GOOD! Your model achieved >90% accuracy!")
    else:
        print("⚠️  Consider feature engineering or more data for better performance")
    
    return best_model, results

# === USAGE ===
if __name__ == "__main__":
    # Make sure you have the optimized dataset
    try:
        model, results = complete_xgboost_training('optimized_phishing_dataset.csv')
        print("\n🚀 Training pipeline completed successfully!")
    except FileNotFoundError:
        print("❌ Error: 'optimized_phishing_dataset.csv' not found!")
        print("   Please run the feature extraction code first.")
    except Exception as e:
        print(f"❌ Error during training: {e}")