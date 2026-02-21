"""
Advanced Training with ALL Features - Target 90%+ Accuracy
Uses text features + LIWC features + sentiment + social features
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = './data/dreaddit-train.csv'

def load_all_features():
    """Load dataset with ALL features"""
    print("Loading dataset with ALL features...")
    df = pd.read_csv(DATA_PATH)
    
    text_col = 'text'
    label_col = 'label'
    
    # Get text features (TF-IDF)
    from feature_engineering import FeatureExtractor
    text_features = FeatureExtractor(max_features=3000, ngram_range=(1, 2))
    text_vectors = text_features.fit_transform(df[text_col].values)
    
    # Get LIWC and other numerical features
    numerical_cols = [col for col in df.columns if col.startswith(('lex_', 'social_', 'syntax_', 'sentiment'))]
    numerical_cols = [col for col in numerical_cols if df[col].dtype in ['float64', 'int64']]
    
    print(f"Found {len(numerical_cols)} numerical features")
    
    # Get numerical features
    numerical_features = df[numerical_cols].fillna(0).values
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    
    # Combine text and numerical features
    from scipy.sparse import hstack, csr_matrix
    combined_features = hstack([text_vectors, csr_matrix(numerical_features_scaled)])
    
    print(f"Total features: {combined_features.shape[1]}")
    
    return combined_features, df[label_col].values, scaler, text_features

def train_advanced_model():
    """Train advanced ensemble model with all features"""
    print("\n" + "="*60)
    print("ADVANCED TRAINING WITH ALL FEATURES")
    print("="*60)
    
    X, y, scaler, text_features = load_all_features()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Try different models
    print("\n--- Testing Individual Models ---")
    
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=2000, random_state=42, class_weight='balanced', solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_name = name
    
    print(f"\n--- Best Model: {best_name} ({best_accuracy:.4f}) ---")
    
    # Classification report
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
    
    # Cross-validation
    print("\n--- Cross-Validation (5-Fold) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Save model
    import pickle
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'text_features': text_features,
        'accuracy': best_accuracy,
        'cv_mean': np.mean(cv_scores)
    }
    
    with open('./models/advanced_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to: ./models/advanced_model.pkl")
    
    return best_accuracy, np.mean(cv_scores)

if __name__ == "__main__":
    from feature_engineering import FeatureExtractor
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    text_col = 'text'
    
    # Run advanced training
    test_acc, cv_acc = train_advanced_model()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"CV Accuracy: {cv_acc:.4f} ({cv_acc*100:.2f}%)")
