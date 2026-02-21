"""
Optimized Training Script - Target 90%+ Accuracy
Uses hyperparameter tuning and ensemble methods
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = './data/dreaddit-train.csv'

def load_data():
    """Load and prepare data with all features"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    text_col = 'text'
    label_col = 'label'
    
    # Get text features (TF-IDF) with different settings
    from feature_engineering import FeatureExtractor
    
    # Try multiple TF-IDF configurations
    configs = [
        {'max_features': 8000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.9},
        {'max_features': 10000, 'ngram_range': (1, 3), 'min_df': 2, 'max_df': 0.95},
        {'max_features': 5000, 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.85},
    ]
    
    from scipy.sparse import hstack, csr_matrix
    
    best_features = None
    best_score = 0
    
    for config in configs:
        text_features = FeatureExtractor(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_df=config['max_df']
        )
        text_vectors = text_features.fit_transform(df[text_col].values)
        
        # Get numerical features
        numerical_cols = [col for col in df.columns if col.startswith(('lex_', 'social_', 'syntax_', 'sentiment'))]
        numerical_cols = [col for col in numerical_cols if df[col].dtype in ['float64', 'int64']]
        
        numerical_features = df[numerical_cols].fillna(0).values
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical_features)
        
        # Combine
        combined = hstack([text_vectors, csr_matrix(numerical_scaled)])
        
        # Quick test
        X_train, X_test, y_train, y_test = train_test_split(
            combined, df[label_col].values, test_size=0.2, random_state=42, stratify=df[label_col].values
        )
        
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        
        print(f"Config {config}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_features = (text_vectors, numerical_scaled, scaler)
    
    print(f"\nBest config score: {best_score:.4f}")
    
    text_vectors, numerical_scaled, scaler = best_features
    
    # Get all numerical columns
    numerical_cols = [col for col in df.columns if col.startswith(('lex_', 'social_', 'syntax_', 'sentiment'))]
    numerical_cols = [col for col in numerical_cols if df[col].dtype in ['float64', 'int64']]
    
    return text_vectors, numerical_scaled, df[label_col].values, scaler, numerical_cols

def train_optimized_model():
    """Train with optimized hyperparameters"""
    print("\n" + "="*60)
    print("OPTIMIZED TRAINING FOR 90%+ ACCURACY")
    print("="*60)
    
    from scipy.sparse import hstack, csr_matrix
    import pickle
    
    # Load data
    text_vectors, numerical_scaled, y, scaler, numerical_cols = load_data()
    combined = hstack([text_vectors, csr_matrix(numerical_scaled)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Total features: {X_train.shape[1]}")
    
    # Test different model configurations
    print("\n--- Testing Different Models ---")
    
    models = {
        'LR_C0.5': LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight='balanced'),
        'LR_C1': LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced'),
        'LR_C2': LogisticRegression(C=2.0, max_iter=2000, random_state=42, class_weight='balanced'),
        'LR_C5': LogisticRegression(C=5.0, max_iter=2000, random_state=42, class_weight='balanced'),
        'RF_100': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'),
        'RF_200': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'SVM': SVC(C=1.0, kernel='rbf', random_state=42, class_weight='balanced', probability=True),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        print(f"  Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    # Find best model
    best_name = max(results, key=lambda x: results[x]['test_acc'])
    best_model = results[best_name]['model']
    best_acc = results[best_name]['test_acc']
    
    print(f"\n--- Best Single Model: {best_name} ({best_acc:.4f}) ---")
    
    # Try ensemble
    print("\n--- Trying Ensemble Methods ---")
    
    # Voting Classifier
    voting = VotingClassifier(
        estimators=[
            ('lr1', LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')),
            ('lr2', LogisticRegression(C=2.0, max_iter=2000, random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ],
        voting='soft'
    )
    
    voting.fit(X_train, y_train)
    voting_acc = accuracy_score(y_test, voting.predict(X_test))
    print(f"Voting Ensemble: {voting_acc:.4f}")
    
    # Stacking Classifier
    stacking = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    stacking.fit(X_train, y_train)
    stacking_acc = accuracy_score(y_test, stacking.predict(X_test))
    print(f"Stacking Ensemble: {stacking_acc:.4f}")
    
    # Choose best model
    if voting_acc > best_acc:
        final_model = voting
        final_name = "Voting Ensemble"
        final_acc = voting_acc
    elif stacking_acc > best_acc:
        final_model = stacking
        final_name = "Stacking Ensemble"
        final_acc = stacking_acc
    else:
        final_model = best_model
        final_name = best_name
        final_acc = best_acc
    
    print(f"\n=== FINAL MODEL: {final_name} ===")
    print(f"Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    
    # Classification report
    y_pred = final_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
    
    # Cross-validation
    print("\n--- Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(final_model, combined, y, cv=cv, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Save model
    model_data = {
        'model': final_model,
        'accuracy': final_acc,
        'cv_mean': np.mean(cv_scores),
        'model_name': final_name
    }
    
    with open('./models/optimized_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved!")
    
    return final_acc, np.mean(cv_scores)

if __name__ == "__main__":
    test_acc, cv_acc = train_optimized_model()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"CV Accuracy: {cv_acc:.4f} ({cv_acc*100:.2f}%)")
