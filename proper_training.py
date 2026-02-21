"""
Proper Training with Cross-Validation to Prevent Overfitting
Tests model generalization on held-out data
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, prepare_dataset
from train_ml_models import MLModelTrainer
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA_PATH = './data/dreaddit-train.csv'
MODEL_DIR = './models'

def check_overfitting():
    """
    Check for overfitting using cross-validation.
    Compare training score vs validation score - large gap = overfitting
    """
    print("="*60)
    print("CHECKING FOR OVERFITTING")
    print("="*60)
    
    # Load dataset
    df = load_dataset(DATA_PATH)
    print(f"Dataset: {len(df)} samples")
    
    text_col = 'text'
    label_col = 'label'
    
    # Use 80/20 split for proper evaluation (not the best split to avoid overfitting to test)
    X_train, X_test, y_train, y_test = prepare_dataset(
        df,
        text_column=text_col,
        label_column=label_col,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Test (held-out): {len(X_test)} samples")
    
    # Train model
    trainer = MLModelTrainer(
        model_type='logistic_regression',
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    # Fit on training data
    trainer.fit(X_train, y_train)
    
    # Get training score
    train_pred = trainer.predict(X_train)
    from sklearn.metrics import accuracy_score
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Get test score (held-out)
    test_accuracy_dict = trainer.evaluate(X_test, y_test)
    test_accuracy = test_accuracy_dict['accuracy']
    
    print(f"\n📊 SCORES:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy:    {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Check for overfitting
    gap = train_accuracy - test_accuracy
    print(f"\n📏 GAP (Train - Test): {gap:.4f}")
    
    if gap > 0.1:
        print("⚠️  WARNING: Possible overfitting! Gap > 10%")
    elif gap > 0.05:
        print("⚠️  CAUTION: Some overfitting detected. Gap > 5%")
    else:
        print("✅ Good! Model generalizes well.")
    
    # Cross-validation for more robust estimate
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-Fold)")
    print("="*60)
    
    # Use cross-validation on full dataset
    X_all = df['text'].values
    y_all = df['label'].values
    
    # Create feature extractor for CV
    from feature_engineering import FeatureExtractor
    vectorizer = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
    X_features = vectorizer.fit_transform(X_all)
    
    # Create model for CV
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(
        model, 
        X_features,
        y_all, 
        cv=cv, 
        scoring='accuracy'
    )
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'gap': gap,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores)
    }

def train_with_regularization():
    """Train with regularization to prevent overfitting."""
    print("\n" + "="*60)
    print("TRAINING WITH REGULARIZATION")
    print("="*60)
    
    df = load_dataset(DATA_PATH)
    text_col = 'text'
    label_col = 'label'
    
    # Try different regularization strengths
    C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    results = []
    
    for C in C_values:
        # Use 5-fold CV for each C
        X = df[text_col].values
        y = df[label_col].values
        
        trainer = MLModelTrainer(
            model_type='logistic_regression',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Set custom C (regularization strength)
        from sklearn.linear_model import LogisticRegression
        trainer.model = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Get features
        from feature_engineering import FeatureExtractor
        trainer.vectorizer = FeatureExtractor(
            max_features=5000,
            ngram_range=(1, 2)
        )
        trainer.vectorizer.fit(X)
        X_features = trainer.vectorizer.transform(X)
        
        cv_scores = cross_val_score(
            trainer.model, X_features, y, cv=cv, scoring='accuracy'
        )
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"C={C:5.3f}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        results.append({
            'C': C,
            'mean_cv': mean_score,
            'std_cv': std_score
        })
    
    # Find best C
    best_result = max(results, key=lambda x: x['mean_cv'])
    best_C = best_result['C']
    
    print(f"\n🏆 Best C: {best_C} with CV accuracy: {best_result['mean_cv']:.4f}")
    
    # Train final model with best C
    print("\n" + "="*60)
    print(f"TRAINING FINAL MODEL WITH C={best_C}")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_dataset(
        df,
        text_column=text_col,
        label_column=label_col,
        test_size=0.2,
        random_state=42
    )
    
    final_trainer = MLModelTrainer(
        model_type='logistic_regression',
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    # Set best C
    from sklearn.linear_model import LogisticRegression
    final_trainer.model = LogisticRegression(
        C=best_C,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    final_trainer.fit(X_train, y_train)
    
    # Evaluate
    train_pred = final_trainer.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    test_metrics = final_trainer.evaluate(X_test, y_test)
    
    print(f"\n📊 FINAL MODEL RESULTS:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:       {test_metrics['precision']:.4f}")
    print(f"  Recall:         {test_metrics['recall']:.4f}")
    print(f"  F1 Score:       {test_metrics['f1_score']:.4f}")
    
    # Check gap
    gap = train_acc - test_metrics['accuracy']
    print(f"\n📏 OVERFITTING GAP: {gap:.4f}")
    
    if gap > 0.1:
        print("⚠️  WARNING: Possible overfitting!")
    else:
        print("✅ Good generalization!")
    
    # Save model
    final_trainer.save(os.path.join(MODEL_DIR, 'proper_model.pkl'))
    print(f"\n✅ Model saved to: {MODEL_DIR}/proper_model.pkl")
    
    return test_metrics, best_C

def main():
    print("="*60)
    print("PROPER TRAINING - PREVENT OVERFITTING")
    print("="*60)
    
    # Step 1: Check for overfitting
    check_overfitting()
    
    # Step 2: Train with regularization
    metrics, best_C = train_with_regularization()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best Regularization (C): {best_C}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Save final results
    results_path = os.path.join(MODEL_DIR, 'proper_training_results.txt')
    with open(results_path, 'w') as f:
        f.write("PROPER TRAINING RESULTS (With Overfitting Prevention)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best C (regularization): {best_C}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    main()
