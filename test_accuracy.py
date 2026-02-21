"""
Comprehensive Accuracy Testing Script
Tests both training and test accuracy with multiple methods
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from data_loader import load_dataset, prepare_dataset
from train_ml_models import MLModelTrainer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = './data/dreaddit-train.csv'

def test_training_vs_test_accuracy():
    """Compare training vs test accuracy on the same split"""
    print("="*70)
    print("TEST 1: Training vs Test Accuracy (Same Split)")
    print("="*70)
    
    # Load dataset
    df = load_dataset(DATA_PATH)
    print(f"Dataset: {len(df)} samples")
    
    # Use 80/20 split
    X_train, X_test, y_train, y_test = prepare_dataset(
        df,
        text_column='text',
        label_column='label',
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    trainer = MLModelTrainer(
        model_type='logistic_regression',
        max_features=5000,
        ngram_range=(1, 2)
    )
    trainer.fit(X_train, y_train)
    
    # Training accuracy
    train_pred = trainer.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test accuracy
    test_pred = trainer.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n📊 Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap:               {train_acc - test_acc:.4f}")
    
    return train_acc, test_acc

def test_multiple_splits():
    """Test on multiple random splits to see variance"""
    print("\n" + "="*70)
    print("TEST 2: Multiple Random Splits (10 different splits)")
    print("="*70)
    
    df = load_dataset(DATA_PATH)
    X = df['text'].values
    y = df['label'].values
    
    train_accuracies = []
    test_accuracies = []
    
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )
        
        trainer = MLModelTrainer(
            model_type='logistic_regression',
            max_features=5000,
            ngram_range=(1, 2)
        )
        trainer.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, trainer.predict(X_train))
        test_acc = accuracy_score(y_test, trainer.predict(X_test))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"  Split {i}: Train={train_acc:.2%}, Test={test_acc:.2%}, Gap={train_acc-test_acc:.2%}")
    
    print(f"\n📊 Summary:")
    print(f"  Avg Training Accuracy: {np.mean(train_accuracies):.2%} (+/- {np.std(train_accuracies):.2%})")
    print(f"  Avg Test Accuracy:     {np.mean(test_accuracies):.2%} (+/- {np.std(test_accuracies):.2%})")
    print(f"  Avg Gap:               {np.mean(np.array(train_accuracies) - np.array(test_accuracies)):.2%}")
    
    return train_accuracies, test_accuracies

def test_cross_validation():
    """5-Fold Cross-Validation"""
    print("\n" + "="*70)
    print("TEST 3: 5-Fold Cross-Validation")
    print("="*70)
    
    df = load_dataset(DATA_PATH)
    X = df['text'].values
    y = df['label'].values
    
    # Get features
    from feature_engineering import FeatureExtractor
    vectorizer = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
    X_features = vectorizer.fit_transform(X)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X_features, y, cv=cv, scoring='accuracy')
    
    print(f"  Fold 1: {cv_scores[0]:.2%}")
    print(f"  Fold 2: {cv_scores[1]:.2%}")
    print(f"  Fold 3: {cv_scores[2]:.2%}")
    print(f"  Fold 4: {cv_scores[3]:.2%}")
    print(f"  Fold 5: {cv_scores[4]:.2%}")
    print(f"\n  Mean CV Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
    
    return cv_scores

def test_different_train_sizes():
    """Test how accuracy changes with different training sizes"""
    print("\n" + "="*70)
    print("TEST 4: Different Training Sizes")
    print("="*70)
    
    df = load_dataset(DATA_PATH)
    X = df['text'].values
    y = df['label'].values
    
    sizes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for size in sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-size, random_state=42, stratify=y
        )
        
        trainer = MLModelTrainer(
            model_type='logistic_regression',
            max_features=5000,
            ngram_range=(1, 2)
        )
        trainer.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, trainer.predict(X_train))
        test_acc = accuracy_score(y_test, trainer.predict(X_test))
        
        print(f"  Train {int(size*100)}%: Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}, Gap={train_acc-test_acc:.2%}")

def test_different_models():
    """Test different ML models"""
    print("\n" + "="*70)
    print("TEST 5: Different ML Models")
    print("="*70)
    
    df = load_dataset(DATA_PATH)
    
    X_train, X_test, y_train, y_test = prepare_dataset(
        df,
        text_column='text',
        label_column='label',
        test_size=0.2,
        random_state=42
    )
    
    models = ['logistic_regression', 'svm', 'random_forest', 'naive_bayes']
    
    for model_type in models:
        trainer = MLModelTrainer(
            model_type=model_type,
            max_features=5000,
            ngram_range=(1, 2)
        )
        trainer.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, trainer.predict(X_train))
        test_acc = accuracy_score(y_test, trainer.predict(X_test))
        
        print(f"  {model_type:25} Train={train_acc:.2%}, Test={test_acc:.2%}, Gap={train_acc-test_acc:.2%}")

def test_with_fresh_data():
    """Test with completely fresh data split"""
    print("\n" + "="*70)
    print("TEST 6: Fresh Random Split (random_state=99)")
    print("="*70)
    
    df = load_dataset(DATA_PATH)
    
    X_train, X_test, y_train, y_test = prepare_dataset(
        df,
        text_column='text',
        label_column='label',
        test_size=0.2,
        random_state=99  # Different random state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    trainer = MLModelTrainer(
        model_type='logistic_regression',
        max_features=5000,
        ngram_range=(1, 2)
    )
    trainer.fit(X_train, y_train)
    
    train_pred = trainer.predict(X_train)
    test_pred = trainer.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n📊 Fresh Split Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap:               {train_acc - test_acc:.4f}")
    
    # Detailed metrics
    print(f"\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, test_pred, target_names=['No Stress', 'Stress']))
    
    return train_acc, test_acc

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE ACCURACY TESTING")
    print("="*70)
    
    # Test 1: Basic train vs test
    train_acc, test_acc = test_training_vs_test_accuracy()
    
    # Test 2: Multiple random splits
    train_accs, test_accs = test_multiple_splits()
    
    # Test 3: Cross-validation
    cv_scores = test_cross_validation()
    
    # Test 4: Different training sizes
    test_different_train_sizes()
    
    # Test 5: Different models
    test_different_models()
    
    # Test 6: Fresh data split
    fresh_train, fresh_test = test_with_fresh_data()
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"✓ Training Accuracy (avg):  {np.mean(train_accs):.2%}")
    print(f"✓ Test Accuracy (avg):      {np.mean(test_accs):.2%}")
    print(f"✓ Cross-Validation:        {np.mean(cv_scores):.2%}")
    print(f"✓ Gap (avg):               {np.mean(np.array(train_accs) - np.array(test_accs)):.2%}")
    
    # Save results
    results_path = './models/comprehensive_test_results.txt'
    with open(results_path, 'w') as f:
        f.write("COMPREHENSIVE ACCURACY TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training Accuracy (avg): {np.mean(train_accs):.4f}\n")
        f.write(f"Test Accuracy (avg): {np.mean(test_accs):.4f}\n")
        f.write(f"Cross-Validation: {np.mean(cv_scores):.4f}\n")
        f.write(f"Gap (avg): {np.mean(np.array(train_accs) - np.array(test_accs)):.4f}\n")
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
