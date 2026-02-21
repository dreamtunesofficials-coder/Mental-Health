"""
Advanced Training Script with Hyperparameter Optimization
Tests different configurations to find the best model accuracy
"""
import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, prepare_dataset
from train_ml_models import MLModelTrainer

# Set paths
DATA_PATH = './data/dreaddit-train.csv'
MODEL_DIR = './models'

os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load dataset and prepare train/test splits."""
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset: {len(df)} samples")
    
    # Use correct columns
    text_col = 'text'
    label_col = 'label'
    
    print(f"Label distribution: {df[label_col].value_counts().to_dict()}")
    
    # Prepare data - using 80/20 split
    X_train, X_test, y_train, y_test = prepare_dataset(
        df, 
        text_column=text_col, 
        label_column=label_col,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def test_different_splits():
    """Test different train/test split ratios."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT TRAIN/TEST SPLITS")
    print("="*60)
    
    # Load full dataset
    df = load_dataset(DATA_PATH)
    text_col = 'text'
    label_col = 'label'
    
    split_ratios = [
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.9, 0.1)
    ]
    
    results = []
    
    for train_size, test_size in split_ratios:
        print(f"\n--- Split: {train_size*100:.0f}% train / {test_size*100:.0f}% test ---")
        
        # Re-split data
        X_train, X_test, y_train, y_test = prepare_dataset(
            df, 
            text_column=text_col, 
            label_column=label_col,
            test_size=test_size,
            random_state=42
        )
        
        # Train logistic regression (fastest)
        trainer = MLModelTrainer(
            model_type='logistic_regression',
            max_features=5000,
            ngram_range=(1, 2)
        )
        trainer.fit(X_train, y_train)
        
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        results.append({
            'split': f"{train_size*100:.0f}/{test_size*100:.0f}",
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        })
    
    return results

def test_different_features():
    """Test different feature configurations."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT FEATURE CONFIGURATIONS")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    feature_configs = [
        {'max_features': 1000, 'ngram_range': (1, 1)},
        {'max_features': 2000, 'ngram_range': (1, 2)},
        {'max_features': 5000, 'ngram_range': (1, 2)},
        {'max_features': 10000, 'ngram_range': (1, 2)},
        {'max_features': 5000, 'ngram_range': (1, 3)},
        {'max_features': 10000, 'ngram_range': (1, 3)},
    ]
    
    results = []
    
    for config in feature_configs:
        print(f"\n--- Config: {config} ---")
        
        trainer = MLModelTrainer(
            model_type='logistic_regression',
            max_features=config['max_features'],
            ngram_range=config['ngram_range']
        )
        trainer.fit(X_train, y_train)
        
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        results.append({
            'config': config,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        })
    
    return results

def test_different_models():
    """Test different model types with optimal configuration."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT MODEL TYPES")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    model_types = [
        'logistic_regression',
        'svm', 
        'random_forest',
        'naive_bayes',
        'gradient_boosting'
    ]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n>>> Training {model_type}...")
        
        trainer = MLModelTrainer(
            model_type=model_type,
            max_features=5000,
            ngram_range=(1, 2)
        )
        trainer.fit(X_train, y_train)
        
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        results[model_type] = metrics
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{model_type}_optimized.pkl')
        trainer.save(model_path)
    
    return results

def test_hyperparameters():
    """Test different hyperparameters for logistic regression."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING FOR LOGISTIC REGRESSION")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Different C values for regularization
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    
    best_accuracy = 0
    best_C = 1
    results = []
    
    for C in C_values:
        print(f"\n--- Testing C={C} ---")
        
        trainer = MLModelTrainer(
            model_type='logistic_regression',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Set custom C value
        from sklearn.linear_model import LogisticRegression
        trainer.model = LogisticRegression(
            C=C,
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        )
        
        trainer.fit(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        results.append({
            'C': C,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        })
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_C = C
    
    print(f"\n*** Best C value: {best_C} with accuracy: {best_accuracy:.4f} ***")
    
    return results, best_C

def find_best_model():
    """Find the overall best model configuration."""
    print("\n" + "="*60)
    print("FINDING BEST MODEL CONFIGURATION")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Try multiple configurations
    configurations = [
        {
            'model_type': 'logistic_regression',
            'max_features': 10000,
            'ngram_range': (1, 2),
            'C': 1
        },
        {
            'model_type': 'logistic_regression',
            'max_features': 5000,
            'ngram_range': (1, 3),
            'C': 10
        },
        {
            'model_type': 'svm',
            'max_features': 5000,
            'ngram_range': (1, 2),
            'C': 1
        },
        {
            'model_type': 'gradient_boosting',
            'max_features': 5000,
            'ngram_range': (1, 2),
        }
    ]
    
    best_accuracy = 0
    best_config = None
    best_trainer = None
    all_results = []
    
    for i, config in enumerate(configurations):
        print(f"\n>>> Testing config {i+1}/{len(configurations)}: {config['model_type']}")
        
        trainer = MLModelTrainer(
            model_type=config['model_type'],
            max_features=config.get('max_features', 5000),
            ngram_range=config.get('ngram_range', (1, 2))
        )
        
        # Apply custom hyperparameters if specified
        if 'C' in config:
            from sklearn.linear_model import LogisticRegression
            trainer.model = LogisticRegression(
                C=config['C'],
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        
        trainer.fit(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        all_results.append({
            'config': config,
            'metrics': metrics
        })
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_config = config
            best_trainer = trainer
    
    print("\n" + "="*60)
    print("BEST MODEL FOUND!")
    print("="*60)
    print(f"Model: {best_config['model_type']}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    # Save best model
    best_trainer.save(os.path.join(MODEL_DIR, 'best_model.pkl'))
    print(f"Model saved to: {MODEL_DIR}/best_model.pkl")
    
    return all_results, best_config, best_trainer

def main():
    print("="*60)
    print("MENTAL STRESS DETECTION - OPTIMIZED TRAINING")
    print("Testing different configurations to find best accuracy")
    print("="*60)
    
    # 1. Test different splits
    print("\n[1/4] Testing different train/test splits...")
    split_results = test_different_splits()
    
    # 2. Test different feature configurations
    print("\n[2/4] Testing different feature configurations...")
    feature_results = test_different_features()
    
    # 3. Test different model types
    print("\n[3/4] Testing different model types...")
    model_results = test_different_models()
    
    # 4. Find best overall model
    print("\n[4/4] Finding best overall model...")
    all_results, best_config, best_trainer = find_best_model()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\nModel Comparison:")
    for model_type, metrics in model_results.items():
        print(f"  {model_type}: {metrics['accuracy']:.4f} accuracy")
    
    print(f"\n*** BEST MODEL: {best_config['model_type']} ***")
    print(f"Configuration: {best_config}")
    
    # Save final results
    results_path = os.path.join(MODEL_DIR, 'optimization_results.txt')
    with open(results_path, 'w') as f:
        f.write("MENTAL STRESS DETECTION - OPTIMIZATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL COMPARISON:\n")
        for model_type, metrics in model_results.items():
            f.write(f"\n{model_type}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\n\nBEST CONFIG: {best_config}\n")
        f.write(f"ACCURACY: {model_results[best_config['model_type']]['accuracy']:.4f}\n")
    
    print(f"\nResults saved to: {results_path}")
    print("\n✅ Training optimization complete!")

if __name__ == "__main__":
    main()
