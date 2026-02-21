"""
Enhanced Training Script for Mental Stress Detection
Uses the larger dreaddit dataset for better model performance
"""
import numpy as np
import pandas as pd
import pickle
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, prepare_dataset
from train_ml_models import MLModelTrainer

# Set paths - use the larger dreaddit dataset
DATA_PATH = './data/dreaddit-train.csv'
MODEL_DIR = './models'

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("MENTAL STRESS DETECTION - MODEL TRAINING")
print("Using larger dreaddit dataset for better accuracy")
print("=" * 60)

# Load dataset
print("\n[1] Loading dataset...")
df = load_dataset(DATA_PATH)
print(f"    Dataset loaded: {len(df)} samples")

# Check for label column
if 'label' in df.columns:
    label_col = 'label'
elif 'LABEL' in df.columns:
    label_col = 'LABEL'
else:
    print("    Looking for label column...")
    print(f"    Available columns: {df.columns.tolist()[:10]}...")
    label_col = 'label'

# Get text and label columns
if 'text' in df.columns:
    text_col = 'text'
elif 'TEXT' in df.columns:
    text_col = 'TEXT'
else:
    # Look for text column
    text_cols = [c for c in df.columns if 'text' in c.lower()]
    if text_cols:
        text_col = text_cols[0]
    else:
        text_col = df.columns[3]  # Usually 'text' is at index 3 based on the data

print(f"    Using text column: {text_col}")
print(f"    Using label column: {label_col}")
label_dist = df[label_col].value_counts().to_dict()
print(f"    Label distribution: {label_dist}")

# Prepare dataset
print("\n[2] Preparing dataset...")
X_train, X_test, y_train, y_test = prepare_dataset(
    df, 
    text_column=text_col, 
    label_column=label_col
)
print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")

# Train multiple models
print("\n[3] Training multiple models...")
print("-" * 60)

model_types = ['logistic_regression', 'svm', 'random_forest', 'naive_bayes']
results = {}

for model_type in model_types:
    print(f"\n>>> Training {model_type.replace('_', ' ').title()}...")
    
    # Create trainer
    trainer = MLModelTrainer(model_type=model_type)
    
    # Fit model
    trainer.fit(X_train, y_train)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    results[model_type] = metrics
    
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:   {metrics['recall']:.4f}")
    print(f"    F1 Score: {metrics['f1_score']:.4f}")
    
    # Cross-validation
    cv_results = trainer.cross_validate(
        np.concatenate([X_train, X_test]), 
        np.concatenate([y_train, y_test]), 
        cv=5
    )
    print(f"    CV F1 Score: {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f'{model_type}.pkl')
    trainer.save(model_path)
    print(f"    Model saved to: {model_path}")

# Find best model
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

best_model = None
best_f1 = 0

for model_type, metrics in results.items():
    print(f"\n{model_type.replace('_', ' ').title()}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        best_model = model_type

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model.replace('_', ' ').title()}")
print(f"   F1 Score: {best_f1:.4f}")
print("=" * 60)

# Save best model as default
print(f"\nSaving best model as default (ml_model.pkl)...")
best_trainer = MLModelTrainer(model_type=best_model)
best_trainer.fit(X_train, y_train)
best_trainer.save(os.path.join(MODEL_DIR, 'ml_model.pkl'))

# Save results summary
results_path = os.path.join(MODEL_DIR, 'training_results.txt')
with open(results_path, 'w') as f:
    f.write("MENTAL STRESS DETECTION - TRAINING RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: {len(df)} samples\n")
    f.write(f"Training: {len(X_train)} samples\n")
    f.write(f"Test: {len(X_test)} samples\n\n")
    f.write("MODEL RESULTS:\n")
    f.write("-" * 50 + "\n")
    for model_type, metrics in results.items():
        f.write(f"\n{model_type}:\n")
        for metric, value in metrics.items():
            if value is not None:
                f.write(f"  {metric}: {value:.4f}\n")
    f.write(f"\nBest Model: {best_model} (F1: {best_f1:.4f})\n")

print(f"\n✅ Training complete! Results saved to: {results_path}")
print(f"\nTo improve further:")
print("  1. Add more data to data/dreaddit-train.csv")
print("  2. Train with BERT using: python train_bert.py")
print("  3. Adjust hyperparameters in train_ml_models.py")
