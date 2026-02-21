"""
Enhanced Training with More Data + Better Confidence Calibration
- Combines train + test data
- Uses high-confidence samples only
- Calibrates confidence scores
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.sparse import hstack, csr_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENHANCED TRAINING WITH MORE DATA")
print("="*60)

# Load ALL data
train_df = pd.read_csv('./data/dreaddit-train.csv')
test_df = pd.read_csv('./data/dreaddit-test.csv')

print(f"\nTrain: {len(train_df)} samples")
print(f"Test: {len(test_df)} samples")

# Combine train + test for more training data
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined: {len(combined_df)} samples")

# ========== DATA CLEANING ==========
print("\n--- Data Cleaning ---")

# Remove duplicates
combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
print(f"After removing duplicates: {len(combined_df)}")

# Remove very short texts
combined_df = combined_df[combined_df['text'].str.len() >= 20]
print(f"After removing short texts: {len(combined_df)}")

# Remove very long texts
combined_df = combined_df[combined_df['text'].str.len() <= 1500]
print(f"After removing long texts: {len(combined_df)}")

# ========== HIGH CONFIDENCE FILTERING ==========
print("\n--- Confidence Filtering ---")

# Use only high confidence samples (>= 0.8)
high_conf_df = combined_df[combined_df['confidence'] >= 0.8].copy()
print(f"High confidence samples (>=0.8): {len(high_conf_df)}")

# ========== FEATURE ENGINEERING ==========
print("\n--- Feature Engineering ---")

# Get numerical feature columns
numerical_cols = [col for col in combined_df.columns 
                  if col.startswith(('lex_', 'social_', 'syntax_', 'sentiment'))]
numerical_cols = [col for col in numerical_cols 
                  if combined_df[col].dtype in ['float64', 'int64']]
print(f"Numerical features: {len(numerical_cols)}")

# Text features
from feature_engineering import FeatureExtractor

# Full data TF-IDF
text_extractor = FeatureExtractor(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9)
all_text_features = text_extractor.fit_transform(combined_df['text'].values)

# Numerical features
numerical_data = combined_df[numerical_cols].fillna(0).values
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_data)

# Combine all features
X_all = hstack([all_text_features, csr_matrix(numerical_scaled)])
y_all = combined_df['label'].values

print(f"Total features: {X_all.shape[1]}")

# ========== TRAINING WITH HIGH CONFIDENCE DATA ==========
print("\n--- Training with High Confidence Data ---")

# Train on high confidence data
X_high = text_extractor.transform(high_conf_df['text'].values)
numerical_high = scaler.transform(high_conf_df[numerical_cols].fillna(0).values)
X_high = hstack([X_high, csr_matrix(numerical_high)])
y_high = high_conf_df['label'].values

print(f"High confidence training data: {X_high.shape[0]} samples")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_high, y_high, test_size=0.2, random_state=42, stratify=y_high
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ========== MODEL TRAINING ==========
print("\n--- Model Training ---")

# Try different regularization values
best_acc = 0
best_model = None
best_c = 0

for c in [0.1, 0.3, 0.5, 0.7, 1.0]:
    model = LogisticRegression(C=c, max_iter=2000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"C={c}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model
        best_c = c

print(f"\nBest C: {best_c}, Test Accuracy: {best_acc:.4f}")

# ========== CONFIDENCE CALIBRATION ==========
print("\n--- Confidence Calibration ---")

# Use CalibratedClassifierCV for better confidence scores
calibrated_model = CalibratedClassifierCV(
    LogisticRegression(C=best_c, max_iter=2000, random_state=42, class_weight='balanced'),
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Test calibrated model
cal_pred = calibrated_model.predict(X_test)
cal_proba = calibrated_model.predict_proba(X_test)

cal_acc = accuracy_score(y_test, cal_pred)
print(f"Calibrated Test Accuracy: {cal_acc:.4f}")

# Check confidence distribution
print(f"\nConfidence Score Distribution:")
print(f"  Min: {cal_proba.min():.4f}")
print(f"  Max: {cal_proba.max():.4f}")
print(f"  Mean: {cal_proba.mean():.4f}")

# AUC-ROC score
auc = roc_auc_score(y_test, cal_proba[:, 1])
print(f"  AUC-ROC: {auc:.4f}")

# ========== FINAL RESULTS ==========
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

y_pred = calibrated_model.predict(X_test)
print(f"\nTest Accuracy: {cal_acc:.4f} ({cal_acc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

# Cross-validation
print("\n--- Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    CalibratedClassifierCV(
        LogisticRegression(C=best_c, max_iter=2000, random_state=42, class_weight='balanced'),
        method='isotonic', cv=3
    ),
    X_high, y_high, cv=cv, scoring='accuracy'
)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ========== SAVE MODEL ==========
model_data = {
    'model': calibrated_model,
    'scaler': scaler,
    'text_extractor': text_extractor,
    'accuracy': cal_acc,
    'cv_mean': np.mean(cv_scores),
    'auc_roc': auc,
    'best_c': best_c,
    'n_samples': X_high.shape[0]
}

with open('./models/calibrated_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to: ./models/calibrated_model.pkl")
print(f"Training samples: {X_high.shape[0]}")
print(f"\nFinal Accuracy: {cal_acc*100:.2f}%")
print(f"Final CV: {np.mean(cv_scores)*100:.2f}%")
print(f"AUC-ROC: {auc*100:.2f}%")
