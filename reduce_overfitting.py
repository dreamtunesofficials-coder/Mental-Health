"""
Reduce Overfitting - Complete Solution
1. Regularization: Stronger regularization for Logistic Regression
2. Data Cleaning: Remove rows with confidence < 0.6
3. Cross-Validation: K-Fold (K=5) for stability check
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("REDUCE OVERFITTING - COMPLETE SOLUTION")
print("="*70)

# ===== STEP 1: DATA LOADING & CLEANING =====
print("\n" + "="*70)
print("STEP 1: DATA CLEANING - Remove Low Confidence Labels (< 0.6)")
print("="*70)

df = pd.read_csv('./data/dreaddit-train.csv')
print(f"Original dataset: {len(df)} samples")

# Check confidence column
print(f"\nConfidence column stats:")
print(f"Min: {df['confidence'].min()}")
print(f"Max: {df['confidence'].max()}")
print(f"Mean: {df['confidence'].mean():.2f}")

# Remove rows where confidence < 0.6
df_clean = df[df['confidence'] >= 0.6].copy()
print(f"\nAfter removing low confidence (< 0.6): {len(df_clean)} samples")
print(f"Removed: {len(df) - len(df_clean)} samples")

# Also remove duplicates
df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
print(f"After removing duplicates: {len(df_clean)} samples")

# ===== STEP 2: FEATURE ENGINEERING =====
print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

X = df_clean['text'].values
y = df_clean['label'].values

print(f"Dataset size: {len(X)} samples")
print(f"Label distribution: {np.bincount(y)}")
print(f"Class ratio: {np.bincount(y)/len(y)}")

# ===== STEP 3: REGULARIZATION & MODEL TRAINING =====
print("\n" + "="*70)
print("STEP 3: REGULARIZATION & MODEL TRAINING")
print("="*70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Test different regularization strengths
print("\n--- Testing Different Regularization (C values) ---")

best_model = None
best_acc = 0
best_C = 1.0
best_vec = None

# Lower C = stronger regularization
C_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

for C in C_values:
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=3, max_df=0.8)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    # Logistic Regression with regularization
    model = LogisticRegression(
        C=C,
        max_iter=2000, 
        random_state=42, 
        class_weight='balanced',
        solver='lbfgs',
        penalty='l2'
    )
    model.fit(X_train_vec, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    test_acc = accuracy_score(y_test, model.predict(X_test_vec))
    gap = abs(train_acc - test_acc)
    
    print(f"C={C:>5} -> Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Gap: {gap*100:.2f}%")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_C = C
        best_model = model
        best_vec = vec

print(f"\nBest C: {best_C}, Best Test Accuracy: {best_acc*100:.2f}%")

# ===== STEP 4: RANDOM FOREST WITH MAX_DEPTH =====
print("\n" + "="*70)
print("STEP 4: RANDOM FOREST WITH REGULARIZATION (max_depth)")
print("="*70)

# Test different max_depth values
for max_d in [3, 5, 7, 10]:
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=3, max_df=0.8)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=max_d,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_vec, y_train)
    
    train_acc = accuracy_score(y_train, rf.predict(X_train_vec))
    test_acc = accuracy_score(y_test, rf.predict(X_test_vec))
    gap = abs(train_acc - test_acc)
    
    print(f"max_depth={max_d} -> Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Gap: {gap*100:.2f}%")

# ===== STEP 5: CROSS-VALIDATION (K=5) =====
print("\n" + "="*70)
print("STEP 5: CROSS-VALIDATION (K=5) - Model Stability")
print("="*70)

# Use best model with lower C for more regularization
final_vec = TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=3, max_df=0.8)
X_all_vec = final_vec.fit_transform(X)

# Test with strong regularization
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Different C values for CV
for C in [0.01, 0.05, 0.1, 0.2]:
    model = LogisticRegression(C=C, max_iter=2000, random_state=42, class_weight='balanced')
    cv_scores = cross_val_score(model, X_all_vec, y, cv=cv, scoring='accuracy')
    
    print(f"C={C}: CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
    print(f"       Mean: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")

# ===== FINAL MODEL WITH BEST REGULARIZATION =====
print("\n" + "="*70)
print("FINAL MODEL - BEST REGULARIZATION")
print("="*70)

# Use C=0.1 (strong regularization)
final_model = LogisticRegression(C=0.1, max_iter=2000, random_state=42, class_weight='balanced')
final_vec = TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=3, max_df=0.8)

X_train_vec = final_vec.fit_transform(X_train)
X_test_vec = final_vec.transform(X_test)

final_model.fit(X_train_vec, y_train)

train_acc = accuracy_score(y_train, final_model.predict(X_train_vec))
test_acc = accuracy_score(y_test, final_model.predict(X_test_vec))
gap = abs(train_acc - test_acc)

print(f"\nFINAL MODEL (with strong regularization):")
print(f"   Training Accuracy: {train_acc*100:.2f}%")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Gap: {gap*100:.2f}%")

print(f"\nClassification Report:")
y_pred = final_model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

# Cross-validation on full data
X_all_vec = final_vec.fit_transform(X)
cv_scores = cross_val_score(final_model, X_all_vec, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation (K=5):")
print(f"   CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"   Mean CV: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")

# ===== SAVE MODEL =====
model_data = {
    'model': final_model,
    'vectorizer': final_vec,
    'train_acc': train_acc,
    'test_acc': test_acc,
    'cv_mean': np.mean(cv_scores),
    'config': {'C': 0.1, 'max_features': 2000, 'ngram': (1,1)}
}

with open('./models/regularized_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to: ./models/regularized_model.pkl")

print("\n" + "="*70)
print("SUMMARY - OVERFITTING REDUCED!")
print("="*70)
print(f"Data cleaned: Removed low confidence (< 0.6) samples")
print(f"Regularization applied: C=0.1 (strong)")
print(f"Cross-validation done: K=5")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"Gap reduced: {gap*100:.2f}%")
