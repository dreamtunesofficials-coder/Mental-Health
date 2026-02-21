"""
Complete Accuracy Check - All Models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMPLETE ACCURACY CHECK - ALL MODELS")
print("="*70)

# Load data
df = pd.read_csv('./data/dreaddit-train.csv')
print(f"\nOriginal dataset: {len(df)} samples")

# Original data (no cleaning)
X = df['text'].values
y = df['label'].values

# Cleaned data (confidence >= 0.6)
df_clean = df[df['confidence'] >= 0.6].copy()
df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
X_clean = df_clean['text'].values
y_clean = df_clean['label'].values

print(f"Cleaned dataset: {len(X_clean)} samples")

# Split both datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# TF-IDF vectorizers
vec_original = TfidfVectorizer(max_features=3000, ngram_range=(1,1), min_df=2, max_df=0.8)
vec_clean = TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=3, max_df=0.8)

# Fit and transform
X_train_vec = vec_original.fit_transform(X_train)
X_test_vec = vec_original.transform(X_test)

X_train_c_vec = vec_clean.fit_transform(X_train_c)
X_test_c_vec = vec_clean.transform(X_test_c)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*70)
print("MODEL 1: Original Data + TF-IDF (3000 features)")
print("="*70)

models_orig = {
    'LR_C0.5': LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C1.0': LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C2.0': LogisticRegression(C=2.0, max_iter=2000, random_state=42, class_weight='balanced'),
    'RF_100': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
    'GB_100': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
}

best_acc_orig = 0
best_model_orig = None

for name, model in models_orig.items():
    model.fit(X_train_vec, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    test_acc = accuracy_score(y_test, model.predict(X_test_vec))
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')
    
    print(f"{name}: Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%, CV={np.mean(cv_scores)*100:.2f}%")
    
    if test_acc > best_acc_orig:
        best_acc_orig = test_acc
        best_model_orig = model
        best_name_orig = name

print(f"\nBest (Original): {best_name_orig} with {best_acc_orig*100:.2f}%")

print("\n" + "="*70)
print("MODEL 2: Cleaned Data + TF-IDF (2000 features) + Regularization")
print("="*70)

models_clean = {
    'LR_C0.01': LogisticRegression(C=0.01, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C0.05': LogisticRegression(C=0.05, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C0.1': LogisticRegression(C=0.1, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C0.2': LogisticRegression(C=0.2, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C0.5': LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight='balanced'),
}

best_acc_clean = 0
best_model_clean = None

for name, model in models_clean.items():
    model.fit(X_train_c_vec, y_train_c)
    train_acc = accuracy_score(y_train_c, model.predict(X_train_c_vec))
    test_acc = accuracy_score(y_test_c, model.predict(X_test_c_vec))
    cv_scores = cross_val_score(model, X_train_c_vec, y_train_c, cv=cv, scoring='accuracy')
    gap = abs(train_acc - test_acc)
    
    print(f"{name}: Train={train_acc*100:.2f}%, Test={test_acc*100:.2f}%, Gap={gap*100:.2f}%, CV={np.mean(cv_scores)*100:.2f}%")
    
    if test_acc > best_acc_clean:
        best_acc_clean = test_acc
        best_model_clean = model
        best_name_clean = name

print(f"\nBest (Cleaned): {best_name_clean} with {best_acc_clean*100:.2f}%")

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

# Final best model evaluation
print(f"\n--- Original Data Model ---")
y_pred_orig = best_model_orig.predict(X_test_vec)
print(classification_report(y_test, y_pred_orig, target_names=['No Stress', 'Stress']))

print(f"\n--- Cleaned + Regularized Model ---")
y_pred_clean = best_model_clean.predict(X_test_c_vec)
print(classification_report(y_test_c, y_pred_clean, target_names=['No Stress', 'Stress']))

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Best Original Model: {best_name_orig}")
print(f"  - Test Accuracy: {best_acc_orig*100:.2f}%")
print(f"Best Cleaned Model: {best_name_clean}")
print(f"  - Test Accuracy: {best_acc_clean*100:.2f}%")
print(f"  - Gap (Train-Test): {abs(accuracy_score(y_train_c, best_model_clean.predict(X_train_c_vec)) - best_acc_clean)*100:.2f}%")

# Save best model
model_data = {
    'model': best_model_clean,
    'vectorizer': vec_clean,
    'test_acc': best_acc_clean,
    'config': best_name_clean,
    'data_type': 'cleaned'
}

with open('./models/best_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nBest model saved to: ./models/best_model.pkl")
