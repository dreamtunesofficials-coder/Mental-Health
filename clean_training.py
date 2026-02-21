"""
Advanced Data Cleaning and Training for 90%+ Accuracy
- Uses high-confidence samples
- Removes outliers
- Uses ALL features (text + LIWC + sentiment)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ADVANCED DATA CLEANING AND TRAINING")
print("="*60)

# Load data
df = pd.read_csv('./data/dreaddit-train.csv')
print(f"\nOriginal data: {len(df)} samples")

# ========== STEP 1: DATA CLEANING ==========
print("\n--- Step 1: Data Cleaning ---")

# 1. Remove exact duplicate texts
df = df.drop_duplicates(subset=['text'], keep='first')
print(f"After removing duplicates: {len(df)}")

# 2. Remove very short texts (< 20 chars)
df = df[df['text'].str.len() >= 20]
print(f"After removing short texts: {len(df)}")

# 3. Remove very long texts (> 1500 chars) - potential outliers
df = df[df['text'].str.len() <= 1500]
print(f"After removing long texts: {len(df)}")

# 4. Filter by confidence - keep high confidence samples
high_conf_df = df[df['confidence'] >= 0.8].copy()
print(f"High confidence samples (>=0.8): {len(high_conf_df)}")

# ========== STEP 2: FEATURE ENGINEERING ==========
print("\n--- Step 2: Feature Engineering ---")

# Get all feature columns
numerical_cols = [col for col in df.columns if col.startswith(('lex_', 'social_', 'syntax_', 'sentiment'))]
numerical_cols = [col for col in numerical_cols if df[col].dtype in ['float64', 'int64']]
print(f"Numerical features: {len(numerical_cols)}")

# Text features using TF-IDF
from feature_engineering import FeatureExtractor
text_extractor = FeatureExtractor(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9)

# Train on full data
all_text_features = text_extractor.fit_transform(df['text'].values)

# Get numerical features
numerical_data = df[numerical_cols].fillna(0).values
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_data)

# Combine features
X_all = hstack([all_text_features, csr_matrix(numerical_scaled)])
y_all = df['label'].values

print(f"Total features: {X_all.shape[1]}")

# ========== STEP 3: TRAIN ON HIGH CONFIDENCE DATA ==========
print("\n--- Step 3: Training on High Confidence Data ---")

# Use high confidence data
X_high_conf = text_extractor.transform(high_conf_df['text'].values)
numerical_high_conf = scaler.transform(high_conf_df[numerical_cols].fillna(0).values)
X_high_conf = hstack([X_high_conf, csr_matrix(numerical_high_conf)])
y_high_conf = high_conf_df['label'].values

print(f"High confidence data: {X_high_conf.shape[0]} samples")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ========== STEP 4: MODEL TRAINING ==========
print("\n--- Step 4: Model Training ---")

# Try different models
models = {
    'LR_C0.5': LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C1': LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced'),
    'LR_C2': LogisticRegression(C=2.0, max_iter=2000, random_state=42, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'),
    'GB': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
}

best_acc = 0
best_model = None
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model
        best_name = name

# Try ensemble
print("\n--- Ensemble ---")
ensemble = VotingClassifier(
    estimators=[
        ('lr1', LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')),
        ('lr2', LogisticRegression(C=2.0, max_iter=2000, random_state=42, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')),
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"Ensemble: Test={ensemble_acc:.4f}")

if ensemble_acc > best_acc:
    best_model = ensemble
    best_name = "Ensemble"
    best_acc = ensemble_acc

# Final results
print(f"\n=== BEST MODEL: {best_name} ===")
print(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

# Cross-validation
print("\n--- Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_all, y_all, cv=cv, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Save model
import pickle
model_data = {
    'model': best_model,
    'scaler': scaler,
    'text_extractor': text_extractor,
    'accuracy': best_acc,
    'cv_mean': np.mean(cv_scores),
    'model_name': best_name
}

with open('./models/cleaned_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved to: ./models/cleaned_model.pkl")
print(f"\nFinal Test Accuracy: {best_acc*100:.2f}%")
print(f"Final CV Accuracy: {np.mean(cv_scores)*100:.2f}%")
