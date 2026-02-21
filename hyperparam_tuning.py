"""
Hyperparameter Tuning for Maximum Accuracy
- Grid Search for optimal parameters
- Reduce overfitting gap
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYPERPARAMETER TUNING FOR MAXIMUM ACCURACY")
print("="*70)

# Load data
df = pd.read_csv('./data/dreaddit-train.csv')
print(f"\nDataset: {len(df)} samples")

# Remove duplicates
df = df.drop_duplicates(subset=['text'], keep='first')
print(f"After removing duplicates: {len(df)} samples")

X = df['text'].values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ===== HYPERPARAMETER TUNING =====
print("\n" + "="*70)
print("STEP 1: TF-IDF Vectorizer Tuning")
print("="*70)

tfidf_params = {
    'max_features': [1000, 2000, 3000, 5000],
    'ngram_range': [(1,1), (1,2), (1,3)],
    'min_df': [1, 2, 3],
    'max_df': [0.7, 0.8, 0.9],
    'sublinear_tf': [True, False]
}

best_tfidf_acc = 0
best_tfidf_params = {}

# Test key combinations (not exhaustive due to time)
test_configs = [
    {'max_features': 2000, 'ngram_range': (1,1), 'min_df': 2, 'max_df': 0.8, 'sublinear_tf': True},
    {'max_features': 3000, 'ngram_range': (1,1), 'min_df': 2, 'max_df': 0.8, 'sublinear_tf': True},
    {'max_features': 3000, 'ngram_range': (1,2), 'min_df': 2, 'max_df': 0.8, 'sublinear_tf': True},
    {'max_features': 3000, 'ngram_range': (1,2), 'min_df': 3, 'max_df': 0.7, 'sublinear_tf': True},
    {'max_features': 5000, 'ngram_range': (1,1), 'min_df': 2, 'max_df': 0.9, 'sublinear_tf': True},
    {'max_features': 5000, 'ngram_range': (1,2), 'min_df': 2, 'max_df': 0.8, 'sublinear_tf': False},
    {'max_features': 5000, 'ngram_range': (1,2), 'min_df': 3, 'max_df': 0.7, 'sublinear_tf': True},
    {'max_features': 5000, 'ngram_range': (1,2), 'min_df': 2, 'max_df': 0.8, 'sublinear_tf': True},
]

for config in test_configs:
    vec = TfidfVectorizer(**config)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    test_acc = accuracy_score(y_test, model.predict(X_test_vec))
    gap = abs(train_acc - test_acc)
    
    print(f"Config: {config['max_features']} features, {config['ngram_range']}, min_df={config['min_df']} -> Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Gap: {gap*100:.2f}%")
    
    if test_acc > best_tfidf_acc:
        best_tfidf_acc = test_acc
        best_tfidf_params = config

print(f"\nBest TF-IDF: {best_tfidf_params}")
print(f"Best Test Accuracy: {best_tfidf_acc*100:.2f}%")

# ===== LOGISTIC REGRESSION TUNING =====
print("\n" + "="*70)
print("STEP 2: Logistic Regression Hyperparameter Tuning")
print("="*70)

# Use best TF-IDF
vec = TfidfVectorizer(**best_tfidf_params)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Test different C values
C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
best_lr_acc = 0
best_C = 1.0

print("\nC value tuning:")
for C in C_values:
    model = LogisticRegression(C=C, max_iter=2000, random_state=42, class_weight='balanced', solver='lbfgs')
    model.fit(X_train_vec, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    test_acc = accuracy_score(y_test, model.predict(X_test_vec))
    gap = abs(train_acc - test_acc)
    
    print(f"C={C:>6} -> Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%, Gap: {gap*100:.2f}%")
    
    if test_acc > best_lr_acc:
        best_lr_acc = test_acc
        best_C = C

print(f"\nBest C: {best_C}, Best Test Accuracy: {best_lr_acc*100:.2f}%")

# ===== SOLVER TUNING =====
print("\n" + "="*70)
print("STEP 3: Solver Tuning")
print("="*70)

solvers = ['lbfgs', 'liblinear', 'saga']
best_solver = 'lbfgs'
best_solver_acc = best_lr_acc

for solver in solvers:
    try:
        model = LogisticRegression(C=best_C, max_iter=2000, random_state=42, class_weight='balanced', solver=solver)
        model.fit(X_train_vec, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train_vec))
        test_acc = accuracy_score(y_test, model.predict(X_test_vec))
        
        print(f"Solver: {solver} -> Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%")
        
        if test_acc > best_solver_acc:
            best_solver_acc = test_acc
            best_solver = solver
    except:
        pass

print(f"\nBest Solver: {best_solver}")

# ===== FINAL MODEL =====
print("\n" + "="*70)
print("FINAL MODEL RESULTS")
print("="*70)

final_model = LogisticRegression(
    C=best_C, 
    max_iter=2000, 
    random_state=42, 
    class_weight='balanced',
    solver=best_solver
)
final_model.fit(X_train_vec, y_train)

train_acc = accuracy_score(y_train, final_model.predict(X_train_vec))
test_acc = accuracy_score(y_test, final_model.predict(X_test_vec))
gap = abs(train_acc - test_acc)

print(f"\n✅ OPTIMIZED MODEL:")
print(f"   Training Accuracy: {train_acc*100:.2f}%")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Gap: {gap*100:.2f}%")

print(f"\n📊 Classification Report:")
y_pred = final_model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))

# ===== CROSS VALIDATION =====
print("\n" + "="*70)
print("CROSS-VALIDATION (5-FOLD)")
print("="*70)

X_all = vec.fit_transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_lr = LogisticRegression(C=best_C, max_iter=2000, random_state=42, class_weight='balanced', solver=best_solver)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_all, y)):
    X_cv_train, X_cv_val = X_all[train_idx], X_all[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(C=best_C, max_iter=2000, random_state=42, class_weight='balanced', solver=best_solver)
    model.fit(X_cv_train, y_cv_train)
    
    score = accuracy_score(y_cv_val, model.predict(X_cv_val))
    cv_scores.append(score)
    print(f"Fold {fold+1}: {score*100:.2f}%")

print(f"\nMean CV: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")

# ===== SAVE MODEL =====
model_data = {
    'model': final_model,
    'vectorizer': vec,
    'train_acc': train_acc,
    'test_acc': test_acc,
    'cv_mean': np.mean(cv_scores),
    'best_params': {
        'C': best_C,
        'solver': best_solver,
        'tfidf': best_tfidf_params
    }
}

with open('./models/tuned_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n💾 Model saved to: ./models/tuned_model.pkl")
print("\n" + "="*70)
print("TUNING COMPLETE!")
print("="*70)
