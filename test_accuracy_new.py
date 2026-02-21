"""
Accuracy Test Script
Shows training and test accuracy
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load model
with open('./models/ml_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

print('=== MODEL INFORMATION ===')
print(f'Model: {type(model_data.get("model"))}')
print(f'Vectorizer: {type(model_data.get("vectorizer"))}')

# Load data
df = pd.read_csv('./data/dreaddit-train.csv')
X = df['text'].values
y = df['label'].values

# Get the vectorizer
vectorizer = model_data.get('vectorizer')

# Transform data
X_vec = vectorizer.transform(X)

# Get model
model = model_data.get('model')

# Predict on full data
y_pred = model.predict(X_vec)
train_acc = accuracy_score(y, y_pred)

print(f'\n=== TRAINING ACCURACY (Full Data) ===')
print(f'Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)')

# Classification report
print(f'\n=== CLASSIFICATION REPORT (Full Data) ===')
print(classification_report(y, y_pred, target_names=['No Stress', 'Stress']))

# Now split and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit new vectorizer on train
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Train model
m = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
m.fit(X_train_vec, y_train)

# Train and test accuracy
train_acc_new = accuracy_score(y_train, m.predict(X_train_vec))
test_acc_new = accuracy_score(y_test, m.predict(X_test_vec))

print(f'\n=== NEW TRAIN/TEST SPLIT ===')
print(f'Train Accuracy: {train_acc_new:.4f} ({train_acc_new*100:.2f}%)')
print(f'Test Accuracy: {test_acc_new:.4f} ({test_acc_new*100:.2f}%)')
print(f'\nClassification Report (Test):')
print(classification_report(y_test, m.predict(X_test_vec), target_names=['No Stress', 'Stress']))
