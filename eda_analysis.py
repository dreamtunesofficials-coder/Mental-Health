"""
Data Cleaning and EDA Script for Mental Stress Detection
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('./data/dreaddit-train.csv')

print('='*60)
print('BASIC DATA OVERVIEW')
print('='*60)
print(f'Shape: {df.shape}')
print(f'Columns: {len(df.columns)}')

print('\n' + '='*60)
print('MISSING VALUES')
print('='*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
print(missing_df[missing_df['Missing'] > 0])

print('\n' + '='*60)
print('LABEL DISTRIBUTION')
print('='*60)
print(df['label'].value_counts())
print(f'\nClass ratio:')
print(df['label'].value_counts(normalize=True))

print('\n' + '='*60)
print('DUPLICATE CHECK')
print('='*60)
print(f'Duplicate rows: {df.duplicated().sum()}')
print(f'Duplicate text entries: {df.duplicated(subset=["text"]).sum()}')

print('\n' + '='*60)
print('TEXT LENGTH ANALYSIS')
print('='*60)
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"Text length - Min: {df['text_length'].min()}, Max: {df['text_length'].max()}")
print(f"Text length - Mean: {df['text_length'].mean():.2f}, Median: {df['text_length'].median()}")
print(f"Word count - Min: {df['word_count'].min()}, Max: {df['word_count'].max()}")
print(f"Word count - Mean: {df['word_count'].mean():.2f}, Median: {df['word_count'].median()}")

# Check for empty/very short texts
short_texts = df[df['text_length'] < 10]
print(f"\nVery short texts (< 10 chars): {len(short_texts)}")

# Check for potential mislabeled data
print('\n' + '='*60)
print('CHECKING FOR MISLABELED DATA')
print('='*60)

# Look at texts labeled as stress but seem non-stressful
stress_texts = df[df['label'] == 1]['text'].head(10)
print("\nSample STRESS texts:")
for i, text in enumerate(stress_texts.head(5)):
    print(f"{i+1}. {text[:200]}...")

non_stress_texts = df[df['label'] == 0]['text'].head(10)
print("\nSample NON-STRESS texts:")
for i, text in enumerate(non_stress_texts.head(5)):
    print(f"{i+1}. {text[:200]}...")

print('\n' + '='*60)
print('SUBREDDIT DISTRIBUTION')
print('='*60)
print(df['subreddit'].value_counts())

print('\n' + '='*60)
print('CONFIDENCE SCORE ANALYSIS')
print('='*60)
print(df['confidence'].describe())

# Check for low confidence labels
low_conf = df[df['confidence'] < 0.6]
print(f"\nLow confidence labels (< 0.6): {len(low_conf)}")

# Save cleaned data
print('\n' + '='*60)
print('SAVING CLEANED DATA')
print('='*60)

# Remove duplicates
df_clean = df.drop_duplicates(subset=['text'], keep='first')
print(f"After removing duplicate texts: {len(df_clean)} (removed {len(df) - len(df_clean)})")

# Remove very short texts
df_clean = df_clean[df_clean['text_length'] >= 10]
print(f"After removing short texts: {len(df_clean)}")

# Save cleaned data
df_clean.to_csv('./data/cleaned_data.csv', index=False)
print("Saved cleaned data to ./data/cleaned_data.csv")

# Now train with cleaned data
print('\n' + '='*60)
print('TRAINING WITH CLEANED DATA')
print('='*60)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Features
X = df_clean['text'].values
y = df_clean['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy with cleaned data: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stress', 'Stress']))
