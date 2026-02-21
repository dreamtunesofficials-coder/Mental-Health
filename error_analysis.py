"""
Error Analysis - Understand where the model makes mistakes
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ERROR ANALYSIS - Understanding Model Mistakes")
print("="*70)

# Load data
df = pd.read_csv('./data/dreaddit-train.csv')
print(f"\nDataset: {len(df)} samples")

# Clean data (remove low confidence)
df_clean = df[df['confidence'] >= 0.6].copy()
df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
print(f"After cleaning: {len(df_clean)} samples")

X = df_clean['text'].values
y = df_clean['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,1), min_df=3, max_df=0.8)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(C=0.1, max_iter=2000, random_state=42, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Confusion Matrix
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"                 No Stress  Stress")
print(f"Actual No Stress    {cm[0][0]:4d}     {cm[0][1]:4d}")
print(f"Actual Stress      {cm[1][0]:4d}     {cm[1][1]:4d}")

# Calculate errors
true_positives = cm[1][1]
false_positives = cm[0][1]
true_negatives = cm[0][0]
false_negatives = cm[1][0]

print(f"\n--- Error Breakdown ---")
print(f"True Positives (Correct Stress): {true_positives}")
print(f"True Negatives (Correct No Stress): {true_negatives}")
print(f"False Positives (Wrong Stress): {false_positives}")
print(f"False Negatives (Missed Stress): {false_negatives}")

# Get misclassified samples
test_df = pd.DataFrame({
    'text': X_test,
    'actual': y_test,
    'predicted': y_pred
})

# False Positives - Predicted Stress but actually No Stress
fp_df = test_df[(test_df['actual'] == 0) & (test_df['predicted'] == 1)]

# False Negatives - Predicted No Stress but actually Stress
fn_df = test_df[(test_df['actual'] == 1) & (test_df['predicted'] == 0)]

print("\n" + "="*70)
print("FALSE POSITIVES - Predicted STRESS but actually NO STRESS")
print("="*70)
print(f"Total: {len(fp_df)} samples\n")
for i, row in fp_df.head(5).iterrows():
    print(f"Text: {row['text'][:150]}...")
    print("-"*50)

print("\n" + "="*70)
print("FALSE NEGATIVES - Predicted NO STRESS but actually STRESS")
print("="*70)
print(f"Total: {len(fn_df)} samples\n")
for i, row in fn_df.head(5).iterrows():
    print(f"Text: {row['text'][:150]}...")
    print("-"*50)

# Analyze patterns
print("\n" + "="*70)
print("ERROR PATTERNS ANALYSIS")
print("="*70)

# Text length analysis
fp_df['text_len'] = fp_df['text'].str.len()
fn_df['text_len'] = fn_df['text'].str.len()

print(f"\n--- Text Length Analysis ---")
print(f"False Positives avg length: {fp_df['text_len'].mean():.0f} chars")
print(f"False Negatives avg length: {fn_df['text_len'].mean():.0f} chars")

# Check for specific keywords
stress_keywords = ['anxious', 'stress', 'worry', 'fear', 'panic', 'overwhelmed', 'nervous', 'depressed']
no_stress_keywords = ['happy', 'good', 'great', 'wonderful', 'relaxed', 'calm', 'enjoy']

def count_keywords(text, keywords):
    return sum(1 for kw in keywords if kw.lower() in text.lower())

fp_df['stress_kw'] = fp_df['text'].apply(lambda x: count_keywords(x, stress_keywords))
fp_df['no_stress_kw'] = fp_df['text'].apply(lambda x: count_keywords(x, no_stress_keywords))
fn_df['stress_kw'] = fn_df['text'].apply(lambda x: count_keywords(x, stress_keywords))
fn_df['no_stress_kw'] = fn_df['text'].apply(lambda x: count_keywords(x, no_stress_keywords))

print(f"\n--- Keyword Analysis ---")
print(f"False Positives: avg stress keywords = {fp_df['stress_kw'].mean():.2f}")
print(f"False Positives: avg no-stress keywords = {fp_df['no_stress_kw'].mean():.2f}")
print(f"False Negatives: avg stress keywords = {fn_df['stress_kw'].mean():.2f}")
print(f"False Negatives: avg no-stress keywords = {fn_df['no_stress_kw'].mean():.2f}")

# Subreddit analysis
test_df_full = test_df.copy()
test_df_full['text_len'] = test_df_full['text'].str.len()

# Merge with original to get subreddit
df_test_indices = df_clean.iloc[X_test].index
test_df_full['subreddit'] = df_clean.iloc[X_test]['subreddit'].values

print(f"\n--- Subreddit Analysis ---")
print("Error rate by subreddit:")
for sub in test_df_full['subreddit'].unique():
    sub_df = test_df_full[test_df_full['subreddit'] == sub]
    error_rate = (sub_df['actual'] != sub_df['predicted']).mean()
    print(f"  {sub}: {error_rate*100:.1f}% error rate")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS TO IMPROVE ACCURACY")
print("="*70)
print("""
1. SUBREDDIT-SPECIFIC FEATURES:
   - Add subreddit as a feature (different subreddits have different styles)
   - Train separate models for high-error subreddits

2. CONTEXT-AWARE FEATURES:
   - Handle sarcasm and irony (common in Reddit)
   - Detect subtle stress signals

3. IMPROVED FEATURE ENGINEERING:
   - Add LIWC psychological features
   - Add sentiment intensity features
   - Use character-level n-grams

4. ENSEMBLE METHODS:
   - Combine multiple models
   - Use BERT for better context understanding
""")

print("\n" + "="*70)
print("MODEL ACCURACY SUMMARY")
print("="*70)
accuracy = (y_pred == y_test).mean()
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Correct predictions: {(y_pred == y_test).sum()}")
print(f"Wrong predictions: {(y_pred != y_test).sum()}")
