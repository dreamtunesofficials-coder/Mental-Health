"""
Retraining Script for Mental Stress Detection Model
Uses collected user data with feedback to improve model performance
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional
import sqlite3
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, preprocess_text
from feature_engineering import FeatureExtractor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')


class ModelRetrainer:
    """
    Handles model retraining with original and collected data.
    """
    
    def __init__(self, db_path: str = './data/stress_detection.db'):
        """
        Initialize the retrainer.
        
        Args:
            db_path: Path to SQLite database with collected predictions
        """
        self.db_path = db_path
        self.models_dir = './models'
        self.data_dir = './data'
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.original_data = None
        self.collected_data = None
        self.combined_data = None
        self.best_model = None
        self.training_results = {}
        
    def load_original_data(self, filepath: str = './data/stress_data.csv') -> pd.DataFrame:
        """
        Load original training data.
        
        Args:
            filepath: Path to original training data
            
        Returns:
            DataFrame with original data
        """
        print(f"Loading original data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Original data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            # Try to identify text and label columns
            text_cols = [col for col in df.columns if 'text' in col.lower()]
            label_cols = [col for col in df.columns if 'label' in col.lower() or 'stress' in col.lower()]
            
            if text_cols and label_cols:
                df = df.rename(columns={
                    text_cols[0]: 'text',
                    label_cols[0]: 'label'
                })
        
        self.original_data = df
        print(f"Loaded {len(df)} original training samples")
        
        return df
    
    def load_collected_data(self, min_confidence: float = 0.7, 
                           require_feedback: bool = True) -> pd.DataFrame:
        """
        Load collected data from database with filtering.
        
        Args:
            min_confidence: Minimum confidence score to include
            require_feedback: Only include data with user feedback
            
        Returns:
            DataFrame with filtered collected data
        """
        print(f"Loading collected data from {self.db_path}...")
        
        if not os.path.exists(self.db_path):
            print("Database not found. Skipping collected data.")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on filters
            query = """
                SELECT user_input_text as text, 
                       predicted_class as label,
                       confidence_score,
                       user_feedback,
                       timestamp
                FROM user_predictions 
                WHERE confidence_score >= ?
            """
            
            params = [min_confidence]
            
            if require_feedback:
                query += " AND user_feedback IS NOT NULL"
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                print("No collected data matching criteria")
                return df
            
            # Convert predicted_class to binary label
            df['label'] = df['label'].map({
                'Stress': 1,
                'No Stress': 0,
                'High Stress': 1,
                'Low Stress': 0
            })
            
            # Remove rows where label couldn't be mapped
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            
            self.collected_data = df
            print(f"Loaded {len(df)} collected samples (confidence >= {min_confidence})")
            
            # Print feedback distribution
            if 'user_feedback' in df.columns:
                feedback_dist = df['user_feedback'].value_counts()
                print(f"Feedback distribution:\n{feedback_dist}")
            
            return df
            
        except Exception as e:
            print(f"Error loading collected data: {e}")
            return pd.DataFrame()
    
    def combine_datasets(self, 
                        use_only_positive_feedback: bool = True,
                        max_collected_ratio: float = 0.3) -> pd.DataFrame:
        """
        Combine original and collected data.
        
        Args:
            use_only_positive_feedback: Only use data with "Yes" feedback
            max_collected_ratio: Maximum ratio of collected to original data
            
        Returns:
            Combined DataFrame
        """
        if self.original_data is None:
            raise ValueError("Original data not loaded. Call load_original_data() first.")
        
        # Start with original data
        combined = self.original_data.copy()
        
        if self.collected_data is not None and not self.collected_data.empty:
            collected = self.collected_data.copy()
            
            # Filter for positive feedback if required
            if use_only_positive_feedback and 'user_feedback' in collected.columns:
                collected = collected[collected['user_feedback'] == 'Yes']
                print(f"Using {len(collected)} samples with positive feedback")
            
            # Limit collected data ratio
            max_collected = int(len(combined) * max_collected_ratio)
            if len(collected) > max_collected:
                collected = collected.sample(n=max_collected, random_state=42)
                print(f"Sampled collected data to {max_collected} samples")
            
            # Keep only necessary columns
            collected = collected[['text', 'label']]
            
            # Combine datasets
            combined = pd.concat([combined, collected], ignore_index=True)
            
            # Remove duplicates
            combined = combined.drop_duplicates(subset=['text'], keep='first')
            
            print(f"Combined dataset: {len(combined)} total samples")
            print(f"  - Original: {len(self.original_data)}")
            print(f"  - Collected: {len(collected)}")
        else:
            print("No collected data to combine")
        
        self.combined_data = combined
        return combined
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare features for training.
        
        Args:
            df: DataFrame with text and label columns
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, vectorizer, scaler)
        """
        print("Preparing features...")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Create dummy numerical features (for compatibility)
        scaler = StandardScaler()
        dummy_numerical = np.zeros((X_train_tfidf.shape[0], 3))
        scaler.fit(dummy_numerical)
        
        print(f"Training samples: {X_train_tfidf.shape[0]}")
        print(f"Test samples: {X_test_tfidf.shape[0]}")
        print(f"Features: {X_train_tfidf.shape[1]}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, scaler
    
    def train_models(self, X_train, X_test, y_train, y_test, 
                    vectorizer, scaler) -> Dict:
        """
        Train multiple models and select the best one.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            vectorizer: Fitted TF-IDF vectorizer
            scaler: Fitted scaler
            
        Returns:
            Dictionary with training results
        """
        print("\nTraining models...")
        
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'NaiveBayes': MultinomialNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Select best model based on CV score
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = {
            'model': results[best_model_name]['model'],
            'vectorizer': vectorizer,
            'scaler': scaler,
            'name': best_model_name,
            'accuracy': results[best_model_name]['accuracy'],
            'cv_score': results[best_model_name]['cv_mean']
        }
        
        print(f"\nBest model: {best_model_name}")
        print(f"  Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"  CV Score: {results[best_model_name]['cv_mean']:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, results[best_model_name]['predictions'],
                                  target_names=['No Stress', 'Stress']))
        
        self.training_results = results
        return results
    
    def save_model(self, filename: str = 'retrained_model.pkl') -> str:
        """
        Save the best model.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to saved model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train models first.")
        
        filepath = os.path.join(self.models_dir, filename)
        
        model_data = {
            'model': self.best_model['model'],
            'vectorizer': self.best_model['vectorizer'],
            'scaler': self.best_model['scaler'],
            'model_name': self.best_model['name'],
            'accuracy': self.best_model['accuracy'],
            'cv_score': self.best_model['cv_score'],
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.combined_data) if self.combined_data is not None else 0
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
        return filepath
    
    def generate_report(self, output_file: str = './data/retraining_report.txt'):
        """
        Generate a training report.
        
        Args:
            output_file: Path to save the report
        """
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL RETRAINING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            if self.original_data is not None:
                f.write(f"Original samples: {len(self.original_data)}\n")
            if self.collected_data is not None:
                f.write(f"Collected samples: {len(self.collected_data)}\n")
            if self.combined_data is not None:
                f.write(f"Combined samples: {len(self.combined_data)}\n")
            f.write("\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for name, result in self.training_results.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Test Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})\n")
            
            if self.best_model:
                f.write(f"\nBEST MODEL: {self.best_model['name']}\n")
                f.write(f"  Test Accuracy: {self.best_model['accuracy']:.4f}\n")
                f.write(f"  CV Score: {self.best_model['cv_score']:.4f}\n")
        
        print(f"Report saved to: {output_file}")
    
    def run_full_retraining(self, 
                          original_data_path: str = './data/stress_data.csv',
                          min_confidence: float = 0.7,
                          use_only_positive_feedback: bool = True) -> str:
        """
        Run complete retraining pipeline.
        
        Args:
            original_data_path: Path to original training data
            min_confidence: Minimum confidence for collected data
            use_only_positive_feedback: Only use positive feedback
            
        Returns:
            Path to saved model
        """
        print("=" * 60)
        print("STARTING MODEL RETRAINING")
        print("=" * 60)
        
        # Load data
        self.load_original_data(original_data_path)
        self.load_collected_data(min_confidence=min_confidence)
        
        # Combine datasets
        self.combine_datasets(use_only_positive_feedback=use_only_positive_feedback)
        
        # Prepare features
        X_train, X_test, y_train, y_test, vectorizer, scaler = self.prepare_features(
            self.combined_data
        )
        
        # Train models
        self.train_models(X_train, X_test, y_train, y_test, vectorizer, scaler)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.save_model(f'retrained_model_{timestamp}.pkl')
        
        # Also save as best_model.pkl if performance is good
        if self.best_model and self.best_model['cv_score'] > 0.75:
            self.save_model('best_model.pkl')
            print("\nModel also saved as best_model.pkl (high performance)")
        
        # Generate report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("RETRAINING COMPLETE")
        print("=" * 60)
        
        return model_path


def main():
    """
    Main function to run retraining.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain stress detection model')
    parser.add_argument('--db-path', default='./data/stress_detection.db',
                       help='Path to database')
    parser.add_argument('--data-path', default='./data/stress_data.csv',
                       help='Path to original training data')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                       help='Minimum confidence for collected data')
    parser.add_argument('--use-feedback', action='store_true',
                       help='Only use data with positive feedback')
    
    args = parser.parse_args()
    
    retrainer = ModelRetrainer(db_path=args.db_path)
    
    model_path = retrainer.run_full_retraining(
        original_data_path=args.data_path,
        min_confidence=args.min_confidence,
        use_only_positive_feedback=args.use_feedback
    )
    
    print(f"\nRetrained model saved to: {model_path}")


if __name__ == "__main__":
    main()
