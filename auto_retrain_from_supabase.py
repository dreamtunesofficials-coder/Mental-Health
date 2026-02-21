"""
🔄 Auto Retrain Model from Supabase Feedback
Fetches data from Supabase cloud, processes feedback, and retrains model
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloud_database import get_cloud_manager, get_database
from data_loader import load_dataset, preprocess_text

from feature_engineering import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class SupabaseFeedbackTrainer:
    """
    Automatically fetches feedback data from Supabase and retrains model
    """
    
    def __init__(self):
        self.cloud_mgr = get_cloud_manager()
        self.local_db = get_database()
        self.training_data_path = './data/feedback_training_data.csv'
        
    def fetch_supabase_data(self) -> pd.DataFrame:
        """
        Fetch all predictions with feedback from Supabase
        """
        try:
            # Get all data from Supabase
            response = self.cloud_mgr.supabase_db.client.table('user_predictions').select('*').execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                print(f"✅ Fetched {len(df)} records from Supabase")
                return df
            else:
                print("⚠️ No data in Supabase")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching from Supabase: {e}")
            return pd.DataFrame()
    
    def process_feedback_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert feedback to correct labels for training
        """
        if df.empty:
            return df
        
        # Filter only records with feedback
        df_with_feedback = df[df['user_feedback'].notna()].copy()
        
        if df_with_feedback.empty:
            print("⚠️ No feedback data found")
            return df_with_feedback
        
        # Create corrected labels based on feedback
        def get_correct_label(row):
            predicted = row['predicted_class']
            feedback = row['user_feedback']
            
            if feedback == 'Yes':
                # Model was correct
                return 1 if predicted == 'Stress' else 0
            elif feedback == 'No':
                # Model was wrong, flip the label
                return 0 if predicted == 'Stress' else 1
            else:
                # Unsure - skip this record
                return None
        
        df_with_feedback['correct_label'] = df_with_feedback.apply(get_correct_label, axis=1)
        
        # Remove records where feedback was 'Unsure'
        df_with_feedback = df_with_feedback[df_with_feedback['correct_label'].notna()]
        
        print(f"✅ Processed {len(df_with_feedback)} feedback records")
        return df_with_feedback
    
    def add_to_training_data(self, df: pd.DataFrame):
        """
        Add processed feedback data to training dataset
        """
        if df.empty:
            return
        
        # Create training data format
        new_training_data = pd.DataFrame({
            'text': df['user_input_text'],
            'label': df['correct_label'].astype(int),
            'source': 'user_feedback',
            'timestamp': df['timestamp'],
            'confidence': df['confidence_score']
        })
        
        # Load existing training data if exists
        if os.path.exists(self.training_data_path):
            existing_data = pd.read_csv(self.training_data_path)
            combined_data = pd.concat([existing_data, new_training_data], ignore_index=True)
            
            # Remove duplicates based on text
            combined_data = combined_data.drop_duplicates(subset=['text'], keep='last')
        else:
            combined_data = new_training_data
        
        # Save updated training data
        combined_data.to_csv(self.training_data_path, index=False)
        print(f"✅ Saved {len(combined_data)} total training records")
        
        return combined_data
    
    def retrain_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Retrain model with feedback data
        """
        if training_data.empty or len(training_data) < 10:
            print("⚠️ Not enough data for retraining (need at least 10 samples)")
            return None
        
        print(f"🔄 Retraining model with {len(training_data)} samples...")
        
        # Prepare features
        texts = training_data['text'].values
        labels = training_data['label'].values
        
        # Extract features
        feature_extractor = FeatureExtractor(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )
        
        X = feature_extractor.fit_transform(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        best_model = None
        best_accuracy = 0
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  📊 {name}: {accuracy:.2%} accuracy")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        print(f"✅ Best model: {best_model_name} ({best_accuracy:.2%} accuracy)")
        
        # Save retrained model
        model_data = {
            'model': best_model,
            'text_extractor': feature_extractor,
            'scaler': None,
            'accuracy': best_accuracy,
            'training_samples': len(training_data),
            'retrained_at': datetime.now().isoformat()
        }
        
        # Save as new model
        model_path = f'./models/retrained_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also update the main model
        with open('./models/best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Updated: ./models/best_model.pkl")
        
        return model_data
    
    def run_auto_retrain(self, min_feedback_count: int = 10):
        """
        Full pipeline: Fetch → Process → Add → Retrain
        """
        print("=" * 60)
        print("🚀 Starting Auto-Retrain from Supabase")
        print("=" * 60)
        
        # Step 1: Fetch from Supabase
        supabase_data = self.fetch_supabase_data()
        
        if supabase_data.empty:
            print("❌ No data available")
            return False
        
        # Step 2: Process feedback
        processed_data = self.process_feedback_labels(supabase_data)
        
        if len(processed_data) < min_feedback_count:
            print(f"⚠️ Need at least {min_feedback_count} feedbacks, got {len(processed_data)}")
            return False
        
        # Step 3: Add to training data
        training_data = self.add_to_training_data(processed_data)
        
        # Step 4: Retrain model
        new_model = self.retrain_model(training_data)
        
        if new_model:
            print("\n" + "=" * 60)
            print("✅ Auto-Retrain Complete!")
            print(f"📊 Model trained on {new_model['training_samples']} samples")
            print(f"🎯 Accuracy: {new_model['accuracy']:.2%}")
            print("=" * 60)
            return True
        
        return False


def schedule_retraining():
    """
    Function to be called periodically (e.g., daily/weekly)
    """
    trainer = SupabaseFeedbackTrainer()
    
    # Check if enough new feedback
    supabase_data = trainer.fetch_supabase_data()
    
    if not supabase_data.empty:
        feedback_count = supabase_data['user_feedback'].notna().sum()
        
        if feedback_count >= 10:  # Minimum 10 feedbacks needed
            print(f"🔄 {feedback_count} feedbacks found. Starting retraining...")
            trainer.run_auto_retrain(min_feedback_count=10)
        else:
            print(f"⏳ Only {feedback_count} feedbacks. Need 10 to retrain.")
    else:
        print("⏳ No data in Supabase yet.")


if __name__ == "__main__":
    # Run auto-retrain immediately
    trainer = SupabaseFeedbackTrainer()
    success = trainer.run_auto_retrain(min_feedback_count=5)  # Lower threshold for testing
    
    if success:
        print("\n🎉 Model successfully retrained with user feedback!")
    else:
        print("\n⚠️ Retraining skipped. Collect more feedback!")
