"""
Model Retraining Module for Mental Stress Detection
Uses collected user data to improve and retrain the model
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Import the database module
from database import StressDetectionDatabase


class ModelRetrainer:
    """
    Handles model retraining using collected user data.
    """
    
    def __init__(self, db_path: str = './data/stress_detection.db'):
        """
        Initialize the retrainer.
        
        Args:
            db_path: Path to the database
        """
        self.db = StressDetectionDatabase(db_path)
        self.min_feedback_samples = 50  # Minimum samples needed for retraining
    
    def get_retraining_data(
        self,
        min_confidence: float = 0.7,
        include_feedback_only: bool = True
    ) -> pd.DataFrame:
        """
        Get data suitable for retraining.
        
        Args:
            min_confidence: Minimum confidence threshold
            include_feedback_only: If True, only use data with user feedback
            
        Returns:
            DataFrame suitable for retraining
        """
        if include_feedback_only:
            # Get only data with user feedback (human-validated)
            df = self.db.get_feedback_data()
            
            if df.empty:
                st.warning("No feedback data available yet. Need more user feedback to retrain.")
                return pd.DataFrame()
            
            # Use "Yes" feedback as high-quality positive samples
            high_quality_df = df[df['user_feedback'] == 'Yes'].copy()
            
            return high_quality_df
            
        else:
            # Get all data with high confidence
            df = self.db.get_all_predictions()
            
            if df.empty:
                return pd.DataFrame()
            
            # Filter by confidence
            high_conf_df = df[df['confidence_score'] >= min_confidence].copy()
            
            return high_conf_df
    
    def prepare_training_data(
        self,
        original_data_path: str = './data/stress_data.csv',
        use_feedback_data: bool = True,
        augmentation_factor: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by combining original and user data.
        
        Args:
            original_data_path: Path to original training data
            use_feedback_data: Whether to include user feedback data
            augmentation_factor: How many times to augment new data
            
        Returns:
            Tuple of (texts, labels)
        """
        all_texts = []
        all_labels = []
        
        # Load original data if exists
        if os.path.exists(original_data_path):
            try:
                orig_df = pd.read_csv(original_data_path)
                if 'text' in orig_df.columns and 'label' in orig_df.columns:
                    all_texts.extend(orig_df['text'].tolist())
                    all_labels.extend(orig_df['label'].tolist())
                    print(f"Loaded {len(orig_df)} samples from original data")
            except Exception as e:
                print(f"Error loading original data: {e}")
        
        # Add user feedback data if available
        if use_feedback_data:
            feedback_df = self.db.get_feedback_data()
            
            if not feedback_df.empty:
                # Use data with positive feedback
                positive_feedback = feedback_df[feedback_df['user_feedback'] == 'Yes']
                
                for _, row in positive_feedback.iterrows():
                    text = row['user_input_text']
                    # Convert predicted class to label
                    label = 1 if row['predicted_class'] == 'Stress' else 0
                    
                    all_texts.append(text)
                    all_labels.append(label)
                    
                    # Data augmentation - add variations
                    for _ in range(augmentation_factor - 1):
                        augmented_text = self._augment_text(text)
                        all_texts.append(augmented_text)
                        all_labels.append(label)
                
                print(f"Added {len(positive_feedback) * augmentation_factor} augmented samples from user feedback")
        
        return np.array(all_texts), np.array(all_labels)
    
    def _augment_text(self, text: str) -> str:
        """
        Simple text augmentation by adding variations.
        
        Args:
            text: Original text
            
        Returns:
            Augmented text
        """
        import random
        
        # Possible augmentations
        augmentations = [
            lambda t: t.lower(),
            lambda t: t.upper(),
            lambda t: t + ".",
            lambda t: t.replace(".", "..."),
            lambda t: "I feel " + t.lower() if not t.lower().startswith("i") else t,
        ]
        
        return random.choice(augmentations)(text)
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistics about collected data.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.db.get_statistics()
        
        # Additional stats
        feedback_df = self.db.get_feedback_data()
        
        if not feedback_df.empty:
            # Feedback breakdown
            feedback_counts = feedback_df['user_feedback'].value_counts()
            
            # High confidence predictions
            high_conf = feedback_df[feedback_df['confidence_score'] >= 0.8]
            
            stats['positive_feedback_count'] = feedback_counts.get('Yes', 0)
            stats['negative_feedback_count'] = feedback_counts.get('No', 0)
            stats['unsure_feedback_count'] = feedback_counts.get('Unsure', 0)
            stats['high_confidence_count'] = len(high_conf)
            stats['ready_for_retraining'] = (
                stats.get('positive_feedback_count', 0) >= self.min_feedback_samples
            )
        else:
            stats['ready_for_retraining'] = False
        
        return stats
    
    def export_for_external_training(self, output_path: str) -> bool:
        """
        Export data in a format suitable for external training scripts.
        
        Args:
            output_path: Path to save the export
            
        Returns:
            True if successful
        """
        try:
            df = self.db.get_all_predictions()
            
            if df.empty:
                return False
            
            # Create export dataframe
            export_df = pd.DataFrame({
                'text': df['user_input_text'],
                'label': df['predicted_class'].apply(lambda x: 1 if x == 'Stress' else 0),
                'confidence': df['confidence_score'],
                'user_feedback': df['user_feedback'],
                'timestamp': df['timestamp']
            })
            
            export_df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False


def show_retraining_dashboard():
    """
    Display a Streamlit dashboard for model retraining.
    """
    st.set_page_config(page_title="Model Retraining", page_icon="refresh")
    
    st.title("Model Retraining Dashboard")
    st.markdown("### Use collected user data to improve your model")
    
    # Initialize retrainer
    retrainer = ModelRetrainer()
    
    # Get statistics
    stats = retrainer.get_data_statistics()
    
    # Display current status
    st.markdown("## Data Collection Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", stats.get('total_predictions', 0))
    
    with col2:
        st.metric("Predictions Today", stats.get('predictions_today', 0))
    
    with col3:
        feedback_count = sum(stats.get('feedback_statistics', {}).values())
        st.metric("Total Feedback", feedback_count)
    
    with col4:
        ready = stats.get('ready_for_retraining', False)
        st.metric("Ready for Retraining", "Yes" if ready else "No")
    
    # Display feedback breakdown
    st.markdown("### Feedback Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"Positive: {stats.get('positive_feedback_count', 0)}")
    
    with col2:
        st.error(f"Negative: {stats.get('negative_feedback_count', 0)}")
    
    with col3:
        st.info(f"Unsure: {stats.get('unsure_feedback_count', 0)}")
    
    # Show data quality
    st.markdown("### Data Quality")
    
    avg_conf = stats.get('average_confidence', 0)
    st.progress(avg_conf)
    st.caption(f"Average Model Confidence: {avg_conf:.2%}")
    
    # Retraining options
    st.markdown("## Retraining Options")
    
    # Option 1: Export data
    st.markdown("### Option 1: Export Data for External Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_path = st.text_input("Export path", "./data/user_predictions_export.csv")
    
    with col2:
        if st.button("Export Data"):
            if retrainer.export_for_external_training(export_path):
                st.success(f"Data exported to {export_path}")
                
                # Provide download button
                with open(export_path, 'rb') as f:
                    st.download_button(
                        "Download Exported Data",
                        f,
                        file_name="retraining_data.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data to export yet.")
    
    # Option 2: Prepare training data
    st.markdown("### Option 2: Prepare Combined Training Data")
    
    use_original = st.checkbox("Include original training data", value=True)
    use_feedback = st.checkbox("Include user feedback data", value=True)
    augment = st.slider("Augmentation factor", 1, 5, 3)
    
    if st.button("Prepare Training Data"):
        with st.spinner("Preparing data..."):
            texts, labels = retrainer.prepare_training_data(
                use_feedback_data=use_feedback
            )
            
            if len(texts) > 0:
                st.success(f"Prepared {len(texts)} training samples!")
                st.write(f"Label distribution: {np.bincount(labels)}")
                
                # Save to file
                combined_df = pd.DataFrame({
                    'text': texts,
                    'label': labels
                })
                combined_path = './data/combined_training_data.csv'
                combined_df.to_csv(combined_path, index=False)
                st.success(f"Saved to {combined_path}")
            else:
                st.warning("Not enough data for training. Need more user feedback.")
    
    # Recommendations
    st.markdown("## Recommendations")
    
    if not stats.get('ready_for_retraining', False):
        needed = retrainer.min_feedback_samples - stats.get('positive_feedback_count', 0)
        st.info(f"Need {needed} more positive feedback samples before retraining.")
        st.markdown("""
        **To get more feedback:**
        1. Encourage users to provide feedback on predictions
        2. Run the app and collect more predictions
        3. Ask users to verify if the prediction was accurate
        """)
    else:
        st.success("You have enough data for retraining!")
        st.markdown("""
        **Next steps:**
        1. Export the combined training data
        2. Use the existing training scripts to retrain the model
        3. Replace the old model with the new one
        4. Deploy the updated app
        """)
    
    # Model improvement tips
    st.markdown("## Model Improvement Tips")
    
    st.markdown("""
    ### 1. Data Quality
    - Focus on getting high-quality user feedback
    - "Yes" feedback indicates the model is correct
    - Use only high-confidence predictions for retraining
    
    ### 2. Data Augmentation
    - Use the augmentation feature to increase training data
    - Add variations of user text (synonyms, paraphrases)
    - Balance the dataset if classes are imbalanced
    
    ### 3. Periodic Retraining
    - Schedule monthly retraining with accumulated data
    - Track model performance over time
    - A/B test new models against old ones
    
    ### 4. Monitoring
    - Monitor prediction distribution over time
    - Alert if model confidence drops significantly
    - Track which types of text have lower accuracy
    """)


def main():
    """
    Main entry point for the retraining dashboard.
    """
    show_retraining_dashboard()


if __name__ == "__main__":
    main()
