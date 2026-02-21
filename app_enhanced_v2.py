"""
Enhanced Streamlit Web Application for Mental Stress Detection
Interactive web interface with database integration for stress prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import json
from typing import Dict, List, Optional
import sys
from datetime import datetime
import uuid

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess_text
from feature_engineering import FeatureExtractor
from cloud_database import StressDetectionDatabase, get_database, get_cloud_manager


# Page configuration
st.set_page_config(
    page_title="Mental Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stress-high {
        background-color: #ffcccc;
        border-left: 5px solid #ff4444;
    }
    .stress-low {
        background-color: #ccffcc;
        border-left: 5px solid #44ff44;
    }
    .stress-medium {
        background-color: #ffffcc;
        border-left: 5px solid #ffaa44;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .feedback-btn {
        margin: 0.5rem;
    }
    .user-info-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_model_path(model_name: str) -> str:
    """Get the correct path for a model file."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    return os.path.join(models_dir, model_name)


def load_ml_model(model_name: str = 'best_model.pkl'):
    """
    Load the ML model from the models directory.
    
    Args:
        model_name: Name of the model file to load
        
    Returns:
        Loaded model or None
    """
    model_path = get_model_path(model_name)
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                return model_data
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # Try alternative models if best_model.pkl doesn't exist
    alternative_models = [
        'proper_model.pkl',
        'optimized_model.pkl',
        'advanced_model.pkl',
        'cleaned_model.pkl',
        'ml_model.pkl'
    ]
    
    for alt_model in alternative_models:
        alt_path = get_model_path(alt_model)
        if os.path.exists(alt_path):
            try:
                with open(alt_path, 'rb') as f:
                    model_data = pickle.load(f)
                    st.info(f"Loaded alternative model: {alt_model}")
                    return model_data
            except Exception as e:
                continue
    
    st.warning("ML model not found. Will use demo mode.")
    return None


def load_bert_model(model_path: str = 'bert_model'):
    """
    Load BERT model.
    
    Args:
        model_path: Path to BERT model
        
    Returns:
        Loaded model or None
    """
    try:
        from train_bert import StressBERTClassifier
        full_path = get_model_path(model_path)
        if os.path.exists(full_path):
            classifier = StressBERTClassifier()
            classifier.model = classifier.model
            return classifier
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
    return None


def predict_stress(
    text: str,
    model_data: dict,
    model_type: str = 'ML'
) -> Dict:
    """
    Predict stress level from text.
    
    Args:
        text: Input text
        model_data: Dictionary with model, vectorizer/text_extractor, scaler
        model_type: Type of model ('ML' or 'BERT')
        
    Returns:
        Dictionary with prediction results
    """
    processed_text = preprocess_text(text)
    
    if model_type == 'ML' and model_data is not None:
        model = model_data.get('model')
        text_extractor = model_data.get('text_extractor') or model_data.get('vectorizer')
        scaler = model_data.get('scaler')
        
        if model is not None and text_extractor is not None:
            text_features = text_extractor.transform([processed_text])
            
            if scaler is not None and 'numerical_cols' in model_data:
                numerical_cols = model_data.get('numerical_cols', [])
                numerical_features = np.zeros((1, len(numerical_cols)))
                numerical_scaled = scaler.transform(numerical_features)
                from scipy.sparse import hstack, csr_matrix
                features = hstack([text_features, csr_matrix(numerical_scaled)])
            else:
                features = text_features
            
            prediction = model.predict(features)[0]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)
                # Create probability dictionary
                class_names = ['No Stress', 'Stress']
                prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(probabilities))}
            else:
                confidence = 0.5
                probabilities = [0.5, 0.5]
                prob_dict = {'No Stress': 0.5, 'Stress': 0.5}
            
            predicted_class = "Stress" if prediction == 1 else "No Stress"
            
            return {
                'prediction': int(prediction),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'probabilities_dict': prob_dict,
                'model_type': 'ML'
            }
    
    # Demo mode fallback
    word_count = len(text.split())
    stress_indicators = ['anxious', 'worried', 'stress', 'overwhelmed', 
                        'pressure', 'nervous', 'tense', 'panic', 'depressed',
                        'sad', 'hopeless', 'anxiety', 'fear', 'scared']
    
    stress_score = sum(1 for word in stress_indicators if word.lower() in text.lower())
    stress_score += min(word_count / 50, 1)
    
    prediction = 1 if stress_score > 1 else 0
    confidence = min(0.5 + stress_score * 0.1, 0.99)
    
    predicted_class = "Stress" if prediction == 1 else "No Stress"
    prob_dict = {
        'No Stress': 1 - confidence,
        'Stress': confidence
    }
    
    return {
        'prediction': prediction,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': [1 - confidence, confidence],
        'probabilities_dict': prob_dict,
        'model_type': 'Demo'
    }


def save_prediction_to_db(
    db: StressDetectionDatabase,
    user_info: Dict,
    text_input: str,
    result: Dict,
    session_id: str
):
    """
    Save prediction to database with cloud support.
    
    Args:
        db: Database instance
        user_info: Dictionary with user information
        text_input: User's text input
        result: Prediction result
        session_id: Session ID
    """
    try:
        # Get client info for tracking (works on both local and cloud)
        ip_address = st.request.remote_addr if hasattr(st, 'request') else None
        user_agent = st.request.headers.get('User-Agent', '') if hasattr(st, 'request') and hasattr(st.request, 'headers') else None
        
        record_id = db.insert_prediction(
            name=user_info.get('name'),
            age=user_info.get('age'),
            gender=user_info.get('gender'),
            user_input_text=text_input,
            predicted_class=result['predicted_class'],
            confidence_score=result['confidence'],
            all_class_probabilities=result['probabilities_dict'],
            model_type=result['model_type'],
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Also save to cloud manager for Streamlit Cloud persistence
        cloud_mgr = get_cloud_manager()
        cloud_mgr.insert_prediction(
            name=user_info.get('name'),
            age=user_info.get('age'),
            gender=user_info.get('gender'),
            user_input_text=text_input,
            predicted_class=result['predicted_class'],
            confidence_score=result['confidence'],
            all_class_probabilities=result['probabilities_dict'],
            model_type=result['model_type'],
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return record_id
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return None



def render_user_info_section():
    """
    Render user information collection section in sidebar.
    
    Returns:
        Dictionary with user information
    """
    st.sidebar.markdown("### 👤 User Information (Optional)")
    
    with st.sidebar.expander("Add Your Details", expanded=False):
        name = st.text_input("Name", placeholder="Your name", key="user_name")
        age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1, key="user_age")
        gender = st.selectbox(
            "Gender",
            options=["Prefer not to say", "Male", "Female", "Other"],
            key="user_gender"
        )
        
        # Convert "Prefer not to say" to None for age and gender
        age_val = None if age == 0 else age
        gender_val = None if gender == "Prefer not to say" else gender
        
        return {
            'name': name if name else None,
            'age': age_val,
            'gender': gender_val
        }


def render_feedback_section(db: StressDetectionDatabase, record_id: int):
    """
    Render feedback collection section.
    
    Args:
        db: Database instance
        record_id: ID of the prediction record
    """
    st.markdown("### 👍 Was this prediction helpful?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Yes, accurate", key=f"feedback_yes_{record_id}"):
            db.update_feedback(record_id, "Yes")
            st.success("Thank you for your feedback!")
            st.rerun()
    
    with col2:
        if st.button("❌ No, incorrect", key=f"feedback_no_{record_id}"):
            db.update_feedback(record_id, "No")
            st.success("Thank you for your feedback! We'll use this to improve.")
            st.rerun()
    
    with col3:
        if st.button("🤔 Unsure", key=f"feedback_unsure_{record_id}"):
            db.update_feedback(record_id, "Unsure")
            st.success("Thank you for your feedback!")
            st.rerun()


def render_analytics_dashboard(db: StressDetectionDatabase):
    """
    Render analytics dashboard with cloud data support.
    """
    st.markdown("## 📊 Analytics Dashboard")
    
    # Check if running on cloud
    cloud_mgr = get_cloud_manager()
    is_cloud = cloud_mgr.is_cloud
    
    if is_cloud:
        st.info("☁️ Running on Streamlit Cloud - Data is being stored in session state")
    
    # Get statistics
    stats = db.get_statistics()

    
    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", stats.get('total_predictions', 0))
        with col2:
            st.metric("Predictions Today", stats.get('predictions_today', 0))
        with col3:
            avg_conf = stats.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_conf:.2%}" if avg_conf else "N/A")
        with col4:
            feedback_count = sum(stats.get('feedback_statistics', {}).values())
            st.metric("Feedback Received", feedback_count)
        
        # Class distribution
        st.markdown("### 📈 Prediction Distribution")
        class_dist = stats.get('class_distribution', {})
        if class_dist:
            dist_df = pd.DataFrame({
                'Class': list(class_dist.keys()),
                'Count': list(class_dist.values())
            })
            st.bar_chart(dist_df.set_index('Class'))
        
        # Feedback statistics
        feedback_stats = stats.get('feedback_statistics', {})
        if feedback_stats:
            st.markdown("### 🗣️ Feedback Distribution")
            feedback_df = pd.DataFrame({
                'Feedback': list(feedback_stats.keys()),
                'Count': list(feedback_stats.values())
            })
            st.bar_chart(feedback_df.set_index('Feedback'))
        
        # Recent predictions
        st.markdown("### 📝 Recent Predictions")
        recent_df = db.get_all_predictions(limit=10)
        if not recent_df.empty:
            display_df = recent_df[['timestamp', 'name', 'predicted_class', 
                                   'confidence_score', 'user_feedback']].copy()
            display_df.columns = ['Timestamp', 'Name', 'Prediction', 
                                'Confidence', 'Feedback']
            st.dataframe(display_df, use_container_width=True)
        
        # Export option
        st.markdown("### 💾 Export Data")
        if st.button("Export to CSV"):
            export_path = f"stress_predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            if db.export_to_csv(export_path):
                st.success(f"Data exported to {export_path}")
                with open(export_path, 'rb') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f,
                        file_name=export_path,
                        mime='text/csv'
                    )
            else:
                st.error("Failed to export data")


def main():
    """
    Main Streamlit application.
    """
    # Initialize database (works for both local and cloud)
    db = get_database()
    cloud_mgr = get_cloud_manager()
    
    # Generate session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Store last record ID for feedback
    if 'last_record_id' not in st.session_state:
        st.session_state.last_record_id = None
    
    # Show deployment info
    if cloud_mgr.is_cloud:
        st.sidebar.success("☁️ Cloud Mode Active")
        st.sidebar.caption("Data stored in session state")
    else:
        st.sidebar.info("💻 Local Mode")
        st.sidebar.caption(f"Database: {db.db_path}")

    
    st.title("🧠 Mental Stress Detection")
    st.markdown("### AI-Powered Stress Detection from Text")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # User information collection
    user_info = render_user_info_section()
    
    st.sidebar.markdown("---")
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Model",
        ["Traditional ML", "Demo Mode", "BERT Model"]
    )
    
    ml_model_data = None
    if model_type == "Traditional ML":
        with st.spinner("Loading ML model..."):
            ml_model_data = load_ml_model()
            if ml_model_data is None:
                st.warning("ML model not found. Using demo mode.")
                model_type = "Demo Mode"
            else:
                st.success("ML model loaded successfully!")
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🧠 Prediction", "📊 Analytics Dashboard"]
    )
    
    st.sidebar.markdown("---")
    
    # Information section
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This application uses machine learning and deep learning "
        "to detect mental stress from text input. Enter some text "
        "to analyze stress levels."
    )
    
    # Privacy notice
    st.sidebar.markdown("### 🔒 Privacy")
    st.sidebar.caption(
        "Your data is stored locally for improving the model. "
        "Personal information is optional and can be anonymized."
    )
    
    # Main content based on page selection
    if page == "📊 Analytics Dashboard":
        render_analytics_dashboard(db)
    else:
        # Prediction Page
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Enter Your Text")
            text_input = st.text_area(
                "Type or paste text to analyze:",
                height=200,
                placeholder="Enter text that expresses your thoughts, feelings, or experiences..."
            )
            
            if st.button("🔍 Analyze Stress Level", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing text..."):
                        if model_type == "Traditional ML" and ml_model_data is not None:
                            result = predict_stress(
                                text_input,
                                model_data=ml_model_data,
                                model_type='ML'
                            )
                        else:
                            result = predict_stress(
                                text_input,
                                model_data=None,
                                model_type=model_type
                            )
                        
                        # Save to database
                        record_id = save_prediction_to_db(
                            db, user_info, text_input, result, 
                            st.session_state.session_id
                        )
                        if record_id:
                            st.session_state.last_record_id = record_id
                        
                        # Display results
                        st.markdown("### 📊 Analysis Results")
                        
                        prediction = result['prediction']
                        confidence = result['confidence']
                        
                        if prediction == 1:
                            st.markdown(
                                f"""
                                <div class="prediction-card stress-high">
                                    <h2>⚠️ High Stress Detected</h2>
                                    <p>Confidence: {confidence:.2%}</p>
                                    <p>Model: {result['model_type']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="prediction-card stress-low">
                                    <h2>✅ Low Stress / No Stress</h2>
                                    <p>Confidence: {confidence:.2%}</p>
                                    <p>Model: {result['model_type']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        if result['probabilities'] is not None:
                            st.markdown("#### Probability Distribution")
                            probs = result['probabilities']
                            
                            prob_df = pd.DataFrame({
                                'Class': ['No Stress', 'Stress'],
                                'Probability': probs
                            })
                            
                            st.bar_chart(prob_df.set_index('Class'))
                        
                        st.markdown("#### 📈 Text Statistics")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Characters", len(text_input))
                        with col_stat2:
                            st.metric("Words", len(text_input.split()))
                        with col_stat3:
                            st.metric("Sentences", len(text_input.split('.')))
                        
                        # Feedback section
                        if record_id:
                            render_feedback_section(db, record_id)
                        
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.markdown("### 📋 Sample Texts")
            
            sample_texts = {
                "Low Stress": "I had a great day at work today. Everything went well and I'm feeling happy and relaxed.",
                "Medium Stress": "I have a lot of work to do and I'm feeling a bit overwhelmed with all the deadlines.",
                "High Stress": "I'm so stressed and anxious about everything. I can't handle this pressure anymore. I'm panicking!"
            }
            
            for label, sample in sample_texts.items():
                if st.button(f"Use {label} Sample"):
                    st.session_state['sample_text'] = sample
            
            if 'sample_text' in st.session_state:
                st.text_area(
                    "Sample Text:",
                    value=st.session_state['sample_text'],
                    height=150,
                    disabled=True
                )
            
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Be honest in your input
            - Longer texts provide better analysis
            - Results are for informational purposes only
            - Consult a professional for medical advice
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>Mental Stress Detection App | For Educational Purposes</p>
            <p><a href='https://github.com/dreamtunesofficials-coder/Mental-Health' target='_blank'>View Source Code on GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


def run_demo():
    """
    Run the demo mode.
    """
    main()


if __name__ == "__main__":
    run_demo()
