"""
Enhanced Streamlit Web Application for Mental Stress Detection
Interactive web interface with database integration for stress prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from typing import Dict, List, Optional
import sys
import uuid

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess_text
from feature_engineering import FeatureExtractor
from database import get_database, StressDetectionDatabase

# Page configuration
st.set_page_config(
    page_title="Mental Stress Detection",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTextArea textarea { font-size: 16px; }
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
    .feedback-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .user-info-section {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize database
db = get_database()


def initialize_session():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'last_prediction_id' not in st.session_state:
        st.session_state.last_prediction_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'user_age' not in st.session_state:
        st.session_state.user_age = None
    if 'user_gender' not in st.session_state:
        st.session_state.user_gender = "Prefer not to say"


def load_ml_model():
    """
    Load the ML model. Uses best_model.pkl for better accuracy.
    
    Returns:
        Loaded model or None
    """
    model_path = './models/best_model.pkl'
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                return model_data
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    st.warning("ML model not found. Will use demo mode.")
    return None


def load_bert_model(model_path: str = './models/bert_model'):
    """
    Load BERT model.
    
    Args:
        model_path: Path to BERT model
        
    Returns:
        Loaded model or None
    """
    try:
        from train_bert import StressBERTClassifier
        if os.path.exists(model_path):
            classifier = StressBERTClassifier()
            classifier._load_model()
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
        # Handle both 'text_extractor' and 'vectorizer' keys
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
            else:
                confidence = None
                probabilities = None
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'probabilities': probabilities,
                'model_type': 'ML'
            }
    
    # Demo mode fallback
    word_count = len(text.split())
    stress_indicators = ['anxious', 'worried', 'stress', 'overwhelmed', 
                        'pressure', 'nervous', 'tense', 'panic']
    
    stress_score = sum(1 for word in stress_indicators if word.lower() in text.lower())
    stress_score += min(word_count / 50, 1)
    
    prediction = 1 if stress_score > 1 else 0
    confidence = min(0.5 + stress_score * 0.1, 0.99)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': [1 - confidence, confidence],
        'model_type': 'Demo'
    }


def save_prediction_to_database(
    name: Optional[str],
    age: Optional[int],
    gender: Optional[str],
    text: str,
    result: Dict
) -> int:
    """
    Save prediction to database.
    
    Args:
        name: User's name
        age: User's age
        gender: User's gender
        text: Input text
        result: Prediction result dictionary
        
    Returns:
        Record ID or -1 if failed
    """
    try:
        # Determine predicted class label
        prediction = result['prediction']
        predicted_class = "Stress" if prediction == 1 else "No Stress"
        
        # Prepare probabilities dictionary
        probabilities = result.get('probabilities')
        if probabilities is not None:
            if isinstance(probabilities, (list, np.ndarray)):
                all_probs = {
                    "No Stress": float(probabilities[0]),
                    "Stress": float(probabilities[1])
                }
            else:
                all_probs = {"Unknown": 0.0}
        else:
            all_probs = {"Demo Mode": result.get('confidence', 0.0)}
        
        # Insert into database
        record_id = db.insert_prediction(
            name=name,
            age=age,
            gender=gender,
            user_input_text=text,
            predicted_class=predicted_class,
            confidence_score=result.get('confidence', 0.0),
            all_class_probabilities=all_probs,
            model_type=result.get('model_type', 'Unknown'),
            session_id=st.session_state.session_id
        )
        
        return record_id
        
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return -1


def render_user_info_section():
    """
    Render user information input section in sidebar.
    """
    st.sidebar.markdown("### User Information")
    st.sidebar.markdown("*Optional - helps us improve the model*")
    
    name = st.sidebar.text_input(
        "Name (optional)",
        value=st.session_state.user_name,
        placeholder="Enter your name",
        key="name_input"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input(
            "Age (optional)",
            min_value=1,
            max_value=120,
            value=st.session_state.user_age if st.session_state.user_age else 25,
            step=1,
            key="age_input"
        )
    
    with col2:
        gender = st.selectbox(
            "Gender (optional)",
            options=["Prefer not to say", "Male", "Female", "Other"],
            index=["Prefer not to say", "Male", "Female", "Other"].index(
                st.session_state.user_gender
            ),
            key="gender_input"
        )
    
    # Update session state
    st.session_state.user_name = name if name else None
    st.session_state.user_age = int(age) if age > 0 else None
    st.session_state.user_gender = gender if gender != "Prefer not to say" else None
    
    return st.session_state.user_name, st.session_state.user_age, st.session_state.user_gender


def render_feedback_section(record_id: int):
    """
    Render feedback section after prediction.
    
    Args:
        record_id: ID of the prediction record
    """
    st.markdown("---")
    st.markdown("### Was this prediction helpful?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Yes, accurate", key="feedback_yes"):
            db.update_feedback(record_id, "Yes")
            st.success("Thank you for your feedback!")
            st.balloons()
    
    with col2:
        if st.button("No, incorrect", key="feedback_no"):
            db.update_feedback(record_id, "No")
            st.info("Thank you for your feedback! We will use this to improve.")
    
    with col3:
        if st.button("Not sure", key="feedback_unsure"):
            db.update_feedback(record_id, "Unsure")
            st.info("Thank you for your feedback!")


def render_analytics_dashboard():
    """
    Render analytics dashboard.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analytics")
    
    if st.sidebar.checkbox("Show Analytics Dashboard", key="show_analytics"):
        st.markdown("## Prediction Analytics Dashboard")
        
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
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            
            with col4:
                feedback_count = sum(stats.get('feedback_statistics', {}).values())
                st.metric("Feedback Received", feedback_count)
            
            # Class distribution
            st.markdown("### Class Distribution")
            class_dist = stats.get('class_distribution', {})
            if class_dist:
                dist_df = pd.DataFrame(
                    list(class_dist.items()), 
                    columns=['Class', 'Count']
                )
                st.bar_chart(dist_df.set_index('Class'))
            
            # Feedback statistics
            feedback_stats = stats.get('feedback_statistics', {})
            if feedback_stats:
                st.markdown("### Feedback Statistics")
                feedback_df = pd.DataFrame(
                    list(feedback_stats.items()),
                    columns=['Feedback', 'Count']
                )
                st.bar_chart(feedback_df.set_index('Feedback'))
            
            # Recent predictions
            st.markdown("### Recent Predictions")
            recent_df = db.get_all_predictions(limit=10)
            if not recent_df.empty:
                st.dataframe(recent_df[[
                    'timestamp', 'predicted_class', 'confidence_score', 
                    'user_feedback', 'model_type'
                ]])
            
            # Export option
            st.markdown("### Export Data")
            if st.button("Export to CSV"):
                export_path = './data/predictions_export.csv'
                if db.export_to_csv(export_path):
                    st.success(f"Data exported to {export_path}")
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            "Download CSV",
                            f,
                            file_name="stress_predictions.csv",
                            mime="text/csv"
                        )


def main():
    """
    Main Streamlit application.
    """
    # Initialize session
    initialize_session()
    
    st.title("Mental Stress Detection")
    st.markdown("### AI-Powered Stress Detection from Text")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # User information section
    name, age, gender = render_user_info_section()
    
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
    
    # Information section
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application uses machine learning and deep learning "
        "to detect mental stress from text input. Enter some text "
        "to analyze stress levels."
    )
    
    # Usage section
    st.sidebar.markdown("### Usage")
    st.sidebar.code("streamlit run app_enhanced.py", language="bash")
    st.sidebar.markdown("The application will open at http://localhost:8501")
    
    # Analytics dashboard
    render_analytics_dashboard()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Your Text")
        text_input = st.text_area(
            "Type or paste text to analyze:",
            height=200,
            placeholder="Enter text that expresses your thoughts, feelings, or experiences..."
        )
        
        if st.button("Analyze Stress Level", type="primary"):
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
                    
                    st.markdown("### Analysis Results")
                    
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if prediction == 1:
                        st.markdown(
                            f"""
                            <div class="prediction-card stress-high">
                                <h2>High Stress Detected</h2>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="prediction-card stress-low">
                                <h2>Low Stress / No Stress</h2>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    if result['probabilities'] is not None:
                        st.markdown("#### Probability Distribution")
                        probs = result['probabilities']
                        
                        prob_df = pd.DataFrame({
                            'Class': ['No Stress', 'Stress'],
                            'Probability': probs if isinstance(probs, list) else [1-confidence, confidence]
                        })
                        
                        st.bar_chart(prob_df.set_index('Class'))
                    
                    st.markdown("#### Text Statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Characters", len(text_input))
                    with col_stat2:
                        st.metric("Words", len(text_input.split()))
                    with col_stat3:
                        st.metric("Sentences", len(text_input.split('.')))
                    
                    # Save to database
                    record_id = save_prediction_to_database(
                        name=name,
                        age=age,
                        gender=gender,
                        text=text_input,
                        result=result
                    )
                    
                    if record_id > 0:
                        st.session_state.last_prediction_id = record_id
                        st.success("Prediction saved to database!")
                        
                        # Show feedback section
                        render_feedback_section(record_id)
                    else:
                        st.warning("Could not save to database, but prediction completed.")
                        
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.markdown("### Sample Texts")
        
        sample_texts = {
            "Low Stress": "I had a great day at work today. Everything went well and I am feeling happy and relaxed.",
            "Medium Stress": "I have a lot of work to do and I am feeling a bit overwhelmed with all the deadlines.",
            "High Stress": "I am so stressed and anxious about everything. I cannot handle this pressure anymore. I am panicking!"
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
        
        st.markdown("### Tips")
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
