"""
Streamlit Web Application for Mental Stress Detection
Interactive web interface for stress prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from typing import Dict, List
import sys

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess_text
from feature_engineering import FeatureExtractor


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
</style>
""", unsafe_allow_html=True)


def load_ml_model():
    """
    Load the ML model. Uses regularized_model.pkl (best for generalization with less overfitting).
    
    Returns:
        Loaded model or None
    """
    # Use best_model.pkl for better accuracy
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
    # Preprocess text
    processed_text = preprocess_text(text)
    
    if model_type == 'ML' and model_data is not None:
        model = model_data.get('model')
        
        # Try to get text_extractor first (new format), then vectorizer (old format)
        text_extractor = model_data.get('text_extractor') or model_data.get('vectorizer')
        scaler = model_data.get('scaler')
        
        if model is not None and text_extractor is not None:
            # Transform text
            text_features = text_extractor.transform([processed_text])
            
            # Get numerical features if scaler is available
            if scaler is not None and 'numerical_cols' in model_data:
                numerical_cols = model_data.get('numerical_cols', [])
                # Use default values for numerical features (mean=0 after scaling)
                numerical_features = np.zeros((1, len(numerical_cols)))
                numerical_scaled = scaler.transform(numerical_features)
                from scipy.sparse import hstack, csr_matrix
                features = hstack([text_features, csr_matrix(numerical_scaled)])
            else:
                features = text_features
            
            # Get prediction
            prediction = model.predict(features)[0]
            
            # Get probability if available
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
    
    # Return mock prediction for demo
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


def main():
    """
    Main Streamlit application.
    """
    # Title and header
    st.title("🧠 Mental Stress Detection")
    st.markdown("### AI-Powered Stress Detection from Text")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Model",
        ["Traditional ML", "Demo Mode", "BERT Model"]
    )
    
    # Load ML model if selected
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
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This application uses machine learning and deep learning "
        "to detect mental stress from text input. Enter some text "
        "to analyze stress levels."
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        st.markdown("### 📝 Enter Your Text")
        text_input = st.text_area(
            "Type or paste text to analyze:",
            height=200,
            placeholder="Enter text that expresses your thoughts, feelings, or experiences..."
        )
        
        # Analyze button
        if st.button("🔍 Analyze Stress Level", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Get prediction based on model type
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
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    # Prediction
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if prediction == 1:
                        st.markdown(
                            f"""
                            <div class="prediction-card stress-high">
                                <h2>⚠️ High Stress Detected</h2>
                                <p>Confidence: {confidence:.2%}</p>
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
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Probability bar
                    if result['probabilities'] is not None:
                        st.markdown("#### Probability Distribution")
                        probs = result['probabilities']
                        
                        prob_df = pd.DataFrame({
                            'Class': ['No Stress', 'Stress'],
                            'Probability': probs
                        })
                        
                        st.bar_chart(prob_df.set_index('Class'))
                    
                    # Text statistics
                    st.markdown("#### 📈 Text Statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Characters", len(text_input))
                    with col_stat2:
                        st.metric("Words", len(text_input.split()))
                    with col_stat3:
                        st.metric("Sentences", len(text_input.split('.')))
                        
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        # Sample texts
        st.markdown("### 📋 Sample Texts")
        
        sample_texts = {
            "Low Stress": "I had a great day at work today. Everything went well and I'm feeling happy and relaxed.",
            "Medium Stress": "I have a lot of work to do and I'm feeling a bit overwhelmed with all the deadlines.",
            "High Stress": "I'm so stressed and anxious about everything. I can't handle this pressure anymore. I'm panicking!"
        }
        
        for label, sample in sample_texts.items():
            if st.button(f"Use {label} Sample"):
                st.session_state['sample_text'] = sample
        
        # Check for sample text in session
        if 'sample_text' in st.session_state:
            st.text_area(
                "Sample Text:",
                value=st.session_state['sample_text'],
                height=150,
                disabled=True
            )
        
        # Tips section
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
