"""
🧠 Mental Stress Detection - Professional Edition
Modern, Interactive Web Application with Advanced UI/UX
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import json
import re
from typing import Dict, List, Optional, Tuple
import sys
from datetime import datetime
import uuid
import time
from collections import Counter

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess_text
from feature_engineering import FeatureExtractor
from cloud_database import StressDetectionDatabase, get_database, get_cloud_manager

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try to import streamlit-lottie for animations
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Mental Stress Detection Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# MODERN CSS STYLING WITH GLASSMORPHISM
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1.5rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        opacity: 0.8;
        margin-top: 3rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-low {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border: 3px solid #7cb342;
    }
    
    .result-medium {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        border: 3px solid #f57c00;
    }
    
    .result-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border: 3px solid #e53935;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'session_id': str(uuid.uuid4()),
        'last_prediction_id': None,
        'user_name': "",
        'user_age': None,
        'user_gender': "Prefer not to say",
        'prediction_history': [],
        'current_text': "",
        'show_welcome': True,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================
@st.cache_resource
def load_ml_model(model_name: str = 'best_model.pkl'):
    """Load ML model with caching for better performance."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # Try alternative models
    alternatives = ['proper_model.pkl', 'optimized_model.pkl', 'advanced_model.pkl', 'ml_model.pkl']
    for alt in alternatives:
        alt_path = os.path.join(models_dir, alt)
        if os.path.exists(alt_path):
            try:
                with open(alt_path, 'rb') as f:
                    return pickle.load(f)
            except:
                continue
    
    return None


# ============================================
# PREDICTION FUNCTIONS
# ============================================
def predict_stress(text: str, model_data: dict, model_type: str = 'ML') -> Dict:
    """Enhanced stress prediction with detailed analysis."""
    processed_text = preprocess_text(text)
    word_count = len(text.split())
    char_count = len(text)
    
    # Extract features for analysis
    stress_keywords = ['anxious', 'worried', 'stress', 'overwhelmed', 'pressure', 
                      'nervous', 'tense', 'panic', 'depressed', 'sad', 'hopeless',
                      'anxiety', 'fear', 'scared', 'tired', 'exhausted', 'burnout',
                      'insomnia', 'headache', 'frustrated', 'angry', 'lonely']
    
    positive_keywords = ['happy', 'joy', 'excited', 'grateful', 'peaceful', 'calm',
                        'relaxed', 'confident', 'optimistic', 'energetic', 'content']
    
    # Count keyword occurrences
    text_lower = text.lower()
    stress_matches = [kw for kw in stress_keywords if kw in text_lower]
    positive_matches = [kw for kw in positive_keywords if kw in text_lower]
    
    # Calculate sentiment scores
    stress_score = len(stress_matches)
    positive_score = len(positive_matches)
    
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
                prob_dict = {
                    'No Stress': float(probabilities[0]),
                    'Stress': float(probabilities[1])
                }
            else:
                confidence = 0.5
                prob_dict = {'No Stress': 0.5, 'Stress': 0.5}
            
            predicted_class = "Stress" if prediction == 1 else "No Stress"
            
            return {
                'prediction': int(prediction),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'model_type': 'ML',
                'word_count': word_count,
                'char_count': char_count,
                'stress_keywords': stress_matches,
                'positive_keywords': positive_matches,
                'stress_score': stress_score,
                'positive_score': positive_score
            }
    
    # Enhanced demo mode
    base_stress = stress_score * 0.15
    length_factor = min(word_count / 100, 0.3)
    sentiment_factor = stress_score / (positive_score + stress_score + 1)
    
    total_stress_score = base_stress + length_factor + (sentiment_factor * 0.3)
    prediction = 1 if total_stress_score > 0.5 else 0
    confidence = min(0.5 + total_stress_score * 0.5, 0.95)
    
    predicted_class = "Stress" if prediction == 1 else "No Stress"
    prob_dict = {
        'No Stress': 1 - confidence,
        'Stress': confidence
    }
    
    return {
        'prediction': prediction,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': prob_dict,
        'model_type': 'Demo',
        'word_count': word_count,
        'char_count': char_count,
        'stress_keywords': stress_matches,
        'positive_keywords': positive_matches,
        'stress_score': stress_score,
        'positive_score': positive_score
    }


# ============================================
# UI COMPONENTS
# ============================================
def render_hero_section():
    """Render the hero section with animated title."""
    st.markdown("""
    <div class="hero-section animate-fade-in">
        <div class="hero-title">🧠 Mental Stress Detection</div>
        <div class="hero-subtitle">
            AI-Powered Analysis for Mental Wellness<br>
            <span style="font-size: 0.9rem; opacity: 0.8;">
                Enter your thoughts and let our advanced AI analyze your stress levels
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_cards():
    """Render feature highlight cards."""
    st.markdown('<div class="feature-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0;">', unsafe_allow_html=True)
    
    features = [
        ("🤖", "AI-Powered", "Advanced ML models"),
        ("⚡", "Real-time", "Instant analysis"),
        ("📊", "Insights", "Detailed reports"),
        ("🔒", "Privacy", "Secure & private"),
        ("📱", "Mobile", "Works everywhere"),
        ("🎯", "Accuracy", "High precision")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card animate-fade-in">
            <div class="feature-icon">{icon}</div>
            <h3 style="margin: 0.5rem 0; color: #333;">{title}</h3>
            <p style="color: #666; font-size: 0.9rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_result_card(result: Dict):
    """Render animated result card."""
    prediction = result['prediction']
    confidence = result['confidence']
    
    if prediction == 1:
        result_class = "high"
        emoji = "😰"
        title = "⚠️ Stress Detected"
        message = "We detected signs of elevated stress in your text."
    else:
        result_class = "low"
        emoji = "😊"
        title = "✅ Low Stress"
        message = "Your text appears to be calm and relaxed."
    
    st.markdown(f"""
    <div class="result-card result-{result_class}" style="animation: fadeIn 0.5s ease-out;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
        <h2 style="margin: 0.5rem 0; color: #333;">{title}</h2>
        <p style="font-size: 1.2rem; color: #666;">{message}</p>
        <p style="font-size: 1.5rem; font-weight: 700; color: #667eea;">
            Confidence: {confidence:.1%}
        </p>
        <p style="font-size: 0.9rem; color: #888; margin-top: 1rem;">
            Model: {result['model_type']}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_chart(probabilities: Dict):
    """Render interactive probability chart."""
    if PLOTLY_AVAILABLE:
        colors = ['#96e6a1', '#ff9a9e']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(probabilities.keys()),
                y=[v * 100 for v in probabilities.values()],
                marker_color=colors,
                text=[f"{v:.1%}" for v in probabilities.values()],
                textposition='auto',
                textfont={'size': 16}
            )
        ])
        
        fig.update_layout(
            title={'text': 'Probability Distribution', 'font': {'size': 18}, 'x': 0.5},
            yaxis_title='Probability (%)',
            yaxis_range=[0, 100],
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#333'},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        prob_df = pd.DataFrame({
            'Class': list(probabilities.keys()),
            'Probability': list(probabilities.values())
        })
        st.bar_chart(prob_df.set_index('Class'))


def render_keyword_analysis(stress_keywords: List, positive_keywords: List):
    """Render keyword analysis visualization."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center;">
            <h4 style="color: #c62828; margin: 0;">⚠️ Stress Indicators</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if stress_keywords:
            unique_stress = list(set(stress_keywords))[:6]
            for kw in unique_stress:
                st.markdown(f"""
                <div style="background: white; padding: 0.5rem; margin: 0.25rem 0; 
                            border-radius: 20px; text-align: center; 
                            border: 2px solid #ff9a9e; color: #c62828;">
                    {kw}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No stress indicators found")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); 
                    padding: 1rem; border-radius: 15px; text-align: center;">
            <h4 style="color: #2d5a3d; margin: 0;">✅ Positive Indicators</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if positive_keywords:
            unique_positive = list(set(positive_keywords))[:6]
            for kw in unique_positive:
                st.markdown(f"""
                <div style="background: white; padding: 0.5rem; margin: 0.25rem 0; 
                            border-radius: 20px; text-align: center; 
                            border: 2px solid #96e6a1; color: #2d5a3d;">
                    {kw}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No positive indicators found")


def render_metrics_cards(result: Dict):
    """Render metric cards with statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        (col1, "📝 Words", result['word_count']),
        (col2, "📊 Characters", result['char_count']),
        (col3, "⚠️ Stress Keywords", result['stress_score']),
        (col4, "✅ Positive Keywords", result['positive_score'])
    ]
    
    for col, label, value in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_sample_buttons():
    """Render sample text buttons."""
    st.markdown("### 📋 Try Sample Texts")
    
    samples = [
        ("😊 Low Stress", "I had a wonderful day today! The weather was perfect, I spent time with friends, and I'm feeling really grateful for all the good things in my life."),
        ("😰 Medium Stress", "Work has been quite demanding lately. I have several deadlines approaching and I'm juggling multiple projects. It's manageable but definitely stressful."),
        ("😫 High Stress", "I can't take this anymore. Everything is overwhelming, I can't sleep, my anxiety is through the roof, and I feel like I'm drowning in responsibilities.")
    ]
    
    cols = st.columns(3)
    for i, (label, text) in enumerate(samples):
        with cols[i]:
            if st.button(label, key=f"sample_{i}", use_container_width=True):
                st.session_state.current_text = text
                st.rerun()


def render_analytics_dashboard(db):
    """Render enhanced analytics dashboard."""
    st.markdown("## 📊 Analytics Dashboard")
    
    stats = db.get_statistics()
    
    if not stats:
        st.warning("No data available yet. Make some predictions first!")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", stats.get('total_predictions', 0))
    with col2:
        st.metric("Today", stats.get('predictions_today', 0))
    with col3:
        avg_conf = stats.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{avg_conf:.1%}" if avg_conf else "N/A")
    with col4:
        feedback_stats = stats.get('feedback_statistics', {})
        st.metric("Feedback", sum(feedback_stats.values()))
    
    # Charts
    if PLOTLY_AVAILABLE:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Class Distribution")
            class_dist = stats.get('class_distribution', {})
            if class_dist:
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    color=list(class_dist.keys()),
                    color_discrete_map={'Stress': '#ff9a9e', 'No Stress': '#96e6a1'},
                    hole=0.4
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🗣️ Feedback Distribution")
            if feedback_stats:
                fig = px.bar(
                    x=list(feedback_stats.keys()),
                    y=list(feedback_stats.values()),
                    color=list(feedback_stats.keys()),
                    color_discrete_sequence=['#96e6a1', '#ff9a9e', '#f6d365']
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    st.markdown("### 📝 Recent Predictions")
    recent_df = db.get_all_predictions(limit=10)
    if not recent_df.empty:
        display_df = recent_df[['timestamp', 'predicted_class', 'confidence_score', 'user_feedback']].copy()
        display_df.columns = ['Time', 'Prediction', 'Confidence', 'Feedback']
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application function."""
    init_session_state()
    
    # Initialize database
    db = get_database()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_type = st.radio(
            "Select Model",
            ["🤖 Traditional ML", "🧪 Demo Mode"],
            index=0
        )
        
        actual_model = "Traditional ML" if "ML" in model_type else "Demo Mode"
        
        st.markdown("---")
        
        # User info
        st.markdown("### 👤 Your Info (Optional)")
        with st.expander("Add Details", expanded=False):
            name = st.text_input("Name", value=st.session_state.user_name)
            age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
            gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
            
            st.session_state.user_name = name if name else None
            st.session_state.user_age = age if age > 0 else None
            st.session_state.user_gender = gender if gender != "Prefer not to say" else None
        
        st.markdown("---")
        
        # Navigation
        page = st.radio("Navigate", ["🧠 Analyze", "📊 Dashboard", "ℹ️ About"])
        
        st.markdown("---")
        
        # Info
        st.markdown("### 🔒 Privacy")
        st.caption("Your data is stored locally. Personal info is optional.")
    
    # Main content
    if page == "ℹ️ About":
        render_hero_section()
        render_feature_cards()
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        ### How It Works
        
        Our Mental Stress Detection system uses advanced Natural Language Processing (NLP) 
        and Machine Learning algorithms to analyze text and detect stress levels.
        
        **Key Features:**
        - **Real-time Analysis**: Get instant results as you type
        - **Multiple Models**: Choose from Traditional ML or Demo mode
        - **Detailed Insights**: View keyword analysis and probability distributions
        - **Privacy Focused**: Your data stays on your device
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif page == "📊 Dashboard":
        render_analytics_dashboard(db)
        
    else:  # Analyze page
        render_hero_section()
        
        # Load model if needed
        ml_model_data = None
        if actual_model == "Traditional ML":
            with st.spinner("🔄 Loading AI Model..."):
                ml_model_data = load_ml_model()
                if ml_model_data is None:
                    st.warning("ML model not found. Using Demo mode.")
                    actual_model = "Demo Mode"
                else:
                    st.success("✅ AI Model Loaded!")
        
        # Main analysis section
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "📝 Share your thoughts...",
            value=st.session_state.current_text,
            height=200,
            placeholder="Type or paste text expressing your feelings, thoughts, or experiences...",
            key="main_text_input"
        )
        
        st.session_state.current_text = text_input
        
        # Real-time stats
        if text_input:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(text_input))
            with col2:
                st.metric("Words", len(text_input.split()))
            with col3:
                sentences = len([s for s in text_input.split('.') if s.strip()])
                st.metric("Sentences", sentences)
        
        # Sample buttons
        render_sample_buttons()
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze Stress Level", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis results
        if analyze_clicked and text_input.strip():
            with st.spinner("🧠 Analyzing your text..."):
                result = predict_stress(
                    text_input,
                    model_data=ml_model_data,
                    model_type=actual_model
                )
                
                # Save to database
                try:
                    record_id = db.insert_prediction(
                        name=st.session_state.user_name,
                        age=st.session_state.user_age,
                        gender=st.session_state.user_gender,
                        user_input_text=text_input,
                        predicted_class=result['predicted_class'],
                        confidence_score=result['confidence'],
                        all_class_probabilities=result['probabilities'],
                        model_type=result['model_type'],
                        session_id=st.session_state.session_id
                    )
                    st.session_state.last_prediction_id = record_id
                except Exception as e:
                    print(f"Database save error: {e}")
                
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                render_result_card(result)
                
                st.markdown("---")
                
                # Probability chart
                render_probability_chart(result['probabilities'])
                
                st.markdown("---")
                
                # Metrics cards
                render_metrics_cards(result)
                
                st.markdown("---")
                
                # Keyword analysis
                render_keyword_analysis(
                    result['stress_keywords'],
                    result['positive_keywords']
                )
                
                # Feedback section
                if st.session_state.last_prediction_id:
                    st.markdown("### 👍 Was this helpful?")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Yes", key="fb_yes"):
                            db.update_feedback(st.session_state.last_prediction_id, "Yes")
                            st.balloons()
                            st.success("Thanks!")
                    
                    with col2:
                        if st.button("❌ No", key="fb_no"):
                            db.update_feedback(st.session_state.last_prediction_id, "No")
                            st.info("Thanks for feedback!")
                    
                    with col3:
                        if st.button("🤔 Not Sure", key="fb_unsure"):
                            db.update_feedback(st.session_state.last_prediction_id, "Unsure")
                            st.info("Thanks!")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🧠 Mental Stress Detection Pro | For Educational Purposes</p>
        <p><a href="https://github.com/dreamtunesofficials-coder/Mental-Health" target="_blank" style="color: white;">View Source Code</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
