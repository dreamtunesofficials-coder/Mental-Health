"""
🧠 Mental Stress Detection - Professional Edition v4
Modern UI with Enhanced v2 Features + Supabase Cloud
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

# Import auto retrain functionality
try:
    from auto_retrain_from_supabase import SupabaseFeedbackTrainer
    RETRAIN_AVAILABLE = True
except ImportError:
    RETRAIN_AVAILABLE = False


# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Mental Stress Detection Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Professional Style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* User info section - Clean, borderless */
    .user-info-section {
        background: transparent;
        padding: 1rem 0;
        margin-bottom: 1rem;
        border: none;
        box-shadow: none;
    }

    
    /* Cards - Clean, borderless */
    .glass-card {
        background: transparent;
        padding: 1rem 0;
        border: none;
        box-shadow: none;
        margin-bottom: 1rem;
    }
    
    /* Remove Streamlit notification box borders */
    .stAlert {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Success message - clean */
    div[data-testid="stSuccessMessage"] {
        border: none !important;
        box-shadow: none !important;
        background: rgba(212, 252, 121, 0.3) !important;
    }
    
    /* Info message - clean */
    div[data-testid="stInfoMessage"] {
        border: none !important;
        box-shadow: none !important;
        background: rgba(227, 242, 253, 0.3) !important;
    }

    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-low {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-left: 5px solid #44ff44;
    }
    
    .result-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-left: 5px solid #ff4444;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
    }
    
    /* Sample buttons */
    .sample-btn {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .sample-btn:hover {
        background: #667eea;
        color: white;
    }
    
    /* Keyword tags */
    .keyword-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .stress-tag {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    
    .positive-tag {
        background: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE
# ============================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'session_id': str(uuid.uuid4()),
        'last_record_id': None,
        'user_name': "",
        'user_age': 0,
        'user_gender': "Prefer not to say",
        'current_text': "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_ml_model(model_name: str = 'best_model.pkl'):
    """Load ML model with caching."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # Try alternatives
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
# PREDICTION
# ============================================
def predict_stress(text: str, model_data: dict, model_type: str = 'ML') -> Dict:
    """Predict stress level from text."""
    processed_text = preprocess_text(text)
    
    # Keyword analysis
    stress_keywords = ['anxious', 'worried', 'stress', 'overwhelmed', 'pressure', 
                      'nervous', 'tense', 'panic', 'depressed', 'sad', 'hopeless',
                      'anxiety', 'fear', 'scared', 'tired', 'exhausted', 'burnout',
                      'insomnia', 'headache', 'frustrated', 'angry', 'lonely']
    
    positive_keywords = ['happy', 'joy', 'excited', 'grateful', 'peaceful', 'calm',
                        'relaxed', 'confident', 'optimistic', 'energetic', 'content']
    
    text_lower = text.lower()
    stress_matches = [kw for kw in stress_keywords if kw in text_lower]
    positive_matches = [kw for kw in positive_keywords if kw in text_lower]
    
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
                prob_dict = {'No Stress': float(probabilities[0]), 'Stress': float(probabilities[1])}
            else:
                confidence = 0.5
                prob_dict = {'No Stress': 0.5, 'Stress': 0.5}
            
            predicted_class = "Stress" if prediction == 1 else "No Stress"
            
            return {
                'prediction': int(prediction),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'probabilities_list': probabilities.tolist() if hasattr(probabilities, 'tolist') else [1-confidence, confidence],
                'model_type': 'ML',
                'stress_keywords': stress_matches,
                'positive_keywords': positive_matches,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
    
    # Demo mode
    stress_score = len(stress_matches) * 0.15 + min(len(text.split()) / 100, 0.3)
    prediction = 1 if stress_score > 0.5 else 0
    confidence = min(0.5 + stress_score * 0.5, 0.95)
    
    predicted_class = "Stress" if prediction == 1 else "No Stress"
    prob_dict = {'No Stress': 1 - confidence, 'Stress': confidence}
    
    return {
        'prediction': prediction,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': prob_dict,
        'probabilities_list': [1 - confidence, confidence],
        'model_type': 'Demo',
        'stress_keywords': stress_matches,
        'positive_keywords': positive_matches,
        'word_count': len(text.split()),
        'char_count': len(text)
    }


# ============================================
# SAVE TO DATABASE (Local + Supabase Cloud)
# ============================================
def save_prediction(db, cloud_mgr, user_info, text_input, result, session_id):
    """Save prediction to both local SQLite and Supabase cloud."""
    try:
        # Save to local SQLite
        record_id = db.insert_prediction(
            name=user_info.get('name'),
            age=user_info.get('age'),
            gender=user_info.get('gender'),
            user_input_text=text_input,
            predicted_class=result['predicted_class'],
            confidence_score=result['confidence'],
            all_class_probabilities=result['probabilities'],
            model_type=result['model_type'],
            session_id=session_id
        )
        
        # Save to Supabase cloud
        cloud_mgr.insert_prediction(
            name=user_info.get('name'),
            age=user_info.get('age'),
            gender=user_info.get('gender'),
            user_input_text=text_input,
            predicted_class=result['predicted_class'],
            confidence_score=result['confidence'],
            all_class_probabilities=result['probabilities'],
            model_type=result['model_type'],
            session_id=session_id
        )
        
        return record_id
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


# ============================================
# UI COMPONENTS
# ============================================
def render_hero():
    """Render hero section."""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧠 Mental Stress Detection</div>
        <div class="hero-subtitle">AI-Powered Analysis for Mental Wellness</div>
    </div>
    """, unsafe_allow_html=True)


def render_user_info():
    """Render user info section like app_enhanced_v2."""
    st.markdown('<div class="user-info-section">', unsafe_allow_html=True)
    st.markdown("### 👤 Your Information (Optional)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("Name", value=st.session_state.user_name, placeholder="Your name")
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.user_age, step=1)
    with col3:
        gender = st.selectbox(
            "Gender",
            ["Prefer not to say", "Male", "Female", "Other"],
            index=["Prefer not to say", "Male", "Female", "Other"].index(st.session_state.user_gender)
        )
    
    # Update session state
    st.session_state.user_name = name if name else None
    st.session_state.user_age = None if age == 0 else age
    st.session_state.user_gender = gender
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'name': st.session_state.user_name,
        'age': st.session_state.user_age,
        'gender': None if gender == "Prefer not to say" else gender
    }


def render_result(result: Dict):
    """Render prediction result."""
    prediction = result['prediction']
    confidence = result['confidence']
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-high">
            <h2>😰 Stress Detected</h2>
            <p style="font-size: 1.2rem;">Confidence: {confidence:.2%}</p>
            <p>Model: {result['model_type']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-low">
            <h2>😊 Low Stress</h2>
            <p style="font-size: 1.2rem;">Confidence: {confidence:.2%}</p>
            <p>Model: {result['model_type']}</p>
        </div>
        """, unsafe_allow_html=True)


def render_probability_chart(probabilities: Dict):
    """Render probability chart."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=list(probabilities.keys()),
                y=[v * 100 for v in probabilities.values()],
                marker_color=['#96e6a1', '#ff9a9e'],
                text=[f"{v:.1%}" for v in probabilities.values()],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='Probability Distribution',
            yaxis_title='Probability (%)',
            yaxis_range=[0, 100],
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        prob_df = pd.DataFrame({
            'Class': list(probabilities.keys()),
            'Probability': list(probabilities.values())
        })
        st.bar_chart(prob_df.set_index('Class'))


def render_keyword_analysis(stress_keywords, positive_keywords):
    """Render keyword analysis."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ Stress Indicators**")
        if stress_keywords:
            for kw in list(set(stress_keywords))[:8]:
                st.markdown(f'<span class="keyword-tag stress-tag">{kw}</span>', unsafe_allow_html=True)
        else:
            st.info("None found")
    
    with col2:
        st.markdown("**✅ Positive Indicators**")
        if positive_keywords:
            for kw in list(set(positive_keywords))[:8]:
                st.markdown(f'<span class="keyword-tag positive-tag">{kw}</span>', unsafe_allow_html=True)
        else:
            st.info("None found")


def render_feedback_section(db, cloud_mgr, record_id):
    """Render feedback buttons - Simple and Direct."""
    st.markdown("---")
    st.markdown("### 👍 Was this prediction helpful?")
    
    # Simple direct approach - check database first
    try:
        conn = db._get_connection()
        cursor = conn.cursor()
        
        # Check current feedback
        cursor.execute("SELECT user_feedback FROM user_predictions WHERE id = ?", (record_id,))
        result = cursor.fetchone()
        
        # If feedback exists in DB, show it
        if result and result[0] and str(result[0]).strip() not in ['', 'None', 'nan']:
            st.success(f"✅ You already gave feedback: **{result[0]}**")
            conn.close()
            return
        
        conn.close()
    except Exception as e:
        st.error(f"Error checking feedback: {e}")
    
    # Feedback buttons - Direct click handler
    st.markdown("**Tap to give feedback:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Yes", key=f"fb_yes_{record_id}", use_container_width=True):
            # Direct save
            try:
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE user_predictions SET user_feedback = ? WHERE id = ?", ("Yes", record_id))
                conn.commit()
                conn.close()
                st.success("✅ Feedback saved: Yes")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error: {e}")
            st.rerun()
    
    with col2:
        if st.button("❌ No", key=f"fb_no_{record_id}", use_container_width=True):
            # Direct save
            try:
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE user_predictions SET user_feedback = ? WHERE id = ?", ("No", record_id))
                conn.commit()
                conn.close()
                st.info("✅ Feedback saved: No")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            st.rerun()
    
    with col3:
        if st.button("🤔 Not Sure", key=f"fb_unsure_{record_id}", use_container_width=True):
            # Direct save
            try:
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE user_predictions SET user_feedback = ? WHERE id = ?", ("Unsure", record_id))
                conn.commit()
                conn.close()
                st.info("✅ Feedback saved: Not Sure")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            st.rerun()










def render_analytics(db):
    """Render analytics dashboard."""
    st.markdown("## 📊 Analytics Dashboard")
    
    stats = db.get_statistics()
    
    if not stats:
        st.warning("No data yet. Make some predictions!")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", stats.get('total_predictions', 0))
    with col2:
        st.metric("Today", stats.get('predictions_today', 0))
    with col3:
        avg = stats.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{avg:.1%}" if avg else "N/A")
    with col4:
        fb = sum(stats.get('feedback_statistics', {}).values())
        st.metric("Feedback", fb)
    
    # Charts
    if PLOTLY_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            class_dist = stats.get('class_distribution', {})
            if class_dist:
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    color=list(class_dist.keys()),
                    color_discrete_map={'Stress': '#ff9a9e', 'No Stress': '#96e6a1'},
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fb_stats = stats.get('feedback_statistics', {})
            if fb_stats:
                fig = px.bar(
                    x=list(fb_stats.keys()),
                    y=list(fb_stats.values()),
                    color=list(fb_stats.keys())
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    st.markdown("### Recent Predictions")
    recent = db.get_all_predictions(limit=10)
    if not recent.empty:
        # Prepare display dataframe
        display_cols = ['timestamp', 'name', 'predicted_class', 'confidence_score', 'user_feedback']
        display = recent[display_cols].copy()
        display.columns = ['Time', 'Name', 'Prediction', 'Confidence', 'Feedback']
        
        # Format feedback - handle None, NaN, empty string
        def format_feedback(val):
            if pd.isna(val) or val is None or str(val).strip() == "" or str(val).lower() == "nan":
                return "⏳ Not given"
            elif str(val).lower() == "yes":
                return "✅ Yes"
            elif str(val).lower() == "no":
                return "❌ No"
            elif str(val).lower() == "unsure":
                return "🤔 Unsure"
            else:
                return str(val)
        
        display['Feedback'] = display['Feedback'].apply(format_feedback)
        
        # Format confidence as percentage
        display['Confidence'] = display['Confidence'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        # Format timestamp
        display['Time'] = display['Time'].apply(lambda x: str(x)[:16] if pd.notna(x) else "N/A")
        
        # Style the dataframe
        def color_feedback(val):
            if "✅" in str(val):
                return 'background-color: #d4edda; color: #155724; font-weight: 600;'
            elif "❌" in str(val):
                return 'background-color: #f8d7da; color: #721c24; font-weight: 600;'
            elif "🤔" in str(val):
                return 'background-color: #fff3cd; color: #856404; font-weight: 600;'
            else:
                return 'background-color: #f8f9fa; color: #6c757d;'
        
        def color_prediction(val):
            if val == "Stress":
                return 'background-color: #f8d7da; color: #721c24; font-weight: 600;'
            else:
                return 'background-color: #d4edda; color: #155724; font-weight: 600;'
        
        styled_df = display.style.map(color_feedback, subset=['Feedback']).map(color_prediction, subset=['Prediction'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Show feedback statistics below the table
        st.markdown("**📊 Feedback Statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        
        # Count feedback values
        yes_count = (recent['user_feedback'].str.lower() == 'yes').sum() if 'user_feedback' in recent.columns else 0
        no_count = (recent['user_feedback'].str.lower() == 'no').sum() if 'user_feedback' in recent.columns else 0
        unsure_count = (recent['user_feedback'].str.lower() == 'unsure').sum() if 'user_feedback' in recent.columns else 0
        
        # Count not given (None, NaN, or empty)
        not_given_mask = recent['user_feedback'].isna() | (recent['user_feedback'] == '') | (recent['user_feedback'].isnull())
        not_given_count = not_given_mask.sum() if 'user_feedback' in recent.columns else len(recent)
        
        with col1:
            st.metric("✅ Yes", int(yes_count))
        with col2:
            st.metric("❌ No", int(no_count))
        with col3:
            st.metric("🤔 Unsure", int(unsure_count))
        with col4:
            st.metric("⏳ Not Given", int(not_given_count))




# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main app function."""
    init_session_state()
    
    # Initialize databases
    db = get_database()
    cloud_mgr = get_cloud_manager()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_type = st.sidebar.radio(
            "Select Model",
            ["Traditional ML", "Demo Mode", "BERT Model"]
        )
        
        st.markdown("---")
        
        # Navigation
        page = st.sidebar.radio("Navigation", ["🧠 Prediction", "📊 Dashboard"])
        
        st.markdown("---")
        
        # Cloud status
        if cloud_mgr.supabase_db.client:
            st.success("☁️ Supabase Connected")
        else:
            st.info("💻 Local Mode")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.caption("AI-powered stress detection. Data saved to local DB and Supabase cloud.")
    
    # Main content
    if page == "📊 Dashboard":
        render_analytics(db)
    else:
        # Prediction Page
        render_hero()
        
        # Load model
        ml_model_data = None
        if model_type == "Traditional ML":
            with st.spinner("Loading ML model..."):
                ml_model_data = load_ml_model()
                if ml_model_data is None:
                    st.warning("ML model not found. Using Demo mode.")
                    model_type = "Demo Mode"
                else:
                    st.success("✅ AI Model Loaded!")
        
        # User Info Section (like app_enhanced_v2)
        user_info = render_user_info()
        
        # Text Input Section
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter Your Text")
        
        text_input = st.text_area(
            "Type or paste text to analyze:",
            value=st.session_state.current_text,
            height=200,
            placeholder="Enter text expressing your thoughts, feelings, or experiences..."
        )
        st.session_state.current_text = text_input
        
        # Stats
        if text_input:
            c1, c2, c3 = st.columns(3)
            c1.metric("Characters", len(text_input))
            c2.metric("Words", len(text_input.split()))
            c3.metric("Sentences", len([s for s in text_input.split('.') if s.strip()]))
        
        # Sample buttons
        st.markdown("**📋 Try Sample Texts:**")
        cols = st.columns(3)
        samples = [
            ("😊 Low Stress", "I had a wonderful day today! The weather was perfect and I'm feeling really grateful."),
            ("😰 Medium Stress", "Work has been demanding lately. I have deadlines approaching and it's stressful."),
            ("😫 High Stress", "I can't take this anymore. Everything is overwhelming and my anxiety is through the roof.")
        ]
        for i, (label, text) in enumerate(samples):
            with cols[i]:
                if st.button(label, key=f"sample_{i}"):
                    st.session_state.current_text = text
                    st.rerun()
        
        # Analyze button
        if st.button("🔍 Analyze Stress Level", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result = predict_stress(
                        text_input,
                        model_data=ml_model_data,
                        model_type='ML' if model_type == "Traditional ML" and ml_model_data else 'Demo'
                    )
                    
                    # Save to both local and cloud
                    record_id = save_prediction(
                        db, cloud_mgr, user_info, text_input, 
                        result, st.session_state.session_id
                    )
                    st.session_state.last_record_id = record_id
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    render_result(result)
                    
                    # Probability chart
                    render_probability_chart(result['probabilities'])
                    
                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("📝 Words", result['word_count'])
                    c2.metric("📊 Characters", result['char_count'])
                    c3.metric("⚠️ Stress Keywords", len(result['stress_keywords']))
                    c4.metric("✅ Positive Keywords", len(result['positive_keywords']))
                    
                    # Keyword analysis
                    st.markdown("---")
                    render_keyword_analysis(result['stress_keywords'], result['positive_keywords'])
                    
                    # Feedback
                    if record_id:
                        st.markdown("---")
                        render_feedback_section(db, cloud_mgr, record_id)

            else:
                st.warning("Please enter some text to analyze.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #888;">
        <p>🧠 Mental Stress Detection Pro | For Educational Purposes</p>
        <p><a href="https://github.com/dreamtunesofficials-coder/Mental-Health" target="_blank">View Source Code</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
