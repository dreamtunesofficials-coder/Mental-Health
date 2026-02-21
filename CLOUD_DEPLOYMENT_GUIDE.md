# 🚀 Streamlit Cloud Deployment Guide

## Overview
This guide explains how to deploy the Mental Stress Detection app to **Streamlit Cloud** (share.streamlit.io) with full database support.

## What Gets Stored in the Database?

When users use your app on Streamlit Cloud, the following data is automatically stored:

### User Information (Optional)
- **Name** - User's name (if provided)
- **Age** - User's age (if provided)
- **Gender** - User's gender (if provided)

### Prediction Data (Automatic)
- **Timestamp** - When the prediction was made
- **Input Text** - The text user submitted
- **Predicted Class** - "Stress" or "No Stress"
- **Confidence Score** - Model's confidence (0-100%)
- **All Probabilities** - JSON with probabilities for all classes
- **Model Type** - Which model was used (ML/Demo/BERT)
- **Session ID** - Unique session identifier
- **IP Address** - User's IP (for analytics)
- **User Agent** - Browser info (for analytics)

### User Feedback (Optional)
- **Feedback** - "Yes", "No", or "Unsure" (thumbs up/down buttons)

## Deployment Steps

### 1. Prepare Your Repository

Ensure your GitHub repository has these files:
```
mental-stress-detection/
├── app_enhanced_v2.py      # Main application
├── cloud_database.py         # Cloud-compatible database
├── data_loader.py           # Text preprocessing
├── feature_engineering.py   # Feature extraction
├── train_bert.py           # BERT model (optional)
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── data/                  # Database will be created here
```

### 2. Update requirements.txt

Add these dependencies for cloud deployment:

```txt
# Core dependencies
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Database (optional - for PostgreSQL support)
# psycopg2-binary>=2.9.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0
requests>=2.31.0

# For text processing
nltk>=3.8.0
```

### 3. Create .streamlit/config.toml

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
serverAddress = "localhost"
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### 4. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Select branch (main/master)
6. Set main file path: `mental-stress-detection/app_enhanced_v2.py`
7. Click "Deploy"

### 5. Verify Database is Working

After deployment:
1. Open your app URL (e.g., `https://your-app-name.streamlit.app`)
2. Make a test prediction
3. Check the Analytics Dashboard in the sidebar
4. Verify data appears in statistics

## How Data Storage Works on Cloud

### Local SQLite Database
- **Location**: `./data/stress_detection.db`
- **Persistence**: Data persists as long as the app is running
- **Limitation**: Data resets when app restarts (Streamlit Cloud limitation)

### Session State Backup
- **Location**: `st.session_state.cloud_predictions`
- **Purpose**: Temporary backup during user session
- **Limitation**: Lost when user closes browser

### For Permanent Cloud Storage

To keep data permanently on Streamlit Cloud, you have these options:

#### Option 1: PostgreSQL Database (Recommended)
1. Create a free PostgreSQL database on:
   - [Supabase](https://supabase.com) (Recommended - 500MB free)
   - [Neon](https://neon.tech) (300MB free)
   - [ElephantSQL](https://www.elephantsql.com) (20MB free)

2. Add connection to `cloud_database.py`:
```python
# In cloud_database.py, uncomment PostgreSQL section
```

3. Store credentials in Streamlit secrets:
```toml
# .streamlit/secrets.toml
[postgres]
host = "your-db-host"
port = 5432
dbname = "your-db-name"
user = "your-username"
password = "your-password"
```

#### Option 2: Supabase (Serverless)
1. Create account at [supabase.com](https://supabase.com)
2. Create a new project
3. Get API keys
4. Add to secrets:
```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"
```

#### Option 3: Google Sheets (Simple)
1. Create a Google Sheet
2. Share with service account
3. Use `gspread` library to store data

## Viewing Collected Data

### Method 1: Analytics Dashboard (Built-in)
- Open your deployed app
- Click "📊 Analytics Dashboard" in sidebar
- View statistics, charts, and recent predictions
- Export data to CSV

### Method 2: Local Database Viewer
```bash
python view_database.py
```

### Method 3: Download from Cloud
1. Go to Analytics Dashboard
2. Click "Export to CSV"
3. Download the file

## Retraining Model with Collected Data

### Automatic Retraining
Run the retraining script locally:
```bash
python retrain_with_collected_data.py --use-feedback --min-confidence 0.7
```

This will:
1. Load original training data
2. Add collected predictions with feedback
3. Train new models
4. Select best model
5. Save as `best_model.pkl`

### Deploy Updated Model
1. Replace `models/best_model.pkl` with new model
2. Push to GitHub
3. Streamlit Cloud will auto-update

## Security & Privacy

### What's Collected
✅ User input text  
✅ Prediction results  
✅ Confidence scores  
✅ Optional user info (name, age, gender)  
✅ Feedback (yes/no/unsure)  
✅ Timestamp  

### What's NOT Collected
❌ No sensitive personal data  
❌ No passwords  
❌ No location data (only IP for analytics)  

### Privacy Best Practices
1. Add a privacy notice in your app
2. Make user info fields optional
3. Allow users to use anonymously
4. Don't store data longer than necessary
5. Use HTTPS (automatic on Streamlit Cloud)

## Troubleshooting

### Issue: Database not persisting on Cloud
**Solution**: This is expected! Streamlit Cloud resets filesystem on restart. Use PostgreSQL/Supabase for permanent storage.

### Issue: Data not showing in dashboard
**Solution**: 
1. Check if predictions were made
2. Refresh the page
3. Check Analytics Dashboard tab

### Issue: App crashes on startup
**Solution**:
1. Check `requirements.txt` has all dependencies
2. Verify `cloud_database.py` is in repository
3. Check Streamlit Cloud logs

## Free Tier Limits

### Streamlit Cloud (Free)
- **Apps**: Unlimited public apps
- **Memory**: 1 GB RAM
- **Storage**: Ephemeral (resets on restart)
- **Uptime**: App sleeps after inactivity

### Supabase (Free Tier)
- **Database**: 500 MB
- **Bandwidth**: 2 GB
- **Connections**: 30 concurrent

### Recommended for Production
- Supabase Pro ($25/month) for 8GB database
- Or self-hosted PostgreSQL

## Summary

✅ **App is Cloud-Ready**: Works on both local and Streamlit Cloud  
✅ **Data Collection**: All predictions stored automatically  
✅ **User Feedback**: Thumbs up/down buttons work  
✅ **Analytics**: Built-in dashboard shows statistics  
✅ **Export**: Download data as CSV anytime  

**Next Steps**:
1. Deploy to Streamlit Cloud
2. Share your app URL
3. Collect user data
4. Retrain model monthly for better accuracy
