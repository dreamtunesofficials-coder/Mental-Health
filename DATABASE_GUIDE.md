# Mental Stress Detection - Database Integration Guide

## Overview

This guide explains how to use the enhanced Streamlit application with database integration for storing user predictions and enabling model retraining.

## Database Choice: SQLite

**Why SQLite?**
- ✅ **Zero Configuration**: No server setup required
- ✅ **Serverless**: Perfect for Streamlit Cloud deployment
- ✅ **Portable**: Single file database (.db)
- ✅ **Python Native**: Built into Python standard library
- ✅ **ACID Compliant**: Reliable transactions

**Alternatives Considered:**
- **PostgreSQL**: Better for high-traffic production, but requires server setup
- **Firestore**: Good for serverless, but adds vendor lock-in and costs

## Database Schema

### Table: `user_predictions`

| Column | Data Type | Description |
|--------|-----------|-------------|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Unique identifier |
| `timestamp` | DATETIME DEFAULT CURRENT_TIMESTAMP | When prediction was made |
| `name` | TEXT | User's name (optional) |
| `age` | INTEGER | User's age (optional) |
| `gender` | TEXT | User's gender (optional) |
| `user_input_text` | TEXT NOT NULL | Text submitted for analysis |
| `predicted_class` | TEXT NOT NULL | Prediction result ("Stress" or "No Stress") |
| `confidence_score` | REAL NOT NULL | Model confidence (0-1) |
| `all_class_probabilities` | TEXT NOT NULL | JSON string of all class probabilities |
| `user_feedback` | TEXT | Feedback: "Yes", "No", or "Unsure" |
| `model_type` | TEXT | Model used: "ML", "BERT", or "Demo" |
| `session_id` | TEXT | Unique session identifier |

## Setup Instructions

### 1. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn scipy
```

### 2. Run the Enhanced Application

```bash
streamlit run app_enhanced_v2.py
```

The database will be automatically created at `./data/stress_detection.db` on first run.

### 3. Database Location

- **Local Development**: `./data/stress_detection.db`
- **Streamlit Cloud**: Data persists for the session, use external storage for production

## Features

### 1. User Information Collection (Optional)
- Name, Age, Gender fields in sidebar
- All fields are optional - users can remain anonymous
- Data helps with demographic analysis

### 2. Automatic Prediction Logging
Every prediction is automatically saved with:
- User input text
- Prediction result and confidence
- All class probabilities
- Timestamp and session ID
- Model type used

### 3. Feedback Collection
After each prediction, users can provide feedback:
- ✅ Yes (prediction was accurate)
- ❌ No (prediction was incorrect)
- 🤔 Unsure

### 4. Analytics Dashboard
Access via sidebar navigation:
- Total predictions count
- Predictions today
- Average confidence score
- Class distribution (Stress vs No Stress)
- Feedback statistics
- Recent predictions table
- Export to CSV functionality

## Using Collected Data for Model Retraining

### Step 1: Export Data
```python
from database import get_database

db = get_database()
db.export_to_csv('training_data.csv')
```

### Step 2: Load Feedback Data
```python
# Get only records with user feedback
feedback_df = db.get_feedback_data()
```

### Step 3: Create Refined Training Dataset
```python
import pandas as pd

# Load original training data
original_data = pd.read_csv('data/stress_data.csv')

# Load collected data with positive feedback
collected_data = db.get_feedback_data()
collected_data = collected_data[collected_data['user_feedback'] == 'Yes']

# Combine datasets
combined = pd.concat([original_data, collected_data[['user_input_text', 'predicted_class']]])
```

### Step 4: Retrain Model
Use the `retrain_with_collected_data.py` script (included) to retrain with new data.

## Security & Privacy

### Local Development
- Database is stored locally in `./data/`
- No external connections required
- User data stays on your machine

### Streamlit Cloud Deployment
For production deployment:

1. **Use Secrets for Sensitive Data**:
   ```toml
   # .streamlit/secrets.toml
   [database]
   path = "/mnt/data/stress_detection.db"
   ```

2. **Data Retention Policy**:
   - Inform users about data collection
   - Provide option to anonymize
   - Implement data deletion if required

3. **Access Control**:
   - Analytics dashboard should be password protected
   - Use Streamlit authentication for admin pages

## API Reference

### Database Class Methods

```python
from database import StressDetectionDatabase

# Initialize
db = StressDetectionDatabase('./data/stress_detection.db')

# Insert prediction
record_id = db.insert_prediction(
    name="John Doe",
    age=30,
    gender="Male",
    user_input_text="I'm feeling stressed about work",
    predicted_class="Stress",
    confidence_score=0.85,
    all_class_probabilities={"No Stress": 0.15, "Stress": 0.85},
    model_type="ML",
    session_id="uuid-here"
)

# Update feedback
db.update_feedback(record_id, "Yes")

# Get all predictions
df = db.get_all_predictions(limit=100)

# Get statistics
stats = db.get_statistics()

# Export to CSV
db.export_to_csv('export.csv')

# Get feedback data for retraining
feedback_df = db.get_feedback_data()
```

## Workflow for Continuous Improvement

### Monthly Retraining Workflow

1. **Collect Data**: Users interact with app, predictions are logged
2. **Review Feedback**: Check `user_feedback` column for validated predictions
3. **Export Data**: Use analytics dashboard to export recent predictions
4. **Retrain Model**: Run retraining script with combined dataset
5. **Evaluate**: Test new model performance
6. **Deploy**: Replace old model with improved version
7. **Monitor**: Track performance improvements in analytics

### A/B Testing Concept

To test new model versions:

1. Deploy both models
2. Randomly assign users to Model A or Model B
3. Track predictions from each model separately
4. Compare feedback scores between models
5. Promote better performing model

## Troubleshooting

### Database Locked Error
- SQLite doesn't support concurrent writes
- Ensure only one Streamlit instance accesses the DB

### Data Not Saving
- Check `./data/` directory exists and is writable
- Check disk space
- Review error messages in terminal

### Model Not Loading
- Verify model files exist in `./models/` directory
- Check model file compatibility with current scikit-learn version
- Use alternative models if primary model fails

## Future Enhancements

1. **PostgreSQL Migration**: For high-traffic production use
2. **Data Encryption**: Encrypt sensitive user information
3. **Automated Retraining**: Schedule monthly retraining jobs
4. **Advanced Analytics**: Time-series analysis, trend detection
5. **User Accounts**: Allow users to view their prediction history

## Support

For issues or questions:
- Check the GitHub repository
- Review the code comments
- Examine the example scripts provided
