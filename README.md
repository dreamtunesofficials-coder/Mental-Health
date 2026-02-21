# 🧠 Mental Stress Detection Pro

🌐 **Live App:** [https://healthstresscheckup.streamlit.app/](https://healthstresscheckup.streamlit.app/)

An advanced AI-powered application for detecting mental stress from text data using Machine Learning and Deep Learning models. Features **continuous learning** from user feedback to improve accuracy over time!

---

## ✨ What's New in Pro Version

- 🔄 **Auto-Retraining**: Model automatically improves with user feedback
- ☁️ **Cloud Database**: Supabase integration for data persistence
- 📊 **Analytics Dashboard**: Real-time insights and statistics
- 🎨 **Modern UI**: Professional glassmorphism design
- 👍 **Feedback System**: Users can validate predictions
- 📈 **Continuous Learning**: Better accuracy with more usage

---

## 🚀 Features

### Core Features
- **🧠 AI Models**: Traditional ML (Random Forest, SVM, Logistic Regression) + BERT
- **📝 Text Analysis**: Stress detection from user input
- **🎯 Confidence Scores**: Probability distribution for predictions
- **🔑 Keyword Analysis**: Automatic stress/positive keyword detection

### Pro Features
- **☁️ Cloud Sync**: All data saved to Supabase
- **👍 Feedback Loop**: Users correct predictions
- **🔄 Auto-Retrain**: Model improves automatically
- **📊 Analytics**: Usage statistics and trends
- **🎨 Modern UI**: Beautiful, responsive design

---

## 📁 Project Structure

```
mental-stress-detection/
├── app_professional_v4.py      🌟 Main Pro Application
├── auto_retrain_from_supabase.py  🔄 Auto-Retraining Engine
├── cloud_database.py           ☁️ Cloud Database Manager
├── database.py                 💾 Local SQLite Database
├── train_ml_models.py          🤖 ML Model Training
├── train_bert.py               🧠 BERT Fine-tuning
├── data_loader.py              📂 Data Preprocessing
├── feature_engineering.py      🔧 Feature Extraction
├── requirements.txt            📦 Dependencies
├── Dockerfile                  🐳 Docker Config
├── README.md                   📖 Documentation
├── models/                     🧠 Saved Models
├── data/                       💾 Datasets & Database
├── notebooks/                  📓 Jupyter Notebooks
└── results/                    📊 Evaluation Results
```

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/dreamtunesofficials-coder/Mental-Health.git
cd mental-stress-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Supabase (Optional - for cloud features)
1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Get API credentials
4. Add to `.streamlit/secrets.toml`:
```toml
[supabase]
url = "your-supabase-url"
key = "your-supabase-key"
```

---

## 💻 Usage

### 🌟 Run Pro Version (Recommended)
```bash
streamlit run app_professional_v4.py
```

App will open at: `http://localhost:8501`

### 📱 App Navigation

| Tab | Description |
|-----|-------------|
| 🧠 **Prediction** | Main stress detection interface |
| 📊 **Dashboard** | Analytics and statistics |
| 🔄 **Retrain** | Model retraining status & manual trigger |

---

## 🔄 How Auto-Retraining Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  User Input │────▶│  Prediction │────▶│   Feedback  │
│    Text     │     │   (AI/ML)     │     │ (Yes/No)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────┐
              │  Supabase   │
              │   Cloud DB  │
              └──────┬──────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  5+ Feedbacks?  │───▶│  Auto-Retrain   │
│   Trigger       │    │    Model        │
└─────────────────┘    └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Better Model   │
                       │   Deployed!     │
                       └─────────────────┘
```

---

## 📊 Dataset Format

CSV file with columns:
- `text`: Input text (user's thoughts/feelings)
- `label`: Target (0 = No Stress, 1 = Stress)

Example:
```csv
text,label
"I feel calm and relaxed today.",0
"I'm very stressed about my work.",1
"Everything is overwhelming me.",1
"Had a great day with friends!",0
```

---

## 🔧 Configuration

### Feature Extraction
```python
from feature_engineering import FeatureExtractor

extractor = FeatureExtractor(
    max_features=5000,      # Max TF-IDF features
    ngram_range=(1, 2),     # Unigrams + Bigrams
    min_df=2,               # Min document frequency
    max_df=0.95,            # Max document frequency
    use_svd=True,           # Dimensionality reduction
    n_components=100        # SVD components
)
```

### Auto-Retrain Settings
```python
# In app_professional_v4.py or auto_retrain_from_supabase.py

# Minimum feedbacks to trigger retrain
min_feedback_count = 5

# Models to train
models = ['RandomForest', 'GradientBoosting', 'LogisticRegression']
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t mental-stress-detection .
```

### Run Container
```bash
docker run -p 8501:8501 mental-stress-detection
```

Access at: `http://localhost:8501`

---

## ☁️ Streamlit Cloud Deployment

### 1. Push to GitHub
```bash
git add .
git commit -m "Pro version with auto-retrain"
git push origin main
```

### 2. Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Select repository
4. Deploy!

### 3. Add Secrets (Supabase)
In Streamlit Cloud dashboard:
- Go to **Settings** → **Secrets**
- Add:
```toml
[supabase]
url = "your-supabase-url"
key = "your-supabase-key"
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~85% | ~84% | ~86% | ~85% |
| BERT | ~90% | ~89% | ~91% | ~90% |
| **Retrained** | **~92%** | **~91%** | **~93%** | **~92%** |

*Performance improves with more user feedback!*

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is for **educational purposes only**.

---

## ⚠️ Disclaimer

> **Important**: This application is for demonstration and educational purposes only. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. 
>
> If you're experiencing mental health issues, please consult a qualified healthcare professional immediately.

---

## 🔗 Resources & Credits

- [Streamlit](https://streamlit.io/) - Web app framework
- [HuggingFace Transformers](https://huggingface.co/transformers/) - BERT models
- [scikit-learn](https://scikit-learn.org/) - ML algorithms
- [Supabase](https://supabase.com/) - Cloud database
- [Plotly](https://plotly.com/) - Interactive charts

---

## 📞 Support

For issues or questions:
- Open an [Issue](https://github.com/dreamtunesofficials-coder/Mental-Health/issues)
- Contact: [Your Contact Info]

---

**Made with ❤️ for mental health awareness**

⭐ Star this repo if you find it helpful!
