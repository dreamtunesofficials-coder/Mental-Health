# Mental Stress Detection

A machine learning and deep learning project for detecting mental stress from text data using traditional ML models (Logistic Regression, SVM, Random Forest) and BERT-based models.

## 📁 Project Structure

```
mental-stress-detection/
├── app.py                  # Streamlit web application
├── train_ml_models.py      # Traditional ML training
├── train_bert.py           # BERT fine-tuning
├── data_loader.py          # Data preprocessing
├── feature_engineering.py  # TF-IDF and feature extraction
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── README.md               # Project documentation
├── models/                 # Saved models directory
├── data/                   # Dataset directory
├── notebooks/              # Jupyter notebooks for EDA
└── results/                # Evaluation results and plots
```

## 🚀 Features

- **Traditional ML Models**: Logistic Regression, SVM, Random Forest, Gradient Boosting, Naive Bayes
- **Deep Learning**: BERT-based model for text classification
- **Web Interface**: Interactive Streamlit application
- **Feature Engineering**: TF-IDF vectorization with SVD dimensionality reduction
- **Additional Features**: Text statistics (word count, sentence count, etc.)
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, ROC-AUC
- **Docker Support**: Containerized application

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- PyTorch 2.0+
- Transformers 4.30+
- Streamlit 1.28+
- scikit-learn 1.3.0+

See `requirements.txt` for complete list of dependencies.

## 🛠️ Installation

1. Clone the repository:
```
bash
git clone <repository-url>
cd mental-stress-detection
```

2. Create a virtual environment:
```
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Streamlit App

```
bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Training Traditional ML Models

```
python
from train_ml_models import MLModelTrainer, train_and_evaluate_models
from data_loader import prepare_dataset
import pandas as pd

# Load and prepare data
df = pd.read_csv('data/stress_data.csv')
X_train, X_test, y_train, y_test = prepare_dataset(df)

# Train multiple models
results = train_and_evaluate_models(
    X_train, X_test, y_train, y_test,
    model_types=['logistic_regression', 'svm', 'random_forest']
)
```

### Training BERT Model

```
python
from train_bert import BertTrainer, train_bert_model
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Train BERT model
trainer = train_bert_model(
    X_train, X_val, y_train, y_val,
    model_name='bert-base-uncased',
    num_epochs=3,
    output_dir='./models/bert_stress'
)
```

## 📊 Dataset Format

The dataset should be a CSV file with at least two columns:
- `text`: Input text data
- `label`: Target variable (0 for no stress, 1 for stress)

Example:
```
csv
text,label
"I feel calm and relaxed today.",0
"I'm very stressed about my work.",1
```

## 🔧 Configuration

### Feature Extraction Parameters

```
python
from feature_engineering import FeatureExtractor

extractor = FeatureExtractor(
    max_features=5000,      # Maximum TF-IDF features
    ngram_range=(1, 2),    # Unigrams and bigrams
    min_df=2,              # Minimum document frequency
    max_df=0.95,           # Maximum document frequency
    use_svd=True,          # Use SVD for dimensionality reduction
    n_components=100       # Number of SVD components
)
```

### Model Parameters

```
python
from train_ml_models import MLModelTrainer

trainer = MLModelTrainer(
    model_type='logistic_regression',
    vectorizer_type='tfidf',
    max_features=5000,
    ngram_range=(1, 2)
)
```

## 🐳 Docker

### Build the Docker image

```
bash
docker build -t mental-stress-detection .
```

### Run the container

```
bash
docker run -p 8501:8501 mental-stress-detection
```

The application will be available at `http://localhost:8501`

## 📈 Results

Model performance can be saved and visualized using the utility functions:

```
python
from train_ml_models import plot_confusion_matrix, plot_model_comparison

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png')

# Compare models
plot_model_comparison(results, save_path='results/model_comparison.png')
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is for educational purposes. 

## ⚠️ Disclaimer

This application is for demonstration and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing mental health issues, please consult a qualified healthcare professional.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
