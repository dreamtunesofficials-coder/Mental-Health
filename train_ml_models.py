"""
Traditional ML Training Module for Mental Stress Detection
Trains and evaluates Logistic Regression, SVM, and Random Forest models
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import FeatureExtractor
from data_loader import load_dataset, prepare_dataset
from typing import Dict, Tuple, Any, Optional


class MLModelTrainer:
    """
    Trainer class for traditional ML models.
    """
    
    def __init__(
        self,
        model_type: str = 'logistic_regression',
        vectorizer_type: str = 'tfidf',
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train
            vectorizer_type: Type of vectorizer (tfidf or count)
            max_features: Maximum number of features
            ngram_range: N-gram range for vectorizer
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.model = None
        self.vectorizer = None
        self.is_fitted = False
        self.training_history = {}
    
    def _get_model(self):
        """Get the model instance based on model_type."""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf', 
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=1.0)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return models[self.model_type]
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        use_svd: bool = False,
        n_components: int = 100
    ) -> 'MLModelTrainer':
        """
        Fit the vectorizer and model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            use_svd: Whether to use SVD for dimensionality reduction
            n_components: Number of SVD components
            
        Returns:
            Self
        """
        # Initialize and fit vectorizer
        self.vectorizer = FeatureExtractor(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            use_svd=use_svd,
            n_components=n_components
        )
        
        X_train_features = self.vectorizer.fit_transform(X_train)
        
        # Get and fit model
        self.model = self._get_model()
        self.model.fit(X_train_features, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Text data
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_features = self.vectorizer.transform(X)
        return self.model.predict(X_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Text data
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_features = self.vectorizer.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_features)
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_features)
            # Convert to pseudo-probabilities using softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            raise RuntimeError("Model does not support probability predictions")
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Try to get ROC-AUC if possible
        try:
            y_proba = self.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_test, y_proba, multi_class='ovr'
                )
        except:
            metrics['roc_auc'] = None
        
        return metrics
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Text data
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        X_features = self.vectorizer.transform(X)
        
        scores = cross_val_score(
            self.model, X_features, y, cv=cv, scoring='f1_weighted'
        )
        
        return {
            'cv_scores': scores,
            'mean_f1': np.mean(scores),
            'std_f1': np.std(scores)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model and vectorizer.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'model_type': self.model_type,
                'vectorizer_type': self.vectorizer_type,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str) -> 'MLModelTrainer':
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.model_type = data['model_type']
        self.vectorizer_type = data['vectorizer_type']
        self.max_features = data['max_features']
        self.ngram_range = data['ngram_range']
        self.is_fitted = data['is_fitted']
        
        return self


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_types: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
        model_types: List of model types to train
        
    Returns:
        Dictionary of model results
    """
    if model_types is None:
        model_types = ['logistic_regression', 'svm', 'random_forest']
    
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        
        trainer = MLModelTrainer(model_type=model_type)
        trainer.fit(X_train, y_train)
        
        metrics = trainer.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model comparison.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    plt.figure(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics]
        plt.bar(x + i * width, values, width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width * (len(models) - 1) / 2, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


if __name__ == "__main__":
    print("ML Model Training Module for Mental Stress Detection")
    print("Available functions:")
    print("- train_and_evaluate_models(X_train, X_test, y_train, y_test)")
    print("- plot_confusion_matrix(y_true, y_pred)")
    print("- plot_model_comparison(results)")
