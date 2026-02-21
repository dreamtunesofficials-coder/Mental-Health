"""
BERT Fine-tuning Module for Mental Stress Detection
Fine-tunes BERT models for stress classification
"""

import numpy as np
import pandas as pd
import torch
import pickle
import os
from typing import Dict, Tuple, Optional, List
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


class BertTrainer:
    """
    Trainer class for BERT-based stress detection.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 128
    ):
        """
        Initialize BERT trainer.
        
        Args:
            model_name: Name of the BERT model
            num_labels: Number of classification labels
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.is_fitted = False
    
    def _prepare_dataset(
        self, 
        texts: List[str], 
        labels: List[int]
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            texts: List of text strings
            labels: List of labels
            
        Returns:
            HuggingFace Dataset
        """
        dataset = Dataset.from_dict({
            'text': texts,
            'label': labels
        })
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return dataset
    
    def _compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted')
        }
    
    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        output_dir: str = './models/bert_output',
        logging_dir: str = './logs/bert_logs'
    ) -> 'BertTrainer':
        """
        Fine-tune BERT model.
        
        Args:
            X_train: Training texts
            X_val: Validation texts
            y_train: Training labels
            y_val: Validation labels
            output_dir: Directory to save model
            logging_dir: Directory for logs
            
        Returns:
            Self
        """
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Prepare datasets
        train_dataset = self._prepare_dataset(
            X_train.tolist(), 
            y_train.tolist()
        )
        val_dataset = self._prepare_dataset(
            X_val.tolist(), 
            y_val.tolist()
        )
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=logging_dir,
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            report_to='none'
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model
        self.trainer.train()
        
        self.is_fitted = True
        return self
    
    def predict(self, texts: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            texts: Text data
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Prepare dataset
        dataset = self._prepare_dataset(texts.tolist(), [0] * len(texts))
        
        # Get predictions
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        
        return preds
    
    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Text data
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Prepare dataset
        dataset = self._prepare_dataset(texts.tolist(), [0] * len(texts))
        
        # Get predictions
        predictions = self.trainer.predict(dataset)
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions.predictions)
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        return probs
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
    
    def save(self, output_dir: str) -> None:
        """
        Save the trained model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load(self, model_dir: str) -> 'BertTrainer':
        """
        Load a trained model.
        
        Args:
            model_dir: Directory to load model from
            
        Returns:
            Self
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        
        self.is_fitted = True
        return self


class StressBERTClassifier:
    """
    Simplified BERT classifier for stress detection.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 128
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of BERT model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def _load_model(self):
        """Load model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict stress level for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.model is None:
            self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return {
            'prediction': pred,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy()
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict stress levels for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results


def train_bert_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    model_name: str = 'bert-base-uncased',
    num_epochs: int = 3,
    batch_size: int = 16,
    output_dir: str = './models/bert_stress'
) -> BertTrainer:
    """
    Train a BERT model for stress detection.
    
    Args:
        X_train: Training texts
        X_val: Validation texts
        y_train: Training labels
        y_val: Validation labels
        model_name: Name of BERT model
        num_epochs: Number of epochs
        batch_size: Batch size
        output_dir: Output directory
        
    Returns:
        Trained BERT trainer
    """
    trainer = BertTrainer(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    trainer.fit(
        X_train, X_val, y_train, y_val,
        output_dir=output_dir
    )
    
    return trainer


def plot_training_logs(logs: List[Dict], save_path: Optional[str] = None):
    """
    Plot training logs.
    
    Args:
        logs: List of training logs
        save_path: Path to save the plot
    """
    epochs = [log['epoch'] for log in logs]
    train_loss = [log['loss'] for log in logs]
    eval_loss = [log['eval_loss'] for log in logs]
    eval_f1 = [log['eval_f1'] for log in logs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(epochs, train_loss, label='Train Loss')
    axes[0].plot(epochs, eval_loss, label='Eval Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Evaluation Loss')
    axes[0].legend()
    
    # Plot F1 score
    axes[1].plot(epochs, eval_f1, label='Eval F1', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Evaluation F1 Score')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


if __name__ == "__main__":
    print("BERT Fine-tuning Module for Mental Stress Detection")
    print("Available classes and functions:")
    print("- BertTrainer: Full BERT training with HuggingFace Trainer")
    print("- StressBERTClassifier: Simplified classifier")
    print("- train_bert_model(X_train, X_val, y_train, y_val)")
    print("- plot_training_logs(logs)")
