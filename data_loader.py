"""
Data Loader Module for Mental Stress Detection
Handles data loading, preprocessing, and dataset preparation
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the mental stress dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def preprocess_text(text: str) -> str:
    """
    Preprocess text data by removing special characters,
    converting to lowercase, and removing extra whitespace.
    
    Args:
        text: Raw text string
        
    Returns:
        Preprocessed text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text
        stopwords: List of stopwords to remove
        
    Returns:
        Text with stopwords removed
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)


def get_default_stopwords() -> List[str]:
    """
    Get default English stopwords.
    
    Returns:
        List of stopwords
    """
    return [
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
        'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
        'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
        "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
        'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
        'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
        "won't", 'wouldn', "wouldn't"
    ]


def prepare_dataset(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for training and testing.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        label_column: Name of the label column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Handle missing values
    df = df.dropna(subset=[text_column, label_column])
    
    X = df['processed_text'].values
    y = df[label_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    filepath: str,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for training.
    
    Args:
        filepath: Path to the dataset
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = load_dataset(filepath)
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df[label_column]
    )
    
    return train_df, test_df


def get_label_distribution(df: pd.DataFrame, label_column: str = 'label') -> Dict:
    """
    Get the distribution of labels in the dataset.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        
    Returns:
        Dictionary with label counts
    """
    return df[label_column].value_counts().to_dict()


if __name__ == "__main__":
    # Example usage
    print("Data Loader Module for Mental Stress Detection")
    print("Available functions:")
    print("- load_dataset(filepath)")
    print("- preprocess_text(text)")
    print("- prepare_dataset(df, text_column, label_column)")
    print("- load_and_prepare_data(filepath)")
    print("- get_label_distribution(df)")
