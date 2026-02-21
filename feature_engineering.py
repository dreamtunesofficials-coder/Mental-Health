"""
Feature Engineering Module for Mental Stress Detection
Handles TF-IDF vectorization and feature extraction
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Optional
import pickle
import os


class FeatureExtractor:
    """
    Feature extractor for text data using TF-IDF and other techniques.
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_svd: bool = False,
        n_components: int = 100
    ):
        """
        Initialize the feature extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_svd: Whether to apply SVD for dimensionality reduction
            n_components: Number of components for SVD
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_svd = use_svd
        self.n_components = n_components
        
        self.tfidf_vectorizer = None
        self.svd = None
        self.is_fitted = False
    
    def fit(self, texts: np.ndarray) -> 'FeatureExtractor':
        """
        Fit the feature extractor on training data.
        
        Args:
            texts: Array of text strings
            
        Returns:
            Self
        """
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit TF-IDF
        self.tfidf_vectorizer.fit(texts)
        
        # Apply SVD if requested
        if self.use_svd:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
        return self
    
    def transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Transform text data to feature vectors.
        
        Args:
            texts: Array of text strings
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted before transform")
        
        # Transform using TF-IDF
        features = self.tfidf_vectorizer.transform(texts)
        
        # Apply SVD if enabled
        if self.use_svd:
            features = self.svd.transform(features)
        
        return features.toarray() if hasattr(features, 'toarray') else features
    
    def fit_transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: Array of text strings
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, filepath: str) -> None:
        """
        Save the feature extractor to disk.
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'svd': self.svd,
                'is_fitted': self.is_fitted,
                'config': {
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'min_df': self.min_df,
                    'max_df': self.max_df,
                    'use_svd': self.use_svd,
                    'n_components': self.n_components
                }
            }, f)
    
    def load(self, filepath: str) -> 'FeatureExtractor':
        """
        Load the feature extractor from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.svd = data['svd']
        self.is_fitted = data['is_fitted']
        config = data['config']
        
        self.max_features = config['max_features']
        self.ngram_range = config['ngram_range']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        self.use_svd = config['use_svd']
        self.n_components = config['n_components']
        
        return self


class CountVectorizerExtractor:
    """
    Count-based vectorizer for bag-of-words features.
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize the count vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
    
    def fit(self, texts: np.ndarray) -> 'CountVectorizerExtractor':
        """
        Fit the vectorizer.
        
        Args:
            texts: Array of text strings
        """
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english'
        )
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Transform text to count vectors.
        
        Args:
            texts: Array of text strings
            
        Returns:
            Count matrix
        """
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        self.fit(texts)
        return self.transform(texts)


def extract_additional_features(texts: np.ndarray) -> np.ndarray:
    """
    Extract additional text-based features.
    
    Args:
        texts: Array of text strings
        
    Returns:
        Array of additional features
    """
    features = []
    
    for text in texts:
        text_features = []
        
        # Text length
        text_features.append(len(text))
        
        # Word count
        word_count = len(text.split())
        text_features.append(word_count)
        
        # Average word length
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        text_features.append(avg_word_len)
        
        # Sentence count
        sentences = text.split('.')
        text_features.append(len([s for s in sentences if s.strip()]))
        
        # Punctuation count
        punctuation_count = sum(1 for c in text if c in '.,!?;:"\'')
        text_features.append(punctuation_count)
        
        # Uppercase ratio
        uppercase_count = sum(1 for c in text if c.isupper())
        text_features.append(uppercase_count / max(len(text), 1))
        
        # Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        text_features.append(digit_count / max(len(text), 1))
        
        features.append(text_features)
    
    return np.array(features)


def combine_features(
    tfidf_features: np.ndarray,
    additional_features: np.ndarray
) -> np.ndarray:
    """
    Combine TF-IDF features with additional features.
    
    Args:
        tfidf_features: TF-IDF feature matrix
        additional_features: Additional text features
        
    Returns:
        Combined feature matrix
    """
    return np.hstack([tfidf_features, additional_features])


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module for Mental Stress Detection")
    print("Available classes and functions:")
    print("- FeatureExtractor: TF-IDF with optional SVD")
    print("- CountVectorizerExtractor: Bag-of-words features")
    print("- extract_additional_features(texts)")
    print("- combine_features(tfidf_features, additional_features)")
