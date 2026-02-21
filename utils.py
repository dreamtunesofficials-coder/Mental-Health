"""
Utility Functions for Mental Stress Detection
Helper functions for data processing, visualization, and model management
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def save_results(results: Dict, filepath: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


def load_results(filepath: str) -> Dict:
    """
    Load results from a JSON file.
    
    Args:
        filepath: Path to load the results from
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_model(model: Any, filepath: str) -> None:
    """
    Save a model using pickle.
    
    Args:
        model: Model to save
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """
    Load a model from pickle file.
    
    Args:
        filepath: Path to load the model from
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_output_directories(base_dir: str = './') -> Dict[str, str]:
    """
    Create output directories for the project.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Dictionary of created directories
    """
    dirs = {
        'models': os.path.join(base_dir, 'models'),
        'data': os.path.join(base_dir, 'data'),
        'logs': os.path.join(base_dir, 'logs'),
        'results': os.path.join(base_dir, 'results'),
        'notebooks': os.path.join(base_dir, 'notebooks')
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in history:
            axes[1].plot(history[metric], label=f'Training {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history:
            axes[1].plot(history[val_metric], label=f'Validation {metric}')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training Metrics')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def compute_text_statistics(texts: np.ndarray) -> pd.DataFrame:
    """
    Compute statistics for text data.
    
    Args:
        texts: Array of text strings
        
    Returns:
        DataFrame with statistics
    """
    stats = []
    
    for text in texts:
        text_stats = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'sentence_count': len(text.split('.')),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        }
        stats.append(text_stats)
    
    return pd.DataFrame(stats)


def hash_text(text: str) -> str:
    """
    Generate hash for text.
    
    Args:
        text: Input text
        
    Returns:
        Hash string
    """
    return hashlib.md5(text.encode()).hexdigest()


def clean_directory(dirpath: str, pattern: str = '*') -> int:
    """
    Clean files in a directory matching pattern.
    
    Args:
        dirpath: Directory path
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    import glob
    
    files = glob.glob(os.path.join(dirpath, pattern))
    count = 0
    
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            count += 1
    
    return count


def get_model_size(filepath: str) -> float:
    """
    Get model file size in MB.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Size in MB
    """
    if not os.path.exists(filepath):
        return 0.0
    
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_section(title: str) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


class ProgressTracker:
    """
    Track progress of training or evaluation.
    """
    
    def __init__(self, total: int, desc: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            desc: Description
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        """
        Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        progress = self.current / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\r{self.desc}: {self.current}/{self.total} ({progress:.1f}%) - {elapsed:.1f}s", 
              end='', flush=True)
    
    def close(self) -> None:
        """Close the progress tracker."""
        print()


if __name__ == "__main__":
    print("Utility Functions for Mental Stress Detection")
    print("Available functions:")
    print("- save_results/results, load_results")
    print("- save_model/load_model")
    print("- plot_training_history")
    print("- plot_class_distribution")
    print("- compute_text_statistics")
    print("- ProgressTracker class")
