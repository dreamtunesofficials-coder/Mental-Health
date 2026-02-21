"""
Cloud-Compatible Database Module for Mental Stress Detection
Supports both local SQLite and cloud PostgreSQL/Supabase
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st
from contextlib import contextmanager

# Try to import cloud database libraries
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class BaseDatabase:
    """Base class for database operations."""
    
    def insert_prediction(self, **kwargs) -> int:
        raise NotImplementedError
        
    def update_feedback(self, record_id: int, feedback: str) -> bool:
        raise NotImplementedError
        
    def get_all_predictions(self, limit: Optional[int] = None) -> pd.DataFrame:
        raise NotImplementedError
        
    def get_statistics(self) -> Dict[str, Any]:
        raise NotImplementedError
        
    def export_to_csv(self, filepath: str) -> bool:
        raise NotImplementedError


class StressDetectionDatabase(BaseDatabase):
    """
    SQLite database manager for local development.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, 'data', 'stress_detection.db')
        
        self.db_path = db_path
        self._ensure_directory_exists()
        self._create_tables()
    
    def _ensure_directory_exists(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Main predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                name TEXT,
                age INTEGER,
                gender TEXT,
                user_input_text TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                all_class_probabilities TEXT NOT NULL,
                user_feedback TEXT,
                model_type TEXT,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON user_predictions(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predicted_class 
            ON user_predictions(predicted_class)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session 
            ON user_predictions(session_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_prediction(
        self,
        name: Optional[str],
        age: Optional[int],
        gender: Optional[str],
        user_input_text: str,
        predicted_class: str,
        confidence_score: float,
        all_class_probabilities: Dict[str, float],
        model_type: str = 'ML',
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> int:
        """Insert a new prediction record."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_predictions 
                (name, age, gender, user_input_text, predicted_class, 
                 confidence_score, all_class_probabilities, model_type, session_id,
                 ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, age, gender, user_input_text, predicted_class,
                confidence_score, json.dumps(all_class_probabilities),
                model_type, session_id, ip_address, user_agent
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return record_id
            
        except Exception as e:
            st.error(f"Database error while saving prediction: {e}")
            return -1
    
    def update_feedback(self, record_id: int, feedback: str) -> bool:
        """Update user feedback for a prediction."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_predictions 
                SET user_feedback = ? 
                WHERE id = ?
            ''', (feedback, record_id))
            
            conn.commit()
            conn.close()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            st.error(f"Database error while updating feedback: {e}")
            return False
    
    def get_all_predictions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve all predictions from the database."""
        try:
            conn = self._get_connection()
            
            query = "SELECT * FROM user_predictions ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['all_class_probabilities'] = df['all_class_probabilities'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
            
        except Exception as e:
            st.error(f"Database error while retrieving predictions: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM user_predictions")
            total_predictions = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT predicted_class, COUNT(*) as count 
                FROM user_predictions 
                GROUP BY predicted_class
            ''')
            class_distribution = dict(cursor.fetchall())
            
            cursor.execute('''
                SELECT AVG(confidence_score) 
                FROM user_predictions
            ''')
            avg_confidence = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT user_feedback, COUNT(*) as count 
                FROM user_predictions 
                WHERE user_feedback IS NOT NULL
                GROUP BY user_feedback
            ''')
            feedback_stats = dict(cursor.fetchall())
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT COUNT(*) FROM user_predictions 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            predictions_today = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_predictions': total_predictions,
                'class_distribution': class_distribution,
                'average_confidence': avg_confidence,
                'feedback_statistics': feedback_stats,
                'predictions_today': predictions_today
            }
            
        except Exception as e:
            st.error(f"Database error while getting statistics: {e}")
            return {}
    
    def export_to_csv(self, filepath: str) -> bool:
        """Export all predictions to CSV file."""
        try:
            df = self.get_all_predictions()
            
            if not df.empty:
                df['all_class_probabilities'] = df['all_class_probabilities'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
                df.to_csv(filepath, index=False)
                return True
            return False
            
        except Exception as e:
            st.error(f"Error exporting to CSV: {e}")
            return False


class CloudDatabaseManager:
    """
    Manages database for both local and cloud deployments.
    Uses session state to persist data on Streamlit Cloud.
    """
    
    def __init__(self):
        self.is_cloud = self._is_streamlit_cloud()
        self.local_db = StressDetectionDatabase()
        
        # Initialize session state for cloud storage
        if 'cloud_predictions' not in st.session_state:
            st.session_state.cloud_predictions = []
    
    def _is_streamlit_cloud(self) -> bool:
        """Check if running on Streamlit Cloud."""
        return os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit-community'
    
    def insert_prediction(self, **kwargs) -> int:
        """Insert prediction - works for both local and cloud."""
        # Always save to local database
        record_id = self.local_db.insert_prediction(**kwargs)
        
        # Also save to session state for cloud persistence
        if self.is_cloud:
            prediction_data = {
                'id': len(st.session_state.cloud_predictions) + 1,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            st.session_state.cloud_predictions.append(prediction_data)
        
        return record_id
    
    def update_feedback(self, record_id: int, feedback: str) -> bool:
        """Update feedback."""
        return self.local_db.update_feedback(record_id, feedback)
    
    def get_all_predictions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all predictions."""
        return self.local_db.get_all_predictions(limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.local_db.get_statistics()
    
    def export_to_csv(self, filepath: str) -> bool:
        """Export to CSV."""
        return self.local_db.export_to_csv(filepath)
    
    def get_cloud_data(self) -> List[Dict]:
        """Get data from session state (for cloud)."""
        return st.session_state.get('cloud_predictions', [])


def get_database() -> StressDetectionDatabase:
    """
    Factory function to get database instance.
    """
    if 'database' not in st.session_state:
        st.session_state.database = StressDetectionDatabase()
    
    return st.session_state.database


def get_cloud_manager() -> CloudDatabaseManager:
    """
    Get cloud database manager.
    """
    if 'cloud_manager' not in st.session_state:
        st.session_state.cloud_manager = CloudDatabaseManager()
    
    return st.session_state.cloud_manager


# For backward compatibility
db = get_database()
