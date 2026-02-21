"""
Database Module for Mental Stress Detection
Handles database connections, schema creation, and data operations
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st


class StressDetectionDatabase:
    """
    SQLite database manager for storing user predictions and feedback.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file (default: ./data/stress_detection.db in project root)
        """
        if db_path is None:
            # Get the directory where this script is located
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

        
        # Create index on timestamp for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON user_predictions(timestamp)
        ''')
        
        # Create index on predicted_class for analytics
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predicted_class 
            ON user_predictions(predicted_class)
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

        """
        Insert a new prediction record.
        
        Args:
            name: User's name (optional)
            age: User's age (optional)
            gender: User's gender (optional)
            user_input_text: The text submitted for analysis
            predicted_class: The predicted mental health condition
            confidence_score: Model's confidence score
            all_class_probabilities: Dictionary of probabilities for all classes
            model_type: Type of model used (ML, BERT, Demo)
            session_id: Session identifier for tracking
            
        Returns:
            ID of the inserted record
        """
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
                name,
                age,
                gender,
                user_input_text,
                predicted_class,
                confidence_score,
                json.dumps(all_class_probabilities),
                model_type,
                session_id,
                ip_address,
                user_agent
            ))

            
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return record_id
            
        except Exception as e:
            st.error(f"Database error while saving prediction: {e}")
            return -1
    
    def update_feedback(self, record_id: int, feedback: str) -> bool:
        """
        Update user feedback for a prediction.
        
        Args:
            record_id: ID of the prediction record
            feedback: User feedback ('Yes', 'No', 'Unsure')
            
        Returns:
            True if successful, False otherwise
        """
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
        """
        Retrieve all predictions from the database.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with all predictions
        """
        try:
            conn = self._get_connection()
            
            query = "SELECT * FROM user_predictions ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Parse JSON probabilities
            if not df.empty:
                df['all_class_probabilities'] = df['all_class_probabilities'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
            
        except Exception as e:
            st.error(f"Database error while retrieving predictions: {e}")
            return pd.DataFrame()
    
    def get_predictions_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get predictions within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with predictions in the date range
        """
        try:
            conn = self._get_connection()
            
            query = '''
                SELECT * FROM user_predictions 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(start_date, end_date)
            )
            conn.close()
            
            if not df.empty:
                df['all_class_probabilities'] = df['all_class_probabilities'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
            
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM user_predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Predictions by class
            cursor.execute('''
                SELECT predicted_class, COUNT(*) as count 
                FROM user_predictions 
                GROUP BY predicted_class
            ''')
            class_distribution = dict(cursor.fetchall())
            
            # Average confidence score
            cursor.execute('''
                SELECT AVG(confidence_score) 
                FROM user_predictions
            ''')
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Feedback statistics
            cursor.execute('''
                SELECT user_feedback, COUNT(*) as count 
                FROM user_predictions 
                WHERE user_feedback IS NOT NULL
                GROUP BY user_feedback
            ''')
            feedback_stats = dict(cursor.fetchall())
            
            # Predictions today
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
        """
        Export all predictions to CSV file.
        
        Args:
            filepath: Path to save CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = self.get_all_predictions()
            
            if not df.empty:
                # Convert probabilities dict to string for CSV
                df['all_class_probabilities'] = df['all_class_probabilities'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
                df.to_csv(filepath, index=False)
                return True
            return False
            
        except Exception as e:
            st.error(f"Error exporting to CSV: {e}")
            return False
    
    def get_feedback_data(self) -> pd.DataFrame:
        """
        Get data with user feedback for model retraining.
        
        Returns:
            DataFrame with feedback data
        """
        try:
            conn = self._get_connection()
            
            query = '''
                SELECT user_input_text, predicted_class, user_feedback,
                       confidence_score, all_class_probabilities
                FROM user_predictions 
                WHERE user_feedback IS NOT NULL
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['all_class_probabilities'] = df['all_class_probabilities'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
            
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection (not needed for SQLite)."""
        pass


def get_database(db_path: str = None) -> StressDetectionDatabase:
    """
    Factory function to get database instance.
    Uses Streamlit session state to maintain single connection.
    
    Args:
        db_path: Path to database file (default: auto-detected based on script location)
        
    Returns:
        StressDetectionDatabase instance
    """
    if 'database' not in st.session_state:
        st.session_state.database = StressDetectionDatabase(db_path)
    
    return st.session_state.database



# Initialize database on module import
db = get_database()
