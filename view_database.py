"""
Database Viewer for Mental Stress Detection
Simple tool to view and manage the SQLite database
"""

import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime

def get_db_path():
    """Get the database path."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'data', 'stress_detection.db')
    return db_path


def check_database():
    """Check if database exists and show info."""
    db_path = get_db_path()
    
    print("=" * 60)
    print("MENTAL STRESS DETECTION - DATABASE VIEWER")
    print("=" * 60)
    print(f"\nDatabase Path: {db_path}")
    print(f"Database Exists: {os.path.exists(db_path)}")
    
    if not os.path.exists(db_path):
        print("\n⚠️  Database does not exist yet!")
        print("   The database will be created automatically when you:")
        print("   1. Run the enhanced app: streamlit run app_enhanced_v2.py")
        print("   2. Make a prediction")
        return False
    
    return True

def show_tables():
    """Show all tables in the database."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n" + "=" * 60)
    print("TABLES IN DATABASE")
    print("=" * 60)
    for table in tables:
        print(f"  - {table[0]}")
    
    conn.close()

def show_schema():
    """Show the schema of user_predictions table."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("USER_PREDICTIONS TABLE SCHEMA")
    print("=" * 60)
    
    cursor.execute("PRAGMA table_info(user_predictions);")
    columns = cursor.fetchall()
    
    print(f"\n{'Column Name':<25} {'Type':<15} {'Nullable':<10}")
    print("-" * 60)
    for col in columns:
        cid, name, dtype, notnull, dflt_value, pk = col
        nullable = "NO" if notnull else "YES"
        print(f"{name:<25} {dtype:<15} {nullable:<10}")
    
    conn.close()

def show_data(limit=10):
    """Show data from user_predictions table."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print("\n⚠️  Database does not exist!")
        return
    
    conn = sqlite3.connect(db_path)
    
    print("\n" + "=" * 60)
    print(f"RECENT PREDICTIONS (Last {limit} records)")
    print("=" * 60)
    
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM user_predictions ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        
        if df.empty:
            print("\nNo data found in the database.")
        else:
            # Display summary
            print(f"\nTotal Records: {len(df)}")
            print(f"\nColumns: {', '.join(df.columns.tolist())}")
            print("\n" + "-" * 60)
            
            # Display data in a readable format
            for idx, row in df.iterrows():
                print(f"\nRecord #{row.get('id', 'N/A')}:")
                print(f"  Timestamp: {row.get('timestamp', 'N/A')}")
                print(f"  Name: {row.get('name', 'N/A')}")
                print(f"  Age: {row.get('age', 'N/A')}")
                print(f"  Gender: {row.get('gender', 'N/A')}")
                print(f"  Input Text: {row.get('user_input_text', 'N/A')[:100]}...")
                print(f"  Prediction: {row.get('predicted_class', 'N/A')}")
                print(f"  Confidence: {row.get('confidence_score', 'N/A')}")
                print(f"  Model: {row.get('model_type', 'N/A')}")
                print(f"  Feedback: {row.get('user_feedback', 'Not provided')}")
                print("-" * 60)
    
    except Exception as e:
        print(f"\nError reading data: {e}")
    
    conn.close()

def show_statistics():
    """Show database statistics."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print("\n⚠️  Database does not exist!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM user_predictions")
    total = cursor.fetchone()[0]
    print(f"\nTotal Predictions: {total}")
    
    if total == 0:
        print("\nNo predictions recorded yet.")
        conn.close()
        return
    
    # Predictions by class
    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count 
        FROM user_predictions 
        GROUP BY predicted_class
    """)
    classes = cursor.fetchall()
    print("\nPredictions by Class:")
    for cls, count in classes:
        print(f"  {cls}: {count}")
    
    # Predictions today
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("""
        SELECT COUNT(*) FROM user_predictions 
        WHERE DATE(timestamp) = ?
    """, (today,))
    today_count = cursor.fetchone()[0]
    print(f"\nPredictions Today: {today_count}")
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence_score) FROM user_predictions")
    avg_conf = cursor.fetchone()[0]
    if avg_conf:
        print(f"Average Confidence: {avg_conf:.2%}")
    
    # Feedback statistics
    cursor.execute("""
        SELECT user_feedback, COUNT(*) as count 
        FROM user_predictions 
        WHERE user_feedback IS NOT NULL
        GROUP BY user_feedback
    """)
    feedback = cursor.fetchall()
    if feedback:
        print("\nFeedback Received:")
        for fb, count in feedback:
            print(f"  {fb}: {count}")
    
    # Model usage
    cursor.execute("""
        SELECT model_type, COUNT(*) as count 
        FROM user_predictions 
        GROUP BY model_type
    """)
    models = cursor.fetchall()
    print("\nModel Usage:")
    for model, count in models:
        print(f"  {model}: {count}")
    
    conn.close()

def export_to_csv():
    """Export all data to CSV."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print("\n⚠️  Database does not exist!")
        return
    
    conn = sqlite3.connect(db_path)
    
    try:
        df = pd.read_sql_query("SELECT * FROM user_predictions", conn)
        
        if df.empty:
            print("\nNo data to export.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stress_predictions_export_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"\n✅ Data exported to: {filename}")
        print(f"   Total records: {len(df)}")
        
    except Exception as e:
        print(f"\nError exporting data: {e}")
    
    conn.close()

def main():
    """Main menu."""
    while True:
        print("\n" + "=" * 60)
        print("DATABASE VIEWER MENU")
        print("=" * 60)
        print("1. Check Database Status")
        print("2. Show Tables")
        print("3. Show Schema")
        print("4. Show Recent Data")
        print("5. Show Statistics")
        print("6. Export to CSV")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            check_database()
        elif choice == '2':
            if check_database():
                show_tables()
        elif choice == '3':
            if check_database():
                show_schema()
        elif choice == '4':
            if check_database():
                limit = input("How many records to show? (default 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                show_data(limit)
        elif choice == '5':
            if check_database():
                show_statistics()
        elif choice == '6':
            if check_database():
                export_to_csv()
        elif choice == '7':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
