"""
Test feedback update functionality
"""
import os
import sys
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_feedback_direct():
    print("=" * 60)
    print("TESTING FEEDBACK UPDATE - DIRECT SQL")
    print("=" * 60)
    
    # Get database path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'data', 'stress_detection.db')
    
    print(f"\nDatabase path: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")
    
    # Connect directly
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check all records
    cursor.execute("SELECT id, predicted_class, user_feedback FROM user_predictions ORDER BY id DESC LIMIT 5")
    records = cursor.fetchall()
    
    print(f"\nLast 5 records:")
    for rec in records:
        print(f"  ID {rec[0]}: {rec[1]} - Feedback: {rec[2]}")
    
    if not records:
        print("No records found!")
        conn.close()
        return
    
    # Get latest ID
    latest_id = records[0][0]
    print(f"\nTesting update on ID: {latest_id}")
    
    # Try direct update
    cursor.execute("UPDATE user_predictions SET user_feedback = ? WHERE id = ?", ("Yes", latest_id))
    conn.commit()
    
    rows_affected = cursor.rowcount
    print(f"Rows affected: {rows_affected}")
    
    # Verify
    cursor.execute("SELECT user_feedback FROM user_predictions WHERE id = ?", (latest_id,))
    result = cursor.fetchone()
    print(f"New feedback value: {result[0] if result else 'Not found'}")
    
    conn.close()
    
    if rows_affected > 0:
        print("\nSUCCESS! Direct SQL update worked!")
    else:
        print("\nFAILED! No rows updated!")

if __name__ == "__main__":
    test_feedback_direct()
