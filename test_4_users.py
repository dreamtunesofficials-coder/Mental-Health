#!/usr/bin/env python3
"""
Test script to verify 4 users giving feedback (Yes/No/Yes/No)
"""

import sqlite3
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_4_users_feedback():
    """Test 4 users giving different feedback."""
    
    db_path = 'data/stress_detection.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("TESTING 4 USERS WITH FEEDBACK")
    print("=" * 60)
    
    # Get last 4 records without feedback
    cursor.execute("""
        SELECT id, name, predicted_class 
        FROM user_predictions 
        WHERE user_feedback IS NULL OR user_feedback = '' 
        ORDER BY id DESC 
        LIMIT 4
    """)
    records = cursor.fetchall()
    
    if len(records) < 4:
        print(f"⚠️  Only {len(records)} records without feedback found")
        print("Creating new predictions first...")
        # We need to create some test predictions
        conn.close()
        return create_test_predictions()
    
    print(f"\n📋 Found {len(records)} records without feedback:")
    for r in records:
        print(f"   ID {r[0]}: {r[1]} - {r[2]}")
    
    # Simulate 4 users giving feedback: Yes, No, Yes, No
    feedbacks = ['Yes', 'No', 'Yes', 'No']
    print(f"\n📝 Simulating 4 users giving feedback: {feedbacks}")
    
    updated_records = []
    for i, (record, feedback) in enumerate(zip(records, feedbacks)):
        record_id = record[0]
        cursor.execute(
            "UPDATE user_predictions SET user_feedback = ? WHERE id = ?",
            (feedback, record_id)
        )
        updated_records.append((record_id, record[1], record[2], feedback))
        print(f"   User {i+1} (ID {record_id}): Feedback = {feedback} ✅")
    
    conn.commit()
    
    # Verify all 4 feedbacks were saved
    print("\n🔍 Verifying feedback in database:")
    all_verified = True
    for record_id, name, pred_class, expected_feedback in updated_records:
        cursor.execute(
            "SELECT user_feedback FROM user_predictions WHERE id = ?",
            (record_id,)
        )
        result = cursor.fetchone()
        actual_feedback = result[0] if result else None
        
        if actual_feedback == expected_feedback:
            print(f"   ID {record_id}: {name} - {pred_class} - Feedback: {actual_feedback} ✅")
        else:
            print(f"   ID {record_id}: {name} - {pred_class} - Feedback: {actual_feedback} ❌ (Expected: {expected_feedback})")
            all_verified = False
    
    conn.close()
    
    print("\n" + "=" * 60)
    if all_verified:
        print("✅ ALL 4 USER FEEDBACKS VERIFIED SUCCESSFULLY!")
    else:
        print("❌ SOME FEEDBACKS NOT VERIFIED")
    print("=" * 60)
    
    return all_verified

def create_test_predictions():
    """Create 4 test predictions if not enough exist."""
    from cloud_database import get_database, get_cloud_manager
    
    db = get_database()
    cloud_mgr = get_cloud_manager()
    
    print("\n🆕 Creating 4 test predictions...")
    
    test_data = [
        {"name": "TestUser1", "age": 25, "gender": "Male", "text": "I am very stressed today", "result": {"predicted_class": "Stress", "confidence": 0.85, "probabilities": {"No Stress": 0.15, "Stress": 0.85}, "model_type": "Test"}},
        {"name": "TestUser2", "age": 30, "gender": "Female", "text": "I am happy today", "result": {"predicted_class": "No Stress", "confidence": 0.90, "probabilities": {"No Stress": 0.90, "Stress": 0.10}, "model_type": "Test"}},
        {"name": "TestUser3", "age": 35, "gender": "Male", "text": "Work is overwhelming", "result": {"predicted_class": "Stress", "confidence": 0.75, "probabilities": {"No Stress": 0.25, "Stress": 0.75}, "model_type": "Test"}},
        {"name": "TestUser4", "age": 28, "gender": "Female", "text": "Feeling relaxed", "result": {"predicted_class": "No Stress", "confidence": 0.95, "probabilities": {"No Stress": 0.95, "Stress": 0.05}, "model_type": "Test"}},
    ]
    
    record_ids = []
    for data in test_data:
        record_id = db.insert_prediction(
            name=data["name"],
            age=data["age"],
            gender=data["gender"],
            user_input_text=data["text"],
            predicted_class=data["result"]["predicted_class"],
            confidence_score=data["result"]["confidence"],
            all_class_probabilities=data["result"]["probabilities"],
            model_type=data["result"]["model_type"]
        )
        record_ids.append(record_id)
        print(f"   Created: {data['name']} (ID: {record_id})")
    
    print(f"\n✅ Created {len(record_ids)} test predictions")
    
    # Now run the feedback test again
    return test_4_users_feedback()

def show_all_feedback():
    """Show all records with feedback."""
    conn = sqlite3.connect('data/stress_detection.db')
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("ALL RECORDS WITH FEEDBACK")
    print("=" * 60)
    
    cursor.execute("""
        SELECT id, name, predicted_class, user_feedback 
        FROM user_predictions 
        WHERE user_feedback IS NOT NULL AND user_feedback != ''
        ORDER BY id DESC
    """)
    records = cursor.fetchall()
    
    if not records:
        print("No records with feedback found.")
    else:
        print(f"\nFound {len(records)} records with feedback:\n")
        for r in records:
            status = "✅" if r[3] else "❌"
            print(f"   {status} ID {r[0]}: {r[1]} - {r[2]} - Feedback: {r[3]}")
    
    # Summary
    cursor.execute("""
        SELECT user_feedback, COUNT(*) 
        FROM user_predictions 
        GROUP BY user_feedback
    """)
    summary = cursor.fetchall()
    
    print("\n📊 Summary:")
    for row in summary:
        feedback_type = row[0] if row[0] else "Not Given"
        count = row[1]
        print(f"   {feedback_type}: {count}")
    
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test 4 users feedback system')
    parser.add_argument('--show', action='store_true', help='Show all feedback records')
    args = parser.parse_args()
    
    if args.show:
        show_all_feedback()
    else:
        success = test_4_users_feedback()
        show_all_feedback()
        sys.exit(0 if success else 1)
