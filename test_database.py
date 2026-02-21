"""
Test script to verify database functionality
Simulates making predictions and storing them in the database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import StressDetectionDatabase
import json
from datetime import datetime

def test_database():
    """Test database creation and data storage."""
    print("=" * 60)
    print("DATABASE TEST")
    print("=" * 60)
    
    # Initialize database
    print("\n1. Initializing database...")
    db = StressDetectionDatabase()
    print(f"   Database path: {db.db_path}")
    print(f"   Database exists: {os.path.exists(db.db_path)}")
    
    # Insert test predictions
    print("\n2. Inserting test predictions...")
    
    test_predictions = [
        {
            'name': 'John Doe',
            'age': 25,
            'gender': 'Male',
            'text': 'I am feeling very stressed about my work deadlines',
            'predicted_class': 'Stress',
            'confidence': 0.85,
            'probabilities': {'No Stress': 0.15, 'Stress': 0.85},
            'model_type': 'ML',
            'session_id': 'test-session-001'
        },
        {
            'name': 'Jane Smith',
            'age': 30,
            'gender': 'Female',
            'text': 'Had a great day today, feeling relaxed and happy',
            'predicted_class': 'No Stress',
            'confidence': 0.92,
            'probabilities': {'No Stress': 0.92, 'Stress': 0.08},
            'model_type': 'Demo',
            'session_id': 'test-session-002'
        },
        {
            'name': None,
            'age': None,
            'gender': None,
            'text': 'Not sure how I feel today, kind of neutral',
            'predicted_class': 'No Stress',
            'confidence': 0.65,
            'probabilities': {'No Stress': 0.65, 'Stress': 0.35},
            'model_type': 'BERT',
            'session_id': 'test-session-003'
        }
    ]
    
    record_ids = []
    for i, pred in enumerate(test_predictions, 1):
        print(f"\n   Inserting prediction {i}...")
        record_id = db.insert_prediction(
            name=pred['name'],
            age=pred['age'],
            gender=pred['gender'],
            user_input_text=pred['text'],
            predicted_class=pred['predicted_class'],
            confidence_score=pred['confidence'],
            all_class_probabilities=pred['probabilities'],
            model_type=pred['model_type'],
            session_id=pred['session_id']
        )
        record_ids.append(record_id)
        print(f"   Record ID: {record_id}")
    
    # Update feedback for some records
    print("\n3. Updating feedback...")
    db.update_feedback(record_ids[0], "Yes")
    print(f"   Updated record {record_ids[0]} with feedback: Yes")
    db.update_feedback(record_ids[1], "No")
    print(f"   Updated record {record_ids[1]} with feedback: No")
    
    # Get all predictions
    print("\n4. Retrieving all predictions...")
    df = db.get_all_predictions()
    print(f"   Total records: {len(df)}")
    print(f"\n   Columns: {list(df.columns)}")
    
    # Display records
    print("\n5. Displaying records:")
    print("-" * 60)
    for idx, row in df.iterrows():
        print(f"\n   Record #{row['id']}:")
        print(f"   - Timestamp: {row['timestamp']}")
        print(f"   - Name: {row['name']}")
        print(f"   - Age: {row['age']}")
        print(f"   - Gender: {row['gender']}")
        print(f"   - Text: {row['user_input_text'][:50]}...")
        print(f"   - Prediction: {row['predicted_class']}")
        print(f"   - Confidence: {row['confidence_score']:.2%}")
        print(f"   - Model: {row['model_type']}")
        print(f"   - Feedback: {row['user_feedback']}")
        print(f"   - Probabilities: {row['all_class_probabilities']}")
    
    # Get statistics
    print("\n6. Getting statistics...")
    stats = db.get_statistics()
    print(f"   Total predictions: {stats.get('total_predictions', 0)}")
    print(f"   Class distribution: {stats.get('class_distribution', {})}")
    print(f"   Average confidence: {stats.get('average_confidence', 0):.2%}")
    print(f"   Feedback statistics: {stats.get('feedback_statistics', {})}")
    print(f"   Predictions today: {stats.get('predictions_today', 0)}")
    
    # Export to CSV
    print("\n7. Exporting to CSV...")
    export_path = f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    success = db.export_to_csv(export_path)
    if success:
        print(f"   ✅ Successfully exported to: {export_path}")
        # Show file size
        file_size = os.path.getsize(export_path)
        print(f"   File size: {file_size} bytes")
        # Clean up
        os.remove(export_path)
        print(f"   Cleaned up test file")
    else:
        print("   ❌ Export failed")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nDatabase location: {db.db_path}")
    print("You can now use the database viewer to inspect the data:")
    print("  python view_database.py")

if __name__ == "__main__":
    test_database()
