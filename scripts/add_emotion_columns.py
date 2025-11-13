#!/usr/bin/env python3
"""
Add emotion detection columns to existing conversation_segments table
"""
import sqlite3
import os

def add_emotion_columns():
    """Add emotion columns to conversation_segments table"""
    # Database path from volumes directory
    db_path = "volumes/speakers.db"

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("Looking for database...")
        # Try alternative locations
        alt_paths = [
            "./volumes/speakers.db",
            "../volumes/speakers.db",
            "/app/volumes/speakers.db"
        ]
        for path in alt_paths:
            if os.path.exists(path):
                db_path = path
                break
        else:
            print("Could not find database file")
            return False

    print(f"Connecting to database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(conversation_segments)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Current columns: {columns}")

    emotion_columns = {
        'emotion_category': 'TEXT',
        'emotion_arousal': 'REAL',
        'emotion_valence': 'REAL',
        'emotion_confidence': 'REAL'
    }

    added = []
    skipped = []

    for col_name, col_type in emotion_columns.items():
        if col_name in columns:
            print(f"✓ Column {col_name} already exists, skipping")
            skipped.append(col_name)
        else:
            print(f"Adding column {col_name} ({col_type})...")
            try:
                cursor.execute(f"ALTER TABLE conversation_segments ADD COLUMN {col_name} {col_type}")
                added.append(col_name)
                print(f"✓ Added {col_name}")
            except Exception as e:
                print(f"✗ Error adding {col_name}: {e}")

    conn.commit()
    conn.close()

    print("\n" + "="*50)
    print("Migration complete!")
    print(f"Added columns: {added if added else 'None (all exist)'}")
    print(f"Skipped (already exist): {skipped if skipped else 'None'}")
    print("="*50)

    return len(added) > 0 or len(skipped) > 0

if __name__ == "__main__":
    import sys
    success = add_emotion_columns()
    sys.exit(0 if success else 1)
