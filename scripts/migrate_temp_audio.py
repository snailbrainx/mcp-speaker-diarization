#!/usr/bin/env python3
"""
Migrate audio files from /tmp/gradio/ to permanent storage
"""
import os
import sys
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Conversation, ConversationSegment

def main():
    db = SessionLocal()
    data_path = os.getenv("DATA_PATH", "data")
    recordings_dir = f"{data_path}/recordings"
    os.makedirs(recordings_dir, exist_ok=True)

    try:
        # Get all conversations with /tmp/gradio/ paths
        conversations = db.query(Conversation).filter(
            Conversation.audio_path.like('/tmp/gradio/%')
        ).all()

        print(f"Found {len(conversations)} conversations with temp audio paths")

        migrated = []
        failed = []

        for conv in conversations:
            old_path = conv.audio_path

            # Check if file exists
            if not os.path.exists(old_path):
                print(f"✗ Conversation {conv.id}: File not found: {old_path}")
                failed.append(conv.id)
                continue

            # Generate new filename
            filename = os.path.basename(old_path)
            new_filename = f"uploaded_{conv.id}_{filename}"
            new_path = os.path.join(recordings_dir, new_filename)

            try:
                # Copy file to permanent location
                shutil.copy2(old_path, new_path)
                print(f"✓ Copied: {old_path}")
                print(f"       -> {new_path}")

                # Update conversation audio_path
                conv.audio_path = new_path

                # Update all segments for this conversation
                segments = db.query(ConversationSegment).filter(
                    ConversationSegment.conversation_id == conv.id
                ).all()

                for seg in segments:
                    if seg.segment_audio_path and seg.segment_audio_path == old_path:
                        seg.segment_audio_path = new_path

                migrated.append(conv.id)

            except Exception as e:
                print(f"✗ Error copying file: {e}")
                failed.append(conv.id)

        # Commit all changes
        db.commit()

        print(f"\n📊 Migration Summary:")
        print(f"  Migrated: {len(migrated)} conversations: {migrated}")
        if failed:
            print(f"  Failed: {len(failed)} conversations: {failed}")

        print("\n✓ Migration complete!")
        print(f"  Audio files now in: {recordings_dir}")
        print("  Database updated with new paths")

    finally:
        db.close()

if __name__ == "__main__":
    main()
