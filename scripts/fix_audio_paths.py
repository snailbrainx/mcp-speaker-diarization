#!/usr/bin/env python3
"""
Fix audio paths in database to use persistent storage locations
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Conversation, ConversationSegment

def main():
    db = SessionLocal()

    try:
        # Get all conversations
        conversations = db.query(Conversation).all()

        print(f"Found {len(conversations)} conversations")
        print("\nChecking audio file availability...")

        broken_conversations = []
        fixed_conversations = []
        working_conversations = []

        for conv in conversations:
            audio_path = conv.audio_path

            # Check if file exists
            if audio_path and os.path.exists(audio_path):
                working_conversations.append(conv.id)
                print(f"✓ Conversation {conv.id}: {audio_path}")
            elif audio_path and '/tmp/gradio/' in audio_path:
                # Temp file that's been deleted
                broken_conversations.append(conv.id)
                print(f"✗ Conversation {conv.id}: MISSING (temp file deleted): {audio_path}")
            else:
                broken_conversations.append(conv.id)
                print(f"✗ Conversation {conv.id}: MISSING: {audio_path}")

        print(f"\n📊 Summary:")
        print(f"  Working: {len(working_conversations)} conversations")
        print(f"  Broken: {len(broken_conversations)} conversations")

        if broken_conversations:
            print(f"\n⚠️  Broken conversations (temp files deleted): {broken_conversations}")
            print("   These were uploaded files that Gradio cleaned up from /tmp/gradio/")
            print("   Audio playback will not work for these conversations.")
            print("   You can either:")
            print("     1. Re-upload the original audio files")
            print("     2. Delete these conversations from the UI")

            # Update segment paths to NULL for broken conversations
            response = input("\nMark broken conversation segments as unavailable? (y/n): ")
            if response.lower() == 'y':
                for conv_id in broken_conversations:
                    # Update conversation audio_path to NULL
                    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                    conv.audio_path = None

                    # Update all segment audio paths to NULL
                    segments = db.query(ConversationSegment).filter(
                        ConversationSegment.conversation_id == conv_id
                    ).all()

                    for seg in segments:
                        seg.segment_audio_path = None

                    print(f"  Cleared audio paths for conversation {conv_id}")

                db.commit()
                print("\n✓ Updated database - broken audio paths cleared")

        print("\n✓ Audio path check complete!")

    finally:
        db.close()

if __name__ == "__main__":
    main()
