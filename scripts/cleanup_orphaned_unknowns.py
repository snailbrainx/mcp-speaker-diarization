#!/usr/bin/env python3
"""
Clean up orphaned Unknown speakers - speakers with no segments assigned
These are created when Unknown speakers get identified/merged but the empty speaker records remain
"""

from app.database import SessionLocal
from app.models import Speaker, ConversationSegment

db = SessionLocal()

print("=" * 80)
print("ORPHANED UNKNOWN SPEAKER CLEANUP")
print("=" * 80)
print()

# Find all Unknown speakers with no segments
all_speakers = db.query(Speaker).filter(Speaker.name.like('Unknown_%')).all()
orphaned = []

for speaker in all_speakers:
    seg_count = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker.id
    ).count()

    if seg_count == 0:
        orphaned.append(speaker)

if not orphaned:
    print("✓ No orphaned Unknown speakers found - database is clean!")
else:
    print(f"Found {len(orphaned)} orphaned Unknown speakers:")
    print()
    for s in orphaned:
        print(f"  - {s.name} (ID: {s.id})")

    print()
    response = input(f"Delete all {len(orphaned)} orphaned Unknown speakers? (yes/no): ").strip().lower()

    if response == 'yes':
        for speaker in orphaned:
            db.delete(speaker)

        db.commit()
        print()
        print(f"✓ Deleted {len(orphaned)} orphaned Unknown speakers!")
        print("✓ Database cleaned successfully!")
    else:
        print()
        print("Cancelled - no changes made")

db.close()
