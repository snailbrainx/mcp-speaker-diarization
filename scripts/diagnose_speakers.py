#!/usr/bin/env python3
"""
Diagnose speaker recognition issues - check if different speakers are being detected
"""

import sys
sys.path.insert(0, '/home/acoloss/nvidia/speaker-diarization-app')

from app.database import SessionLocal
from app.models import Conversation, ConversationSegment, Speaker
from collections import Counter

db = SessionLocal()

# Get most recent conversation
conv = db.query(Conversation).order_by(Conversation.id.desc()).first()

if not conv:
    print("No conversations found")
    sys.exit(1)

print(f"=" * 80)
print(f"CONVERSATION {conv.id}: {conv.title or 'Untitled'}")
print(f"=" * 80)
print(f"Started: {conv.start_time}")
print(f"Duration: {conv.duration:.1f}s" if conv.duration else "Duration: N/A")
print(f"Segments: {conv.num_segments}")
print()

# Get all segments
segments = db.query(ConversationSegment).filter(
    ConversationSegment.conversation_id == conv.id
).order_by(ConversationSegment.id).all()

# Count speakers
speaker_counts = Counter()
for seg in segments:
    speaker_counts[seg.speaker_name or "Unknown"] += 1

print(f"SPEAKER DISTRIBUTION:")
print(f"-" * 80)
for speaker, count in speaker_counts.most_common():
    print(f"{speaker:30s} : {count:3d} segments")

print()
print(f"=" * 80)
print(f"SEGMENT DETAILS (first 20)")
print(f"=" * 80)

for i, seg in enumerate(segments[:20], 1):
    print(f"\n{i:2d}. ID={seg.id:3d} | {seg.start_offset:6.2f}s - {seg.end_offset:6.2f}s")
    print(f"    Speaker: {seg.speaker_name or 'Unknown'}")
    print(f"    Text: {seg.text[:60] if seg.text else 'N/A'}")
    print(f"    Confidence: {seg.confidence:.2f}" if seg.confidence else "    Confidence: N/A")

print()
print(f"=" * 80)
print(f"DIAGNOSIS")
print(f"=" * 80)

unique_speakers = len(speaker_counts)
print(f"\nUnique speakers detected: {unique_speakers}")

if unique_speakers == 1:
    speaker_name = list(speaker_counts.keys())[0]
    if speaker_name.startswith("SPEAKER_"):
        print(f"\n❌ PROBLEM: All segments labeled as '{speaker_name}'")
        print(f"   This means diarization detected only 1 speaker")
        print(f"   OR all segments failed to auto-enroll")
        print()
        print(f"   Possible causes:")
        print(f"   - Very similar voices (hard to distinguish)")
        print(f"   - Diarization threshold too low")
        print(f"   - All embeddings had NaN values")
    elif speaker_name.startswith("Unknown_"):
        print(f"\n❌ PROBLEM: All segments labeled as '{speaker_name}'")
        print(f"   This means diarization detected multiple speakers (SPEAKER_00, SPEAKER_01)")
        print(f"   BUT they all got mapped to the same Unknown")
        print()
        print(f"   This is a BUG in the unknown_speaker_map logic!")
else:
    print(f"\n✅ Multiple speakers detected correctly")

    # Check if we have both SPEAKER_xx and Unknown_xxx
    has_speaker_labels = any(s.startswith("SPEAKER_") for s in speaker_counts.keys())
    has_unknown_labels = any(s.startswith("Unknown_") for s in speaker_counts.keys())

    if has_speaker_labels and has_unknown_labels:
        print(f"   ⚠️  Mix of SPEAKER_xx and Unknown_xxx")
        print(f"   Some segments got auto-enrolled, others didn't")
    elif has_speaker_labels:
        print(f"   ⚠️  All segments show SPEAKER_xx labels")
        print(f"   Auto-enrollment didn't work")
    else:
        print(f"   ✅ All segments auto-enrolled as Unknown_xxx")

print()

db.close()
