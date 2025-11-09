#!/usr/bin/env python3
"""
Fix segments that have speaker_name but no speaker_id
This happens when reassignment was done with the old buggy code
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import ConversationSegment, Speaker
from app.diarization import SpeakerRecognitionEngine
import numpy as np

# Initialize
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("ERROR: HF_TOKEN environment variable not set")
    print("Please set it with: export HF_TOKEN=your_token_here")
    sys.exit(1)
engine = SpeakerRecognitionEngine(hf_token=hf_token)
db = SessionLocal()

print("=" * 80)
print("FIX ORPHANED SEGMENTS - Create Missing Speakers")
print("=" * 80)
print()

# Find segments with speaker_name but no speaker_id
orphaned = db.query(ConversationSegment).filter(
    ConversationSegment.speaker_id == None,
    ConversationSegment.speaker_name != None,
    ConversationSegment.speaker_name != ''
).all()

if not orphaned:
    print("✓ No orphaned segments found - all good!")
    db.close()
    sys.exit(0)

# Group by speaker_name
from collections import defaultdict
by_speaker = defaultdict(list)
for seg in orphaned:
    by_speaker[seg.speaker_name].append(seg)

print(f"Found {len(orphaned)} orphaned segments for {len(by_speaker)} speakers:")
print()

for speaker_name, segments in by_speaker.items():
    print(f"  {speaker_name}: {len(segments)} segments")

print()
print("Will create speaker records with embeddings for each...")
print()

for speaker_name, segments in by_speaker.items():
    print(f"Creating speaker: {speaker_name}")

    # Extract embeddings from all segments
    embeddings = []
    for seg in segments:
        conv = seg.conversation
        audio_file = seg.segment_audio_path if seg.segment_audio_path and os.path.exists(seg.segment_audio_path) else conv.audio_path

        if not audio_file or not os.path.exists(audio_file):
            print(f"  ⚠️  Seg {seg.id}: No audio file")
            continue

        try:
            emb = engine.extract_segment_embedding(
                audio_file,
                seg.start_offset,
                seg.end_offset
            )
            if not np.isnan(emb).any():
                embeddings.append(emb)
                print(f"  ✓ Seg {seg.id}: Extracted embedding")
            else:
                print(f"  ⚠️  Seg {seg.id}: NaN embedding")
        except Exception as e:
            print(f"  ⚠️  Seg {seg.id}: Error - {e}")
            continue

    if not embeddings:
        print(f"  ❌ Could not extract any embeddings for {speaker_name} - skipping")
        continue

    # Create speaker with average embedding
    avg_embedding = np.mean(embeddings, axis=0)
    speaker = Speaker(name=speaker_name)
    speaker.set_embedding(avg_embedding)
    db.add(speaker)
    db.flush()

    # Assign all segments to this speaker
    for seg in segments:
        seg.speaker_id = speaker.id

    db.commit()

    print(f"  ✓ Created speaker '{speaker_name}' (ID {speaker.id}) with {len(embeddings)} embeddings")
    print(f"  ✓ Assigned {len(segments)} segments")
    print()

print("=" * 80)
print("✓ All orphaned segments fixed!")
print("=" * 80)

db.close()
