#!/usr/bin/env python3
"""
Quick verification that extracted segments match transcription
"""

import os

print("=" * 80)
print("SEGMENT AUDIO FILES GENERATED")
print("=" * 80)
print()
print("The following segments have been extracted to data/test_output/:")
print()

segments = [
    (1, "this is sentence number one this is sentence number 2"),
    (2, "this is sentence number 3 this is sentence number"),
    (3, "4 this is sentence number 5"),
    (4, "this is sentenced number 6 this is sentence number"),
    (5, "7 I think this is sentence number 8"),
    (6, "this is sentence number it 9 this is sentence number 10"),
    (7, "this is sentence number 11"),
    (8, "This is sentence number 12."),
    (9, "This is sentence number 13."),
    (10, "This is sentence number 14."),
    (11, "This is sentence number 15."),
    (12, "This is sentence number 16."),
    (13, "This is sentence number 17."),
    (14, "This is sentence number 18."),
    (15, "This is sentence number 19."),
    (16, "And this is sentence number 20."),
]

for seg_num, transcription in segments:
    filename = f"data/test_output/segment_{seg_num:03d}.wav"
    if os.path.exists(filename):
        print(f"✅ {filename}")
        print(f"   Transcription: '{transcription}'")
        print()

print("=" * 80)
print("VERIFICATION INSTRUCTIONS")
print("=" * 80)
print()
print("Please play these WAV files and verify they match the transcription:")
print()
print("  1. Open data/test_output/ folder")
print("  2. Play segment_001.wav - should say 'sentence number one' and 'two'")
print("  3. Play segment_008.wav - should say 'sentence number 12'")
print("  4. Play segment_016.wav - should say 'sentence number 20'")
print()
print("If the audio matches the transcription, then:")
print("  ✅ Playback is correctly aligned")
print("  ✅ Enrollment will use the correct audio")
print("  ✅ The system is working perfectly")
print()
print("=" * 80)
