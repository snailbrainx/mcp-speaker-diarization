import gradio as gr
from sqlalchemy.orm import Session
from typing import List, Tuple
import pandas as pd
import os
from datetime import datetime, timedelta

from .database import SessionLocal
from .models import Speaker, Recording, Conversation, ConversationSegment
from .diarization import SpeakerRecognitionEngine
from .streaming_recorder import StreamingRecorder
from .audio_utils import convert_to_mp3
import requests
import time
import json

# Try to import live recording (requires PortAudio)
try:
    from .live_recording import LiveRecordingManager
    LIVE_RECORDING_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Live recording not available: {e}")
    print("Install PortAudio: sudo apt-get install -y portaudio19-dev")
    LiveRecordingManager = None
    LIVE_RECORDING_AVAILABLE = False


class GradioInterface:
    def __init__(self, preload_models: bool = True):
        self.engine = SpeakerRecognitionEngine()
        self.live_recorder = None
        self.current_conversation_id = None
        self.live_segments = []

        # Load configuration from environment
        self.speaker_threshold = float(os.getenv("SPEAKER_THRESHOLD", "0.20"))
        self.filter_hallucinations = os.getenv("FILTER_HALLUCINATIONS", "true").lower() == "true"

        # Streaming recorder
        self.streaming_recorder = StreamingRecorder()
        self.streaming_recorder.on_segment_processed = self._on_stream_segment_processed
        self.streaming_results = []  # Store processed segments for display
        self.streaming_conversation_id = None
        self.audio_level = 0.0
        self.last_update_time = time.time()  # Track when we last updated display

        # Preload all models at startup for faster first request
        if preload_models:
            print("\n=== Preloading AI models ===")
            _ = self.engine.whisper_model  # Loads Whisper
            _ = self.engine.diarization_pipeline  # Loads diarization
            _ = self.engine.embedding_model  # Loads embeddings
            print("=== All models loaded and ready! ===\n")

    def get_db(self):
        db = SessionLocal()
        try:
            return db
        finally:
            pass  # Will be closed by caller

    def enroll_speaker(self, name: str, audio_file: str) -> str:
        """Enroll a new speaker"""
        if not name or not audio_file:
            return "Error: Please provide both name and audio file"

        db = self.get_db()
        try:
            # Check if speaker exists
            existing = db.query(Speaker).filter(Speaker.name == name).first()
            if existing:
                return f"Error: Speaker '{name}' already exists"

            # Extract embedding
            embedding = self.engine.extract_embedding(audio_file)

            # Save speaker
            speaker = Speaker(name=name)
            speaker.set_embedding(embedding)
            db.add(speaker)
            db.commit()

            # Clear GPU cache after embedding extraction
            self.engine.clear_gpu_cache()

            return f"✓ Successfully enrolled speaker '{name}'"

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            db.close()

    def list_speakers(self) -> pd.DataFrame:
        """List all enrolled speakers"""
        db = self.get_db()
        try:
            speakers = db.query(Speaker).all()
            if not speakers:
                return pd.DataFrame(columns=["ID", "Name", "Created At"])

            data = [
                {"ID": s.id, "Name": s.name, "Created At": s.created_at.strftime("%Y-%m-%d %H:%M")}
                for s in speakers
            ]
            return pd.DataFrame(data)

        finally:
            db.close()

    def delete_speaker(self, speaker_id: int) -> str:
        """Delete a single speaker"""
        if not speaker_id:
            return "Error: Please provide speaker ID"

        db = self.get_db()
        try:
            speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                return f"Error: Speaker ID {speaker_id} not found"

            name = speaker.name
            db.delete(speaker)
            db.commit()

            return f"✓ Successfully deleted speaker '{name}'"

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            db.close()

    def get_speakers_for_selection(self):
        """Get speakers formatted for CheckboxGroup and current selections"""
        db = self.get_db()
        try:
            speakers = db.query(Speaker).all()
            # Format as "ID: Name" for display
            choices = [f"{s.id}: {s.name}" for s in speakers]
            # Return as CheckboxGroup update
            return gr.CheckboxGroup(choices=choices, value=[])

        finally:
            db.close()

    def delete_multiple_speakers(self, selected_speakers: list) -> str:
        """Delete multiple speakers from checkbox selection"""
        if not selected_speakers:
            return "Please select speakers to delete"

        db = self.get_db()
        try:
            # Parse IDs from "ID: Name" format
            ids = []
            for item in selected_speakers:
                try:
                    speaker_id = int(item.split(':')[0].strip())
                    ids.append(speaker_id)
                except (ValueError, IndexError):
                    return f"Error: Invalid selection format: {item}"

            deleted = []
            errors = []

            for speaker_id in ids:
                speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
                if speaker:
                    deleted.append(speaker.name)
                    db.delete(speaker)
                else:
                    errors.append(f"ID {speaker_id} not found")

            db.commit()

            result = f"✓ Deleted {len(deleted)} speaker(s): {', '.join(deleted)}"
            if errors:
                result += f"\n⚠️ Errors: {', '.join(errors)}"

            return result

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            db.close()

    def delete_all_unknown_speakers(self) -> str:
        """Delete all Unknown_xxx speakers at once"""
        db = self.get_db()
        try:
            # Find all speakers with names starting with "Unknown_"
            unknown_speakers = db.query(Speaker).filter(Speaker.name.like("Unknown_%")).all()

            if not unknown_speakers:
                return "No Unknown speakers found"

            count = len(unknown_speakers)
            names = [s.name for s in unknown_speakers]

            # Delete all unknown speakers
            for speaker in unknown_speakers:
                db.delete(speaker)

            db.commit()

            return f"✓ Deleted {count} Unknown speaker(s)"

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            db.close()

    def _verify_and_enroll_unknown_speaker(self, seg, unknown_speaker_map, db, conversation, segment_audio_path=None, threshold=None):
        """
        Handle auto-enrollment of unknown speakers with embedding verification.

        Returns:
            Tuple of (speaker_id, speaker_name, segment_to_add_or_None)
            - If segment_to_add_or_None is not None, caller should add it and continue
            - Otherwise, caller should create a new segment with returned speaker_id/name
        """
        # Use instance threshold if not specified
        if threshold is None:
            threshold = self.speaker_threshold

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        import time
        import uuid

        speaker_name = seg.get("speaker")
        embedding = seg.get("embedding")

        if embedding is None:
            return None, speaker_name, None

        # Check if we already created this unknown speaker
        if speaker_name in unknown_speaker_map:
            existing_id, existing_name, existing_embedding = unknown_speaker_map[speaker_name]

            # Skip if either has NaN
            if np.isnan(embedding).any() or np.isnan(existing_embedding).any():
                print(f"⚠️  NaN embeddings - creating new Unknown despite same label")
            else:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    existing_embedding.reshape(1, -1)
                )[0][0]

                print(f"  Embedding similarity to existing {existing_name}: {similarity:.3f}")

                # If similar enough, reuse the Unknown
                # Optimal threshold 0.20 based on ground truth testing (0 misidentifications + 50% matching)
                if similarity > threshold:
                    print(f"  ✓ Reusing {existing_name} (similar voice, similarity={similarity:.3f})")
                    # Return segment ready to add
                    segment = ConversationSegment(
                        conversation_id=conversation.id,
                        speaker_id=existing_id,
                        speaker_name=existing_name,
                        text=seg.get("text", ""),
                        start_time=conversation.start_time + timedelta(seconds=seg["start"]),
                        end_time=conversation.start_time + timedelta(seconds=seg["end"]),
                        start_offset=seg["start"],
                        end_offset=seg["end"],
                        confidence=seg.get("confidence", 0.0),
                        segment_audio_path=segment_audio_path
                    )
                    return existing_id, existing_name, segment
                else:
                    print(f"  ✗ Different voice ({similarity:.3f}) - creating new Unknown")
                    print(f"    (Original diarization label: '{seg.get('speaker')}')")

        # Create new Unknown speaker (first time OR different voice)
        max_attempts = 5
        for attempt in range(max_attempts):
            timestamp_name = f"Unknown_{int(time.time() * 1000000)}"
            existing = db.query(Speaker).filter(Speaker.name == timestamp_name).first()
            if not existing:
                break
            time.sleep(0.001)
        else:
            timestamp_name = f"Unknown_{uuid.uuid4().hex[:12]}"

        new_speaker = Speaker(name=timestamp_name)
        new_speaker.set_embedding(embedding)
        db.add(new_speaker)
        db.flush()

        speaker_id = new_speaker.id
        speaker_name = new_speaker.name
        unknown_speaker_map[seg.get("speaker")] = (speaker_id, speaker_name, embedding)
        print(f"✓ Auto-enrolled: {speaker_name} (ID: {speaker_id})")

        return speaker_id, speaker_name, None

    def process_audio(self, audio_file: str, enable_transcription: bool = False) -> Tuple[str, pd.DataFrame]:
        """Process audio with speaker diarization and recognition"""
        if not audio_file:
            return "Error: Please provide an audio file", pd.DataFrame()

        # Convert MP3 to WAV if needed (pyannote has issues with MP3 duration metadata)
        # Use high-quality conversion - preserve original sample rate and convert to mono
        if audio_file.lower().endswith('.mp3'):
            import subprocess
            wav_path = audio_file.rsplit('.', 1)[0] + '_converted.wav'
            print(f"Converting MP3 to WAV for reliable processing: {wav_path}")
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', audio_file,
                    '-ac', '1',      # Convert to mono (mix down stereo)
                    '-c:a', 'pcm_s16le',  # PCM 16-bit
                    # Don't specify -ar, let it preserve original sample rate
                    wav_path
                ], check=True, capture_output=True)
                audio_file = wav_path
                print(f"✓ Converted to WAV successfully")
            except subprocess.CalledProcessError as e:
                print(f"Warning: MP3 to WAV conversion failed: {e.stderr.decode()}")
                # Continue with original file

        # Copy uploaded file to permanent storage location
        # Gradio temp files get cleaned up, so we need to persist them
        import shutil
        from datetime import datetime
        data_path = os.getenv("DATA_PATH", "data")
        recordings_dir = f"{data_path}/recordings"
        os.makedirs(recordings_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.basename(audio_file)
        permanent_filename = f"{timestamp}_{original_filename}"
        permanent_path = os.path.join(recordings_dir, permanent_filename)

        # Copy file to permanent location
        shutil.copy2(audio_file, permanent_path)
        print(f"✓ Saved audio to permanent storage: {permanent_path}")

        # Use permanent path for processing and database storage
        audio_file = permanent_path

        db = self.get_db()
        try:
            # Get known speakers
            speakers = db.query(Speaker).all()
            known_speakers = [
                (s.id, s.name, s.get_embedding())
                for s in speakers
            ]

            if enable_transcription:
                # Process with transcription
                # Threshold 0.20 - optimal based on ground truth testing (0 misidentifications + 50% matching)
                result = self.engine.transcribe_with_diarization(
                    audio_file,
                    known_speakers,
                    threshold=self.speaker_threshold
                )

                # Create summary
                summary = f"""
✓ Processing Complete (with Transcription)!

📊 Statistics:
- Total speakers detected: {result['num_speakers']}
- Total segments: {len(result['segments'])}

💡 Saved to Conversations - Check the 💬 Conversations tab to view!
"""

                # Create segments dataframe with transcription
                segments_data = [
                    {
                        "Speaker": seg["speaker"],
                        "Start": f"{seg['start']:.1f}s",
                        "End": f"{seg['end']:.1f}s",
                        "Text": seg["text"],
                        "Confidence": f"{seg['confidence']:.1%}" if seg.get('confidence', 0) > 0 else ""
                    }
                    for seg in result["segments"]
                ]
            else:
                # Process without transcription (diarization only)
                # Threshold 0.20 - optimal based on ground truth testing (0 misidentifications + 50% matching)
                result = self.engine.process_audio_with_recognition(
                    audio_file,
                    known_speakers,
                    threshold=self.speaker_threshold
                )

                # Create summary
                summary = f"""
✓ Processing Complete!

📊 Statistics:
- Total speakers detected: {result['num_speakers']}
- Known speakers: {result['num_known']}
- Unknown speakers: {result['num_unknown']}
- Total segments: {len(result['segments'])}

💡 Saved to Conversations - Check the 💬 Conversations tab to view!
"""

                # Create segments dataframe
                segments_data = [
                    {
                        "Speaker": seg["speaker_name"],
                        "Start (s)": f"{seg['start']:.2f}",
                        "End (s)": f"{seg['end']:.2f}",
                        "Duration (s)": f"{seg['duration']:.2f}",
                        "Confidence": f"{seg['confidence']:.2%}" if seg.get('confidence') else "N/A"
                    }
                    for seg in result["segments"]
                ]

            segments_df = pd.DataFrame(segments_data)

            # Save recording to database (old Recording table)
            filename = os.path.basename(audio_file)
            recording = Recording(
                filename=filename,
                status="completed",
                duration=max(s["end"] for s in result["segments"]) if result["segments"] else 0
            )
            db.add(recording)

            # ALSO save as Conversation (new system)
            from datetime import timedelta
            conv_start = datetime.now()
            duration = max(s["end"] for s in result["segments"]) if result["segments"] else 0

            conversation = Conversation(
                title=f"Recording {conv_start.strftime('%Y-%m-%d %H:%M')}",
                start_time=conv_start - timedelta(seconds=duration),
                end_time=conv_start,
                duration=duration,
                status="completed",
                audio_path=audio_file,
                audio_format="wav",
                num_segments=len(result["segments"]),
                num_speakers=result["num_speakers"]
            )
            db.add(conversation)
            db.flush()  # Get conversation ID

            # Auto-enroll unknown speakers and save segments
            unknown_speaker_map = {}  # Map unknown labels to created speaker IDs
            segments_added = 0

            # For uploaded files, segment_audio_path is the same as audio_file
            segment_audio_path = audio_file

            for seg in result["segments"]:
                speaker_id = None
                speaker_name = seg["speaker"]

                if seg.get("is_known"):
                    # Find speaker by name
                    speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
                    if speaker:
                        speaker_id = speaker.id
                else:
                    # AUTO-ENROLL UNKNOWN SPEAKER
                    speaker_id, speaker_name, ready_segment = self._verify_and_enroll_unknown_speaker(
                        seg, unknown_speaker_map, db, conversation, segment_audio_path, threshold=self.speaker_threshold
                    )
                    if ready_segment:
                        db.add(ready_segment)
                        segments_added += 1
                        continue

                segment = ConversationSegment(
                    conversation_id=conversation.id,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    text=seg.get("text", ""),
                    start_time=conversation.start_time + timedelta(seconds=seg["start"]),
                    end_time=conversation.start_time + timedelta(seconds=seg["end"]),
                    start_offset=seg["start"],
                    end_offset=seg["end"],
                    confidence=seg.get("confidence", 0.0),
                    segment_audio_path=segment_audio_path  # Needed for speaker identification
                )
                db.add(segment)

            db.commit()

            # Clear GPU cache after full audio processing
            self.engine.clear_gpu_cache()

            return summary, segments_df

        except Exception as e:
            return f"Error: {str(e)}", pd.DataFrame()
        finally:
            db.close()

    def add_audio_to_conversation(self, audio_file: str, conversation_id: int, enable_transcription: bool = True, segment_audio_path: str = None) -> str:
        """Add an audio segment to an existing conversation"""
        # Common Whisper hallucinations to filter out
        HALLUCINATION_PHRASES = [
            "thank you.",
            "thanks for watching",
            "thanks for watching.",
            "bye.",
            "bye-bye.",
            "you",
            ".",
            "",
            " "
        ]

        db = self.get_db()
        try:
            # Get conversation
            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                return f"❌ Conversation {conversation_id} not found"

            # Get known speakers
            speakers = db.query(Speaker).all()
            known_speakers = [(s.id, s.name, s.get_embedding()) for s in speakers]

            # Process audio with transcription
            # Threshold 0.20 - optimal based on ground truth testing (0 misidentifications + 50% matching)
            if enable_transcription:
                result = self.engine.transcribe_with_diarization(audio_file, known_speakers, threshold=self.speaker_threshold)
            else:
                result = self.engine.process_audio_with_recognition(audio_file, known_speakers, threshold=self.speaker_threshold)

            # Auto-enroll unknown speakers
            unknown_speaker_map = {}
            segments_added = 0

            for seg in result["segments"]:
                # Filter out hallucinations (if enabled in config)
                if self.filter_hallucinations:
                    text = seg.get("text", "").strip().lower()
                    if text in HALLUCINATION_PHRASES:
                        print(f"🚫 Filtered hallucination: '{seg.get('text')}'")
                        continue  # Skip this segment

                    # Also skip very short transcriptions (likely hallucinations)
                    if len(seg.get("text", "").strip()) < 3:
                        print(f"🚫 Filtered short text: '{seg.get('text')}'")
                        continue

                speaker_id = None
                speaker_name = seg.get("speaker")

                if seg.get("is_known"):
                    # Find speaker by name
                    speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
                    if speaker:
                        speaker_id = speaker.id
                else:
                    # AUTO-ENROLL UNKNOWN SPEAKER
                    speaker_id, speaker_name, ready_segment = self._verify_and_enroll_unknown_speaker(
                        seg, unknown_speaker_map, db, conversation, segment_audio_path, threshold=self.speaker_threshold
                    )
                    if ready_segment:
                        db.add(ready_segment)
                        segments_added += 1
                        continue

                # Add segment to conversation
                segment = ConversationSegment(
                    conversation_id=conversation_id,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    text=seg.get("text", ""),
                    start_time=conversation.start_time + timedelta(seconds=seg["start"]),
                    end_time=conversation.start_time + timedelta(seconds=seg["end"]),
                    start_offset=seg["start"],
                    end_offset=seg["end"],
                    confidence=seg.get("confidence", 0.0),
                    segment_audio_path=segment_audio_path  # Needed for speaker identification with offsets
                )
                db.add(segment)
                segments_added += 1

            # Update conversation metadata
            all_segments = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conversation_id
            ).all()
            conversation.num_segments = len(all_segments)
            conversation.num_speakers = len(set(s.speaker_name for s in all_segments if s.speaker_name))

            db.commit()

            return f"✅ Added {segments_added} segment(s)\n📊 Speakers: {result['num_speakers']}\n🎯 Total in conversation: {conversation.num_segments} segments"

        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_conversation_stats(self, conv_id) -> str:
        """Get conversation statistics"""
        if not conv_id:
            return "No active conversation"

        db = self.get_db()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if not conv:
                return "Conversation not found"

            segments_count = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).count()

            unique_speakers = db.query(ConversationSegment.speaker_name).filter(
                ConversationSegment.conversation_id == conv_id
            ).distinct().count()

            duration = conv.duration or 0
            if conv.start_time and not conv.end_time:
                duration = (datetime.now() - conv.start_time).total_seconds()

            return f"""
**Conversation #{conv_id}**
Status: {conv.status}
Duration: {duration/60:.1f} minutes
Segments: {segments_count}
Speakers: {unique_speakers}
"""
        finally:
            db.close()

    def get_conversation_segments_df(self, conv_id) -> pd.DataFrame:
        """Get segments for a conversation as dataframe"""
        if not conv_id:
            return pd.DataFrame()

        db = self.get_db()
        try:
            segments = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).order_by(ConversationSegment.id.desc()).all()

            if not segments:
                return pd.DataFrame(columns=["ID", "Speaker", "Text", "Time"])

            data = [{
                "ID": s.id,
                "Speaker": s.speaker_name or "Unknown",
                "Text": (s.text[:60] + "...") if s.text and len(s.text) > 60 else s.text or "",
                "Time": f"{s.start_offset:.1f}s"
            } for s in segments]

            return pd.DataFrame(data)
        finally:
            db.close()

    def get_speaker_names(self):
        """Get list of speaker names for dropdown (excluding Unknown speakers)"""
        db = self.get_db()
        try:
            speakers = db.query(Speaker).all()
            # Filter out Unknown_xxx speakers - they're for internal matching only
            names = [s.name for s in speakers if not s.name.startswith("Unknown_")]
            # Return as Dropdown update
            return gr.Dropdown(choices=names)
        finally:
            db.close()

    def get_segment_audio(self, segment_id: int) -> Tuple[str, str]:
        """Get audio file for a specific segment"""
        db = self.get_db()
        try:
            segment = db.query(ConversationSegment).filter(ConversationSegment.id == segment_id).first()
            if not segment:
                return None, "Segment not found"

            conversation = segment.conversation

            # Determine source audio file
            # IMPORTANT: Prefer segment_audio_path because offsets are relative to it!
            # Only use conversation.audio_path if no segment file exists (legacy/uploaded files)
            source_audio = None
            if segment.segment_audio_path and os.path.exists(segment.segment_audio_path):
                source_audio = segment.segment_audio_path
            elif conversation.audio_path and os.path.exists(conversation.audio_path):
                source_audio = conversation.audio_path
            else:
                return None, "⚠️ Audio file not found"

            # Extract just this segment's portion using ffmpeg
            import subprocess
            import tempfile

            # Create temporary file for the extracted segment
            temp_dir = "data/temp_segments"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = f"{temp_dir}/segment_{segment_id}.wav"

            # Use ffmpeg to extract the specific time range
            try:
                duration = segment.end_offset - segment.start_offset
                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(segment.start_offset),
                    "-t", str(duration),
                    "-i", source_audio,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    temp_path
                ], check=True, capture_output=True)

                return temp_path, f"Loaded segment {segment_id} ({duration:.1f}s)"
            except subprocess.CalledProcessError as e:
                return None, f"Error extracting audio: {e.stderr.decode()}"

        except Exception as e:
            return None, f"Error: {str(e)}"
        finally:
            db.close()

    def get_conversation_choices(self):
        """Get conversation choices for dropdown (returns Gradio update)"""
        db = self.get_db()
        try:
            convs = db.query(Conversation).order_by(Conversation.start_time.desc()).limit(50).all()
            if not convs:
                return gr.update(choices=[])

            choices = []
            for c in convs:
                # Format: ("Title - ID 123 (50 segments)", 123)
                label = f"{c.title or 'Untitled'} - ID {c.id} ({c.num_segments or 0} segments)"
                choices.append((label, c.id))
            return gr.update(choices=choices)
        finally:
            db.close()

    def list_conversations(self) -> pd.DataFrame:
        """List all conversations"""
        db = self.get_db()
        try:
            convs = db.query(Conversation).order_by(Conversation.start_time.desc()).limit(50).all()
            if not convs:
                return pd.DataFrame(columns=["ID", "Title", "Start", "Duration (min)", "Segments", "Speakers"])

            data = []
            for c in convs:
                # Calculate duration - either final or elapsed if still recording
                if c.duration:
                    duration_str = f"{c.duration/60:.1f}"
                elif c.start_time and c.status == "recording":
                    elapsed = (datetime.now() - c.start_time).total_seconds()
                    duration_str = f"{elapsed/60:.1f} ⏺️"  # Recording indicator
                else:
                    duration_str = "N/A"

                data.append({
                    "ID": c.id,
                    "Title": c.title or "Untitled",
                    "Start": c.start_time.strftime("%Y-%m-%d %H:%M") if c.start_time else "N/A",
                    "Duration (min)": duration_str,
                    "Segments": c.num_segments or 0,
                    "Speakers": c.num_speakers or 0
                })
            return pd.DataFrame(data)
        finally:
            db.close()

    def view_conversation(self, conv_id: int):
        """View conversation details and return timer state"""
        if not conv_id:
            return "Please enter a conversation ID", pd.DataFrame(), gr.update(active=False)

        db = self.get_db()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if not conv:
                return f"Conversation {conv_id} not found", pd.DataFrame(), gr.update(active=False)

            # Calculate duration - either final duration or elapsed time if still recording
            if conv.duration:
                duration_text = f"{conv.duration/60:.1f} minutes"
            elif conv.start_time and conv.status == "recording":
                elapsed = (datetime.now() - conv.start_time).total_seconds()
                duration_text = f"{elapsed/60:.1f} minutes (recording...)"
            else:
                duration_text = "N/A"

            details = f"""
**{conv.title}**
Started: {conv.start_time.strftime('%Y-%m-%d %H:%M:%S') if conv.start_time else 'N/A'}
Duration: {duration_text}
Speakers: {conv.num_speakers or 0} | Segments: {conv.num_segments or 0}
Status: {conv.status} | Format: {conv.audio_format or 'N/A'}
"""

            segments = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).order_by(ConversationSegment.id).all()  # Order by ID for chronological order

            # Activate timer if conversation is recording
            timer_active = conv.status == "recording"

            if not segments:
                # Check if conversation is recording - show helpful message
                if conv.status == "recording":
                    details += "\n\n⏳ Recording in progress - segments updating automatically..."
                return details, pd.DataFrame(columns=["ID", "Speaker", "Text", "Time", "Confidence"]), gr.update(active=timer_active)

            seg_data = [{
                "ID": s.id,
                "Speaker": f"⚠️ {s.speaker_name or 'Unknown'}" if s.is_misidentified else (s.speaker_name or "Unknown"),
                "Text": s.text[:100] + "..." if s.text and len(s.text) > 100 else s.text or "",
                "Time": f"{s.start_offset:.1f}-{s.end_offset:.1f}s",
                "Confidence": f"{s.confidence:.2%}" if s.confidence else "N/A"
            } for s in segments]

            return details, pd.DataFrame(seg_data), gr.update(active=timer_active)
        finally:
            db.close()

    def identify_speaker_in_segment(self, conv_id: int, segment_id: int, speaker_name: str) -> str:
        """Identify speaker in a segment"""
        print(f"DEBUG: identify_speaker_in_segment called with:")
        print(f"  conv_id: {conv_id} (type: {type(conv_id)})")
        print(f"  segment_id: {segment_id} (type: {type(segment_id)})")
        print(f"  speaker_name: '{speaker_name}' (type: {type(speaker_name)})")

        if not conv_id or not segment_id or not speaker_name:
            return "Error: Please provide conversation ID, segment ID, and speaker name"

        try:
            # Call API endpoint
            print(f"DEBUG: Sending POST to /api/v1/conversations/{conv_id}/segments/{segment_id}/identify")
            print(f"DEBUG: Payload: {{'speaker_name': '{speaker_name}', 'enroll': True}}")

            response = requests.post(
                f"http://localhost:8000/api/v1/conversations/{conv_id}/segments/{segment_id}/identify",
                json={"speaker_name": speaker_name, "enroll": True}
            )

            print(f"DEBUG: Response status: {response.status_code}")
            print(f"DEBUG: Response body: {response.text}")

            if response.status_code == 200:
                return f"✓ {response.json()['message']}"
            else:
                return f"Error: {response.json().get('detail', 'Unknown error')}"

        except Exception as e:
            print(f"DEBUG: Exception: {e}")
            return f"Error: {str(e)}"

    def get_conversations_for_selection(self):
        """Get conversations formatted for CheckboxGroup"""
        db = self.get_db()
        try:
            convs = db.query(Conversation).order_by(Conversation.start_time.desc()).limit(100).all()
            # Format as "ID: Title (Date)"
            choices = []
            for c in convs:
                date_str = c.start_time.strftime("%Y-%m-%d %H:%M") if c.start_time else "N/A"
                title = c.title or "Untitled"
                choices.append(f"{c.id}: {title} ({date_str})")

            return gr.CheckboxGroup(choices=choices, value=[])

        finally:
            db.close()

    def delete_multiple_conversations(self, selected_conversations: list) -> str:
        """Delete multiple conversations from checkbox selection"""
        if not selected_conversations:
            return "Please select conversations to delete"

        db = self.get_db()
        try:
            # Parse IDs from "ID: Title (Date)" format
            ids = []
            for item in selected_conversations:
                try:
                    conv_id = int(item.split(':')[0].strip())
                    ids.append(conv_id)
                except (ValueError, IndexError):
                    return f"Error: Invalid selection format: {item}"

            deleted = []
            errors = []
            segments_deleted = 0

            for conv_id in ids:
                conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                if conv:
                    title = conv.title or f"Conv {conv_id}"

                    # Delete audio file if it exists
                    if conv.audio_path and os.path.exists(conv.audio_path):
                        try:
                            os.remove(conv.audio_path)
                        except Exception as e:
                            print(f"Warning: Could not delete audio file {conv.audio_path}: {e}")

                    # Delete segment directory if it exists
                    segment_dir = f"data/stream_segments/conv_{conv_id}"
                    if os.path.exists(segment_dir):
                        try:
                            import shutil
                            shutil.rmtree(segment_dir)
                        except Exception as e:
                            print(f"Warning: Could not delete segment directory {segment_dir}: {e}")

                    # Count segments before deletion (cascade will delete them)
                    seg_count = db.query(ConversationSegment).filter(
                        ConversationSegment.conversation_id == conv_id
                    ).count()
                    segments_deleted += seg_count

                    # Delete conversation (cascade deletes segments)
                    db.delete(conv)
                    deleted.append(title)
                else:
                    errors.append(f"ID {conv_id} not found")

            db.commit()

            result = f"✓ Deleted {len(deleted)} conversation(s) and {segments_deleted} segment(s)"
            if errors:
                result += f"\n⚠ Errors: {', '.join(errors)}"

            return result

        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            db.close()

    def get_audio_devices(self):
        """List available audio input devices"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = []

            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    # Format: "0: Built-in Microphone (2 channels)"
                    name = f"{idx}: {device['name']} ({device['max_input_channels']} ch)"
                    input_devices.append(name)

            if not input_devices:
                input_devices = ["No input devices found"]

            # Return as Dropdown update
            return gr.Dropdown(choices=input_devices)
        except Exception as e:
            return gr.Dropdown(choices=[f"Error listing devices: {str(e)}"])

    def start_live_recording(self, device_str: str = None):
        """Start continuous live recording"""
        if not LIVE_RECORDING_AVAILABLE:
            return "❌ Live recording not available. Install PortAudio: sudo apt-get install -y portaudio19-dev"

        if self.live_recorder and self.live_recorder.is_recording:
            return "❌ Already recording"

        try:
            # Parse device index from string like "0: Built-in Microphone (2 ch)"
            device_index = None
            if device_str and device_str.strip():
                try:
                    device_index = int(device_str.split(':')[0].strip())
                except (ValueError, IndexError):
                    return f"❌ Invalid device selection: {device_str}"

            db = self.get_db()

            # Create conversation
            conv = Conversation(
                title=f"Live Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                start_time=datetime.now(),
                status="recording"
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            self.current_conversation_id = conv.id
            db.close()

            # Initialize recorder with device
            self.live_recorder = LiveRecordingManager(self.engine, device=device_index)
            self.live_recorder.on_segment_ready = self._on_segment_ready
            self.live_recorder.start_recording(conv.id)
            self.live_segments = []

            device_msg = f" on device {device_index}" if device_index is not None else ""
            return f"✅ Live recording started{device_msg} (ID: {conv.id})"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def stop_live_recording(self):
        """Stop live recording"""
        if not self.live_recorder or not self.live_recorder.is_recording:
            return "❌ Not recording"

        try:
            self.live_recorder.stop_recording()

            # Update conversation
            db = self.get_db()
            conv = db.query(Conversation).filter(Conversation.id == self.current_conversation_id).first()
            if conv:
                conv.end_time = datetime.now()
                conv.duration = (conv.end_time - conv.start_time).total_seconds()
                conv.status = "completed"
                db.commit()
            db.close()

            return f"✅ Recording stopped ({len(self.live_segments)} segments processed)"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _on_segment_ready(self, segment_info: dict, conversation_id: int):
        """Callback when a segment is ready for processing"""
        try:
            db = self.get_db()

            # Get known speakers
            speakers = db.query(Speaker).all()
            known_speakers = [(s.id, s.name, s.get_embedding()) for s in speakers]

            # Process segment
            # Threshold 0.20 - optimal based on ground truth testing (0 misidentifications + 50% matching)
            result = self.engine.transcribe_with_diarization(
                segment_info["audio_path"],
                known_speakers,
                threshold=self.speaker_threshold
            )

            # Save segments
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            for seg in result["segments"]:
                speaker_id = None
                speaker_name = seg["speaker"]

                if seg.get("is_known"):
                    speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
                    if speaker:
                        speaker_id = speaker.id
                else:
                    # Auto-enroll unknown
                    embedding = seg.get("embedding")
                    if embedding is not None:
                        new_speaker = Speaker(name=f"Unknown_{int(time.time())}")
                        new_speaker.set_embedding(embedding)
                        db.add(new_speaker)
                        db.flush()
                        speaker_id = new_speaker.id
                        speaker_name = new_speaker.name

                segment = ConversationSegment(
                    conversation_id=conversation_id,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    text=seg.get("text", ""),
                    start_time=segment_info["timestamp"],
                    end_time=segment_info["timestamp"],
                    start_offset=0,
                    end_offset=segment_info["duration"],
                    confidence=seg.get("confidence", 0.0)
                )
                db.add(segment)
                self.live_segments.append({
                    "speaker": speaker_name,
                    "text": seg.get("text", "")[:100]
                })

            db.commit()
            db.close()

            # Clear GPU cache after segment processing
            self.engine.clear_gpu_cache()

            # Cleanup temp file
            if os.path.exists(segment_info["audio_path"]):
                os.remove(segment_info["audio_path"])

        except Exception as e:
            print(f"Error processing segment: {e}")

    def get_live_status(self):
        """Get current live recording status"""
        if not self.live_recorder or not self.live_recorder.is_recording:
            return "Status: Stopped", pd.DataFrame()

        status = self.live_recorder.get_status()
        status_text = f"""
**Status**: Recording 🔴
**Duration**: {status['duration']:.1f}s
**Last Speech**: {status['last_speech']:.1f}s ago
**Segments Processed**: {len(self.live_segments)}
"""

        if self.live_segments:
            df = pd.DataFrame(self.live_segments[-10:])  # Last 10 segments
        else:
            df = pd.DataFrame(columns=["speaker", "text"])

        return status_text, df

    def start_streaming(self):
        """Start streaming audio recording"""
        if self.streaming_recorder.is_recording:
            return "❌ Already recording"

        # Create new conversation
        db = self.get_db()
        try:
            conv = Conversation(
                title=f"Live Stream {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                start_time=datetime.now(),
                status="recording"
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            self.streaming_conversation_id = conv.id
            self.streaming_results = []

            # Start recorder
            self.streaming_recorder.start_recording(conv.id)

            return f"✅ Started streaming (Conversation #{conv.id})\n\n🎤 Speak now... Processing happens after {self.streaming_recorder.silence_duration} seconds of silence."

        finally:
            db.close()

    def stop_streaming(self):
        """Stop streaming audio recording"""
        if not self.streaming_recorder.is_recording:
            return "❌ Not recording"

        conv_id = self.streaming_conversation_id
        self.streaming_recorder.stop_recording()

        # Concatenate all segment files into full conversation audio
        print("🔗 Concatenating streaming segments...")
        full_audio_path = self.streaming_recorder.concatenate_segments()

        # Convert to MP3 for space efficiency
        mp3_path = None
        if full_audio_path and os.path.exists(full_audio_path):
            try:
                print("🎵 Converting to MP3...")
                mp3_path = convert_to_mp3(
                    full_audio_path,
                    bitrate="192k",
                    delete_original=True  # Delete WAV after conversion
                )
                print(f"✅ MP3 saved: {mp3_path}")
            except Exception as e:
                print(f"⚠️ MP3 conversion failed: {e}")
                mp3_path = full_audio_path  # Fall back to WAV

        # Mark conversation as completed
        db = self.get_db()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if conv:
                conv.status = "completed"
                conv.end_time = datetime.now()
                if conv.start_time:
                    conv.duration = (conv.end_time - conv.start_time).total_seconds()

                # Set audio path and format
                if mp3_path:
                    conv.audio_path = mp3_path
                    conv.audio_format = "mp3" if mp3_path.endswith(".mp3") else "wav"

                # Update counts
                segments = db.query(ConversationSegment).filter(
                    ConversationSegment.conversation_id == conv_id
                ).all()
                conv.num_segments = len(segments)
                conv.num_speakers = len(set(s.speaker_name for s in segments if s.speaker_name))

                db.commit()

            self.streaming_conversation_id = None

            return f"✅ Streaming stopped\n\n📊 Conversation #{conv_id} saved with {len(self.streaming_results)} segments\n🎵 Audio: {mp3_path}"

        finally:
            db.close()

    def process_streaming_chunk(self, audio_chunk):
        """Process audio chunk from streaming input"""
        if audio_chunk is None:
            return None, "⚪ Idle", 0.0

        # Process through streaming recorder
        stats = self.streaming_recorder.process_audio_chunk(audio_chunk)
        self.audio_level = stats.get("audio_level", 0.0)

        # Update VAD indicator
        if stats.get("speech_detected", False):
            vad_text = "🟢 SPEECH DETECTED"
        else:
            vad_text = "🔴 Silence"

        return audio_chunk, vad_text, self.audio_level

    def _on_stream_segment_processed(self, segment_info: dict):
        """Callback when a segment finishes processing"""
        try:
            # Process the audio file
            result = self.add_audio_to_conversation(
                segment_info["path"],
                segment_info["conversation_id"],
                enable_transcription=True,
                segment_audio_path=segment_info["path"]
            )

            # Add to results for display
            self.streaming_results.append({
                "segment_id": segment_info["id"],
                "timestamp": segment_info["timestamp"].strftime("%H:%M:%S"),
                "duration": segment_info["duration"],
                "result": result
            })

            print(f"✅ Segment {segment_info['id']} processed and saved")

            # Clear GPU cache after processing to prevent memory accumulation
            self.engine.clear_gpu_cache()

        except RuntimeError as e:
            # Whisper tensor errors - likely due to model state issues
            if "key" in str(e).lower() or "value" in str(e).lower():
                print(f"⚠️ Whisper model error on segment {segment_info['id']} (duration: {segment_info['duration']:.1f}s)")
                print(f"   Error: {e}")
                print(f"   Skipping this segment - try shorter pauses between speech")
            else:
                print(f"❌ Runtime error processing segment {segment_info['id']}: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error processing stream segment {segment_info['id']}: {e}")
            import traceback
            traceback.print_exc()

    def get_streaming_display(self):
        """Get current streaming status and results for display"""
        if not self.streaming_recorder.is_recording and len(self.streaming_results) == 0:
            return "Not recording", pd.DataFrame(), 0.0, "0 / 0 / 0", "⚪ Idle"

        stats = self.streaming_recorder.get_stats()

        # VAD indicator
        if self.streaming_recorder.speech_detected:
            vad_indicator = "🟢 SPEECH DETECTED"
        else:
            vad_indicator = "🔴 Silence"

        # Status text
        if stats["is_recording"]:
            status = f"""🎤 **RECORDING**

Conversation: #{self.streaming_conversation_id}
Segments queued: {stats['segments_queued']}
Segments processed: {stats['segments_processed']}
Queue size: {stats['queue_size']}
Buffer: {stats['buffer_chunks']} chunks
"""
        else:
            status = "⏹️ Not recording"
            vad_indicator = "⚪ Idle"

        # Results dataframe
        if len(self.streaming_results) > 0:
            # Get latest segments from database
            db = self.get_db()
            try:
                segments = db.query(ConversationSegment).filter(
                    ConversationSegment.conversation_id == self.streaming_conversation_id
                ).order_by(ConversationSegment.id.desc()).limit(20).all()

                if segments:
                    data = [{
                        "Time": (
                            f"{s.start_time.strftime('%m/%d %H:%M:%S')}-{s.end_time.strftime('%H:%M:%S')}"
                            if s.start_time and s.end_time else ""
                        ),
                        "Speaker": s.speaker_name or "Unknown",
                        "Text": (s.text[:80] + "...") if s.text and len(s.text) > 80 else s.text or ""
                    } for s in segments]  # Already ordered by ID desc (newest first)
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(columns=["Time", "Speaker", "Text"])
            finally:
                db.close()
        else:
            df = pd.DataFrame(columns=["Time", "Speaker", "Text"])

        # Stats summary
        stats_text = f"{stats['segments_queued']} / {stats['segments_processed']} / {len(self.streaming_results)}"

        return status, df, self.audio_level, stats_text, vad_indicator

    def load_segment_for_labeling(self, conv_id, segment_idx):
        """Load a segment for ground truth labeling"""
        from app.models import GroundTruthLabel
        import pandas as pd

        if not conv_id:
            return "Select a conversation first", None, "", 0, pd.DataFrame()

        db = self.get_db()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if not conv:
                return "Conversation not found", None, "", 0, pd.DataFrame()

            segments = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).order_by(ConversationSegment.id).all()

            if not segments:
                return "No segments found", None, "", 0, pd.DataFrame()

            # Get current segment
            segment_idx = max(0, min(segment_idx, len(segments) - 1))
            seg = segments[segment_idx]

            # Segment info
            info = f"Segment {segment_idx + 1}/{len(segments)}\nID: {seg.id}\nCurrent Speaker: {seg.speaker_name}\nTime: {seg.start_offset:.1f}s - {seg.end_offset:.1f}s"

            # Get audio file path - extract just this segment for playback
            # Check if segment has its own audio file (different from conversation audio)
            has_individual_segment_file = (
                seg.segment_audio_path and
                os.path.exists(seg.segment_audio_path) and
                seg.segment_audio_path != conv.audio_path
            )

            if has_individual_segment_file:
                audio_path = seg.segment_audio_path
            elif conv.audio_path and os.path.exists(conv.audio_path):
                # Extract segment from full audio
                from app.audio_utils import extract_segment
                try:
                    audio_path = extract_segment(
                        conv.audio_path,
                        seg.start_offset,
                        seg.end_offset
                    )
                    print(f"Extracted segment {seg.id}: {seg.start_offset:.1f}s - {seg.end_offset:.1f}s")
                except Exception as e:
                    print(f"Error extracting segment audio: {e}")
                    audio_path = None
            else:
                audio_path = None

            # Get all labels for this conversation
            labels = db.query(GroundTruthLabel).join(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).all()

            labels_data = [{
                "Seg ID": label.segment_id,
                "True Speaker": label.true_speaker_name,
                "Current Speaker": label.segment.speaker_name,
                "Time": f"{label.segment.start_offset:.1f}s",
                "Text": (label.segment.text[:50] + "...") if label.segment.text and len(label.segment.text) > 50 else label.segment.text or ""
            } for label in labels]

            labels_df = pd.DataFrame(labels_data) if labels_data else pd.DataFrame()

            return info, audio_path, seg.text or "", segment_idx, labels_df

        finally:
            db.close()

    def save_ground_truth_label(self, conv_id, segment_idx, true_speaker):
        """Save ground truth label for a segment"""
        from app.models import GroundTruthLabel
        import pandas as pd

        if not conv_id or not true_speaker:
            return "❌ Please select conversation and enter speaker name", pd.DataFrame()

        db = self.get_db()
        try:
            segments = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).order_by(ConversationSegment.id).all()

            if not segments or segment_idx >= len(segments):
                return "❌ Invalid segment", pd.DataFrame()

            seg = segments[segment_idx]

            # Check if label already exists
            existing = db.query(GroundTruthLabel).filter(
                GroundTruthLabel.segment_id == seg.id
            ).first()

            if existing:
                existing.true_speaker_name = true_speaker
                existing.labeled_at = datetime.utcnow()
                status = f"✓ Updated label for segment {seg.id}"
            else:
                label = GroundTruthLabel(
                    segment_id=seg.id,
                    true_speaker_name=true_speaker
                )
                db.add(label)
                status = f"✓ Saved label for segment {seg.id}"

            db.commit()

            # Get updated labels
            labels = db.query(GroundTruthLabel).join(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).all()

            labels_data = [{
                "Seg ID": label.segment_id,
                "True Speaker": label.true_speaker_name,
                "Current Speaker": label.segment.speaker_name,
                "Time": f"{label.segment.start_offset:.1f}s",
                "Text": (label.segment.text[:50] + "...") if label.segment.text and len(label.segment.text) > 50 else label.segment.text or ""
            } for label in labels]

            labels_df = pd.DataFrame(labels_data) if labels_data else pd.DataFrame()

            return status, labels_df

        finally:
            db.close()

    def next_segment(self, conv_id, current_idx):
        """Move to next segment"""
        return self.load_segment_for_labeling(conv_id, current_idx + 1)

    def prev_segment(self, conv_id, current_idx):
        """Move to previous segment"""
        return self.load_segment_for_labeling(conv_id, max(0, current_idx - 1))

    def clear_ground_truth_labels(self, conv_id):
        """Clear all ground truth labels for a conversation"""
        from app.models import GroundTruthLabel
        import pandas as pd

        if not conv_id:
            return "❌ Select a conversation first", pd.DataFrame()

        db = self.get_db()
        try:
            # Delete all labels for this conversation
            db.query(GroundTruthLabel).filter(
                GroundTruthLabel.segment_id.in_(
                    db.query(ConversationSegment.id).filter(
                        ConversationSegment.conversation_id == conv_id
                    )
                )
            ).delete(synchronize_session=False)
            db.commit()

            return "✓ Cleared all ground truth labels", pd.DataFrame()

        finally:
            db.close()

    def export_ground_truth_labels(self, conv_id):
        """Export ground truth labels to JSON"""
        from app.models import GroundTruthLabel
        import json

        if not conv_id:
            return "❌ Select a conversation first"

        db = self.get_db()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if not conv:
                return "❌ Conversation not found"

            labels = db.query(GroundTruthLabel).join(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).all()

            if not labels:
                return "❌ No labels found for this conversation"

            export_data = {
                'conversation_id': conv_id,
                'conversation_title': conv.title,
                'audio_path': conv.audio_path,
                'total_labels': len(labels),
                'labels': [{
                    'segment_id': label.segment_id,
                    'true_speaker': label.true_speaker_name,
                    'current_speaker': label.segment.speaker_name,
                    'start': label.segment.start_offset,
                    'end': label.segment.end_offset,
                    'text': label.segment.text
                } for label in labels]
            }

            # Save to file
            output_file = f"tests/ground_truth_labels_conv_{conv_id}.json"
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            return f"✓ Exported {len(labels)} labels to: {output_file}"

        finally:
            db.close()

    def get_segments_table(self, conv_id):
        """Get segments table for a conversation (for refreshing after updates)"""
        if not conv_id:
            return pd.DataFrame()

        db = self.get_db()
        try:
            segments = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conv_id
            ).order_by(ConversationSegment.id).all()

            if not segments:
                return pd.DataFrame(columns=["ID", "Speaker", "Text", "Time", "Confidence"])

            seg_data = [{
                "ID": s.id,
                "Speaker": f"⚠️ {s.speaker_name or 'Unknown'}" if s.is_misidentified else (s.speaker_name or "Unknown"),
                "Text": s.text[:100] + "..." if s.text and len(s.text) > 100 else s.text or "",
                "Time": f"{s.start_offset:.1f}-{s.end_offset:.1f}s",
                "Confidence": f"{s.confidence:.2%}" if s.confidence else "N/A"
            } for s in segments]

            return pd.DataFrame(seg_data)
        finally:
            db.close()

    def toggle_segment_misidentified(self, conv_id, segment_id, is_misidentified):
        """Toggle misidentified flag on a segment and recalculate speaker embedding"""
        if not conv_id or segment_id is None:
            return "❌ Invalid parameters", self.get_segments_table(conv_id), ""

        db = self.get_db()
        try:
            segment = db.query(ConversationSegment).filter(
                ConversationSegment.id == segment_id,
                ConversationSegment.conversation_id == conv_id
            ).first()

            if not segment:
                return "❌ Segment not found", self.get_segments_table(conv_id), ""

            # Check if the value is already set to what we're trying to set it to
            # This prevents unnecessary recalculations when just clicking on the segment
            if segment.is_misidentified == is_misidentified:
                # No change needed - just return current state
                marker = "⚠️ " if segment.is_misidentified else ""
                updated_info = f"Segment ID: {segment_id}\nSpeaker: {marker}{segment.speaker_name}\nTime: {segment.start_offset:.1f}s - {segment.end_offset:.1f}s"
                return "", self.get_segments_table(conv_id), updated_info

            speaker_id = segment.speaker_id
            segment.is_misidentified = is_misidentified
            db.commit()

            # CRITICAL: Recalculate speaker embedding to exclude/include this segment
            recalc_msg = ""
            if speaker_id:
                self.recalculate_speaker_embedding(speaker_id, db)
                recalc_msg = " (embedding recalculated)"

            # Get updated segment info
            db.refresh(segment)
            marker = "⚠️ " if segment.is_misidentified else ""
            updated_info = f"Segment ID: {segment_id}\nSpeaker: {marker}{segment.speaker_name}\nTime: {segment.start_offset:.1f}s - {segment.end_offset:.1f}s"

            status = "marked as misidentified" if is_misidentified else "unmarked"
            return f"✓ Segment {segment_id} {status}{recalc_msg}", self.get_segments_table(conv_id), updated_info

        finally:
            db.close()

    def reassign_segment(self, conv_id, segment_id, new_speaker_name):
        """Reassign a segment to a different speaker"""
        if not conv_id or segment_id is None or not new_speaker_name:
            return "❌ Invalid parameters", self.get_segments_table(conv_id)

        db = self.get_db()
        try:
            segment = db.query(ConversationSegment).filter(
                ConversationSegment.id == segment_id,
                ConversationSegment.conversation_id == conv_id
            ).first()

            if not segment:
                return "❌ Segment not found", self.get_segments_table(conv_id)

            old_speaker_id = segment.speaker_id
            old_speaker_name = segment.speaker_name

            # Find or create the target speaker
            target_speaker = db.query(Speaker).filter(Speaker.name == new_speaker_name).first()

            if not target_speaker:
                # Create new speaker with embedding from this segment
                conv = segment.conversation
                audio_file = segment.segment_audio_path if segment.segment_audio_path and os.path.exists(segment.segment_audio_path) else conv.audio_path

                if audio_file and os.path.exists(audio_file):
                    try:
                        # Extract embedding from this segment
                        embedding = self.engine.extract_segment_embedding(
                            audio_file,
                            segment.start_offset,
                            segment.end_offset
                        )

                        # Create speaker with embedding
                        target_speaker = Speaker(name=new_speaker_name)
                        target_speaker.set_embedding(embedding)
                        db.add(target_speaker)
                        db.flush()  # Get ID
                        print(f"✓ Created new speaker '{new_speaker_name}' with embedding from segment {segment_id}")
                    except Exception as e:
                        print(f"Error creating speaker embedding: {e}")
                        # Fallback: create without proper embedding (will be calculated later)
                        import numpy as np
                        target_speaker = Speaker(name=new_speaker_name)
                        target_speaker.set_embedding(np.zeros(512, dtype=np.float32))
                        db.add(target_speaker)
                        db.flush()
                else:
                    # No audio available - create placeholder
                    import numpy as np
                    target_speaker = Speaker(name=new_speaker_name)
                    target_speaker.set_embedding(np.zeros(512, dtype=np.float32))
                    db.add(target_speaker)
                    db.flush()

            # Assign to speaker
            segment.speaker_id = target_speaker.id
            segment.speaker_name = target_speaker.name

            # Unmark as misidentified since we're correcting it
            segment.is_misidentified = False

            db.commit()

            # Recalculate embeddings for both old and new speakers
            if old_speaker_id:
                self.recalculate_speaker_embedding(old_speaker_id, db)
            if target_speaker:
                self.recalculate_speaker_embedding(target_speaker.id, db)

            # Auto-cleanup: Delete orphaned Unknown speakers
            cleanup_msg = ""
            if old_speaker_name and old_speaker_name.startswith("Unknown_") and old_speaker_id:
                # Check if any segments still use this Unknown speaker
                remaining_segments = db.query(ConversationSegment).filter(
                    ConversationSegment.speaker_id == old_speaker_id
                ).count()

                if remaining_segments == 0:
                    # No segments use this Unknown speaker anymore - delete it
                    orphaned_speaker = db.query(Speaker).filter(Speaker.id == old_speaker_id).first()
                    if orphaned_speaker:
                        db.delete(orphaned_speaker)
                        db.commit()
                        cleanup_msg = f" (auto-deleted orphaned {old_speaker_name})"

            # Get updated segment info to display
            db.refresh(segment)
            updated_info = f"Segment ID: {segment_id}\nSpeaker: {segment.speaker_name}\nTime: {segment.start_offset:.1f}s - {segment.end_offset:.1f}s"

            return (
                f"✓ Segment {segment_id} reassigned from '{old_speaker_name}' to '{new_speaker_name}'{cleanup_msg}",
                self.get_segments_table(conv_id),
                updated_info
            )

        finally:
            db.close()

    def recalculate_speaker_embedding(self, speaker_id, db=None):
        """Recalculate speaker embedding excluding misidentified segments"""
        import numpy as np

        should_close = db is None
        if db is None:
            db = self.get_db()

        try:
            speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                return

            # Get all segments for this speaker that are NOT misidentified
            segments = db.query(ConversationSegment).filter(
                ConversationSegment.speaker_id == speaker_id,
                ConversationSegment.is_misidentified == False
            ).all()

            if not segments:
                print(f"⚠️  No valid segments for speaker {speaker.name} - cannot recalculate embedding")
                return

            # Extract embeddings from all non-misidentified segments
            embeddings = []
            for seg in segments:
                conv = seg.conversation
                audio_file = seg.segment_audio_path if seg.segment_audio_path and os.path.exists(seg.segment_audio_path) else conv.audio_path

                if not audio_file or not os.path.exists(audio_file):
                    continue

                try:
                    emb = self.engine.extract_segment_embedding(
                        audio_file,
                        seg.start_offset,
                        seg.end_offset
                    )
                    if not np.isnan(emb).any():
                        embeddings.append(emb)
                except Exception as e:
                    print(f"Error extracting embedding for segment {seg.id}: {e}")
                    continue

            if not embeddings:
                print(f"⚠️  Could not extract any valid embeddings for speaker {speaker.name}")
                return

            # Calculate average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            speaker.set_embedding(avg_embedding)
            db.commit()

            print(f"✓ Recalculated embedding for '{speaker.name}' using {len(embeddings)} non-misidentified segments")

            # Clear GPU cache after all embedding extractions
            self.engine.clear_gpu_cache()

        finally:
            if should_close:
                db.close()

    def create_backup_snapshot(self):
        """Create a timestamped backup of all speakers, embeddings, and segment assignments"""
        import json
        from datetime import datetime
        import numpy as np
        import os

        db = self.get_db()
        try:
            # Create backups directory if it doesn't exist
            os.makedirs("backups", exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_file = f"backups/backup_{timestamp}.json"

            # Collect all data
            backup_data = {
                "timestamp": timestamp,
                "speakers": [],
                "segments": []
            }

            # Export all speakers with embeddings
            speakers = db.query(Speaker).all()
            for speaker in speakers:
                embedding = speaker.get_embedding()
                backup_data["speakers"].append({
                    "id": speaker.id,
                    "name": speaker.name,
                    "embedding": embedding.tolist() if embedding is not None else None
                })

            # Export all segment assignments
            segments = db.query(ConversationSegment).all()
            for seg in segments:
                backup_data["segments"].append({
                    "id": seg.id,
                    "conversation_id": seg.conversation_id,
                    "speaker_id": seg.speaker_id,
                    "speaker_name": seg.speaker_name,
                    "is_misidentified": seg.is_misidentified,
                    "start_offset": seg.start_offset,
                    "end_offset": seg.end_offset
                })

            # Write to file
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            return f"✓ Backup created: {backup_file}\n{len(speakers)} speakers, {len(segments)} segments saved"

        except Exception as e:
            return f"❌ Backup failed: {str(e)}"
        finally:
            db.close()

    def list_backups(self):
        """List all available backup files"""
        import os
        import gradio as gr

        if not os.path.exists("backups"):
            return gr.update(choices=[])

        backups = [f for f in os.listdir("backups") if f.startswith("backup_") and f.endswith(".json")]
        backups.sort(reverse=True)  # Most recent first
        return gr.update(choices=backups)

    def restore_from_backup(self, backup_file):
        """Restore speakers, embeddings, and segment assignments from a backup file"""
        import json
        import numpy as np
        import os

        if not backup_file:
            return "❌ No backup file selected"

        backup_path = os.path.join("backups", backup_file)
        if not os.path.exists(backup_path):
            return f"❌ Backup file not found: {backup_path}"

        db = self.get_db()
        try:
            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            # Restore speakers and embeddings
            speaker_id_map = {}  # Map old IDs to new IDs
            for speaker_data in backup_data["speakers"]:
                old_id = speaker_data["id"]

                # Find existing speaker by name or create new
                speaker = db.query(Speaker).filter(Speaker.name == speaker_data["name"]).first()

                if speaker:
                    # Update existing speaker's embedding
                    if speaker_data["embedding"]:
                        embedding = np.array(speaker_data["embedding"], dtype=np.float32)
                        speaker.set_embedding(embedding)
                    speaker_id_map[old_id] = speaker.id
                else:
                    # Create new speaker
                    speaker = Speaker(name=speaker_data["name"])
                    if speaker_data["embedding"]:
                        embedding = np.array(speaker_data["embedding"], dtype=np.float32)
                        speaker.set_embedding(embedding)
                    db.add(speaker)
                    db.flush()
                    speaker_id_map[old_id] = speaker.id

            db.commit()

            # Restore segment assignments
            segments_updated = 0
            for seg_data in backup_data["segments"]:
                segment = db.query(ConversationSegment).filter(
                    ConversationSegment.id == seg_data["id"]
                ).first()

                if segment:
                    # Map old speaker_id to new speaker_id
                    old_speaker_id = seg_data["speaker_id"]
                    new_speaker_id = speaker_id_map.get(old_speaker_id) if old_speaker_id else None

                    segment.speaker_id = new_speaker_id
                    segment.speaker_name = seg_data["speaker_name"]
                    segment.is_misidentified = seg_data.get("is_misidentified", False)
                    segments_updated += 1

            db.commit()

            return f"✓ Restored from {backup_file}\n{len(speaker_id_map)} speakers, {segments_updated} segments updated"

        except Exception as e:
            db.rollback()
            return f"❌ Restore failed: {str(e)}"
        finally:
            db.close()

    def create_interface(self):
        """Create Gradio interface"""

        with gr.Blocks(title="Speaker Diarization & Recognition", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎤 Speaker Diarization & Recognition System")
            gr.Markdown("Powered by pyannote.audio with GPU acceleration")

            with gr.Tabs():
                # Tab 1: Live Recording (Continuous Streaming)
                with gr.Tab("🔴 Live Recording"):
                    gr.Markdown("### Continuous streaming with real-time transcription")
                    gr.Markdown(f"🎤 Speak naturally - segments auto-process after {self.streaming_recorder.silence_duration} seconds of silence")

                    with gr.Row():
                        start_stream_btn = gr.Button("▶️ Start Streaming", variant="primary", size="lg")
                        stop_stream_btn = gr.Button("⏹️ Stop Streaming", variant="stop", size="lg")

                    # Status and audio level
                    with gr.Row():
                        with gr.Column(scale=2):
                            stream_status = gr.Textbox(
                                label="Status",
                                value="Not recording",
                                lines=6,
                                interactive=False
                            )

                        with gr.Column(scale=1):
                            vad_indicator = gr.Textbox(
                                label="🎙️ Voice Activity Detection (VAD)",
                                value="⚪ Idle",
                                interactive=False
                            )
                            audio_level_display = gr.Number(
                                label="🎚️ Audio Level",
                                value=0.0,
                                interactive=False
                            )
                            stats_display = gr.Textbox(
                                label="📊 Stats (Queued/Processed/Total)",
                                value="0 / 0 / 0",
                                interactive=False
                            )

                    # Streaming audio input (hidden, auto-starts with button)
                    stream_audio = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="",
                        visible=False
                    )

                    # Live transcription feed
                    gr.Markdown("### 📝 Live Transcription Feed (Latest 20 segments)")
                    gr.Markdown("*Auto-updates every 2 seconds during recording*")
                    transcription_feed = gr.Dataframe(
                        headers=["Time", "Speaker", "Text"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )

                    # Hidden timer for auto-refresh (triggers every 2 seconds)
                    auto_refresh_timer = gr.Timer(value=2, active=False)

                    # Event handlers
                    def start_and_show():
                        msg = self.start_streaming()
                        return msg, gr.Audio(visible=True), gr.update(active=True)

                    def stop_and_hide():
                        msg = self.stop_streaming()
                        return msg, gr.Audio(visible=False), gr.update(active=False)

                    start_stream_btn.click(
                        fn=start_and_show,
                        outputs=[stream_status, stream_audio, auto_refresh_timer]
                    )

                    stop_stream_btn.click(
                        fn=stop_and_hide,
                        outputs=[stream_status, stream_audio, auto_refresh_timer]
                    )

                    # Auto-refresh on timer tick (every 2 seconds while recording)
                    auto_refresh_timer.tick(
                        fn=self.get_streaming_display,
                        outputs=[stream_status, transcription_feed, audio_level_display, stats_display, vad_indicator]
                    )

                    # Process streaming chunks and update VAD in real-time
                    stream_audio.stream(
                        fn=self.process_streaming_chunk,
                        inputs=[stream_audio],
                        outputs=[stream_audio, vad_indicator, audio_level_display]
                    )

                    # Initial load
                    demo.load(
                        fn=self.get_streaming_display,
                        outputs=[stream_status, transcription_feed, audio_level_display, stats_display, vad_indicator]
                    )

                # Tab 2: Process Audio
                with gr.Tab("🎵 Process Audio"):
                    gr.Markdown("### Upload or record audio for speaker diarization")

                    with gr.Row():
                        with gr.Column():
                            audio_input = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Audio Input"
                            )
                            enable_transcription = gr.Checkbox(
                                label="🎯 Enable Transcription (Whisper)",
                                value=False,
                                info="Convert speech to text with speaker labels (takes longer)"
                            )
                            process_btn = gr.Button("🚀 Process Audio", variant="primary")

                        with gr.Column():
                            process_output = gr.Textbox(
                                label="Processing Results",
                                lines=10
                            )

                    segments_output = gr.Dataframe(
                        label="Speaker Segments"
                    )

                    process_btn.click(
                        fn=self.process_audio,
                        inputs=[audio_input, enable_transcription],
                        outputs=[process_output, segments_output]
                    )

                # Tab 2: Enroll Speakers
                with gr.Tab("➕ Enroll Speaker"):
                    gr.Markdown("### Enroll a new speaker by providing their name and voice sample")
                    gr.Markdown("**Tip:** Provide 10-30 seconds of clear audio for best results")

                    with gr.Row():
                        with gr.Column():
                            enroll_name = gr.Textbox(
                                label="Speaker Name",
                                placeholder="e.g., John Doe"
                            )
                            enroll_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Voice Sample"
                            )
                            enroll_btn = gr.Button("➕ Enroll Speaker", variant="primary")

                        with gr.Column():
                            enroll_output = gr.Textbox(
                                label="Enrollment Status",
                                lines=3
                            )

                    enroll_btn.click(
                        fn=self.enroll_speaker,
                        inputs=[enroll_name, enroll_audio],
                        outputs=[enroll_output]
                    )

                # Tab 3: Manage Speakers
                with gr.Tab("👥 Manage Speakers"):
                    gr.Markdown("### View and manage enrolled speakers")

                    with gr.Row():
                        with gr.Column(scale=3):
                            speaker_checkboxes = gr.CheckboxGroup(
                                label="Select speakers to delete",
                                choices=[],
                                interactive=True
                            )

                        with gr.Column(scale=1):
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                            delete_unknown_btn = gr.Button("🧹 Delete All Unknown Speakers", variant="secondary")
                            delete_output = gr.Textbox(label="Status", lines=2)

                    refresh_btn.click(
                        fn=self.get_speakers_for_selection,
                        outputs=[speaker_checkboxes]
                    )

                    delete_btn.click(
                        fn=self.delete_multiple_speakers,
                        inputs=[speaker_checkboxes],
                        outputs=[delete_output]
                    ).then(
                        fn=self.get_speakers_for_selection,
                        outputs=[speaker_checkboxes]
                    )

                    delete_unknown_btn.click(
                        fn=self.delete_all_unknown_speakers,
                        outputs=[delete_output]
                    ).then(
                        fn=self.get_speakers_for_selection,
                        outputs=[speaker_checkboxes]
                    )

                    # Auto-load speakers on tab load
                    demo.load(
                        fn=self.get_speakers_for_selection,
                        outputs=[speaker_checkboxes]
                    )

                # Tab 4: Conversations
                with gr.Tab("💬 Conversations"):
                    gr.Markdown("### Browse past conversations and identify speakers")

                    with gr.Row():
                        refresh_conv_btn = gr.Button("🔄 Refresh Conversations")

                    # Delete conversations section
                    with gr.Accordion("🗑️ Delete Conversations", open=False):
                        with gr.Row():
                            with gr.Column(scale=3):
                                conv_checkboxes = gr.CheckboxGroup(
                                    label="Select conversations to delete",
                                    choices=[],
                                    interactive=True
                                )

                            with gr.Column(scale=1):
                                refresh_del_btn = gr.Button("🔄 Refresh", size="sm")
                                delete_conv_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                                delete_conv_output = gr.Textbox(label="Status", lines=2)

                    conv_list = gr.Dataframe(
                        label="Conversations (click to select)",
                        interactive=False,
                        wrap=True
                    )

                    # Hidden state to store selected conversation ID
                    selected_conv_id = gr.State(value=None)

                    conv_details = gr.Textbox(label="Conversation Details", lines=3)

                    with gr.Row():
                        with gr.Column():
                            segments_table = gr.Dataframe(
                                label="Segments (click to select a segment)",
                                interactive=False,
                                wrap=True
                            )
                        with gr.Column(scale=0, min_width=100):
                            refresh_segments_btn = gr.Button("🔄 Refresh Segments", size="sm")

                    # Hidden timer for auto-refresh of segments during recording
                    conv_refresh_timer = gr.Timer(value=3, active=False)

                    # Audio playback and speaker identification section
                    gr.Markdown("### Selected Segment")

                    with gr.Row():
                        with gr.Column(scale=2):
                            # Hidden state for selected segment ID
                            selected_segment_id = gr.State(value=None)

                            segment_info = gr.Textbox(
                                label="Selected Segment Info",
                                value="Click a segment in the table above",
                                interactive=False
                            )

                            segment_audio = gr.Audio(
                                label="Segment Audio Playback",
                                interactive=False
                            )

                        with gr.Column(scale=1):
                            speaker_dropdown = gr.Dropdown(
                                label="Identify as Speaker",
                                choices=[],
                                allow_custom_value=True,
                                interactive=True
                            )
                            refresh_speakers_btn = gr.Button("🔄 Refresh Speakers", size="sm")
                            identify_btn = gr.Button("🏷️ Identify Speaker", variant="primary")
                            identify_output = gr.Textbox(label="Status", lines=2)

                            gr.Markdown("---")
                            gr.Markdown("### Misidentification Controls")

                            misidentified_checkbox = gr.Checkbox(
                                label="⚠️ Mark as Misidentified",
                                info="Check if this segment is assigned to wrong speaker",
                                value=False
                            )

                            reassign_dropdown = gr.Dropdown(
                                label="Reassign to Speaker",
                                choices=[],
                                allow_custom_value=True,
                                interactive=True
                            )
                            refresh_reassign_btn = gr.Button("🔄 Refresh", size="sm")
                            reassign_btn = gr.Button("🔄 Reassign Segment", variant="secondary")
                            misid_output = gr.Textbox(label="Misidentification Status", lines=2)

                            gr.Markdown("---")
                            gr.Markdown("### 💾 Backup & Restore")
                            gr.Markdown("Create snapshots of all speaker embeddings and segment assignments")

                            with gr.Row():
                                create_backup_btn = gr.Button("📸 Create Backup Snapshot", variant="primary")
                            backup_status = gr.Textbox(label="Backup Status", lines=2)

                            backup_dropdown = gr.Dropdown(
                                label="Select Backup to Restore",
                                choices=[],
                                interactive=True
                            )
                            with gr.Row():
                                refresh_backups_btn = gr.Button("🔄 Refresh Backups", size="sm")
                                restore_btn = gr.Button("♻️ Restore from Backup", variant="secondary")
                            restore_status = gr.Textbox(label="Restore Status", lines=2)

                    # Event handlers
                    def select_conversation(evt: gr.SelectData, df):
                        """When a conversation is clicked, load its details"""
                        if evt.index is None or df is None or len(df) == 0:
                            return None

                        row_index = evt.index[0]
                        # Get ID using column name - works correctly even after sorting
                        conv_id = df.iloc[row_index]['ID']
                        return int(conv_id)

                    def select_segment(evt: gr.SelectData, conv_id, df):
                        """When a segment is clicked, load its audio and info"""
                        if evt.index is None or conv_id is None:
                            return None, "No segment selected", None, "Please select a conversation first", False

                        if df is None or len(df) == 0:
                            return None, "No segments available", None, "", False

                        row_index = evt.index[0]

                        # Get the ID from the DataFrame row - this works even with duplicate text
                        # The ID column is always the first column
                        try:
                            segment_id = int(df.iloc[row_index]['ID'])
                        except (ValueError, TypeError, KeyError, IndexError) as e:
                            return None, f"Could not get segment ID from row {row_index}: {e}", None, "", False

                        # Get segment audio using the correct ID
                        audio_file, message = self.get_segment_audio(segment_id)

                        # Get segment's misidentified status
                        db = self.get_db()
                        try:
                            segment = db.query(ConversationSegment).filter(ConversationSegment.id == segment_id).first()
                            is_misidentified = segment.is_misidentified if segment else False
                        finally:
                            db.close()

                        info_text = f"Segment ID: {segment_id}\n{message}"
                        return segment_id, info_text, audio_file, "", is_misidentified

                    # Delete conversations handlers
                    refresh_del_btn.click(
                        fn=self.get_conversations_for_selection,
                        outputs=[conv_checkboxes]
                    )

                    delete_conv_btn.click(
                        fn=self.delete_multiple_conversations,
                        inputs=[conv_checkboxes],
                        outputs=[delete_conv_output]
                    ).then(
                        fn=self.get_conversations_for_selection,
                        outputs=[conv_checkboxes]
                    ).then(
                        fn=self.list_conversations,
                        outputs=[conv_list]
                    )

                    refresh_conv_btn.click(fn=self.list_conversations, outputs=[conv_list])

                    # Manual refresh segments button
                    refresh_segments_btn.click(
                        fn=self.view_conversation,
                        inputs=[selected_conv_id],
                        outputs=[conv_details, segments_table, conv_refresh_timer]
                    )

                    # Auto-refresh on timer tick (every 3 seconds for active recordings)
                    conv_refresh_timer.tick(
                        fn=self.view_conversation,
                        inputs=[selected_conv_id],
                        outputs=[conv_details, segments_table, conv_refresh_timer]
                    )

                    # When conversation row is clicked, extract ID and load details
                    conv_list.select(
                        fn=select_conversation,
                        inputs=[conv_list],
                        outputs=[selected_conv_id]
                    ).then(
                        fn=lambda: (None, None, None, "Click a segment in the table above", None),  # Clear old data first
                        outputs=[conv_details, segments_table, selected_segment_id, segment_info, segment_audio]
                    ).then(
                        fn=self.view_conversation,
                        inputs=[selected_conv_id],
                        outputs=[conv_details, segments_table, conv_refresh_timer]
                    )

                    # When segment row is clicked, load audio and info
                    segments_table.select(
                        fn=select_segment,
                        inputs=[selected_conv_id, segments_table],
                        outputs=[selected_segment_id, segment_info, segment_audio, identify_output, misidentified_checkbox]
                    )

                    # Refresh speaker dropdown
                    refresh_speakers_btn.click(
                        fn=self.get_speaker_names,
                        outputs=[speaker_dropdown]
                    )

                    # Identify speaker
                    identify_btn.click(
                        fn=self.identify_speaker_in_segment,
                        inputs=[selected_conv_id, selected_segment_id, speaker_dropdown],
                        outputs=[identify_output]
                    ).then(
                        fn=self.view_conversation,
                        inputs=[selected_conv_id],
                        outputs=[conv_details, segments_table, conv_refresh_timer]
                    )

                    # Misidentification controls
                    misidentified_checkbox.change(
                        fn=self.toggle_segment_misidentified,
                        inputs=[selected_conv_id, selected_segment_id, misidentified_checkbox],
                        outputs=[misid_output, segments_table, segment_info]
                    )

                    refresh_reassign_btn.click(
                        fn=self.get_speaker_names,
                        outputs=[reassign_dropdown]
                    )

                    reassign_btn.click(
                        fn=self.reassign_segment,
                        inputs=[selected_conv_id, selected_segment_id, reassign_dropdown],
                        outputs=[misid_output, segments_table, segment_info]
                    )

                    # Backup & Restore event handlers
                    create_backup_btn.click(
                        fn=self.create_backup_snapshot,
                        outputs=[backup_status]
                    )

                    refresh_backups_btn.click(
                        fn=self.list_backups,
                        outputs=[backup_dropdown]
                    )

                    restore_btn.click(
                        fn=self.restore_from_backup,
                        inputs=[backup_dropdown],
                        outputs=[restore_status]
                    ).then(
                        fn=self.list_conversations,
                        outputs=[conv_list]
                    )

                    # Auto-load on tab open
                    demo.load(fn=self.list_conversations, outputs=[conv_list])
                    demo.load(fn=self.get_speaker_names, outputs=[speaker_dropdown])

                # Tab 5: Ground Truth Labeling
                with gr.Tab("🏷️ Ground Truth Labeling"):
                    gr.Markdown("""
                    ### Label Segments for Testing

                    Use this to manually identify speakers in segments for testing optimal settings.
                    **This does NOT change the actual conversation** - labels are saved separately for testing only.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            label_conv_list = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                            refresh_label_conv_btn = gr.Button("🔄 Refresh List")

                            gr.Markdown("### Current Segment")
                            segment_info = gr.Textbox(label="Segment Info", lines=3, interactive=False)
                            segment_audio = gr.Audio(label="Play Segment", type="filepath")
                            segment_text = gr.Textbox(label="Transcription", lines=2, interactive=False)

                            true_speaker_input = gr.Textbox(label="True Speaker Name", placeholder="e.g., Tommy Eagan")
                            save_label_btn = gr.Button("💾 Save Label", variant="primary")

                            with gr.Row():
                                prev_segment_btn = gr.Button("⬅️ Previous")
                                next_segment_btn = gr.Button("➡️ Next")

                            label_status = gr.Textbox(label="Status", interactive=False)

                        with gr.Column(scale=2):
                            gr.Markdown("### Labeled Segments")
                            labels_table = gr.Dataframe(
                                headers=["Seg ID", "True Speaker", "Current Speaker", "Time", "Text"],
                                interactive=False
                            )

                            gr.Markdown("### Actions")
                            with gr.Row():
                                clear_labels_btn = gr.Button("🗑️ Clear All Labels")
                                export_labels_btn = gr.Button("📥 Export Labels (JSON)")

                            export_output = gr.Textbox(label="Export Status", interactive=False)

                    # Store current segment index
                    current_segment_idx = gr.State(value=0)

                    # Event handlers
                    refresh_label_conv_btn.click(
                        fn=self.get_conversation_choices,
                        outputs=[label_conv_list]
                    )

                    label_conv_list.change(
                        fn=self.load_segment_for_labeling,
                        inputs=[label_conv_list, current_segment_idx],
                        outputs=[segment_info, segment_audio, segment_text, current_segment_idx, labels_table]
                    )

                    save_label_btn.click(
                        fn=self.save_ground_truth_label,
                        inputs=[label_conv_list, current_segment_idx, true_speaker_input],
                        outputs=[label_status, labels_table]
                    )

                    next_segment_btn.click(
                        fn=self.next_segment,
                        inputs=[label_conv_list, current_segment_idx],
                        outputs=[segment_info, segment_audio, segment_text, current_segment_idx, labels_table]
                    )

                    prev_segment_btn.click(
                        fn=self.prev_segment,
                        inputs=[label_conv_list, current_segment_idx],
                        outputs=[segment_info, segment_audio, segment_text, current_segment_idx, labels_table]
                    )

                    clear_labels_btn.click(
                        fn=self.clear_ground_truth_labels,
                        inputs=[label_conv_list],
                        outputs=[export_output, labels_table]
                    )

                    export_labels_btn.click(
                        fn=self.export_ground_truth_labels,
                        inputs=[label_conv_list],
                        outputs=[export_output]
                    )

                # Tab 6: API Info
                with gr.Tab("📡 API Information"):
                    gr.Markdown("""
                    ### REST API Endpoints

                    This application provides a REST API for programmatic access (perfect for MCP server integration!):

                    **Base URL:** `http://localhost:8000/api/v1`

                    #### Speaker Management
                    - `GET /speakers` - List all enrolled speakers
                    - `POST /speakers/enroll` - Enroll new speaker (multipart/form-data)
                    - `PATCH /speakers/{id}/rename` - Rename speaker (useful for AI agents)
                    - `DELETE /speakers/{id}` - Delete speaker

                    #### Audio Processing
                    - `POST /process` - Process audio file with diarization
                    - `GET /recordings` - List all processed recordings
                    - `GET /recordings/{id}` - Get recording details

                    #### System
                    - `GET /status` - Check system status and GPU availability

                    ### Example: Rename Unknown Speaker (MCP)
                    ```bash
                    curl -X PATCH http://localhost:8000/api/v1/speakers/5/rename \\
                      -H "Content-Type: application/json" \\
                      -d '{"new_name": "Alice"}'
                    ```

                    ### OpenAPI Documentation
                    Visit `http://localhost:8000/docs` for interactive API documentation.

                    ---

                    ### MCP Server (AI Agent Interface)

                    This application includes a **Model Context Protocol (MCP)** server for AI agent integration.

                    **MCP Endpoint:** `http://localhost:8000/mcp` (or `http://YOUR_IP:8000/mcp` for network access)

                    #### Connection Example (Flowise, etc.)
                    ```json
                    {
                      "url": "http://10.x.x.x:8000/mcp",
                      "transport": "http"
                    }
                    ```

                    #### Available MCP Tools (10 total)
                    - `list_conversations` - Get conversations with IDs (use FIRST)
                    - `get_latest_segments` - Get segments with speaker names and segment_id
                    - `identify_speaker_in_segment` - Identify unknown speakers (auto-updates ALL past segments)
                    - `list_speakers` - Get all enrolled speaker profiles with IDs
                    - `rename_speaker` - Rename speaker (auto-updates all past segments)
                    - `delete_speaker` - Delete single speaker by ID
                    - `delete_all_unknown_speakers` - Cleanup: delete ALL "Unknown_*" speakers
                    - `get_conversation` - Get full conversation with all segments
                    - `reprocess_conversation` - Re-run recognition (only for NEW speakers in OLD conversations)
                    - `update_conversation_title` - Update conversation title

                    #### Protocol Details
                    - **Protocol:** MCP 2024-11-05 (JSON-RPC 2.0)
                    - **Transport:** HTTP with Server-Sent Events (SSE)
                    - **Info:** Visit `http://localhost:8000/mcp` for server capabilities
                    """)

        return demo


def create_gradio_app():
    """Factory function to create Gradio app"""
    interface = GradioInterface()
    return interface.create_interface()
