import torch
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import os
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import time
from pydub import AudioSegment


def auto_enroll_unknown_speaker(embedding: np.ndarray, db_session, threshold: float = 0.30):
    """
    Auto-enroll unknown speaker with timestamp-based naming and clustering.

    CRITICAL: This is a FALLBACK for segments that didn't match in transcribe_with_diarization().
    Checks ALL speakers (not just Unknown_*) with a slightly lower threshold as a second chance.
    This prevents creating duplicate Unknown speakers for people who are already enrolled.

    Args:
        embedding: Numpy array of speaker embedding
        db_session: SQLAlchemy database session
        threshold: Similarity threshold from settings (will use threshold-0.05 as fallback)

    Returns:
        Tuple of (speaker_id, speaker_name) for the enrolled/matched speaker
    """
    from .models import Speaker

    # FIRST: Check ALL speakers (including enrolled ones like "Bob", "Alice") as safety fallback
    # Use slightly LOWER threshold (5% less) to catch borderline cases that transcribe_with_diarization missed
    # This prevents creating duplicate Unknowns for people already in the database
    fallback_threshold = max(threshold - 0.05, 0.20)  # At least 0.20 (20%)

    all_speakers = db_session.query(Speaker).all()
    best_match = None
    best_similarity = 0.0
    best_above_threshold = None

    if all_speakers and embedding is not None:
        print(f"  🔍 Auto-enroll fallback: Checking {len(all_speakers)} speakers (fallback threshold: {fallback_threshold:.2%})")

        for speaker in all_speakers:
            existing_embedding = speaker.get_embedding()

            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                existing_embedding.reshape(1, -1)
            )[0][0]

            # Track best overall (for logging)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker

            # Track best above threshold (for matching)
            if similarity > fallback_threshold and (best_above_threshold is None or similarity > best_similarity):
                best_above_threshold = speaker
                print(f"    → Match: '{speaker.name}' (similarity: {similarity:.2%})")

    if best_above_threshold:
        print(f"✓ Fallback match: Identified as '{best_above_threshold.name}' (similarity: {best_similarity:.2%}, fallback threshold: {fallback_threshold:.2%})")
        return (best_above_threshold.id, best_above_threshold.name)
    elif best_match:
        print(f"  ℹ️ Best match was '{best_match.name}' with {best_similarity:.2%} (below fallback threshold {fallback_threshold:.2%})")

    # No match found - create new Unknown speaker with timestamp
    # Use microsecond precision for better uniqueness
    max_attempts = 5
    timestamp_name = None

    for attempt in range(max_attempts):
        timestamp_name = f"Unknown_{int(time.time() * 1000000)}"

        # Check if name already exists in DB
        existing = db_session.query(Speaker).filter(Speaker.name == timestamp_name).first()
        if not existing:
            break
        time.sleep(0.001)  # Wait 1ms and try again
    else:
        # Fallback to UUID if all attempts fail
        import uuid
        timestamp_name = f"Unknown_{uuid.uuid4().hex[:12]}"

    # Create new speaker with embedding
    new_speaker = Speaker(name=timestamp_name)
    new_speaker.set_embedding(embedding)
    db_session.add(new_speaker)
    db_session.flush()  # Get the ID without committing

    print(f"✓ Auto-enrolled new speaker: {timestamp_name} (ID: {new_speaker.id})")

    return (new_speaker.id, new_speaker.name)


class SpeakerRecognitionEngine:
    """
    Speaker diarization and recognition engine using pyannote.audio
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize the speaker recognition engine

        Args:
            hf_token: HuggingFace token for accessing pyannote models
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuration from environment
        self.context_padding = float(os.getenv("CONTEXT_PADDING", "0.15"))

        # Initialize models (lazy loading)
        self._diarization_pipeline = None
        self._embedding_model = None
        self._whisper_model = None
        self._emotion_model = None

        print(f"Speaker Recognition Engine initialized on device: {self.device}")

    def clear_gpu_cache(self):
        """Clear GPU memory cache to prevent memory accumulation"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def diarization_pipeline(self):
        """Lazy load diarization pipeline"""
        if self._diarization_pipeline is None:
            print("Loading pyannote diarization pipeline...")
            # Set HF_TOKEN environment variable for authentication
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )
            self._diarization_pipeline.to(self.device)
        return self._diarization_pipeline

    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            print("Loading pyannote embedding model...")
            # Set HF_TOKEN environment variable for authentication
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
            # In pyannote.audio 4.0, load Model first then create Inference
            model = Model.from_pretrained("pyannote/embedding")
            self._embedding_model = Inference(model, window="whole")
            self._embedding_model.to(self.device)
        return self._embedding_model

    @property
    def whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            # Get model from environment variable (default: large-v3)
            model_name = os.getenv("WHISPER_MODEL", "large-v3")
            print(f"Loading faster-whisper model ({model_name})...")
            # Load faster-whisper model with FP16 for GPU acceleration
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            self._whisper_model = WhisperModel(model_name, device=device_name, compute_type=compute_type)
            print(f"faster-whisper model '{model_name}' loaded on {device_name} with {compute_type}")
        return self._whisper_model

    @property
    def emotion_model(self):
        """Lazy load emotion2vec model via FunASR"""
        if self._emotion_model is None:
            print("Loading emotion2vec model via FunASR...")
            try:
                from funasr import AutoModel
                # Load emotion2vec+ model from ModelScope/HuggingFace
                # Using 'plus_large' for better accuracy (can switch to 'plus_base' for speed)
                model_name = os.getenv("EMOTION_MODEL", "iic/emotion2vec_plus_large")
                self._emotion_model = AutoModel(
                    model=model_name,
                    hub="hf",  # Use HuggingFace hub (fix for overseas users)
                    disable_update=True  # Don't auto-update models
                )
                print(f"emotion2vec model loaded successfully ({model_name})")
            except ImportError:
                print("Warning: FunASR not installed. Emotion detection will be disabled.")
                print("Install with: pip install funasr")
                return None
            except Exception as e:
                print(f"Warning: Failed to load emotion2vec model: {e}")
                return None
        return self._emotion_model

    def transcribe(self, audio_file: str) -> List[Dict]:
        """
        Transcribe audio file with timestamps

        Args:
            audio_file: Path to audio file

        Returns:
            List of transcription segments with timestamps
        """
        print(f"Transcribing {audio_file}...")
        # Get language from environment variable (default: "en")
        # Use "auto" for auto-detection, or specify language code (e.g., "es", "fr", "de")
        language = os.getenv("WHISPER_LANGUAGE", "en")
        # faster-whisper transcription with word-level timestamps and confidence
        segments_generator, info = self.whisper_model.transcribe(
            audio_file,
            language=None if language == "auto" else language,  # None = auto-detect
            task="transcribe",
            beam_size=5,
            vad_filter=True,  # Use VAD to filter out non-speech
            word_timestamps=True  # Enable word-level timestamps and probabilities
        )

        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        transcription_segments = []
        # Convert generator to list - transcription happens during iteration
        for segment in segments_generator:
            # Extract word-level data if available
            words_data = []
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words_data.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability  # Word-level confidence
                    })

            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words_data,  # Include word-level data with confidence
                "avg_logprob": segment.avg_logprob  # Segment-level confidence indicator
            })

        return transcription_segments

    def extract_embedding(self, audio_file: str) -> np.ndarray:
        """
        Extract speaker embedding from audio file

        Args:
            audio_file: Path to audio file

        Returns:
            Speaker embedding as numpy array
        """
        with torch.no_grad():
            embedding = self.embedding_model(audio_file)
        return np.array(embedding)

    def diarize(self, audio_file: str) -> Dict:
        """
        Perform speaker diarization on audio file

        Args:
            audio_file: Path to audio file

        Returns:
            Diarization result with speaker segments
        """
        # Convert MP3 to WAV if needed for reliable processing
        temp_wav_file = None
        if audio_file.endswith('.mp3'):
            try:
                print(f"Converting MP3 to WAV: {audio_file}...")
                audio = AudioSegment.from_file(audio_file)
                temp_wav_file = audio_file.rsplit('.', 1)[0] + '_temp_converted.wav'
                audio.export(temp_wav_file, format='wav')
                audio_file = temp_wav_file
                print(f"Conversion successful: {audio_file}")
            except Exception as e:
                print(f"Warning: Failed to convert MP3 to WAV: {e}")
                # Continue with original MP3 file

        print(f"Running diarization on {audio_file}...")
        try:
            with torch.no_grad():
                output = self.diarization_pipeline(audio_file)
        finally:
            # Clean up temp WAV file if created
            if temp_wav_file and os.path.exists(temp_wav_file):
                try:
                    os.remove(temp_wav_file)
                except:
                    pass

        # Convert to dictionary format
        segments = []
        # In pyannote.audio 4.0+, use the speaker_diarization attribute
        for turn, speaker in output.speaker_diarization:
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })

        return {
            "segments": segments,
            "num_speakers": len(set(s["speaker"] for s in segments))
        }

    def match_speaker(
        self,
        segment_embedding: np.ndarray,
        known_speakers: List[Tuple[int, str, np.ndarray]],
        threshold: float = 0.7
    ) -> Optional[Tuple[int, str, float]]:
        """
        Match a segment embedding to known speakers

        Args:
            segment_embedding: Embedding to match
            known_speakers: List of (id, name, embedding) tuples
            threshold: Minimum similarity threshold for a match

        Returns:
            (speaker_id, speaker_name, confidence) or None if no match
        """
        if not known_speakers:
            return None

        # Validate segment embedding - check for NaN values
        if np.isnan(segment_embedding).any():
            print(f"  ⚠️ Segment embedding contains NaN values - skipping matching")
            return None

        # Calculate similarities
        best_match = None
        best_similarity = threshold

        for speaker_id, speaker_name, speaker_embedding in known_speakers:
            # Validate known speaker embedding
            if np.isnan(speaker_embedding).any():
                print(f"  ⚠️ Speaker '{speaker_name}' embedding contains NaN - skipping")
                continue

            similarity = cosine_similarity(
                segment_embedding.reshape(1, -1),
                speaker_embedding.reshape(1, -1)
            )[0][0]

            # Debug: Always print similarity score
            print(f"  Similarity with {speaker_name}: {similarity:.4f} (threshold: {threshold})")

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (speaker_id, speaker_name, float(similarity))

        return best_match

    def extract_segment_embedding(
        self,
        audio_file: str,
        start_time: float,
        end_time: float,
        context_padding: float = None
    ) -> np.ndarray:
        """
        Extract embedding from a specific segment of audio with context padding

        Args:
            audio_file: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            context_padding: Seconds to include before/after for more reliable embeddings (default: 0.15s)
                           Optimal for movie audio with background music/effects
                           Based on comprehensive ground truth testing: 67.4% matching + only 3 misidentifications
                           Lower padding reduces background music corruption

        Returns:
            Segment embedding as numpy array
        """
        from pyannote.core import Segment
        import soundfile as sf

        # Use instance padding if not specified
        if context_padding is None:
            context_padding = self.context_padding

        # Get actual audio duration to prevent out-of-bounds
        try:
            info = sf.info(audio_file)
            duration = info.duration

            # MP3 files can have duration discrepancies between libraries
            # Use more conservative margins for MP3 files (pyannote often sees slightly shorter duration)
            # Measured discrepancy: ~23ms, using 50ms margin for safety (covers padding + discrepancy)
            is_mp3 = audio_file.lower().endswith('.mp3')
            safety_margin = 0.05 if is_mp3 else 0.01  # 50ms for MP3, 10ms for WAV

            # Add context padding for more reliable embeddings
            padded_start = start_time - context_padding
            padded_end = end_time + context_padding

            # Clamp times to valid range - use aggressive margins for MP3
            # Pyannote sometimes fails with times too close to file boundaries
            start_time = max(0, min(padded_start, duration - 0.5))
            end_time = min(padded_end, duration - safety_margin)

            # If start is beyond end after clamping, adjust start
            if start_time >= end_time:
                start_time = max(0, end_time - 0.5)

            # Ensure segment is at least 0.1s
            if end_time - start_time < 0.1:
                # Try to extend end
                end_time = min(start_time + 0.1, duration - safety_margin)
                # If still too short, move start back
                if end_time - start_time < 0.1:
                    start_time = max(0, end_time - 0.1)

            # Final safety check - ensure we're well within bounds
            if end_time > duration - safety_margin:
                end_time = duration - safety_margin
                if start_time >= end_time:
                    start_time = max(0, end_time - 0.5)

        except Exception as e:
            print(f"Warning: Could not get audio duration, using original times: {e}")
            # Reduce end time slightly as fallback
            end_time = end_time - 0.1

        segment = Segment(start_time, end_time)
        try:
            with torch.no_grad():
                embedding = self.embedding_model.crop(audio_file, segment)
            return np.array(embedding)
        except Exception as e:
            # Pyannote can throw sample mismatch errors at file boundaries
            error_msg = str(e)
            if "samples instead of the expected" in error_msg or "requested chunk" in error_msg:
                print(f"⚠️ Pyannote boundary error for {start_time:.2f}s-{end_time:.2f}s: {error_msg}")
                raise RuntimeError(f"Segment {start_time:.2f}s-{end_time:.2f}s beyond file duration")
            else:
                raise

    def extract_emotion(
        self,
        audio_file: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Extract emotion from audio file or segment using emotion2vec via FunASR

        Args:
            audio_file: Path to audio file
            start_time: Optional start time for segment (seconds)
            end_time: Optional end time for segment (seconds)

        Returns:
            Dictionary with emotion data or None if extraction fails:
            {
                'emotion_category': str,  # Primary emotion label
                'emotion_arousal': float,  # 0-1
                'emotion_valence': float,  # 0-1
                'emotion_confidence': float  # 0-1
            }
        """
        if self.emotion_model is None:
            return None

        try:
            # FunASR emotion2vec can process audio file directly or with timestamps
            # Format: model.generate(input, granularity="utterance")

            if start_time is not None and end_time is not None:
                # For segments, we need to extract the audio first
                import torchaudio
                from pydub import AudioSegment
                import tempfile

                # Extract segment to temporary file
                audio = AudioSegment.from_file(audio_file)
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                segment = audio[start_ms:end_ms]

                # Resample to 16kHz if needed (emotion2vec requirement)
                # This won't affect faster-whisper or pyannote which handle their own resampling
                if segment.frame_rate != 16000:
                    segment = segment.set_frame_rate(16000)

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    segment.export(temp_file.name, format='wav')
                    temp_path = temp_file.name

                # Process with emotion2vec
                result = self.emotion_model.generate(
                    temp_path,
                    granularity="utterance",
                    extract_embedding=False
                )

                # Clean up temp file
                import os
                os.unlink(temp_path)
            else:
                # Process entire file - need to check/resample first
                from pydub import AudioSegment
                import tempfile

                audio = AudioSegment.from_file(audio_file)

                # Resample to 16kHz if needed
                if audio.frame_rate != 16000:
                    audio = audio.set_frame_rate(16000)
                    # Create temp file with resampled audio
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        audio.export(temp_file.name, format='wav')
                        temp_path = temp_file.name

                    result = self.emotion_model.generate(
                        temp_path,
                        granularity="utterance",
                        extract_embedding=False
                    )

                    # Clean up
                    import os
                    os.unlink(temp_path)
                else:
                    # Already 16kHz, use directly
                    result = self.emotion_model.generate(
                        audio_file,
                        granularity="utterance",
                        extract_embedding=False
                    )

            # FunASR emotion2vec returns list of results
            # Format: [[{'labels': ['angry'], 'scores': [0.9]}]]
            if result and len(result) > 0:
                # Unwrap nested structure
                res = result[0] if isinstance(result, list) else result
                if isinstance(res, list) and len(res) > 0:
                    res = res[0]

                # Extract emotion label and scores
                if isinstance(res, dict):
                    labels = res.get('labels', ['unknown'])
                    scores = res.get('scores', [0.0])

                    # Find the index with highest score
                    if isinstance(scores, list) and len(scores) > 0:
                        max_idx = scores.index(max(scores))
                        max_score = scores[max_idx]

                        # Get the label at that index
                        if isinstance(labels, list) and len(labels) > max_idx:
                            raw_label = labels[max_idx]
                        else:
                            raw_label = labels[0] if isinstance(labels, list) else str(labels)
                    else:
                        raw_label = labels[0] if isinstance(labels, list) else str(labels)
                        max_score = 0.0

                    # Extract English label (format: "中文/english")
                    if '/' in raw_label:
                        emotion_label = raw_label.split('/')[-1].strip()
                    else:
                        emotion_label = raw_label
                else:
                    emotion_label = 'unknown'
                    max_score = 0.0

                # Map emotion labels to arousal/valence (approximate mapping)
                # emotion2vec supports: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
                arousal_map = {
                    'angry': 0.8, 'disgusted': 0.6, 'fearful': 0.7, 'happy': 0.7,
                    'neutral': 0.5, 'sad': 0.3, 'surprised': 0.8, 'other': 0.5, 'unknown': 0.5
                }
                valence_map = {
                    'angry': 0.2, 'disgusted': 0.2, 'fearful': 0.2, 'happy': 0.9,
                    'neutral': 0.5, 'sad': 0.1, 'surprised': 0.6, 'other': 0.5, 'unknown': 0.5
                }

                return {
                    'emotion_category': emotion_label,
                    'emotion_arousal': arousal_map.get(emotion_label, 0.5),
                    'emotion_valence': valence_map.get(emotion_label, 0.5),
                    'emotion_confidence': float(max_score)
                }

            return None

        except Exception as e:
            print(f"Warning: Failed to extract emotion: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_audio_with_recognition(
        self,
        audio_file: str,
        known_speakers: List[Tuple[int, str, np.ndarray]],
        threshold: float = 0.7
    ) -> Dict:
        """
        Full pipeline: diarization + speaker recognition

        Args:
            audio_file: Path to audio file
            known_speakers: List of (id, name, embedding) tuples
            threshold: Similarity threshold for speaker matching

        Returns:
            Dictionary with segments and speaker identifications
        """
        # Step 1: Diarize
        diarization_result = self.diarize(audio_file)

        # Step 2: Match each segment to known speakers
        segments_with_recognition = []
        unknown_counter = 1

        for segment in diarization_result["segments"]:
            # Extract embedding for this segment
            segment_embedding = self.extract_segment_embedding(
                audio_file,
                segment["start"],
                segment["end"]
            )

            # Try to match to known speaker
            match = self.match_speaker(segment_embedding, known_speakers, threshold)

            if match:
                speaker_id, speaker_name, confidence = match
                segment_info = {
                    **segment,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "confidence": confidence,
                    "is_known": True,
                    "embedding": None  # Don't need to return embedding for known speakers
                }
            else:
                # Unknown speaker - return embedding for auto-enrollment
                segment_info = {
                    **segment,
                    "speaker_id": None,
                    "speaker_name": f"Unknown_{unknown_counter:02d}",
                    "confidence": 0.0,
                    "is_known": False,
                    "embedding": segment_embedding  # Return embedding for auto-enrollment
                }
                unknown_counter += 1

            segments_with_recognition.append(segment_info)

        return {
            "segments": segments_with_recognition,
            "num_speakers": diarization_result["num_speakers"],
            "num_known": sum(1 for s in segments_with_recognition if s["is_known"]),
            "num_unknown": sum(1 for s in segments_with_recognition if not s["is_known"])
        }

    def transcribe_with_diarization(
        self,
        audio_file: str,
        known_speakers: List[Tuple[int, str, np.ndarray]] = None,
        threshold: float = 0.7
    ) -> Dict:
        """
        Full pipeline: transcription + diarization + speaker recognition

        Args:
            audio_file: Path to audio file
            known_speakers: Optional list of (id, name, embedding) tuples
            threshold: Similarity threshold for speaker matching

        Returns:
            Dictionary with transcribed segments and speaker labels
        """
        known_speakers = known_speakers or []
        print(f"Known speakers: {len(known_speakers)}")
        for speaker_id, speaker_name, _ in known_speakers:
            print(f"  - ID: {speaker_id}, Name: {speaker_name}")

        # Run transcription and diarization IN PARALLEL for speed
        print("Running transcription and diarization in parallel...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks to run concurrently
            transcription_future = executor.submit(self.transcribe, audio_file)
            diarization_future = executor.submit(self.diarize, audio_file)

            # Wait for both to complete
            transcription_segments = transcription_future.result()
            diarization_result = diarization_future.result()

        elapsed = time.time() - start_time
        print(f"Parallel processing completed in {elapsed:.2f}s")

        # Step 3: Match transcription segments to speakers
        # Map diarization labels (SPEAKER_00, SPEAKER_01) to Unknown_XX
        unknown_speaker_map = {}  # Maps SPEAKER_XX -> Unknown_YY
        unknown_counter = 1

        transcribed_with_speakers = []

        for trans_seg in transcription_segments:
            # Find which speaker was talking during this transcription segment
            # Use the middle of the transcription segment for matching
            mid_time = (trans_seg["start"] + trans_seg["end"]) / 2

            # Find the diarization segment that contains this time
            speaker_label = None
            for diar_seg in diarization_result["segments"]:
                if diar_seg["start"] <= mid_time <= diar_seg["end"]:
                    speaker_label = diar_seg["speaker"]
                    break

            if speaker_label is None:
                # If no exact match, find the closest diarization segment
                min_distance = float('inf')
                for diar_seg in diarization_result["segments"]:
                    diar_mid = (diar_seg["start"] + diar_seg["end"]) / 2
                    distance = abs(diar_mid - mid_time)
                    if distance < min_distance:
                        min_distance = distance
                        speaker_label = diar_seg["speaker"]

            # Try to match to known speaker if provided
            speaker_name = speaker_label  # Default to diarization label (will be converted to Unknown_XX)
            is_known = False
            confidence = 0.0
            embedding = None

            # ALWAYS extract embeddings (needed for embedding verification and speaker matching)
            # Check if segment is long enough for embedding extraction
            segment_duration = trans_seg["end"] - trans_seg["start"]
            if segment_duration < 0.5:
                print(f"⏭️ Skipping embedding extraction (segment too short: {segment_duration:.2f}s)")
            else:
                try:
                    # Extract embedding for this segment
                    segment_embedding = self.extract_segment_embedding(
                        audio_file,
                        trans_seg["start"],
                        trans_seg["end"]
                    )
                    # Validate embedding
                    if np.isnan(segment_embedding).any():
                        print(f"⏭️ Segment embedding has NaN - skipping")
                    else:
                        # Store valid embedding for later use (auto-enrollment or verification)
                        embedding = segment_embedding

                        # Try to match to known speakers if provided
                        if known_speakers:
                            match = self.match_speaker(segment_embedding, known_speakers, threshold)
                            if match:
                                _, speaker_name, confidence = match
                                is_known = True
                                print(f"Matched segment to {speaker_name} with confidence {confidence:.2%}")
                                embedding = None  # Don't return embedding for known speakers
                            else:
                                print(f"No match found for segment (threshold: {threshold})")
                                # Keep embedding for auto-enrollment
                except RuntimeError as e:
                    error_str = str(e)
                    if "Kernel size" in error_str or "input size" in error_str:
                        print(f"⏭️ Skipping embedding extraction (segment too short for embedding model)")
                    elif "beyond file duration" in error_str:
                        print(f"⏭️ Skipping embedding extraction (segment at file boundary)")
                    else:
                        raise
                except Exception as e:
                    # Catch pyannote errors like sample mismatches at file boundaries
                    if "samples instead of the expected" in str(e) or "requested chunk" in str(e):
                        print(f"⏭️ Skipping embedding extraction (segment beyond file duration: {trans_seg['start']:.2f}s-{trans_seg['end']:.2f}s)")
                    else:
                        raise

            # If still not matched to known speaker (or embedding failed), map to Unknown_XX
            if not is_known and speaker_label:
                # Create consistent mapping from diarization label to Unknown_XX
                if speaker_label not in unknown_speaker_map:
                    unknown_speaker_map[speaker_label] = f"Unknown_{unknown_counter:02d}"
                    unknown_counter += 1
                speaker_name = unknown_speaker_map[speaker_label]

            # Extract emotion for this segment
            emotion_data = self.extract_emotion(
                audio_file,
                trans_seg["start"],
                trans_seg["end"]
            )

            segment_data = {
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "text": trans_seg["text"],
                "speaker": speaker_name,
                "speaker_label": speaker_label,
                "is_known": is_known,
                "confidence": confidence,
                "embedding": embedding,  # Include embedding for unknown speakers
                "words": trans_seg.get("words", []),  # Include word-level data
                "avg_logprob": trans_seg.get("avg_logprob")  # Include segment confidence
            }

            # Add emotion data if available
            if emotion_data:
                segment_data.update({
                    "emotion_category": emotion_data.get("emotion_category"),
                    "emotion_arousal": emotion_data.get("emotion_arousal"),
                    "emotion_valence": emotion_data.get("emotion_valence"),
                    "emotion_confidence": emotion_data.get("emotion_confidence")
                })

            transcribed_with_speakers.append(segment_data)

        return {
            "segments": transcribed_with_speakers,
            "num_speakers": diarization_result["num_speakers"]
        }
