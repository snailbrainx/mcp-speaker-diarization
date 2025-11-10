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
            print("Loading faster-whisper model (large-v3)...")
            # Load faster-whisper model with FP16 for GPU acceleration
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            self._whisper_model = WhisperModel("large-v3", device=device_name, compute_type=compute_type)
            print(f"faster-whisper model loaded on {device_name} with {compute_type}")
        return self._whisper_model

    def transcribe(self, audio_file: str) -> List[Dict]:
        """
        Transcribe audio file with timestamps

        Args:
            audio_file: Path to audio file

        Returns:
            List of transcription segments with timestamps
        """
        print(f"Transcribing {audio_file}...")
        # faster-whisper transcription
        segments_generator, info = self.whisper_model.transcribe(
            audio_file,
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True  # Use VAD to filter out non-speech
        )

        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        transcription_segments = []
        # Convert generator to list - transcription happens during iteration
        for segment in segments_generator:
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
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

            # Add context padding for more reliable embeddings
            padded_start = start_time - context_padding
            padded_end = end_time + context_padding

            # Clamp times to valid range - use more aggressive margins
            # Pyannote sometimes fails with times too close to file boundaries
            start_time = max(0, min(padded_start, duration - 0.5))
            end_time = min(padded_end, duration - 0.1)

            # If start is beyond end after clamping, adjust start
            if start_time >= end_time:
                start_time = max(0, end_time - 0.5)

            # Ensure segment is at least 0.1s
            if end_time - start_time < 0.1:
                # Try to extend end
                end_time = min(start_time + 0.1, duration - 0.1)
                # If still too short, move start back
                if end_time - start_time < 0.1:
                    start_time = max(0, end_time - 0.1)

            # Final safety check - ensure we're well within bounds
            if end_time > duration - 0.05:
                end_time = duration - 0.05
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
            speaker_name = speaker_label  # Default to diarization label
            is_known = False
            confidence = 0.0
            embedding = None

            # ALWAYS extract embeddings (needed for embedding verification in gradio_ui.py)
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

            transcribed_with_speakers.append({
                "start": trans_seg["start"],
                "end": trans_seg["end"],
                "text": trans_seg["text"],
                "speaker": speaker_name,
                "speaker_label": speaker_label,
                "is_known": is_known,
                "confidence": confidence,
                "embedding": embedding  # Include embedding for unknown speakers
            })

        return {
            "segments": transcribed_with_speakers,
            "num_speakers": diarization_result["num_speakers"]
        }
