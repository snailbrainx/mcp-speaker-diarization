"""
Streaming audio recorder with queue-based background processing
"""
import numpy as np
import queue
import threading
import time
import wave
import os
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor


class StreamingRecorder:
    """
    Handles continuous audio streaming with background processing
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        silence_threshold: float = 0.005,
        silence_duration: float = None,
        max_workers: int = 2
    ):
        """
        Initialize streaming recorder

        Args:
            sample_rate: Audio sample rate (Hz)
            silence_threshold: Energy threshold for silence detection
            silence_duration: Seconds of silence before processing segment (default from .env or 0.5s)
            max_workers: Number of parallel processing threads
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold

        # Load settings from config manager (which checks env vars and config file)
        from .config import get_config
        config = get_config()
        settings = config.get_settings()

        # Load silence duration from config if not specified
        if silence_duration is None:
            silence_duration = settings.silence_duration
        self.silence_duration = silence_duration

        # State
        self.is_recording = False
        self.conversation_id = None

        # Audio buffering
        self.current_buffer = []
        self.last_speech_time = None
        self.chunk_count = 0
        self.speech_detected = False  # VAD state for UI display

        # Track cumulative offset for segments
        self.cumulative_offset = 0.0  # Total duration of all processed segments

        # Queue system
        self.segment_queue = queue.Queue()  # Segments waiting to be processed
        self.results_queue = queue.Queue()  # Processed results

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_futures = []

        # Callbacks
        self.on_segment_processed: Optional[Callable] = None
        self.on_audio_level: Optional[Callable] = None

        # Stats
        self.total_segments = 0
        self.segments_processed = 0
        self.segments_queued = 0

        # Track segment paths for concatenation
        self.segment_paths = []

    def start_recording(self, conversation_id: int):
        """Start recording for a conversation"""
        # Reload settings (in case they changed in UI)
        from .config import get_config
        config = get_config()
        settings = config.get_settings()
        self.silence_duration = settings.silence_duration

        self.is_recording = True
        self.conversation_id = conversation_id
        self.current_buffer = []
        self.last_speech_time = time.time()
        self.chunk_count = 0
        self.total_segments = 0
        self.segments_processed = 0
        self.segments_queued = 0
        self.segment_paths = []
        self.cumulative_offset = 0.0

        # Create directory for this conversation's segments
        os.makedirs(f"data/stream_segments/conv_{conversation_id}", exist_ok=True)
        print(f"🎤 Streaming recorder started for conversation {conversation_id} (silence: {self.silence_duration}s)")

    def stop_recording(self):
        """Stop recording and wait for processing to complete"""
        print("⏹️ Stopping streaming recorder...")

        # Process any remaining buffer
        if len(self.current_buffer) > 0:
            self._queue_segment()

        self.is_recording = False

        # Wait for all processing to complete
        print(f"⏳ Waiting for {self.segments_queued - self.segments_processed} segments to finish processing...")
        while self.segments_queued > self.segments_processed:
            time.sleep(0.5)

        print("✅ Streaming recorder stopped")

    def process_audio_chunk(self, audio_chunk: tuple) -> Dict:
        """
        Process incoming audio chunk from Gradio

        Args:
            audio_chunk: Tuple of (sample_rate, audio_data)

        Returns:
            Dict with status and stats
        """
        if not self.is_recording or audio_chunk is None:
            return {
                "status": "not_recording",
                "audio_level": 0.0,
                "segments_queued": 0,
                "segments_processed": 0
            }

        sample_rate, audio_data = audio_chunk

        # Handle mono/stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Take first channel

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Calculate energy for VAD and level display
        energy = np.sqrt(np.mean(audio_data ** 2))

        # Trigger audio level callback
        if self.on_audio_level:
            self.on_audio_level(energy)

        # Add to buffer
        self.current_buffer.append(audio_data)
        self.chunk_count += 1

        # Check for speech/silence
        if energy > self.silence_threshold:
            # Speech detected
            self.last_speech_time = time.time()
            self.speech_detected = True
        else:
            # Silence detected
            self.speech_detected = False
            silence_elapsed = time.time() - self.last_speech_time

            # Process segment after enough silence
            if silence_elapsed >= self.silence_duration and len(self.current_buffer) > 10:
                self._queue_segment()

        return {
            "status": "recording",
            "audio_level": float(energy),
            "speech_detected": self.speech_detected,
            "segments_queued": self.segments_queued,
            "segments_processed": self.segments_processed,
            "buffer_size": len(self.current_buffer)
        }

    def _queue_segment(self):
        """Queue current buffer for background processing"""
        if len(self.current_buffer) == 0:
            return

        # Combine buffer into single array
        segment_audio = np.concatenate(self.current_buffer)

        # Check minimum duration (0.5 seconds minimum for embedding model)
        duration = len(segment_audio) / self.sample_rate
        if duration < 0.5:
            print(f"⏭️ Skipping segment (too short: {duration:.2f}s, minimum 0.5s)")
            self.current_buffer = []
            return

        # Warn if segment is very long (but still process it)
        if duration > 60.0:
            print(f"⚠️ Long segment detected: {duration:.1f}s - processing may take longer")

        # Check if segment has enough actual speech (not just silence)
        avg_energy = np.sqrt(np.mean(segment_audio ** 2))
        if avg_energy < self.silence_threshold * 2:
            print(f"⏭️ Skipping segment (mostly silence, energy: {avg_energy:.4f})")
            self.current_buffer = []
            return

        # Normalize audio
        max_val = np.abs(segment_audio).max()
        if max_val > 0:
            segment_audio = segment_audio * (0.9 / max_val)

        # Save to persistent file
        segment_id = self.total_segments + 1
        segment_path = f"data/stream_segments/conv_{self.conversation_id}/seg_{segment_id:04d}.wav"

        with wave.open(segment_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes((segment_audio * 32767).astype(np.int16).tobytes())

        # Track segment path for later concatenation
        self.segment_paths.append(segment_path)

        # Calculate duration and offsets
        duration = len(segment_audio) / self.sample_rate
        start_offset = self.cumulative_offset
        end_offset = self.cumulative_offset + duration

        # Queue for processing
        segment_info = {
            "id": segment_id,
            "path": segment_path,
            "segment_file": segment_path,  # Also include for compatibility
            "conversation_id": self.conversation_id,
            "duration": duration,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "timestamp": datetime.now()
        }

        self.segment_queue.put(segment_info)
        self.total_segments += 1
        self.segments_queued += 1

        # Update cumulative offset for next segment
        self.cumulative_offset = end_offset

        # Submit to thread pool
        future = self.executor.submit(self._process_segment_worker, segment_info)
        self.processing_futures.append(future)

        # Clear buffer
        self.current_buffer = []
        self.last_speech_time = time.time()

        print(f"📦 Queued segment {segment_id} ({duration:.1f}s, offset {start_offset:.1f}-{end_offset:.1f}s) - Queue: {self.segment_queue.qsize()}")

    def _process_segment_worker(self, segment_info: Dict):
        """
        Background worker that processes a segment
        This will be called by the processing engine
        """
        try:
            # This is where the actual processing happens
            # The callback will handle transcription + diarization
            if self.on_segment_processed:
                self.on_segment_processed(segment_info)

            self.segments_processed += 1
            print(f"✅ Processed segment {segment_info['id']} ({self.segments_processed}/{self.segments_queued})")

        except Exception as e:
            print(f"❌ Error processing segment {segment_info['id']}: {e}")
            import traceback
            traceback.print_exc()

    def get_stats(self) -> Dict:
        """Get current recording stats"""
        return {
            "is_recording": self.is_recording,
            "total_segments": self.total_segments,
            "segments_queued": self.segments_queued,
            "segments_processed": self.segments_processed,
            "queue_size": self.segment_queue.qsize(),
            "buffer_chunks": len(self.current_buffer)
        }

    def concatenate_segments(self) -> Optional[str]:
        """
        Concatenate all segment WAV files into a single conversation file

        Returns:
            Path to the concatenated WAV file, or None if no segments
        """
        if not self.segment_paths or self.conversation_id is None:
            return None

        try:
            print(f"🔗 Concatenating {len(self.segment_paths)} segments...")

            # Create recordings directory if it doesn't exist
            os.makedirs("data/recordings", exist_ok=True)

            # Output path for full conversation
            output_path = f"data/recordings/conv_{self.conversation_id}_full.wav"

            # Read all segments and concatenate
            all_audio = []
            sample_rate = None

            for seg_path in self.segment_paths:
                if not os.path.exists(seg_path):
                    print(f"⚠️ Segment not found: {seg_path}")
                    continue

                with wave.open(seg_path, 'rb') as wf:
                    if sample_rate is None:
                        sample_rate = wf.getframerate()

                    # Read audio data
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    all_audio.append(audio_data)

            if not all_audio:
                print("⚠️ No valid segments to concatenate")
                return None

            # Concatenate all audio
            full_audio = np.concatenate(all_audio)

            # Write concatenated audio to file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(full_audio.tobytes())

            duration = len(full_audio) / sample_rate
            print(f"✅ Concatenated conversation saved: {output_path} ({duration:.1f}s)")

            return output_path

        except Exception as e:
            print(f"❌ Error concatenating segments: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
