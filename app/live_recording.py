"""
Live continuous recording with VAD-based segmentation
"""
import os

# Fix for PyTorch 2.6+ weights_only=True breaking change
# Required for loading pyannote.audio model checkpoints which contain
# pickled objects (omegaconf, pytorch_lightning callbacks, etc.)
# This must be set BEFORE importing torch
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import torch

import sounddevice as sd
import numpy as np
import wave
import threading
import queue
import time
import os
from datetime import datetime, timedelta
from typing import Optional, Callable
from pyannote.audio import Pipeline

from .diarization import SpeakerRecognitionEngine


class LiveRecordingManager:
    """
    Manages continuous live recording with automatic speech detection and segmentation
    """

    def __init__(
        self,
        engine: SpeakerRecognitionEngine,
        sample_rate: int = 48000,  # 48kHz for high quality audio (was 16kHz)
        channels: int = 1,
        chunk_duration: float = 0.5,
        silence_threshold_sec: float = 2.0,
        conversation_end_threshold_sec: float = 300.0,
        vad_threshold: float = 0.5,
        device: int = None  # Audio device index (None = default)
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration)
        self.silence_threshold = silence_threshold_sec
        self.conversation_end_threshold = conversation_end_threshold_sec
        self.vad_threshold = vad_threshold
        self.device = device

        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.current_segment = []
        self.last_speech_time = None
        self.conversation_start_time = None
        self.conversation_id = None

        # VAD pipeline
        self.vad_pipeline = None

        # Threading
        self.record_thread = None
        self.process_thread = None

        # Callbacks
        self.on_segment_ready: Optional[Callable] = None
        self.on_conversation_end: Optional[Callable] = None

    def _load_vad(self):
        """Lazy load VAD pipeline"""
        if self.vad_pipeline is None:
            print("Loading VAD pipeline...")
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            try:
                # Try to load VAD pipeline
                self.vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.vad_pipeline.to(device)
                print(f"VAD pipeline loaded on {device}")
            except Exception as e:
                print(f"Warning: Could not load VAD pipeline: {e}")
                print("Falling back to energy-based VAD")
                self.vad_pipeline = None

    def _detect_speech_energy(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based VAD as fallback"""
        energy = np.sqrt(np.mean(audio_chunk**2))
        # Lower threshold for better sensitivity (0.005 instead of 0.01)
        return energy > 0.005

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice audio capture"""
        if status:
            print(f"Audio callback status: {status}")

        # Copy data to avoid buffer issues
        audio_data = indata.copy()
        self.audio_queue.put(audio_data)

    def start_recording(self, conversation_id: Optional[int] = None):
        """Start live recording"""
        if self.is_recording:
            raise RuntimeError("Recording already in progress")

        self.conversation_id = conversation_id
        self.conversation_start_time = datetime.now()
        self.last_speech_time = datetime.now()
        self.is_recording = True
        self.audio_buffer = []
        self.current_segment = []

        # Load VAD
        self._load_vad()

        # Start audio capture thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        print(f"✓ Live recording started (conversation_id: {conversation_id})")

    def stop_recording(self):
        """Stop live recording"""
        if not self.is_recording:
            return

        print("Stopping live recording...")
        self.is_recording = False

        # Wait for threads
        if self.record_thread:
            self.record_thread.join(timeout=5)
        if self.process_thread:
            self.process_thread.join(timeout=5)

        # Process any remaining audio
        if self.current_segment:
            self._process_segment()

        # Trigger conversation end callback
        if self.on_conversation_end:
            self.on_conversation_end(self.conversation_id)

        print("✓ Live recording stopped")

    def _record_loop(self):
        """Audio capture loop"""
        try:
            device_msg = f" device {self.device}" if self.device is not None else " default device"
            with sd.InputStream(
                device=self.device,  # Use selected device or None for default
                callback=self._audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            ):
                print(f"Recording from{device_msg} (sample rate: {self.sample_rate}Hz)...")
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in recording loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_recording = False

    def _process_loop(self):
        """Process audio chunks and detect speech/silence"""
        while self.is_recording:
            try:
                # Get audio chunk from queue (with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.5)

                # Detect speech
                has_speech = self._detect_speech_energy(audio_chunk[:, 0])  # Use first channel

                if has_speech:
                    # Speech detected
                    self.last_speech_time = datetime.now()
                    self.current_segment.append(audio_chunk)
                    self.audio_buffer.append(audio_chunk)
                else:
                    # Silence detected
                    silence_duration = (datetime.now() - self.last_speech_time).total_seconds()

                    if silence_duration >= self.silence_threshold and self.current_segment:
                        # Enough silence to trigger segment processing
                        self._process_segment()
                        self.current_segment = []

                    elif silence_duration >= self.conversation_end_threshold:
                        # Long silence - end conversation
                        print(f"Long silence detected ({silence_duration:.1f}s) - ending conversation")
                        self.stop_recording()
                        break

            except queue.Empty:
                # No audio data, continue
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def _process_segment(self):
        """Process accumulated audio segment"""
        if not self.current_segment:
            return

        try:
            # Combine audio chunks
            segment_audio = np.concatenate(self.current_segment, axis=0)

            # Normalize audio to improve quality (use 90% of dynamic range to prevent clipping)
            max_val = np.abs(segment_audio).max()
            if max_val > 0:
                segment_audio = segment_audio * (0.9 / max_val)

            # Save to temporary WAV file
            temp_path = f"/tmp/segment_{int(time.time())}.wav"
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes((segment_audio * 32767).astype(np.int16).tobytes())

            # Trigger callback with segment path
            if self.on_segment_ready:
                segment_info = {
                    "audio_path": temp_path,
                    "duration": len(segment_audio) / self.sample_rate,
                    "timestamp": datetime.now()
                }
                self.on_segment_ready(segment_info, self.conversation_id)

            print(f"✓ Segment processed: {len(segment_audio)/self.sample_rate:.2f}s")

        except Exception as e:
            print(f"Error processing segment: {e}")

    def get_status(self) -> dict:
        """Get current recording status"""
        if not self.is_recording:
            return {"status": "stopped"}

        return {
            "status": "recording",
            "duration": (datetime.now() - self.conversation_start_time).total_seconds(),
            "last_speech": (datetime.now() - self.last_speech_time).total_seconds(),
            "buffer_size": len(self.audio_buffer)
        }
