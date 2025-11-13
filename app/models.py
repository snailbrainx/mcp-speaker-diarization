from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base
import json

class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stored as numpy array bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to segments
    segments = relationship("Segment", back_populates="speaker")

    def get_embedding(self):
        """Convert binary embedding back to numpy array"""
        import numpy as np
        return np.frombuffer(self.embedding, dtype=np.float32)

    def set_embedding(self, embedding_array):
        """Convert numpy array to binary for storage"""
        import numpy as np
        self.embedding = embedding_array.astype(np.float32).tobytes()


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    duration = Column(Float)  # Duration in seconds
    processed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="processing")  # processing, completed, failed

    # Relationship to segments
    segments = relationship("Segment", back_populates="recording", cascade="all, delete-orphan")


class Segment(Base):
    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=False)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)  # Null if unknown
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)  # End time in seconds
    confidence = Column(Float)  # Confidence score for speaker match
    speaker_label = Column(String)  # e.g., "SPEAKER_00", "Unknown_01", or actual name

    # Relationships
    recording = relationship("Recording", back_populates="segments")
    speaker = relationship("Speaker", back_populates="segments")


class Conversation(Base):
    """
    Represents a continuous recording session (e.g., a meeting, interview, etc.)
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)  # Auto-generated or user-set
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)  # Null while recording
    duration = Column(Float, nullable=True)  # Duration in seconds
    status = Column(String, default="recording")  # recording, processing, completed, failed
    audio_path = Column(String, nullable=True)  # Path to WAV or MP3 file
    audio_format = Column(String, default="wav")  # wav or mp3
    num_segments = Column(Integer, default=0)
    num_speakers = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    transcript_segments = relationship("ConversationSegment", back_populates="conversation", cascade="all, delete-orphan")


class ConversationSegment(Base):
    """
    Individual speech segment within a conversation with transcription
    """
    __tablename__ = "conversation_segments"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)  # Null for unknown
    speaker_name = Column(String, nullable=True)  # Denormalized for quick access
    text = Column(Text, nullable=True)  # Transcription text

    # Absolute timestamps (for AI context)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)

    # Relative timestamps (seconds from conversation start, for audio playback)
    start_offset = Column(Float, nullable=False)
    end_offset = Column(Float, nullable=False)

    # Individual segment audio file (for streaming playback before concatenation)
    segment_audio_path = Column(String, nullable=True)

    confidence = Column(Float, nullable=True)  # Speaker identification confidence

    # Emotion detection (from emotion2vec)
    emotion_category = Column(String, nullable=True)  # Primary emotion label (happy, sad, angry, etc.)
    emotion_arousal = Column(Float, nullable=True)  # Arousal level (0-1, calm to excited)
    emotion_valence = Column(Float, nullable=True)  # Valence (0-1, negative to positive)
    emotion_confidence = Column(Float, nullable=True)  # Confidence score for emotion prediction

    # Word-level transcription data with confidence scores (JSON)
    words_data = Column(Text, nullable=True)  # Stores JSON array of {word, start, end, probability}
    avg_logprob = Column(Float, nullable=True)  # Segment-level average log probability

    processed_at = Column(DateTime, default=datetime.utcnow)

    # Misidentification tracking
    is_misidentified = Column(Boolean, default=False, nullable=False)  # True if this segment was wrongly assigned to current speaker

    # Relationships
    conversation = relationship("Conversation", back_populates="transcript_segments")
    speaker = relationship("Speaker")

    @property
    def words(self):
        """Parse words_data JSON and return as list"""
        if self.words_data:
            try:
                import json
                return json.loads(self.words_data)
            except:
                return None
        return None


class GroundTruthLabel(Base):
    """
    Ground truth speaker labels for testing and optimization
    Stores manual identifications WITHOUT affecting the actual segments
    """
    __tablename__ = "ground_truth_labels"

    id = Column(Integer, primary_key=True, index=True)
    segment_id = Column(Integer, ForeignKey("conversation_segments.id"), nullable=False)
    true_speaker_name = Column(String, nullable=False)  # Manual identification
    labeled_by = Column(String, default="user")  # Who labeled it
    labeled_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)  # Optional notes about the segment

    # Relationship
    segment = relationship("ConversationSegment")
