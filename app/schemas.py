from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class SpeakerBase(BaseModel):
    name: str

class SpeakerCreate(SpeakerBase):
    pass

class SpeakerResponse(SpeakerBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class SpeakerRename(BaseModel):
    new_name: str

class SegmentResponse(BaseModel):
    id: int
    start_time: float
    end_time: float
    speaker_label: str
    speaker_id: Optional[int]
    confidence: Optional[float]

    class Config:
        from_attributes = True

class RecordingResponse(BaseModel):
    id: int
    filename: str
    duration: Optional[float]
    status: str
    processed_at: datetime
    segments: List[SegmentResponse] = []

    class Config:
        from_attributes = True

class DiarizationResult(BaseModel):
    recording_id: int
    num_speakers: int
    num_known: int
    num_unknown: int
    segments: List[SegmentResponse]

class StatusResponse(BaseModel):
    status: str
    message: str
    gpu_available: bool
    device: str


# Conversation Schemas
class Word(BaseModel):
    """Word-level transcription data with confidence"""
    word: str
    start: float
    end: float
    probability: float

class ConversationSegmentResponse(BaseModel):
    id: int
    conversation_id: int
    speaker_id: Optional[int]
    speaker_name: Optional[str]
    text: Optional[str]
    start_time: datetime
    end_time: datetime
    start_offset: float
    end_offset: float
    confidence: Optional[float]
    words: Optional[List[Word]] = None
    avg_logprob: Optional[float] = None
    is_misidentified: bool = False

    class Config:
        from_attributes = True


class ConversationListItem(BaseModel):
    """Lightweight conversation summary for list views (no segments)"""
    id: int
    title: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str
    audio_format: str
    num_segments: int
    num_speakers: int

    class Config:
        from_attributes = True


class ConversationsListResponse(BaseModel):
    """Response for list conversations endpoint with pagination"""
    conversations: List[ConversationListItem]
    total: int
    skip: int
    limit: int


class ConversationResponse(BaseModel):
    """Full conversation details with all segments"""
    id: int
    title: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str
    audio_format: str
    num_segments: int
    num_speakers: int
    transcript_segments: List[ConversationSegmentResponse] = []

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None


class IdentifySpeakerRequest(BaseModel):
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    enroll: bool = True


class ToggleMisidentifiedRequest(BaseModel):
    is_misidentified: bool
