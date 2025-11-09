from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from datetime import datetime
import torch
from pydub import AudioSegment

from .database import get_db
from .models import Speaker, Recording, Segment
from .schemas import (
    SpeakerCreate, SpeakerResponse, SpeakerRename,
    RecordingResponse, SegmentResponse, DiarizationResult,
    StatusResponse
)
from .diarization import SpeakerRecognitionEngine

router = APIRouter()

# Initialize speaker recognition engine (singleton)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SpeakerRecognitionEngine()
    return engine


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    return StatusResponse(
        status="online",
        message="Speaker diarization service is running",
        gpu_available=torch.cuda.is_available(),
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )


@router.get("/speakers", response_model=List[SpeakerResponse])
async def list_speakers(db: Session = Depends(get_db)):
    """List all enrolled speakers"""
    speakers = db.query(Speaker).all()
    return speakers


@router.post("/speakers/enroll", response_model=SpeakerResponse)
async def enroll_speaker(
    name: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Enroll a new speaker with audio sample

    Args:
        name: Speaker name
        audio_file: Audio file containing speaker's voice (10-30 seconds recommended)
    """
    # Check if speaker already exists
    existing = db.query(Speaker).filter(Speaker.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Speaker '{name}' already exists")

    # Save audio file temporarily
    data_path = os.getenv("DATA_PATH", "/app/data")
    os.makedirs(f"{data_path}/temp", exist_ok=True)
    temp_path = f"{data_path}/temp/{audio_file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # Extract embedding
        embedding = engine.extract_embedding(temp_path)

        # Create speaker in database
        speaker = Speaker(name=name)
        speaker.set_embedding(embedding)

        db.add(speaker)
        db.commit()
        db.refresh(speaker)

        return speaker

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.patch("/speakers/{speaker_id}/rename", response_model=SpeakerResponse)
async def rename_speaker(
    speaker_id: int,
    rename_data: SpeakerRename,
    db: Session = Depends(get_db)
):
    """
    Rename a speaker (useful for AI agents to label unknown speakers)

    Args:
        speaker_id: Speaker ID
        rename_data: New name for the speaker
    """
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    # Check if new name already exists
    existing = db.query(Speaker).filter(Speaker.name == rename_data.new_name).first()
    if existing and existing.id != speaker_id:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker '{rename_data.new_name}' already exists"
        )

    old_name = speaker.name
    speaker.name = rename_data.new_name
    speaker.updated_at = datetime.utcnow()

    # UPDATE ALL PAST CONVERSATION SEGMENTS
    from .models import ConversationSegment
    updated_segments = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker_id
    ).update({"speaker_name": rename_data.new_name})

    db.commit()
    db.refresh(speaker)

    print(f"✓ Renamed speaker '{old_name}' → '{rename_data.new_name}' (updated {updated_segments} past segments)")

    return speaker


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: int, db: Session = Depends(get_db)):
    """Delete a speaker"""
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    db.delete(speaker)
    db.commit()

    return {"message": f"Speaker '{speaker.name}' deleted successfully"}


@router.post("/process", response_model=DiarizationResult)
async def process_audio(
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Process audio file with speaker diarization and recognition

    Args:
        audio_file: Audio file to process
    """
    # Save audio file
    data_path = os.getenv("DATA_PATH", "/app/data")
    os.makedirs(f"{data_path}/recordings", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save uploaded file temporarily
    temp_filename = f"{timestamp}_{audio_file.filename}"
    temp_path = f"{data_path}/recordings/{temp_filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Convert to WAV for reliable processing
    if not temp_path.endswith('.wav'):
        try:
            audio = AudioSegment.from_file(temp_path)
            wav_filename = temp_filename.rsplit('.', 1)[0] + '_converted.wav'
            file_path = f"{data_path}/recordings/{wav_filename}"
            audio.export(file_path, format='wav')
            filename = wav_filename
            # Remove temp file after conversion
            os.remove(temp_path)
        except Exception as e:
            # If conversion fails, use original file
            print(f"Warning: Failed to convert to WAV: {e}")
            file_path = temp_path
            filename = temp_filename
    else:
        file_path = temp_path
        filename = temp_filename

    # Create recording entry
    recording = Recording(filename=filename, status="processing")
    db.add(recording)
    db.commit()
    db.refresh(recording)

    try:
        # Get known speakers
        speakers = db.query(Speaker).all()
        known_speakers = [
            (s.id, s.name, s.get_embedding())
            for s in speakers
        ]

        # Process audio
        result = engine.process_audio_with_recognition(
            file_path,
            known_speakers,
            threshold=0.7
        )

        # Store segments
        for seg_data in result["segments"]:
            segment = Segment(
                recording_id=recording.id,
                speaker_id=seg_data.get("speaker_id"),
                start_time=seg_data["start"],
                end_time=seg_data["end"],
                confidence=seg_data.get("confidence"),
                speaker_label=seg_data["speaker_name"]
            )
            db.add(segment)

        # Update recording status
        recording.status = "completed"
        recording.duration = max(s["end"] for s in result["segments"])
        db.commit()
        db.refresh(recording)

        # Return result
        return DiarizationResult(
            recording_id=recording.id,
            num_speakers=result["num_speakers"],
            num_known=result["num_known"],
            num_unknown=result["num_unknown"],
            segments=[SegmentResponse(
                id=s.id,
                start_time=s.start_time,
                end_time=s.end_time,
                speaker_label=s.speaker_label,
                speaker_id=s.speaker_id,
                confidence=s.confidence
            ) for s in recording.segments]
        )

    except Exception as e:
        recording.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recordings", response_model=List[RecordingResponse])
async def list_recordings(db: Session = Depends(get_db)):
    """List all recordings"""
    recordings = db.query(Recording).all()
    return recordings


@router.get("/recordings/{recording_id}", response_model=RecordingResponse)
async def get_recording(recording_id: int, db: Session = Depends(get_db)):
    """Get recording details with segments"""
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording
