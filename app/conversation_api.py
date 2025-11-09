"""
API endpoints for conversation management
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
import os

from .database import get_db
from .models import Conversation, ConversationSegment, Speaker
from .schemas import (
    ConversationResponse,
    ConversationCreate,
    ConversationUpdate,
    ConversationSegmentResponse,
    IdentifySpeakerRequest
)
from .diarization import SpeakerRecognitionEngine
from .audio_utils import convert_to_mp3, batch_convert_to_mp3
from .api import get_engine

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all conversations with pagination and filtering"""
    query = db.query(Conversation).order_by(Conversation.start_time.desc())

    if status:
        query = query.filter(Conversation.status == status)

    conversations = query.offset(skip).limit(limit).all()
    return conversations


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get conversation details with all segments"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    update_data: ConversationUpdate,
    db: Session = Depends(get_db)
):
    """Update conversation metadata"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if update_data.title:
        conversation.title = update_data.title
    if update_data.status:
        conversation.status = update_data.status

    db.commit()
    db.refresh(conversation)
    return conversation


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete conversation and associated audio"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete audio file
    if conversation.audio_path and os.path.exists(conversation.audio_path):
        os.remove(conversation.audio_path)

    db.delete(conversation)
    db.commit()

    return {"message": f"Conversation {conversation_id} deleted"}


@router.post("/{conversation_id}/reprocess")
async def reprocess_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """Re-process conversation with current speaker profiles (works with MP3)"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not conversation.audio_path or not os.path.exists(conversation.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Get known speakers
    speakers = db.query(Speaker).all()
    known_speakers = [(s.id, s.name, s.get_embedding()) for s in speakers]

    # Process audio (works with both WAV and MP3!)
    # Threshold from .env - optimal default 0.20 based on ground truth testing (0 misidentifications + 50% matching)
    threshold = float(os.getenv("SPEAKER_THRESHOLD", "0.20"))
    result = engine.transcribe_with_diarization(
        conversation.audio_path,
        known_speakers,
        threshold=threshold
    )

    # Delete old segments
    db.query(ConversationSegment).filter(
        ConversationSegment.conversation_id == conversation_id
    ).delete()

    # Create new segments
    conv_start = conversation.start_time

    for seg in result["segments"]:
        # Determine speaker
        speaker_id = None
        speaker_name = seg["speaker"]
        confidence = seg.get("confidence", 0.0)

        if seg.get("is_known"):
            # Find speaker by name
            speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
            if speaker:
                speaker_id = speaker.id

        segment = ConversationSegment(
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=seg["text"],
            start_time=conv_start + timedelta(seconds=seg["start"]),
            end_time=conv_start + timedelta(seconds=seg["end"]),
            start_offset=seg["start"],
            end_offset=seg["end"],
            confidence=confidence
        )
        db.add(segment)

    # Update conversation stats
    conversation.status = "completed"
    conversation.num_segments = len(result["segments"])
    conversation.num_speakers = result["num_speakers"]

    db.commit()

    return {"message": "Conversation reprocessed", "segments": len(result["segments"])}


@router.post("/{conversation_id}/segments/{segment_id}/identify")
async def identify_speaker_in_segment(
    conversation_id: int,
    segment_id: int,
    request: IdentifySpeakerRequest,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Identify speaker in segment and optionally enroll them

    Args:
        request: Request body with speaker_id, speaker_name, and enroll flag
    """
    speaker_id = request.speaker_id
    speaker_name = request.speaker_name
    enroll = request.enroll
    segment = db.query(ConversationSegment).filter(
        ConversationSegment.id == segment_id,
        ConversationSegment.conversation_id == conversation_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    conversation = segment.conversation

    # Determine which audio file to use
    # IMPORTANT: Offsets are ALWAYS relative to the segment_audio_path if it exists
    # Segment files contain multiple speech segments at their recorded offsets
    audio_file = None
    start_time = segment.start_offset
    end_time = segment.end_offset

    # Prefer segment_audio_path (offsets are correct for this file)
    if segment.segment_audio_path and os.path.exists(segment.segment_audio_path):
        audio_file = segment.segment_audio_path
    elif conversation.audio_path and os.path.exists(conversation.audio_path):
        audio_file = conversation.audio_path
    else:
        raise HTTPException(status_code=404, detail="Audio file not found (neither conversation audio nor segment audio exists)")

    # Store the old speaker name and ID for propagation and embedding recalculation
    old_speaker_name = segment.speaker_name
    old_speaker_id = segment.speaker_id

    # Extract embedding FIRST if enrolling (needed for new speakers)
    embedding = None
    if enroll:
        try:
            # Extract from specific time range
            embedding = engine.extract_segment_embedding(
                audio_file,
                start_time,
                end_time
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract embedding: {str(e)}"
            )

    # Get or create speaker
    speaker = None
    merge_msg = ""

    if speaker_id:
        # Existing speaker by ID - load from DB
        speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
    elif speaker_name:
        # Try to find existing speaker by name
        speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()

    # At this point, speaker is either:
    # - Existing speaker (found by ID or name)
    # - None (need to create new)

    if speaker:
        # Existing speaker - we'll recalculate embedding after updating segments
        merge_msg = ""
    else:
        # New speaker - must have name and embedding
        if not speaker_name:
            raise HTTPException(status_code=400, detail="speaker_name required for new speaker")

        if not enroll or embedding is None:
            raise HTTPException(status_code=400, detail="Must enroll new speaker (enroll=True)")

        # Create new speaker with embedding
        speaker = Speaker(name=speaker_name)
        speaker.set_embedding(embedding)
        db.add(speaker)
        db.flush()  # Get ID without committing
        merge_msg = " (initial enrollment)"

    # Update THIS segment
    segment.speaker_id = speaker.id
    segment.speaker_name = speaker.name
    segment.confidence = 1.0  # Manually identified

    # UPDATE ALL OTHER SEGMENTS with the same old speaker name (retroactive identification!)
    # SAFETY: Only do retroactive updates for Unknown speakers!
    # If old speaker is already identified (Tommy, Diamond, etc.), only update THIS segment
    updated_count = 0
    if old_speaker_name and old_speaker_name != speaker.name and old_speaker_name.startswith("Unknown_"):
        updated_count = db.query(ConversationSegment).filter(
            ConversationSegment.speaker_name == old_speaker_name,
            ConversationSegment.id != segment_id  # Don't update the one we just did
        ).update({
            "speaker_id": speaker.id,
            "speaker_name": speaker.name
        })

    # Recalculate speaker embedding from ALL non-misidentified segments
    if enroll:
        # Get all segments for this speaker that are NOT misidentified
        speaker_segments = db.query(ConversationSegment).filter(
            ConversationSegment.speaker_id == speaker.id,
            ConversationSegment.is_misidentified == False
        ).all()

        if speaker_segments:
            import numpy as np
            embeddings = []

            for seg in speaker_segments:
                conv = seg.conversation
                seg_audio = seg.segment_audio_path if seg.segment_audio_path and os.path.exists(seg.segment_audio_path) else conv.audio_path

                if not seg_audio or not os.path.exists(seg_audio):
                    continue

                try:
                    emb = engine.extract_segment_embedding(
                        seg_audio,
                        seg.start_offset,
                        seg.end_offset
                    )
                    if not np.isnan(emb).any():
                        embeddings.append(emb)
                except Exception as e:
                    print(f"Warning: Could not extract embedding for segment {seg.id}: {e}")
                    continue

            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                speaker.set_embedding(avg_embedding)
                merge_msg = f" (recalculated from {len(embeddings)} non-misidentified segments)"

        # CRITICAL: Also recalculate OLD speaker's embedding to exclude this segment
        if old_speaker_id and old_speaker_id != speaker.id:
            old_speaker = db.query(Speaker).filter(Speaker.id == old_speaker_id).first()
            if old_speaker:
                # Get remaining segments for old speaker (excluding the one we just moved)
                old_speaker_segments = db.query(ConversationSegment).filter(
                    ConversationSegment.speaker_id == old_speaker_id,
                    ConversationSegment.is_misidentified == False
                ).all()

                if old_speaker_segments:
                    import numpy as np
                    old_embeddings = []
                    for seg in old_speaker_segments:
                        conv = seg.conversation
                        seg_audio = seg.segment_audio_path if seg.segment_audio_path and os.path.exists(seg.segment_audio_path) else conv.audio_path
                        if seg_audio and os.path.exists(seg_audio):
                            try:
                                emb = engine.extract_segment_embedding(seg_audio, seg.start_offset, seg.end_offset)
                                if not np.isnan(emb).any():
                                    old_embeddings.append(emb)
                            except Exception as e:
                                print(f"Warning: Could not extract embedding for segment {seg.id}: {e}")

                    if old_embeddings:
                        old_avg_embedding = np.mean(old_embeddings, axis=0)
                        old_speaker.set_embedding(old_avg_embedding)
                        print(f"✓ Recalculated embedding for '{old_speaker.name}' (removed segment {segment_id})")
                    else:
                        print(f"⚠️ No valid segments remaining for '{old_speaker.name}' after removing segment {segment_id}")

        # Auto-cleanup: Delete orphaned Unknown speakers
        if old_speaker_name.startswith("Unknown_"):
            # Check if any segments still use this Unknown speaker
            remaining_segments = db.query(ConversationSegment).filter(
                ConversationSegment.speaker_name == old_speaker_name
            ).count()

            if remaining_segments == 0:
                # No segments use this Unknown speaker anymore - delete it
                orphaned_speaker = db.query(Speaker).filter(Speaker.name == old_speaker_name).first()
                if orphaned_speaker:
                    db.delete(orphaned_speaker)
                    merge_msg += f" (auto-deleted orphaned {old_speaker_name})"

    db.commit()
    db.refresh(segment)

    return {
        "message": f"Speaker identified as {speaker.name}{merge_msg}. Updated {updated_count + 1} segment(s) total.",
        "speaker_id": speaker.id,
        "enrolled": enroll,
        "segments_updated": updated_count + 1
    }


@router.post("/batch-convert-mp3")
async def batch_convert_conversations_to_mp3(
    delete_originals: bool = True,
    db: Session = Depends(get_db)
):
    """Batch convert all WAV conversations to MP3"""
    conversations = db.query(Conversation).filter(
        Conversation.audio_format == "wav",
        Conversation.audio_path.isnot(None)
    ).all()

    converted = []
    errors = []

    for conv in conversations:
        if not os.path.exists(conv.audio_path):
            errors.append(f"File not found: {conv.audio_path}")
            continue

        try:
            mp3_path = convert_to_mp3(
                conv.audio_path,
                bitrate="192k",
                delete_original=delete_originals
            )

            conv.audio_path = mp3_path
            conv.audio_format = "mp3"
            converted.append(conv.id)

        except Exception as e:
            errors.append(f"Conv {conv.id}: {str(e)}")

    db.commit()

    return {
        "converted": len(converted),
        "conversation_ids": converted,
        "errors": errors
    }


