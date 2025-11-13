"""
API endpoints for conversation management
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
import json
import os
import subprocess
import tempfile

from .database import get_db
from .models import Conversation, ConversationSegment, Speaker
from .schemas import (
    ConversationResponse,
    ConversationListItem,
    ConversationsListResponse,
    ConversationCreate,
    ConversationUpdate,
    ConversationSegmentResponse,
    IdentifySpeakerRequest,
    ToggleMisidentifiedRequest
)
from .diarization import SpeakerRecognitionEngine
from .audio_utils import convert_to_mp3, batch_convert_to_mp3
from .api import get_engine

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=ConversationsListResponse)
async def list_conversations(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all conversations with pagination and filtering.
    Returns lightweight summaries without segments for better performance.
    """
    query = db.query(Conversation).order_by(Conversation.start_time.desc())

    if status:
        query = query.filter(Conversation.status == status)

    # Get total count
    total = query.count()

    # Get paginated results (no segments loaded)
    conversations = query.offset(skip).limit(limit).all()

    return ConversationsListResponse(
        conversations=conversations,
        total=total,
        skip=skip,
        limit=limit
    )


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

        # Serialize word-level data if available
        words_json = json.dumps(seg["words"]) if seg.get("words") else None

        segment = ConversationSegment(
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=seg["text"],
            start_time=conv_start + timedelta(seconds=seg["start"]),
            end_time=conv_start + timedelta(seconds=seg["end"]),
            start_offset=seg["start"],
            end_offset=seg["end"],
            confidence=confidence,
            emotion_category=seg.get("emotion_category"),
            emotion_arousal=seg.get("emotion_arousal"),
            emotion_valence=seg.get("emotion_valence"),
            emotion_confidence=seg.get("emotion_confidence"),
            words_data=words_json,  # Include word-level confidence
            avg_logprob=seg.get("avg_logprob")
        )
        db.add(segment)

    # Update conversation stats
    conversation.status = "completed"
    conversation.num_segments = len(result["segments"])
    conversation.num_speakers = result["num_speakers"]

    db.commit()

    # Clear GPU cache after reprocessing
    engine.clear_gpu_cache()

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

    # Determine which audio file to use for embedding extraction
    # CRITICAL: Database offsets (start_offset/end_offset) are ALWAYS conversation-relative!
    # They represent seconds from the conversation start, NOT from individual segment files.
    # Therefore, we MUST use the full conversation audio file where these offsets are valid.
    audio_file = None
    start_time = segment.start_offset
    end_time = segment.end_offset

    # PREFER full conversation audio (where conversation-relative offsets are valid)
    if conversation.audio_path and os.path.exists(conversation.audio_path):
        audio_file = conversation.audio_path
    elif segment.segment_audio_path and os.path.exists(segment.segment_audio_path):
        # Fallback: segment audio only if conversation audio missing
        # NOTE: This will likely fail because offsets don't match segment file
        audio_file = segment.segment_audio_path
        print(f"⚠️ WARNING: Using segment audio with conversation-relative offsets - may extract wrong audio!")
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

    # ALWAYS recalculate NEW speaker embedding from ALL their non-misidentified segments
    # This ensures the speaker profile improves with every identification
    speaker_segments = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker.id,
        ConversationSegment.is_misidentified == False
    ).all()

    if speaker_segments:
        import numpy as np
        embeddings = []

        for seg in speaker_segments:
            conv = seg.conversation

            # CRITICAL: Prefer full conversation audio over segment audio!
            # Database offsets (start_offset/end_offset) are CONVERSATION-RELATIVE.
            # Segment audio files (seg_0001.wav) are individual clips with different offsets.
            # Using segment audio file with conversation-relative offsets extracts WRONG audio!
            seg_audio = conv.audio_path if conv.audio_path and os.path.exists(conv.audio_path) else None

            # Fallback to segment audio only if no conversation audio exists
            if not seg_audio and seg.segment_audio_path and os.path.exists(seg.segment_audio_path):
                seg_audio = seg.segment_audio_path

            if not seg_audio:
                print(f"  ⚠️ Skipping segment {seg.id}: No audio file available")
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
                print(f"  ⚠️ Could not extract embedding for segment {seg.id} from {os.path.basename(seg_audio)}: {e}")
                continue

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            speaker.set_embedding(avg_embedding)
            print(f"✓ Recalculated embedding for '{speaker.name}' (added segment {segment_id}, now {len(embeddings)} total segments)")
            merge_msg = f" (recalculated from {len(embeddings)} non-misidentified segments)"

    # CRITICAL: Also recalculate OLD speaker's embedding to exclude this segment
    # SKIP if old speaker is Unknown_* (will be auto-deleted below, no point recalculating)
    if old_speaker_id and old_speaker_id != speaker.id and not (old_speaker_name and old_speaker_name.startswith("Unknown_")):
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

                    # CRITICAL: Prefer full conversation audio over segment audio (see above comment)
                    seg_audio = conv.audio_path if conv.audio_path and os.path.exists(conv.audio_path) else None
                    if not seg_audio and seg.segment_audio_path and os.path.exists(seg.segment_audio_path):
                        seg_audio = seg.segment_audio_path

                    if seg_audio:
                        try:
                            emb = engine.extract_segment_embedding(seg_audio, seg.start_offset, seg.end_offset)
                            if not np.isnan(emb).any():
                                old_embeddings.append(emb)
                        except Exception as e:
                            print(f"  ⚠️ Could not extract embedding for segment {seg.id}: {e}")

                if old_embeddings:
                    old_avg_embedding = np.mean(old_embeddings, axis=0)
                    old_speaker.set_embedding(old_avg_embedding)
                    print(f"✓ Recalculated embedding for '{old_speaker.name}' (removed segment {segment_id})")
                else:
                    print(f"⚠️ No valid segments remaining for '{old_speaker.name}' after removing segment {segment_id}")

    # Auto-cleanup: Delete orphaned Unknown speakers
    if old_speaker_name and old_speaker_name.startswith("Unknown_"):
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

    # Clear GPU cache after all embedding extractions
    engine.clear_gpu_cache()

    return {
        "message": f"Speaker identified as {speaker.name}{merge_msg}. Updated {updated_count + 1} segment(s) total.",
        "speaker_id": speaker.id,
        "enrolled": enroll,
        "segments_updated": updated_count + 1
    }


@router.patch("/{conversation_id}/segments/{segment_id}/misidentified")
async def toggle_segment_misidentified(
    conversation_id: int,
    segment_id: int,
    request: ToggleMisidentifiedRequest,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Toggle misidentification status for a segment and recalculate speaker embedding

    When a segment is marked as misidentified, it's excluded from the speaker's
    embedding calculation, improving recognition accuracy.
    """
    import numpy as np

    segment = db.query(ConversationSegment).filter(
        ConversationSegment.id == segment_id,
        ConversationSegment.conversation_id == conversation_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Update misidentification status
    old_status = segment.is_misidentified
    segment.is_misidentified = request.is_misidentified

    # If segment has a speaker, recalculate their embedding
    if segment.speaker_id:
        speaker = db.query(Speaker).filter(Speaker.id == segment.speaker_id).first()

        if speaker:
            # Get all non-misidentified segments for this speaker
            speaker_segments = db.query(ConversationSegment).filter(
                ConversationSegment.speaker_id == speaker.id,
                ConversationSegment.is_misidentified == False
            ).all()

            if speaker_segments:
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
                    print(f"✓ Recalculated embedding for '{speaker.name}' from {len(embeddings)} non-misidentified segments")
                else:
                    print(f"⚠️ No valid segments remaining for '{speaker.name}' after marking segment {segment_id} as misidentified")

    db.commit()
    db.refresh(segment)

    # Clear GPU cache after embedding extractions
    engine.clear_gpu_cache()

    status_text = "marked as misidentified" if request.is_misidentified else "unmarked as misidentified"
    return {
        "message": f"Segment {segment_id} {status_text}",
        "is_misidentified": segment.is_misidentified,
        "embedding_recalculated": segment.speaker_id is not None
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


@router.get("/segments/{segment_id}/audio")
async def get_segment_audio(
    segment_id: int,
    db: Session = Depends(get_db)
):
    """
    Extract and serve audio for a specific conversation segment.

    Uses ffmpeg to extract the segment's time range from the full conversation audio.
    Returns WAV audio file.
    """
    print(f"🎵 Audio request for segment {segment_id}")

    segment = db.query(ConversationSegment).filter(ConversationSegment.id == segment_id).first()
    if not segment:
        print(f"❌ Segment {segment_id} not found in database")
        raise HTTPException(status_code=404, detail="Segment not found")

    conversation = segment.conversation

    # Determine source audio file and check if we need extraction
    # CRITICAL: Streaming segment files (seg_XXXX.wav) contain the RAW VAD-triggered audio chunk.
    # After diarization, ONE segment file may contain MULTIPLE speaker segments.
    # We MUST extract the specific time range, not serve the whole file!

    # First check: Can we use full conversation audio? (Best option)
    use_conversation_audio = conversation.audio_path and os.path.exists(conversation.audio_path)

    # Second check: Use segment file if conversation audio doesn't exist yet (during streaming)
    use_segment_audio = segment.segment_audio_path and os.path.exists(segment.segment_audio_path)

    if not use_conversation_audio and not use_segment_audio:
        print(f"❌ No audio file found for segment {segment_id}")
        print(f"  segment_audio_path: {segment.segment_audio_path}")
        print(f"  conversation.audio_path: {conversation.audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Prefer full conversation audio (offsets are conversation-relative)
    if use_conversation_audio:
        source_audio = conversation.audio_path
        start_time = segment.start_offset
        end_time = segment.end_offset
        print(f"  Using conversation audio: {source_audio}")
        print(f"  Offsets: {start_time:.2f}s - {end_time:.2f}s (conversation-relative)")
    else:
        # Fallback: Use segment file with file-relative offsets
        # Need to calculate the segment's position within its segment file
        source_audio = segment.segment_audio_path
        # TODO: Calculate file-relative offsets from segment file metadata
        # For now, serve entire segment file (may contain extra audio)
        print(f"  ⚠️ Using segment audio (may contain multiple segments): {source_audio}")
        start_time = 0  # Start of segment file
        # Get duration from file
        from pydub import AudioSegment as AS
        audio = AS.from_file(source_audio)
        end_time = len(audio) / 1000.0  # Convert ms to seconds
        print(f"  Serving entire segment file: 0s - {end_time:.2f}s")

    # Create temporary directory for extracted segments
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"segment_{segment_id}_{int(datetime.now().timestamp())}.wav")

    try:
        # Use ffmpeg to extract the specific time range with small padding at end
        duration = end_time - start_time
        duration_with_padding = duration + 0.25  # Add 250ms to avoid cutting off last word
        print(f"  Extracting {duration_with_padding:.2f}s from offset {start_time:.2f}s")
        print(f"  Output: {temp_path}")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration_with_padding),
            "-i", source_audio,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            temp_path
        ], check=True, capture_output=True, text=True)

        if not os.path.exists(temp_path):
            print(f"❌ Extraction failed - temp file not created")
            raise HTTPException(status_code=500, detail="Audio extraction failed")

        file_size = os.path.getsize(temp_path)
        print(f"✅ Extracted successfully ({file_size} bytes)")

        # Return the extracted audio file with cache control headers
        from starlette.background import BackgroundTask

        # Clean up temp file after sending
        def cleanup():
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"🗑️  Cleaned up {temp_path}")
            except Exception as e:
                print(f"Failed to cleanup temp file {temp_path}: {e}")

        return FileResponse(
            path=temp_path,
            media_type="audio/wav",
            filename=f"segment_{segment_id}.wav",
            background=BackgroundTask(cleanup),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Error extracting audio: {e.stderr}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


