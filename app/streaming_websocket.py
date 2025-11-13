"""
WebSocket endpoint for real-time audio streaming and transcription.
Integrates with StreamingRecorder for live speaker diarization.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

from .database import get_db
from .models import Conversation, ConversationSegment, Speaker
from .streaming_recorder import StreamingRecorder
from .audio_utils import convert_to_mp3
from .config import get_config
import os

router = APIRouter(prefix="/streaming", tags=["Streaming"])

# Active WebSocket connections (conversation_id -> WebSocket)
active_connections: Dict[int, WebSocket] = {}

# Active recorders (conversation_id -> StreamingRecorder)
active_recorders: Dict[int, StreamingRecorder] = {}


def get_engine():
    """Get shared speaker recognition engine (preloaded at startup)"""
    from .api import get_engine as get_api_engine
    return get_api_engine()


async def send_message(websocket: WebSocket, message_type: str, data: dict):
    """Send JSON message to WebSocket client"""
    try:
        # Check if WebSocket is still connected
        if websocket.client_state.CONNECTED:
            await websocket.send_json({
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
    except Exception as e:
        # Silently fail if connection is closed
        pass


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:
    - Client → Server: Binary audio chunks (ArrayBuffer)
    - Server → Client: JSON messages (status, segment, error)
    """
    await websocket.accept()
    conversation_id: Optional[int] = None
    recorder: Optional[StreamingRecorder] = None

    try:
        # Wait for initial "start" message
        init_message = await websocket.receive_json()

        if init_message.get("type") != "start":
            await send_message(websocket, "error", {"message": "Expected 'start' message"})
            await websocket.close()
            return

        # Create conversation
        conversation = Conversation(
            title=f"Live Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            start_time=datetime.utcnow(),
            status="recording"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        conversation_id = conversation.id
        active_connections[conversation_id] = websocket

        # Initialize recorder
        config = get_config()
        settings = config.get_settings()

        recorder = StreamingRecorder(
            sample_rate=48000,
            silence_threshold=0.005,
            silence_duration=settings.silence_duration,
            max_workers=2
        )

        # Get event loop for scheduling async tasks from background threads
        loop = asyncio.get_event_loop()

        # Set callback after initialization
        # Use asyncio.run_coroutine_threadsafe to schedule from background thread
        def segment_callback(seg_info):
            asyncio.run_coroutine_threadsafe(
                _handle_segment_processed(websocket, conversation_id, seg_info, db, get_engine()),
                loop
            )

        recorder.on_segment_processed = segment_callback

        recorder.start_recording(conversation_id)
        active_recorders[conversation_id] = recorder

        # Send confirmation
        await send_message(websocket, "started", {
            "conversation_id": conversation_id,
            "sample_rate": 48000,
            "message": "Recording started"
        })

        # Main loop: receive and process audio chunks
        while True:
            try:
                # Receive message
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary audio chunk
                    audio_bytes = message["bytes"]
                    print(f"📦 Received audio chunk: {len(audio_bytes)} bytes")

                    # Convert bytes to numpy array (assuming float32)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    print(f"🔊 Converted to audio array: {len(audio_array)} samples")

                    # Process chunk (StreamingRecorder expects tuple of (sample_rate, audio_data))
                    result = recorder.process_audio_chunk((48000, audio_array))
                    print(f"📊 VAD: {result['speech_detected']}, Level: {result['audio_level']:.3f}")

                    # Send status update
                    await send_message(websocket, "status", {
                        "vad_active": result["speech_detected"],
                        "audio_level": float(result["audio_level"]),
                        "stats": {
                            "buffer_size": result.get("buffer_size", 0),
                            "segments_processed": result.get("segments_processed", 0),
                            "total_audio_seconds": result.get("segments_processed", 0) * 2.0  # Estimate
                        }
                    })

                elif "text" in message:
                    # JSON message (e.g., stop command)
                    data = json.loads(message["text"])

                    if data.get("type") == "stop":
                        # Stop recording
                        break

            except WebSocketDisconnect:
                print(f"WebSocket disconnected for conversation {conversation_id}")
                break
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
                import traceback
                traceback.print_exc()
                try:
                    await send_message(websocket, "error", {"message": str(e)})
                except:
                    pass  # Connection already closed
                break  # Exit loop on error

        # Cleanup: stop recording and finalize
        await _finalize_recording(conversation_id, recorder, conversation, db, websocket)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected during initialization")
        if conversation_id and recorder:
            await _finalize_recording(conversation_id, recorder, conversation, db, None)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket.client_state.CONNECTED:
            await send_message(websocket, "error", {"message": str(e)})
    finally:
        # Cleanup
        if conversation_id:
            active_connections.pop(conversation_id, None)
            active_recorders.pop(conversation_id, None)

        # Close WebSocket if still open (ignore if already closed)
        try:
            if websocket.client_state.CONNECTED:
                await websocket.close()
        except RuntimeError:
            # WebSocket already closed, ignore
            pass


async def _handle_segment_processed(
    websocket: WebSocket,
    conversation_id: int,
    segment_info: dict,
    db: Session,
    engine
):
    """
    Callback when StreamingRecorder finishes processing a segment.
    Runs diarization + transcription, saves to DB, sends to client.
    """
    try:
        # Get conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if not conversation:
            return

        # Get segment file path
        segment_file = segment_info["segment_file"]
        start_offset = segment_info["start_offset"]
        end_offset = segment_info["end_offset"]

        if not os.path.exists(segment_file):
            print(f"Segment file not found: {segment_file}")
            return

        # Get known speakers
        speakers = db.query(Speaker).all()
        known_speakers = [(s.id, s.name, s.get_embedding()) for s in speakers]

        # Get threshold from config
        config = get_config()
        settings = config.get_settings()
        threshold = settings.speaker_threshold

        # Process with diarization + transcription
        result = engine.transcribe_with_diarization(
            segment_file,
            known_speakers,
            threshold=threshold,
            db_session=db
        )

        # Save segments to database
        conv_start = conversation.start_time
        segments_data = []

        for seg in result["segments"]:
            # Determine speaker
            speaker_id = None
            speaker_name = seg["speaker"]
            confidence = seg.get("confidence", 0.0)

            if seg.get("is_known"):
                speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
                if speaker:
                    speaker_id = speaker.id
            else:
                # Auto-enroll unknown speakers with embeddings (enables clustering)
                embedding = seg.get("embedding")
                if embedding is not None and speaker_name.startswith("Unknown_"):
                    from .diarization import auto_enroll_unknown_speaker
                    speaker_id, speaker_name = auto_enroll_unknown_speaker(
                        embedding, db, threshold=threshold
                    )
                    # Update confidence since we're using the enrolled speaker
                    confidence = 1.0 if speaker_id else confidence

            # Adjust offsets relative to conversation start
            seg_start_offset = start_offset + seg["start"]
            seg_end_offset = start_offset + seg["end"]

            # Serialize word-level data as JSON if available
            words_json = None
            if "words" in seg and seg["words"]:
                words_json = json.dumps(seg["words"])

            segment = ConversationSegment(
                conversation_id=conversation_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                text=seg["text"],
                start_time=conv_start + timedelta(seconds=seg_start_offset),
                end_time=conv_start + timedelta(seconds=seg_end_offset),
                start_offset=seg_start_offset,
                end_offset=seg_end_offset,
                confidence=confidence,
                emotion_category=seg.get("emotion_category"),
                emotion_confidence=seg.get("emotion_confidence"),
                segment_audio_path=segment_file,
                words_data=words_json,
                avg_logprob=seg.get("avg_logprob")
            )
            db.add(segment)
            db.flush()

            segments_data.append({
                "segment_id": segment.id,
                "speaker_name": speaker_name,
                "text": seg["text"],
                "start_offset": seg_start_offset,
                "end_offset": seg_end_offset,
                "confidence": confidence,
                "emotion_category": seg.get("emotion_category"),
                "emotion_confidence": seg.get("emotion_confidence"),
                "is_known": seg.get("is_known", False),
                "words": seg.get("words", []),  # Include word-level data
                "avg_logprob": seg.get("avg_logprob")
            })

        # Update conversation stats
        conversation.num_segments = db.query(ConversationSegment).filter(
            ConversationSegment.conversation_id == conversation_id
        ).count()

        db.commit()

        # Send segments to client
        for seg_data in segments_data:
            await send_message(websocket, "segment", seg_data)

        # Clear GPU cache
        engine.clear_gpu_cache()

    except Exception as e:
        print(f"Error processing segment: {e}")
        await send_message(websocket, "error", {"message": f"Segment processing error: {str(e)}"})


async def _finalize_recording(
    conversation_id: int,
    recorder: StreamingRecorder,
    conversation: Conversation,
    db: Session,
    websocket: Optional[WebSocket]
):
    """
    Finalize recording: stop recorder, concatenate segments, convert to MP3.
    """
    try:
        print(f"Finalizing recording for conversation {conversation_id}")

        # Stop recorder and wait for queue to finish
        recorder.stop_recording()

        # Concatenate segments
        full_audio_path = recorder.concatenate_segments()

        if full_audio_path and os.path.exists(full_audio_path):
            # Convert to MP3
            try:
                mp3_path = convert_to_mp3(full_audio_path, delete_original=True)
                conversation.audio_path = mp3_path
                conversation.audio_format = "mp3"
            except Exception as e:
                print(f"MP3 conversion failed: {e}, keeping WAV")
                conversation.audio_path = full_audio_path
                conversation.audio_format = "wav"

        # Update conversation status
        conversation.status = "completed"
        conversation.end_time = datetime.utcnow()

        # Calculate duration
        if conversation.num_segments and conversation.num_segments > 0:
            last_segment = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conversation_id
            ).order_by(ConversationSegment.end_offset.desc()).first()

            if last_segment:
                conversation.duration = last_segment.end_offset

        # Count speakers
        speaker_count = db.query(ConversationSegment.speaker_name).filter(
            ConversationSegment.conversation_id == conversation_id
        ).distinct().count()
        conversation.num_speakers = speaker_count

        db.commit()

        # Send completion message
        if websocket and websocket.client_state.CONNECTED:
            await send_message(websocket, "completed", {
                "conversation_id": conversation_id,
                "num_segments": conversation.num_segments,
                "num_speakers": conversation.num_speakers,
                "duration": conversation.duration,
                "message": "Recording completed and saved"
            })

        print(f"Recording finalized: {conversation.num_segments} segments, {conversation.num_speakers} speakers")

    except Exception as e:
        print(f"Error finalizing recording: {e}")
        conversation.status = "failed"
        db.commit()

        if websocket and websocket.client_state.CONNECTED:
            await send_message(websocket, "error", {"message": f"Finalization error: {str(e)}"})
