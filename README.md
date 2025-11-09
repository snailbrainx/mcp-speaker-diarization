# MCP Speaker Diarization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An all-in-one complete package combining GPU-accelerated speaker diarization and recognition with web interface and REST API. Integrates pyannote.audio speaker diarization with faster-whisper transcription, designed for AI agent integration and hobby projects.

## Screenshots

<details>
<summary>Click to view screenshots</summary>

<table>
  <tr>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/1.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/1.png" width="300" alt="Conversations Tab"/>
      </a>
      <br/>Conversations Tab
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/2.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/2.png" width="300" alt="Ground Truth Labeling"/>
      </a>
      <br/>Ground Truth Labeling
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/3.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/3.png" width="300" alt="Segment Audio Playback"/>
      </a>
      <br/>Segment Audio Playback
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/4.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/4.png" width="300" alt="Manage Speakers"/>
      </a>
      <br/>Manage Speakers
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/5.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/5.png" width="300" alt="Process Audio Tab"/>
      </a>
      <br/>Process Audio Tab
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/6.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/6.png" width="300" alt="Enroll Speaker"/>
      </a>
      <br/>Enroll Speaker
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/7.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/7.png" width="300" alt="Live Recording"/>
      </a>
      <br/>Live Recording
    </td>
  </tr>
</table>

</details>

## What Makes This Different

This combines pyannote.audio speaker diarization with faster-whisper transcription, adding a complete speaker recognition layer:

- **Speaker Enrollment & Recognition**: Store speaker profiles, recognize them in future recordings
- **Unknown Speaker Auto-Clustering**: Automatically group unknown speakers, identify retroactively
- **Embedding Management**: Continuous profile improvement through embedding merging
- **Misidentification Correction**: Mark and fix incorrect identifications, recalculate embeddings
- **Retroactive Updates**: Rename any speaker → all past segments automatically update
- **Conversation Structure**: Maintain "who said what" context across sessions
- **MCP Integration**: Built-in MCP server for AI agent integration
- **Complete Web UI**: Live recording, speaker management, playback, backup/restore

While other projects combine pyannote.audio and faster-whisper for basic diarization+transcription, this application adds the complete speaker recognition layer: enrolling known speakers, automatically identifying them in recordings, and maintaining speaker identity across sessions—essential for AI agents that need to remember and distinguish between multiple speakers.

## Key Features

### Core Functionality
- **Speaker Diarization**: Automatically detect "who spoke when" in audio recordings
- **Speaker Recognition**: Enroll speakers once, recognize them in all future recordings
- **Transcription**: Optional high-quality speech-to-text using faster-whisper (large-v3 model)
- **Live Recording**: Real-time streaming with voice activity detection and instant processing
- **Unknown Speaker Handling**: Automatic clustering and enrollment of new speakers
- **Conversation Management**: Organize recordings with speaker context and full transcripts

### Advanced Features
- **Misidentification Correction**: Mark and fix incorrectly identified segments
- **Embedding Merging**: Continuously improve speaker profiles with each recording
- **Retroactive Updates**: Rename a speaker → all past segments automatically update
- **Backup/Restore**: Export and restore speaker profiles and identifications
- **Ground Truth Labeling**: Test and optimize recognition accuracy
- **Bulk Operations**: Process multiple recordings, convert to MP3, delete conversations

### AI Agent Integration
- **REST API**: Full programmatic access to all features
- **MCP-Ready**: API structure designed for Model Context Protocol servers
- **Speaker Context**: Maintain "who said what" context for AI assistants
- **Auto-Enrollment**: AI can identify and label unknown speakers during conversations
- **Conversation Queries**: Retrieve full transcripts with speaker labels for context

## Use Cases

- **AI Assistant Calls**: Enable AI agents to identify and remember multiple speakers across sessions
- **Meeting Transcription**: Automatic speaker labeling for team meetings
- **Interview Processing**: Identify host vs. guests in podcasts and interviews
- **Customer Support**: Separate agent and customer in support calls
- **Research**: Analyze multi-party conversations with speaker attribution

## Technical Stack

- **Diarization**: pyannote.audio 4.0.1 (`pyannote/speaker-diarization-community-1`)
- **Embeddings**: pyannote.audio (`pyannote/embedding`)
- **Transcription**: faster-whisper 1.2.1 (large-v3 model, CTranslate2 backend)
- **Web Framework**: FastAPI 0.115.5 + Gradio 5.49.1
- **ML Framework**: PyTorch 2.5.1 with CUDA 12.4 support
- **Database**: SQLAlchemy 2.0.36 with SQLite + Pydantic 2.11.0
- **Audio Processing**: pydub, soundfile, ffmpeg
- **MCP Integration**: MCP 1.21.0 for AI agent connectivity

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 12.x support (4GB+ VRAM recommended, 8GB+ for optimal performance)
  - **Tested on**: NVIDIA RTX 3090 (24GB VRAM) - excellent performance
  - **Minimum**: 4GB VRAM for diarization + transcription
  - **Works on**: Consumer GPUs (1080, 2080, 3060, 3090, 4080, 4090, etc.)
- **CPU Fallback**: Runs on CPU but significantly slower (GPU strongly recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~10GB for models, plus space for audio recordings

### Software
- **Operating System**: Linux (tested on Ubuntu), macOS (via Docker), Windows (via WSL2 + Docker)
- **Python**: 3.11 or 3.12
- **CUDA**: 12.4 (included in Docker image)
- **cuDNN**: 9.x (auto-installed)
- **Docker** (optional but recommended): 20.10+ with NVIDIA Container Toolkit

### System Dependencies
- **ffmpeg**: Audio processing and format conversion
- **git**: HuggingFace model downloads
- **portaudio19-dev**: Live microphone recording (optional)

## Quick Start

### Prerequisites

1. **Get a HuggingFace Token**
   - Create account at [huggingface.co](https://huggingface.co/)
   - Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Accept model terms:
     - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
     - [pyannote/embedding](https://huggingface.co/pyannote/embedding)

2. **Install NVIDIA Container Toolkit** (Docker deployment)
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Option 1: Docker Deployment (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd speaker-diarization-app

# Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

**Access the application:**
- Web UI: http://localhost:8000/gradio
- API Documentation: http://localhost:8000/docs
- API Endpoint: http://localhost:8000/api/v1

### Option 2: Local Development (Python venv)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg git portaudio19-dev

# Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Run application
./run_local.sh

# Or run manually:
# export HF_TOKEN="your_token_here"
# python -m app.main
```

**First Run:**
- Models will auto-download (~3-5GB)
- Startup may take 2-3 minutes for model loading
- GPU memory will be allocated (check with `nvidia-smi`)

## Remote Access

If you're running the application on a remote server (e.g., headless Ubuntu server with GPU), you can access the web interface via SSH port forwarding.

### SSH Tunnel (Windows)

**Using PowerShell or Command Prompt:**

```powershell
ssh -L 8000:localhost:8000 username@remote-server-ip
```

**Using PuTTY:**

1. Open PuTTY and enter your server hostname/IP
2. Navigate to: **Connection → SSH → Tunnels**
3. Add forwarding rule:
   - Source port: `8000`
   - Destination: `localhost:8000`
   - Click "Add"
4. Return to Session tab and connect

**After connecting:**
- Open browser on your Windows machine
- Navigate to: `http://localhost:8000/gradio`
- The web interface will load as if running locally

### SSH Tunnel (Linux/Mac)

```bash
ssh -L 8000:localhost:8000 username@remote-server-ip
```

Then access `http://localhost:8000/gradio` in your local browser.

### Important Notes

- **Security Warning**: This application has no built-in authentication or encryption. Do NOT expose it on open/public networks. Only use on trusted local networks or via SSH tunneling.
- The SSH connection must remain open while using the application
- All audio processing happens on the remote server (utilizes remote GPU)
- Your local machine only displays the web interface
- Microphone recording uses your local browser's microphone, uploads to server
- For network deployments, consider proper HTTPS with nginx reverse proxy and authentication

## Configuration

All settings are configured via environment variables in `.env` file:

### Required
```bash
# HuggingFace token for model access
HF_TOKEN=your_huggingface_token_here
```

### Optional (with optimized defaults)
```bash
# Database location
DATABASE_URL=sqlite:////app/volumes/speakers.db

# Speaker recognition threshold (0.0-1.0)
# Lower = more strict, fewer false positives
# Optimal: 0.20 (based on ground truth testing: 0 misidentifications + 50% matching)
SPEAKER_THRESHOLD=0.20

# Context padding for embedding extraction (seconds)
# Adds time before/after segment for robust embeddings
# Optimal: 0.15s (67.4% matching + only 3 misidentifications in movie audio)
CONTEXT_PADDING=0.15

# Silence duration before processing segment (seconds)
# For live recording only
# Lower = more responsive, Higher = more complete segments
SILENCE_DURATION=0.5

# Filter common Whisper hallucinations
# Set to false if real speech is being filtered
FILTER_HALLUCINATIONS=true
```

### Optimal Settings

These settings have been validated through ground truth testing on real audio with background music and effects:

- **SPEAKER_THRESHOLD=0.20**: Achieves zero misidentifications while maintaining 50% matching rate
- **CONTEXT_PADDING=0.15**: Optimal for audio with background noise/music (67.4% matching, 3 misidentifications)
- **SILENCE_DURATION=0.5**: Balances responsiveness with complete sentence capture

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                            │
│              (Upload Audio / Live Recording)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────┐              ┌───────────────┐
│  Diarization   │              │ Transcription │
│  (pyannote)    │              │ (Whisper)     │
│                │              │               │
│ "Who spoke     │              │ "What was     │
│  when"         │              │  said"        │
└───────┬────────┘              └───────┬───────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Segment Alignment    │
            │  (Match text to       │
            │   speaker by time)    │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Embedding Extraction │
            │  (Extract speaker     │
            │   voice signature)    │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Speaker Matching     │
            │  (Compare to known    │
            │   speakers)           │
            └───────────┬───────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌────────────────┐          ┌──────────────────┐
│  Known Speaker │          │ Unknown Speaker  │
│  "Alice"       │          │ "Unknown_01"     │
└────────────────┘          └────────┬─────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Auto-Clustering     │
                          │  (Group similar      │
                          │   unknowns)          │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  User Identifies     │
                          │  → Embedding Merging │
                          │  → Retroactive       │
                          │     Updates          │
                          └──────────────────────┘
```

### Processing Pipeline

1. **Audio Input**
   - Upload: MP3/WAV files automatically converted and saved to `data/recordings/`
   - Live: Browser microphone → streaming chunks saved to `data/stream_segments/`

2. **Parallel Processing** (faster than sequential)
   - **Diarization** (pyannote): Detects speaker turns, outputs segments with anonymous labels (SPEAKER_00, SPEAKER_01, etc.)
   - **Transcription** (Whisper): Converts speech to text with timestamps
   - Both run simultaneously using ThreadPoolExecutor

3. **Segment Alignment**
   - Match transcription segments to speaker labels by timestamp overlap
   - Uses segment midpoint for matching: `(start + end) / 2`
   - Falls back to closest segment if no exact overlap

4. **Embedding Extraction**
   - For each segment, extract 192-dimensional voice signature using pyannote embedding model
   - **Context padding** (0.15s) added before/after for robustness with background noise
   - Minimum segment duration: 0.5 seconds

5. **Speaker Matching**
   - Compare segment embedding to known speaker embeddings
   - **Cosine similarity** calculation (0.0-1.0)
   - If similarity > threshold (0.20): Identified as known speaker
   - If similarity ≤ threshold: Labeled as "Unknown_XX"

6. **Unknown Speaker Handling**
   - **Embedding verification**: Check if multiple Unknown segments are the same person
   - Group similar unknowns (same threshold: 0.20)
   - Each unique voice gets unique Unknown_XX identifier
   - Embeddings stored for future auto-enrollment

7. **Auto-Enrollment** (when user identifies unknown)
   - User provides speaker name for any segment
   - If new name: Creates speaker profile automatically
   - **Embedding merging**: Averages embeddings from all segments of same speaker
   - **Retroactive updates**: All past segments with same Unknown label get updated
   - **Continuous improvement**: Each identification strengthens speaker profile

### Voice Activity Detection (VAD)

Two independent VAD systems work together:

1. **Live Recording VAD** (energy-based)
   - Calculates RMS energy: `sqrt(mean(audio^2))`
   - Threshold: 0.005 (configurable)
   - Detects speech vs. silence in real-time
   - Shows live indicator in UI: "🟢 Speech Detected" or "⚪ Idle"
   - After X seconds silence (default 0.5s), triggers segment processing

2. **Transcription VAD** (Whisper's built-in)
   - Uses Silero VAD model
   - Filters non-speech before transcription
   - Reduces hallucinations ("thank you.", "thanks for watching")
   - Enabled via `vad_filter=True` parameter

### Misidentification Correction

1. **Mark as Misidentified**: Exclude segment from embedding calculations
2. **Reassign to Correct Speaker**: Updates both speakers' embeddings
3. **Automatic Recalculation**: Embedding averaged from all non-misidentified segments
4. **Prevents Embedding Corruption**: Ensures speaker profiles remain accurate

## Usage

### Web Interface

#### 1. Enroll Speakers

Before processing recordings, enroll known speakers:

1. Navigate to **"➕ Enroll Speaker"** tab
2. Enter speaker name (e.g., "Alice")
3. Provide voice sample (10-30 seconds recommended):
   - Upload audio file, OR
   - Click "Record" and speak
4. Click **"Enroll Speaker"**

**Tips:**
- Clear audio with minimal background noise works best
- Longer samples (20-30s) provide more robust embeddings
- Multiple enrollments for same speaker will merge embeddings

#### 2. Process Audio (Upload)

1. Navigate to **"🎵 Process Audio"** tab
2. Upload audio file or record
3. Check **"Enable Transcription"** for text (optional, slower)
4. Click **"Process Audio"**
5. View results: Speaker segments with timestamps and confidence scores
6. Find full conversation in **"💬 Conversations"** tab

#### 3. Live Recording

1. Navigate to **"🔴 Live Recording"** tab
2. Click **"Start Recording"** (grant browser microphone permission)
3. Speak naturally - watch VAD indicator for speech detection
4. Monitor live stats: Audio level, segments queued/processed
5. Click **"Stop Recording"** when finished
6. Recording auto-processes and appears in Conversations

**Live Recording Features:**
- Real-time VAD indicator
- Automatic silence detection
- Parallel segment processing
- Instant transcription updates
- Auto-converts to MP3 for storage

#### 4. Manage Conversations

Navigate to **"💬 Conversations"** tab:

**View Conversations:**
- Select conversation from dropdown
- View full transcript with speaker labels
- Click segments to play audio
- See confidence scores for identifications

**Identify Unknown Speakers:**
1. Select conversation with Unknown_XX speakers
2. Find segment with unknown speaker
3. Enter correct speaker name
4. Check **"Auto-enroll if new speaker"** (creates speaker profile automatically)
5. Click **"Identify Speaker in Segment"**
6. All segments with same Unknown label update retroactively

**Fix Misidentifications:**
1. Select misidentified segment
2. Check **"Mark as Misidentified"** checkbox
3. Optionally provide correct speaker name
4. Click **"Identify Speaker in Segment"**
5. Speaker embeddings recalculate excluding misidentified segments

**Other Actions:**
- **Reprocess**: Re-run diarization with current speaker profiles
- **Delete**: Remove conversation and associated audio
- **Backup/Restore**: Export/import speaker profiles (JSON format)

### REST API

Full interactive API documentation available at: `http://localhost:8000/docs` (Swagger/OpenAPI)

Base URL: `http://localhost:8000/api/v1`

#### System Endpoints

**GET /status** - System health and GPU status
```bash
curl http://localhost:8000/api/v1/status
```
Response:
```json
{
  "status": "ok",
  "gpu_available": true,
  "device": "cuda:0",
  "speaker_count": 5,
  "conversation_count": 12
}
```

---

#### Speaker Management

**GET /speakers** - List all enrolled speakers
```bash
curl http://localhost:8000/api/v1/speakers
```
Response:
```json
[
  {
    "id": 1,
    "name": "Alice",
    "created_at": "2025-01-09T10:30:00",
    "embedding_count": 1
  },
  {
    "id": 2,
    "name": "Bob",
    "created_at": "2025-01-09T11:00:00",
    "embedding_count": 3
  }
]
```

**POST /speakers/enroll** - Enroll new speaker or merge with existing
```bash
curl -X POST http://localhost:8000/api/v1/speakers/enroll \
  -F "name=Alice" \
  -F "audio_file=@voice_sample.wav"
```
Parameters:
- `name` (form field): Speaker name
- `audio_file` (file): Audio file (WAV/MP3, 10-30s recommended)

Response:
```json
{
  "id": 1,
  "name": "Alice",
  "embedding_shape": [192],
  "message": "Speaker enrolled successfully"
}
```

**PATCH /speakers/{speaker_id}/rename** - Rename speaker (updates all past segments)
```bash
curl -X PATCH http://localhost:8000/api/v1/speakers/5/rename \
  -H "Content-Type: application/json" \
  -d '{"new_name": "Alice Smith"}'
```
Response:
```json
{
  "message": "Speaker renamed successfully",
  "updated_segments": 42
}
```

**DELETE /speakers/{speaker_id}** - Delete speaker profile
```bash
curl -X DELETE http://localhost:8000/api/v1/speakers/5
```
Response:
```json
{
  "message": "Speaker deleted successfully",
  "affected_segments": 42
}
```

---

#### Audio Processing

**POST /process** - Process audio file (diarization + optional transcription)
```bash
# With transcription
curl -X POST http://localhost:8000/api/v1/process \
  -F "audio_file=@recording.mp3" \
  -F "enable_transcription=true"

# Diarization only (faster)
curl -X POST http://localhost:8000/api/v1/process \
  -F "audio_file=@recording.mp3"
```
Parameters:
- `audio_file` (file): Audio file (WAV/MP3)
- `enable_transcription` (form field, optional): `true` or `false` (default: false)

Response:
```json
{
  "conversation_id": 15,
  "num_speakers": 3,
  "duration": 125.5,
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "Alice",
      "speaker_id": 1,
      "confidence": 0.92,
      "text": "Hello, how are you today?"
    },
    {
      "start": 3.5,
      "end": 5.8,
      "speaker": "Unknown_01",
      "speaker_id": null,
      "confidence": null,
      "text": "I'm doing well, thanks for asking."
    }
  ]
}
```

---

#### Conversation Management

**GET /conversations** - List all conversations (with pagination)
```bash
# Get first 10 conversations
curl "http://localhost:8000/api/v1/conversations?skip=0&limit=10"
```
Query parameters:
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Max records to return (default: 100)

Response:
```json
[
  {
    "id": 15,
    "title": "Conversation 15",
    "start_time": "2025-01-09T14:30:00",
    "duration": 125.5,
    "num_speakers": 3,
    "audio_path": "data/recordings/conv_15_full.mp3"
  }
]
```

**GET /conversations/{conversation_id}** - Get conversation with full transcript
```bash
curl http://localhost:8000/api/v1/conversations/15
```
Response:
```json
{
  "id": 15,
  "title": "Conversation 15",
  "start_time": "2025-01-09T14:30:00",
  "duration": 125.5,
  "num_speakers": 3,
  "audio_path": "data/recordings/conv_15_full.mp3",
  "segments": [
    {
      "id": 234,
      "start": 0.5,
      "end": 3.2,
      "speaker_name": "Alice",
      "speaker_id": 1,
      "text": "Hello, how are you today?",
      "confidence": 0.92,
      "is_misidentified": false
    }
  ]
}
```

**PATCH /conversations/{conversation_id}** - Update conversation metadata
```bash
curl -X PATCH http://localhost:8000/api/v1/conversations/15 \
  -H "Content-Type: application/json" \
  -d '{"title": "Team Meeting - 2025-01-09"}'
```

**DELETE /conversations/{conversation_id}** - Delete conversation and audio
```bash
curl -X DELETE http://localhost:8000/api/v1/conversations/15
```

**POST /conversations/{conversation_id}/reprocess** - Re-run diarization with current speakers
```bash
curl -X POST http://localhost:8000/api/v1/conversations/15/reprocess
```
Use case: After enrolling new speakers, reprocess past conversations to identify them

---

#### Speaker Identification

**POST /conversations/{conversation_id}/segments/{segment_id}/identify** - Identify speaker in segment
```bash
# Identify and auto-enroll new speaker
curl -X POST "http://localhost:8000/api/v1/conversations/15/segments/234/identify?speaker_name=Charlie&enroll=true"

# Identify as existing speaker
curl -X POST "http://localhost:8000/api/v1/conversations/15/segments/234/identify?speaker_name=Alice&enroll=false"

# Mark as misidentified
curl -X POST "http://localhost:8000/api/v1/conversations/15/segments/234/identify?speaker_name=Alice&enroll=false&mark_misidentified=true"
```
Query parameters:
- `speaker_name` (required): Name to assign to segment
- `enroll` (optional): Create speaker profile if doesn't exist (default: false)
- `mark_misidentified` (optional): Mark segment as misidentified (default: false)

Response:
```json
{
  "message": "Speaker identified successfully",
  "segment_id": 234,
  "speaker_name": "Charlie",
  "retroactive_updates": 8,
  "speaker_created": true
}
```

---

#### Recordings

**GET /recordings** - List all recording files
```bash
curl http://localhost:8000/api/v1/recordings
```

**GET /recordings/{recording_id}** - Get recording details
```bash
curl http://localhost:8000/api/v1/recordings/5
```

---

#### Bulk Operations

**POST /conversations/batch-convert-mp3** - Convert all WAV recordings to MP3
```bash
curl -X POST "http://localhost:8000/api/v1/conversations/batch-convert-mp3?delete_originals=true"
```
Query parameters:
- `delete_originals` (optional): Delete WAV files after conversion (default: false)

Response:
```json
{
  "converted": 5,
  "failed": 0,
  "space_saved_mb": 234.5
}
```

#### AI Agent Integration

The API includes a built-in **MCP (Model Context Protocol) server** for seamless AI agent integration.

##### MCP Server

The system includes a native MCP server accessible at `/mcp` endpoint:

**Endpoint:** `http://localhost:8000/mcp`
**Transport:** HTTP with Server-Sent Events (SSE)
**Protocol:** JSON-RPC 2.0 (MCP 2024-11-05)

**Connection from AI Agents:**

```json
{
  "url": "http://10.x.x.x:8000/mcp",
  "transport": "http"
}
```

Replace `10.x.x.x` with your server's IP address for network access (e.g., from Flowise, Claude Desktop, or other MCP clients).

**Available MCP Tools:**

1. **get_latest_segments**(conversation_id?, limit=20)
   - Get recent conversation segments with speaker labels and transcripts
   - Returns segment IDs, speaker names, text, timestamps
   - Use to see what was just said

2. **identify_speaker_in_segment**(conversation_id, segment_id, speaker_name, auto_enroll=True)
   - Identify or correct speaker in a segment
   - KEY tool for when user says "Unknown_01 is Bob"
   - Automatically updates all matching past segments

3. **list_speakers**()
   - Get all enrolled speaker profiles with IDs and names

4. **get_conversation**(conversation_id)
   - Get full conversation with all segments and transcript

5. **list_conversations**(skip=0, limit=10)
   - Get list of conversations with pagination

6. **rename_speaker**(speaker_id, new_name)
   - Rename speaker (automatically updates all past segments)

7. **delete_speaker**(speaker_id)
   - Delete speaker profile

8. **delete_all_unknown_speakers**()
   - Cleanup tool: Delete ALL speakers with names starting with 'Unknown_'
   - Useful after identifying all unknowns in conversations
   - No parameters needed

9. **reprocess_conversation**(conversation_id)
   - Re-analyze conversation with current speaker profiles
   - Useful after enrolling new speakers

10. **update_conversation_title**(conversation_id, title)
   - Update conversation title

**Example AI Agent Workflow:**

```python
# AI agent connects to MCP server at http://10.x.x.x:8000/mcp

# 1. Get latest segments
result = mcp_client.call_tool("get_latest_segments", {"limit": 20})

# 2. AI sees "Unknown_01" in transcript
# User: "Unknown_01 is actually Bob"

# 3. AI identifies the speaker
mcp_client.call_tool("identify_speaker_in_segment", {
    "conversation_id": 15,
    "segment_id": 234,
    "speaker_name": "Bob",
    "auto_enroll": True  # Creates speaker profile automatically
})

# 4. System automatically:
#    - Creates "Bob" speaker profile
#    - Updates all past "Unknown_01" segments to "Bob"
#    - Returns count of retroactively updated segments
```

**Example: AI Assistant with Speaker Context**

```python
import requests

# Get conversation transcript with speakers
response = requests.get("http://localhost:8000/api/v1/conversations/123")
conversation = response.json()

# Build context for AI
context = f"Meeting on {conversation['start_time']}:\n\n"
for segment in conversation['segments']:
    speaker = segment['speaker_name']
    text = segment['text']
    context += f"{speaker}: {text}\n"

# AI now has full speaker context
# AI can identify unknowns: "Unknown_01 sounds like Bob from engineering"
requests.post(
    "http://localhost:8000/api/v1/conversations/123/segments/456/identify",
    params={"speaker_name": "Bob", "enroll": True}
)
```

## Advanced Features

### Embedding Merging

When identifying unknown speakers or re-identifying existing speakers:

- **Never replaces** embeddings (would lose historical data)
- **Always merges** using averaging: `(existing_embedding + new_embedding) / 2`
- **Continuous improvement**: Each recording strengthens speaker profile
- **Handles variability**: Averages across different audio conditions, emotions, etc.

### Retroactive Identification

Rename any speaker → all past segments automatically update:

```bash
# User identifies Unknown_01 as "Alice" in conversation 5
curl -X POST "http://localhost:8000/api/v1/conversations/5/segments/123/identify?speaker_name=Alice&enroll=true"

# System automatically:
# 1. Creates "Alice" speaker profile (if new)
# 2. Updates segment 123
# 3. Finds ALL segments with speaker_name="Unknown_01"
# 4. Updates ALL to speaker_name="Alice"
# 5. Merges embeddings from all segments
# 6. Returns count of updated segments
```

### Backup & Restore

Export and restore speaker profiles:

**Backup:**
- Exports all speakers and their embeddings to JSON
- Includes segment assignments for full state recovery
- Saves to `backups/backup_YYYYMMDD_HHMMSS.json`
- **Does NOT include audio files** (only speaker data)

**Restore:**
- Reconstructs speaker database from backup
- Restores embeddings and segment assignments
- Useful for testing different configurations
- Useful for migrating between deployments

### Ground Truth Labeling

Test and optimize recognition accuracy:

1. Manually label segments with true speaker identities
2. Labels stored separately (doesn't affect actual segments)
3. Run tests comparing predictions vs. labels
4. Optimize threshold and padding parameters
5. Current optimal settings derived from this testing

## Data Persistence

### Directory Structure

```
speaker-diarization-app/
├── data/
│   ├── recordings/              # Permanent audio storage
│   │   ├── conv_7_full.mp3     # Live recordings (MP3)
│   │   ├── uploaded_1_tommy_converted.wav  # Uploads
│   │   └── 20251109_160230_meeting.wav    # Timestamped uploads
│   │
│   ├── stream_segments/         # Live recording segments (temporary)
│   │   └── conv_7/
│   │       ├── seg_0001.wav
│   │       ├── seg_0002.wav
│   │       └── ...
│   │
│   └── temp/                    # Temporary segment extractions
│       └── segment_123_456.wav
│
├── volumes/
│   ├── speakers.db              # SQLite database
│   └── huggingface_cache/       # Downloaded models
│
├── backups/                     # Backup snapshots (JSON)
│   └── backup_20251109_120000.json
│
├── scripts/                     # Utility scripts
│   ├── migrate_temp_audio.py   # Fix audio paths
│   ├── diagnose_speakers.py    # Debug issues
│   └── ...
│
└── tests/                       # Test files
    └── test_*.py
```

### Docker Volumes

All data persists via volume mounts in `docker-compose.yml`:

```yaml
volumes:
  - ./volumes:/app/volumes          # Database + model cache
  - ./data:/app/data                # Audio files
  - ./backups:/app/backups          # Backup snapshots
```

**What Persists:**
- ✅ Speaker profiles and embeddings
- ✅ All conversations and segments
- ✅ Audio recordings
- ✅ Downloaded models (~3-5GB)
- ✅ Backup snapshots

**What Doesn't Persist:**
- ❌ Container state (rebuild-safe)
- ❌ Logs (use `docker-compose logs -f` to monitor)

## Troubleshooting

### Installation Issues

**"HuggingFace token not found"**
- Ensure `HF_TOKEN` set in `.env` file
- Accept model terms at HuggingFace (links in Prerequisites)
- Check token has no extra spaces/quotes

**"Unable to load libcudnn_cnn.so.9"**
- Standalone: `run_local.sh` sets LD_LIBRARY_PATH automatically
- Docker: Dockerfile installs cuDNN via pip
- Manual: `pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12`

**Permission errors**
```bash
sudo chown -R $USER:docker data/ volumes/ backups/
```

**Docker GPU not detected**
```bash
# Verify NVIDIA Container Toolkit installed
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit
```

### Processing Issues

**"CUDA out of memory"**
- Close other GPU applications
- Process shorter audio segments
- Enable transcription selectively (disable for diarization-only)
- Fallback: Run on CPU (set `CUDA_VISIBLE_DEVICES=""` - very slow)

**Speaker not recognized**
- Enrollment audio should be 10-30 seconds minimum
- Use clear audio with minimal background noise
- Check threshold: Lower = more strict (try 0.15-0.25 range)
- Re-enroll with better quality audio

**"Audio file not found" errors**
- Old uploads: Run `python scripts/migrate_temp_audio.py`
- New uploads: Should auto-save to `data/recordings/`
- Check `allowed_paths` in `app/main.py` (required for Gradio)

**Whisper hallucinations ("thank you.", "thanks for watching")**
- Already filtered via energy thresholding and text filtering
- Set `FILTER_HALLUCINATIONS=true` in `.env`
- Ensure `vad_filter=True` in transcription (default)

### Performance Issues

**Slow processing**
- Verify GPU in use: Check `nvidia-smi` during processing
- Docker: Ensure `runtime: nvidia` in docker-compose.yml
- Check CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- First run: Models download (~3-5GB), subsequent runs much faster

**High memory usage**
- Normal: Models load ~4-6GB VRAM
- Transcription adds ~2-3GB
- Multiple simultaneous processes multiply memory usage
- Reduce batch size or process sequentially

### Audio Issues

**No audio playback in UI**
- Docker: Ensure `allowed_paths` set in `gr.mount_gradio_app()`
- Check audio files exist: `ls data/recordings/`
- Check browser console for errors
- Try different browser (tested: Chrome, Firefox)

**Live recording not working**
- Browser permission: Allow microphone access
- Standalone: Install PortAudio: `sudo apt-get install portaudio19-dev`
- Check browser microphone settings
- Try different browser

### Dependency Compatibility Issues

**Gradio 500 Error: "TypeError: argument of type 'bool' is not iterable"**
- **Cause**: Pydantic 2.12+ is incompatible with Gradio 5.x
- **Solution**: Use Pydantic 2.11.0 (already set in requirements.txt)
- **Check version**: `pip show pydantic`
- **Fix if needed**: `pip install pydantic==2.11.0 --force-reinstall`
- **Important**: Gradio 5.49.1 requires `pydantic>=2.0,<2.12` and MCP 1.21.0 requires `pydantic>=2.11.0`, so Pydantic 2.11.0 is the compatible version for both

**Root Cause**: Pydantic 2.12+ changed JSON schema generation to use `additionalProperties: false` (boolean value), but Gradio's schema introspection code expects dictionary objects and attempts `if "const" in schema`, which fails when schema is a boolean.

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

### Dependency Licenses

All major dependencies use permissive open-source licenses compatible with MIT:

- **pyannote.audio** (4.0.1): MIT License
  - Models require HuggingFace token and terms acceptance
  - Models themselves remain open-source and MIT licensed
- **faster-whisper** (1.2.1): MIT License (SYSTRAN)
- **Gradio** (5.49.1): Apache License 2.0
- **FastAPI** (0.115.5): MIT License
- **PyTorch** (2.5.1): BSD 3-Clause License
- **SQLAlchemy** (2.0.36): MIT License
- **Pydantic** (2.11.0): MIT License
- **MCP** (1.21.0): MIT License

**Note:** While the software licenses are permissive, pyannote's pretrained models require:
1. HuggingFace account
2. Access token
3. Acceptance of model terms of use

This is an authentication requirement, not a licensing restriction. The models remain open-source.

## Credits

This project builds upon exceptional open-source work:

- **[pyannote.audio](https://github.com/pyannote/pyannote-audio)** by Hervé Bredin - State-of-the-art speaker diarization and embedding models
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** by SYSTRAN - Optimized Whisper implementation using CTranslate2
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Original speech recognition model
- **[FastAPI](https://github.com/tiangolo/fastapi)** by Sebastián Ramírez - Modern web framework
- **[Gradio](https://github.com/gradio-app/gradio)** - ML web interfaces made simple

Thank you to these projects and their contributors for making this application possible.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional language support (currently English-only)
- Performance optimizations
- UI/UX improvements
- Documentation improvements

## Disclaimer

This software is provided "as-is" without warranty of any kind. The developers make no guarantees about the accuracy of speaker identification or transcription. While we've implemented best practices and extensive testing, speaker recognition is inherently probabilistic and may produce errors.

**Use responsibly:**
- Verify important identifications manually
- Test thoroughly in your environment
- Respect privacy and obtain consent before recording
- This is a tool to assist, not replace, human judgment

Some portions of this codebase were developed collaboratively with Claude Code (AI pair programming assistant). While thoroughly tested, we recommend reviewing code before deploying in critical applications.

---

**Questions or issues?** Open an issue on GitHub or check existing issues for solutions.

**Want to use this with AI agents?** See the API Reference section for MCP integration guidance.
