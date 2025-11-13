# MCP Speaker Diarization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An all-in-one complete package combining GPU-accelerated speaker diarization and recognition with web interface and REST API. Integrates pyannote.audio speaker diarization with faster-whisper transcription, designed for AI agent integration and hobby projects.

## Screenshots

Example Next.js frontend interface (available at [github.com/snailbrainx/speaker_identity_nextjs](https://github.com/snailbrainx/speaker_identity_nextjs)):

<table>
  <tr>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/1.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/1.png" width="300" alt="Settings - Voice Profile Management"/>
      </a>
      <br/>Settings - Voice Profile Management
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/2.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/2.png" width="300" alt="Process Audio - Upload Files"/>
      </a>
      <br/>Process Audio - Upload Files
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/3.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/3.png" width="300" alt="Conversation Detail - Segments & Transcription"/>
      </a>
      <br/>Conversation Detail - Segments & Transcription
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/4.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/4.png" width="300" alt="Conversations List"/>
      </a>
      <br/>Conversations List
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/5.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/5.png" width="300" alt="Speaker Management - Enroll & Manage"/>
      </a>
      <br/>Speaker Management - Enroll & Manage
    </td>
    <td align="center">
      <a href="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/6.png">
        <img src="https://raw.githubusercontent.com/snailbrainx/mcp-speaker-diarization/master/screenshots/6.png" width="300" alt="Live Recording - Real-time Transcription"/>
      </a>
      <br/>Live Recording - Real-time Transcription
    </td>
  </tr>
</table>

## What Makes This Different

This combines pyannote.audio speaker diarization with faster-whisper transcription, adding a complete speaker recognition layer:

- **Speaker Enrollment & Recognition**: Store speaker profiles, recognize them in future recordings
- **Emotion Detection**: Real-time emotion recognition (angry, happy, sad, neutral, etc.) per speech segment
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
- **Emotion Detection**: Real-time emotion recognition using emotion2vec+ (ACL 2024) - detects angry, happy, sad, neutral, fearful, surprised, disgusted emotions per segment
- **Transcription**: Optional high-quality speech-to-text using faster-whisper (large-v3 model) with word-level confidence scores
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
- **Meeting Transcription**: Automatic speaker labeling for team meetings with word-level confidence scores
- **Subtitle Generation**: Create accurate subtitles for movies and TV shows with speaker identification
- **Security & Authentication**: Voice identification at scale for security systems and access control
- **Interview Processing**: Identify host vs. guests in podcasts and interviews
- **Customer Support**: Separate agent and customer in support calls
- **Research**: Analyze multi-party conversations with speaker attribution
- **Agent Integration**: Seamless integration with AI agents via REST API and MCP server

## Technical Stack

- **Diarization**: pyannote.audio 4.0.1 (`pyannote/speaker-diarization-community-1`)
- **Embeddings**: pyannote.audio (`pyannote/embedding`)
- **Emotion Recognition**: emotion2vec_plus_large via FunASR (ACL 2024, 9 emotion categories)
- **Transcription**: faster-whisper 1.2.1 (configurable models: tiny/base/small/medium/large-v3, supports 99 languages, CTranslate2 backend)
- **Backend API**: FastAPI 0.115.5 with WebSocket streaming support
- **ML Framework**: PyTorch 2.5.1 with CUDA 12.4 support
- **Database**: SQLAlchemy 2.0.36 with SQLite + Pydantic 2.11.0
- **Audio Processing**: pydub, soundfile, ffmpeg
- **MCP Integration**: MCP 1.21.0 for AI agent connectivity

## Emotion Detection

Real-time emotion recognition for each speech segment using **emotion2vec+** (ACL 2024, published paper).

### Features
- **9 Emotion Categories**: angry, happy, sad, neutral, fearful, surprised, disgusted, other, unknown
- **Per-Segment Analysis**: Emotion detected for each speaker turn
- **Dimensional Mapping**: Automatic arousal/valence scores (0-1 scale)
- **Confidence Scores**: Per-prediction confidence levels
- **Real-Time Processing**: Works with both live recording and file upload
- **API Integration**: All emotion data returned via REST API and WebSocket

### Supported Emotions
- **angry**: High arousal, negative valence
- **happy**: High arousal, positive valence
- **sad**: Low arousal, negative valence
- **neutral**: Mid arousal, mid valence
- **fearful**: High arousal, negative valence
- **surprised**: High arousal, mid-to-positive valence
- **disgusted**: Mid-to-high arousal, negative valence
- **other**: Catch-all for ambiguous emotions
- **unknown**: Unable to classify

### API Response Format
Each conversation segment includes emotion fields:
```json
{
  "speaker": "Andy",
  "transcription": "Hey, is anyone home?",
  "emotion_category": "happy",
  "emotion_confidence": 0.87,
  "emotion_arousal": 0.7,
  "emotion_valence": 0.9
}
```

### Performance
- **Accuracy**: 100% on test set (angry, happy, neutral)
- **Speed**: ~32ms per segment
- **Model Size**: ~300M parameters
- **VRAM**: ~2GB

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 12.x support
  - **Tested on**: NVIDIA RTX 3090 (24GB VRAM) - excellent performance
  - **VRAM Requirements** (faster-whisper is very efficient):
    - Diarization + embeddings: ~2-3GB base requirement
    - Emotion detection: ~2GB (emotion2vec_plus_large)
    - **Whisper model adds** (choose based on available VRAM):
      - `tiny`/`base`: ~400-500MB (total: ~5GB minimum with emotion)
      - `small`: ~1GB (total: ~6GB recommended with emotion)
      - `medium`: ~2GB (total: ~7GB recommended with emotion)
      - `large-v3`: ~3-4GB (total: ~8-9GB recommended with emotion, default)
  - **Works on**: Consumer GPUs (GTX 1060 6GB+, 1080, 2060, 3060, 3090, 4080, 4090, etc.)
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
- API Documentation: http://localhost:8000/docs
- API Endpoint: http://localhost:8000/api/v1
- MCP Server: http://localhost:8000/mcp

For a web interface, see the separate [Next.js frontend repository](https://github.com/snailbrainx/speaker_identity_nextjs).

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
- Navigate to: `http://localhost:8000/docs` (API documentation)

### SSH Tunnel (Linux/Mac)

```bash
ssh -L 8000:localhost:8000 username@remote-server-ip
```

Then access API docs at `http://localhost:8000/docs`.

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
# Recommended: 0.30 for normal home usage (good balance of accuracy and matching)
# Alternative: 0.20 for stricter matching with movie audio/background music
SPEAKER_THRESHOLD=0.30

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

# Whisper transcription model (faster-whisper with CTranslate2)
# Choose based on GPU capabilities:
# - tiny.en / tiny: ~400MB VRAM, fastest, lowest accuracy
# - base.en / base: ~500MB VRAM, very fast, basic accuracy
# - small.en / small: ~1GB VRAM, fast, good accuracy
# - medium.en / medium: ~2GB VRAM, slower, better accuracy
# - large-v3 / large-v2: ~3-4GB VRAM, slowest, best accuracy
WHISPER_MODEL=large-v3

# Whisper language setting
# - "en" = English only (default, fastest)
# - "auto" = Auto-detect language (99 languages supported)
# - Or specify: "es", "fr", "de", "zh", "ja", etc.
WHISPER_LANGUAGE=en
```

### Recommended Settings

Default settings are optimized for normal home usage:

- **SPEAKER_THRESHOLD=0.30**: Good balance of accuracy and matching for home conversations
- **CONTEXT_PADDING=0.15**: Optimal for audio with background noise/music
- **SILENCE_DURATION=0.5**: Balances responsiveness with complete sentence capture
- **WHISPER_MODEL=large-v3**: Best accuracy, requires ~3-4GB VRAM. Use `small` (~1GB) or `base` (~500MB) for weaker GPUs.
- **WHISPER_LANGUAGE=en**: English only (fastest). Use `auto` for multilingual auto-detection or specify language code.

For stricter matching with movie audio or challenging conditions, reduce SPEAKER_THRESHOLD to 0.20.

## How It Works

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          User Input                              │
│                (Upload Audio / Live Recording)                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Audio Format         │
               │  Conversion           │
               │  (if needed)          │
               │                       │
               │  MP3/M4A → WAV        │
               │  Live: 48kHz chunks   │
               └───────────┬───────────┘
                           │
            ╔══════════════╧════════════════╗
            ║  PARALLEL PROCESSING          ║  ← ~50% faster!
            ║  ThreadPoolExecutor           ║     Both run
            ║  (2 workers)                  ║     simultaneously
            ╚══════════════╤════════════════╝
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌────────────────────┐           ┌───────────────────────┐
│  Transcription     │           │  Diarization          │
│  (faster-whisper)  │           │  (pyannote.audio)     │
│                    │           │                       │
│  "What was said"   │           │  "Who spoke when"     │
│                    │           │                       │
│  • Speech → Text   │           │  • Detect speaker     │
│  • Word timestamps │           │    turns              │
│  • Confidence      │           │  • Assign labels      │
│    scores          │           │    (SPEAKER_00, etc.) │
│  • VAD filtering   │           │  • Time boundaries    │
│                    │           │                       │
│  ~2-10 seconds     │           │  ~2-10 seconds        │
└─────────┬──────────┘           └───────────┬───────────┘
          │                                  │
          └──────────────┬───────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Segment Alignment    │
             │                       │
             │  Match transcription  │
             │  to speaker labels    │
             │  by timestamp overlap │
             └───────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────────┐      ┌────────────────────────┐
│ Embedding          │      │  Speaker Matching      │
│ Extraction         │      │  (Cosine Similarity)   │
│ (pyannote)         │      │                        │
│                    │      │  Compare embeddings    │
│ • Extract voice    │──────→  to known speakers     │
│   signature        │      │                        │
│ • 512-D vectors    │      │  Threshold: 0.20-0.30  │
│ • Context padding  │      │                        │
│   (0.15s)          │      │  Match or Unknown?     │
│ • Skip if <0.5s    │      │                        │
└────────────────────┘      └───────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                ┌─────────────────┐         ┌──────────────────┐
                │  Known Speaker  │         │ Unknown Speaker  │
                │  "Alice"        │         │ "Unknown_01"     │
                │                 │         │                  │
                │  • Has ID       │         │  • No ID yet     │
                │  • Confidence   │         │  • Auto-enrolled │
                │    score        │         │  • Embedding     │
                │                 │         │    stored        │
                └────────┬────────┘         └────────┬─────────┘
                         │                           │
                         └──────────┬────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Emotion Detection    │
                        │  (emotion2vec+)       │
                        │                       │
                        │  "How they felt"      │
                        │                       │
                        │  Per segment:         │
                        │  • Extract audio      │
                        │  • Resample to 16kHz  │
                        │  • Run inference      │
                        │  • Parse results      │
                        │                       │
                        │  Output (4 fields):   │
                        │  • Category (9 types) │
                        │    angry, happy, sad, │
                        │    neutral, fearful,  │
                        │    surprised, etc.    │
                        │  • Arousal (0-1)      │
                        │  • Valence (0-1)      │
                        │  • Confidence (0-1)   │
                        │                       │
                        │  ~32ms per segment    │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Database Storage     │
                        │                       │
                        │  ConversationSegment: │
                        │  • text               │
                        │  • speaker_name       │
                        │  • speaker_id         │
                        │  • confidence         │
                        │  • emotion_category   │
                        │  • emotion_arousal    │
                        │  • emotion_valence    │
                        │  • emotion_confidence │
                        │  • start/end times    │
                        │  • word-level data    │
                        └───────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐          ┌──────────────────────┐
          │  Auto-Clustering │          │  User Identifies     │
          │                  │          │  Unknown Speaker     │
          │  Group similar   │          │                      │
          │  Unknown speakers│          │  "Unknown_01 is Bob" │
          │  by embedding    │          │                      │
          │  similarity      │          │  → Embedding Merging │
          └──────────────────┘          │  → Retroactive       │
                                        │     Updates (all     │
                                        │     past segments)   │
                                        └──────────────────────┘
```

**Key Points:**
- **Parallel Processing**: Transcription (Whisper) and Diarization (Pyannote) run simultaneously using ThreadPoolExecutor, achieving ~50% speedup
- **Audio Conversion**: Automatic format conversion (MP3→WAV) before processing; live recording saves 48kHz chunks
- **Sequential Operations**: Alignment → Embedding Extraction → Speaker Matching → Emotion Detection (in order)
- **Emotion Detection**: Runs AFTER speaker identification, per segment, with automatic 16kHz resampling
- **Sample Rates**: Browser (48kHz) → Whisper/Pyannote (auto-resample) → Emotion (16kHz) → Storage (MP3 192k)

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
   - If similarity > threshold (default 0.30): Identified as known speaker
   - If similarity ≤ threshold: Labeled as "Unknown_XX"

6. **Unknown Speaker Handling**
   - **Embedding verification**: Check if multiple Unknown segments are the same person
   - Group similar unknowns (same threshold)
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

A modern Next.js web interface is available as a separate project with full voice management capabilities:

**Repository:** https://github.com/snailbrainx/speaker_identity_nextjs

**Features:**
- Live recording with real-time transcription and speaker identification
- Speaker enrollment and management
- Conversation browsing with audio playback
- Speaker identification and correction
- Profile management and backup/restore
- Ground truth labeling for testing

The frontend connects to this API backend and provides a complete user interface for all speaker diarization features. See the frontend repository for installation and usage instructions.

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

#### Settings Management

**GET /settings/voice** - Get current voice processing settings
```bash
curl http://localhost:8000/api/v1/settings/voice
```
Response:
```json
{
  "speaker_threshold": 0.30,
  "context_padding": 0.15,
  "silence_duration": 0.5,
  "filter_hallucinations": true
}
```

**POST /settings/voice** - Update voice processing settings
```bash
curl -X POST http://localhost:8000/api/v1/settings/voice \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_threshold": 0.25,
    "context_padding": 0.15,
    "silence_duration": 0.5,
    "filter_hallucinations": true
  }'
```

**POST /settings/voice/reset** - Reset settings to defaults
```bash
curl -X POST http://localhost:8000/api/v1/settings/voice/reset
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

#### WebSocket Streaming

**WS /streaming/ws** - Real-time audio streaming with live transcription

WebSocket endpoint for browser-based live recording with real-time processing.

**Protocol:**
- Client → Server: Binary audio chunks (Float32Array)
- Server → Client: JSON messages (status, segment, error, completed)

**Usage:**
```javascript
// 1. Connect WebSocket
const ws = new WebSocket('ws://localhost:8000/api/v1/streaming/ws');

// 2. Send start message
ws.send(JSON.stringify({ type: 'start' }));

// 3. Stream audio chunks (48kHz, Float32)
const mediaRecorder = new MediaRecorder(stream);
mediaRecorder.ondataavailable = (e) => {
  const arrayBuffer = await e.data.arrayBuffer();
  const float32Array = new Float32Array(arrayBuffer);
  ws.send(float32Array.buffer);
};

// 4. Receive real-time segments
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'segment') {
    console.log(`${data.data.speaker_name}: ${data.data.text}`);
  }
};

// 5. Stop recording
ws.send(JSON.stringify({ type: 'stop' }));
```

**Message Types:**
- `started` - Recording initiated, returns `conversation_id`
- `status` - Real-time VAD status and audio level
- `segment` - Completed segment with transcription and speaker
- `completed` - Recording finished successfully
- `error` - Processing error

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

---

#### Voice Profiles & Backup

**GET /profiles/** - List all voice profiles
```bash
curl http://localhost:8000/api/v1/profiles/
```

**POST /profiles/** - Create new empty profile
```bash
curl -X POST http://localhost:8000/api/v1/profiles/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Home Profile", "description": "Family members"}'
```

**POST /profiles/duplicate** - Duplicate current state to new profile
```bash
curl -X POST http://localhost:8000/api/v1/profiles/duplicate \
  -H "Content-Type: application/json" \
  -d '{"name": "Backup 2025-01", "description": "Monthly backup"}'
```

**PATCH /profiles/{profile_name}** - Update profile with current state
```bash
curl -X PATCH http://localhost:8000/api/v1/profiles/MyProfile \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'
```

**POST /profiles/restore** - Restore speakers and settings from profile
```bash
curl -X POST http://localhost:8000/api/v1/profiles/restore \
  -H "Content-Type: application/json" \
  -d '{"filename": "profile_Home_Profile.json"}'
```

**GET /profiles/download/{profile_name}** - Download profile as JSON
```bash
curl http://localhost:8000/api/v1/profiles/download/MyProfile > myprofile.json
```

**GET /profiles/download-all** - Download all profiles as ZIP
```bash
curl http://localhost:8000/api/v1/profiles/download-all > profiles.zip
```

**POST /profiles/import** - Import profile from JSON file
```bash
curl -X POST http://localhost:8000/api/v1/profiles/import \
  -F "file=@myprofile.json"
```

**DELETE /profiles/{profile_name}** - Delete profile and checkpoints
```bash
curl -X DELETE http://localhost:8000/api/v1/profiles/MyProfile
```

**Checkpoint Management:**
```bash
# Create checkpoint snapshot
curl -X POST http://localhost:8000/api/v1/profiles/MyProfile/checkpoints

# List checkpoints
curl http://localhost:8000/api/v1/profiles/MyProfile/checkpoints

# Delete checkpoint
curl -X DELETE http://localhost:8000/api/v1/profiles/MyProfile/checkpoints/20250109_143000
```

---

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

11. **search_conversations_by_speaker**(speaker_name, limit=50, skip=0)
   - Search conversation history by speaker name
   - Returns all conversations where speaker appears with IDs, titles, datetimes, durations, segment counts
   - Ordered by most recent first
   - Use case: "What was I talking about with Nick last week?"
   - Returns error if speaker doesn't exist

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

## AI Assistant Integration Examples

This section shows how to build a conversational AI assistant that uses this speaker diarization system for continuous voice interaction with speaker memory.

### Example Interaction Flow

```
[Unknown speaker detected in conversation]
User: "Alright mate, how are you doing?"
Friend: "Good mate, you?"

AI Assistant: "Who are you speaking to?"
User: "Oh, that's Nick"
AI Assistant: "Oh hi Nick!"
[Background: AI calls MCP tool to identify Unknown_17627242 as "Nick"]

[Future conversations automatically recognize Nick]
Nick: "Hey, remember what we discussed yesterday?"
AI Assistant: "Yes Nick, you mentioned the project deadline..."
```

### Integration Architecture

**How data flows:**

```
┌─────────────────────────────────────────────────────────┐
│  Your AI Assistant Application (you build this)         │
│                                                          │
│  [1] Continuous audio recording                         │
│         ↓                                                │
│  [2] POST audio → /api/v1/process-audio                 │
│         ↓                                                │
│  [3] Receive transcription response                     │
│         ↓                                                │
│  [4] Format to JSON & send to LLM                       │
│         ↓                                                │
│  [5] LLM responds (voice/text)                          │
│                                                          │
│  [6] When AI needs speaker info/history:                │
│      LLM calls MCP tools (on-demand only!)              │
│      - identify_unknown_speaker()                       │
│      - get_conversation_transcript()                    │
│      - update_speaker()                                 │
└─────────────────────────────────────────────────────────┘
              ↓ REST API (continuous)
              ↓ MCP (on-demand)
┌─────────────────────────────────────────────────────────┐
│  MCP Speaker Diarization (this system)                  │
│  - Processes audio                                       │
│  - Identifies speakers                                   │
│  - Returns transcriptions                                │
│  - Stores conversation history                           │
└─────────────────────────────────────────────────────────┘
```

**Key point:**
- **REST API** = continuous data flow (your app gets transcriptions)
- **MCP** = tools AI calls only when needed (query history, update speakers)

---

### Implementation Guide

#### 1. Continuous Recording & Transcription (Your App)

**Your application sends audio to REST API:**

```python
# your_ai_app.py - Continuous recording loop

import httpx
import sounddevice as sd
import numpy as np
from datetime import datetime

DIARIZATION_API = "http://localhost:8000/api/v1"

async def record_and_process():
    """Continuous recording, sends to diarization API"""

    while True:
        # Record audio chunk (e.g., 5 seconds)
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
        sd.wait()

        # Send to diarization API
        files = {"file": ("audio.wav", audio.tobytes())}
        data = {"enable_transcription": "true"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DIARIZATION_API}/process-audio",
                files=files,
                data=data
            )

        result = response.json()

        # Process each segment returned
        for segment in result.get("segments", []):
            await handle_transcription(segment)
```

#### 2. Format Transcriptions for Your LLM

**Convert API response to your LLM format:**

```python
# your_ai_app.py - Format for LLM

async def handle_transcription(segment: dict):
    """Format transcription and send to LLM"""

    # Format as your structured JSON
    message = {
        "method": "voice",
        "speaker": segment["speaker"],
        "timestamp": segment["start_time"],  # or use current time
        "content": segment.get("transcription", "")
    }

    # Example segment from API:
    # {
    #   "speaker": "Andy",
    #   "start_time": "2025-11-09T12:43:00Z",
    #   "transcription": "Hey, is anyone home?",
    #   "emotion_category": "happy",
    #   "emotion_confidence": 0.87,
    #   "emotion_arousal": 0.7,
    #   "emotion_valence": 0.9
    # }

    # OR for unknown speaker:
    # {
    #   "speaker": "Unknown_17627242",
    #   "start_time": "2025-11-09T12:45:00Z",
    #   "transcription": "Good mate, you?",
    #   "emotion_category": "neutral",
    #   "emotion_confidence": 0.76,
    #   "emotion_arousal": 0.5,
    #   "emotion_valence": 0.5
    # }

    # Decide if this should go to AI
    if await should_forward_to_ai(message):
        await send_to_llm(message)
    else:
        # Store for potential later retrieval
        await store_ambient_conversation(message)
```

#### 3. Intelligent Conversation Routing

**Filter what goes to your main AI (saves cost/latency):**

```python
# your_ai_app.py - Conversation filter

async def should_forward_to_ai(message: dict) -> bool:
    """
    Use small/fast LLM to detect if speech is directed at AI.
    Only forwards AI-directed speech to main (expensive) LLM.
    """

    # Use lightweight model (GPT-3.5-turbo, Claude Haiku, etc.)
    filter_prompt = f"""Is this person talking TO the AI assistant, or having a conversation with someone else?

Speaker: {message['speaker']}
Said: "{message['content']}"

Reply with only: AI_DIRECTED or AMBIENT_CONVERSATION"""

    result = await lightweight_llm.complete(filter_prompt)
    return "AI_DIRECTED" in result

async def send_to_llm(message: dict):
    """Send to main AI for processing"""

    # Your LLM conversation with system prompt
    response = await main_llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_voice_message(message)}
    ])

    # Respond with voice/text
    await respond(response)
```

#### 4. System Prompt for Unknown Speaker Handling

**Add to your AI assistant's system prompt:**

```python
SYSTEM_PROMPT = """
You are a helpful AI assistant with access to speaker diarization via MCP tools.

When you detect an unknown speaker (speaker name starts with "Unknown_"):
1. Ask the user who they are speaking to
2. When the user identifies the speaker, call the identify_unknown_speaker MCP tool
3. Greet the newly identified person naturally

Example interaction:
User says: "Oh, that's Nick"
You should:
- Call: identify_unknown_speaker(unknown_speaker_id="Unknown_17627242", real_name="Nick")
- Respond: "Oh hi Nick! Nice to meet you."

The system will automatically update all past segments from this unknown speaker.

Available MCP tools for speaker management:
- identify_unknown_speaker(unknown_speaker_id, real_name)
- get_conversation_transcript(conversation_id)
- update_speaker(speaker_id, new_name)
- list_speakers()
- delete_all_unknown_speakers()
- search_conversations_by_speaker(speaker_name, limit, skip)

Use these tools when users reference past conversations or update speaker identities.
"""
```

#### 5. MCP Tool Calls (On-Demand Only)

**When AI needs to query or update speaker data:**

```python
# your_ai_app.py - MCP client integration

import httpx

MCP_ENDPOINT = "http://localhost:8000/mcp"

async def call_mcp_tool(tool_name: str, arguments: dict):
    """Call MCP tool when AI needs it"""

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(MCP_ENDPOINT, json=payload)

    return response.json()

# Example 1: User says "that's Nick"
# Your LLM extracts the intent and calls:

result = await call_mcp_tool(
    "identify_unknown_speaker",
    {
        "unknown_speaker_id": "Unknown_17627242",
        "real_name": "Nick"
    }
)
# Response: "Updated speaker Unknown_17627242 to Nick. Updated 5 past segments."

# Example 2: User says "what did Nick say about the meeting?"
# Your LLM calls:

result = await call_mcp_tool(
    "get_conversation_transcript",
    {
        "conversation_id": "3",  # You track which conversation
        "include_transcription": "true"
    }
)
# Response: Full transcript with all segments, times, speakers
```

#### 6. Retroactive Context Retrieval

**When user references past conversations:**

```python
# User says: "I was talking to you earlier about the project"

# Your AI flow:
# 1. LLM recognizes need for context
# 2. Calls MCP to get recent conversation
conversation = await call_mcp_tool(
    "get_conversation_transcript",
    {"conversation_id": recent_conversation_id}
)

# 3. Parse transcript and add to context
for segment in conversation["segments"]:
    if segment["speaker"] == "Andy":
        context += f"{segment['speaker']}: {segment['transcription']}\n"

# 4. Respond with full context
# Now AI can answer: "You mentioned the deadline was Friday..."
```

#### 7. Conversational Speaker Updates

**Complete example of the Nick conversation:**

```python
# Conversation flow in your app:

# [Segment 1 arrives from REST API]
{
    "speaker": "Andy",
    "transcription": "Alright mate, how are you doing?",
    "start_time": "12:43:00"
}
→ Send to LLM (no response needed, not directed at AI)

# [Segment 2 arrives]
{
    "speaker": "Unknown_17627242",
    "transcription": "Good mate, you?",
    "start_time": "12:43:05"
}
→ Send to LLM (triggers AI's unknown speaker protocol)
→ AI responds: "Who are you speaking to?"

# [Segment 3 arrives]
{
    "speaker": "Andy",
    "transcription": "Oh, that's Nick",
    "start_time": "12:43:10"
}
→ Send to LLM
→ LLM extracts: speaker_id="Unknown_17627242", real_name="Nick"
→ LLM calls MCP tool:
await call_mcp_tool(
    "identify_unknown_speaker",
    {
        "unknown_speaker_id": "Unknown_17627242",
        "real_name": "Nick"
    }
)
→ AI responds: "Oh hi Nick!"

# [All future segments now show:]
{
    "speaker": "Nick",  # ← Updated automatically
    "transcription": "...",
}
```

---

### Advanced: Visual Speaker Recognition

**Extend your app with camera-based identification:**

```python
# your_ai_app.py - Add visual recognition

async def handle_unknown_speaker(unknown_id: str):
    """When unknown speaker detected, try visual identification"""

    # 1. Trigger camera (USB webcam, IP camera, etc.)
    photo = await camera.capture()

    # 2. Send to vision LLM (runs separately, doesn't block main AI)
    vision_result = await vision_llm.identify_face(photo)
    # Example: {"name": "Nick", "confidence": 0.92}

    # 3. If high confidence, auto-identify
    if vision_result.confidence > 0.85:
        await call_mcp_tool(
            "identify_unknown_speaker",
            {
                "unknown_speaker_id": unknown_id,
                "real_name": vision_result.name
            }
        )
        await respond(f"Hi {vision_result.name}!")
    else:
        # Fall back to asking
        await respond("Who are you speaking to?")

# Security layer for sensitive commands
async def verify_before_action(command: str, speaker: str):
    """Visual verification before executing sensitive commands"""

    if is_sensitive(command):
        photo = await camera.capture()
        person = await vision_llm.identify(photo)

        if person.name == speaker and person.confidence > 0.95:
            await execute_command(command)
        else:
            await respond("I need to visually verify it's you first")
```

---

### What's Included vs. What You Build

**✅ Included in this system:**
- REST API endpoints for audio processing
- Speaker enrollment and recognition
- Audio transcription (faster-whisper)
- Unknown speaker auto-clustering
- Conversation storage and retrieval
- MCP server with 10 tools for AI integration

**🔨 You build (your separate application):**
- Continuous audio recording loop
- REST API client (sends audio, receives transcriptions)
- LLM integration (your choice: OpenAI, Anthropic, etc.)
- Conversation routing/filtering logic
- Voice output (TTS - text-to-speech)
- MCP client (calls tools when AI needs them)
- Camera integration (optional)
- Visual recognition LLM (optional)
- Your application-specific logic

---

### Example Use Cases

**1. Smart Home Assistant**
```
Scenario: Household members interact naturally
- Continuous recording in main living area
- Recognizes family members by voice
- Executes commands only from authorized speakers
- Asks visitors to identify themselves first
```

**2. Personal AI Companion**
```
Scenario: Always-on conversational assistant
- Records all conversations (with consent)
- Maintains context across days/weeks
- Remembers who said what and when
- Can be queried: "What did Nick say about the project?"
```

**3. Meeting Room Assistant**
```
Scenario: Automated meeting transcription
- Identifies all participants as they speak
- Generates real-time transcript with speaker labels
- Answers questions: "Who mentioned the deadline?"
- Creates action items assigned to speakers
```

**4. Multi-User AI System**
```
Scenario: Personalized AI for each user
- Switches personality/context based on speaker
- Maintains separate conversation histories
- Different permissions per user
- Personalized responses and memory
```

---

### Quick Start Integration

```bash
# 1. Start this diarization system
./run_local.sh

# 2. In your AI app, install client libraries
pip install httpx sounddevice

# 3. Create your AI application with structure above
# your_ai_app/
#   ├── main.py              # Continuous recording loop
#   ├── mcp_client.py        # MCP tool calls
#   ├── llm_integration.py   # Your LLM (OpenAI/Anthropic/etc)
#   └── config.py            # API endpoints

# 4. Run your application
python your_ai_app/main.py
```

**Minimal working example:**

```python
# minimal_assistant.py

import httpx
import sounddevice as sd

async def main():
    while True:
        # Record 5 seconds
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
        sd.wait()

        # Send to diarization
        files = {"file": ("audio.wav", audio.tobytes())}
        response = httpx.post(
            "http://localhost:8000/api/v1/process-audio",
            files=files,
            data={"enable_transcription": "true"}
        )

        # Get transcription
        for segment in response.json().get("segments", []):
            message = {
                "speaker": segment["speaker"],
                "content": segment.get("transcription", "")
            }

            # Send to your LLM
            llm_response = await your_llm.chat(message)
            print(f"AI: {llm_response}")
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
- Check threshold: Lower = more strict (try 0.20-0.35 range, default 0.30)
- Re-enroll with better quality audio

**"Audio file not found" errors**
- Old uploads: Run `python scripts/migrate_temp_audio.py`
- New uploads: Should auto-save to `data/recordings/`
- Verify `data/` directory is accessible

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
- Check audio files exist: `ls data/recordings/`
- Verify API endpoint returns audio: `/api/v1/conversations/segments/{id}/audio`
- Check browser console for errors
- Try different browser (tested: Chrome, Firefox, Safari)

**Live recording not working**
- Browser permission: Allow microphone access
- Standalone: Install PortAudio: `sudo apt-get install portaudio19-dev`
- Check browser microphone settings
- Try different browser

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

### Dependency Licenses

All major dependencies use permissive open-source licenses compatible with MIT:

- **pyannote.audio** (4.0.1): MIT License
  - Models require HuggingFace token and terms acceptance
  - Models themselves remain open-source and MIT licensed
- **faster-whisper** (1.2.1): MIT License (SYSTRAN)
- **FastAPI** (0.115.5): MIT License
- **Next.js** (15.x): MIT License
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

Thank you to these projects and their contributors for making this application possible.

## Planned Features

The following features are planned for future releases:

### Automatic Conversation Summarization and Titling
- AI-powered conversation summarization when recording finishes
- Automatic title generation based on conversation content
- Triggers when current conversation ends and new one begins
- Replaces generic "Conversation 15" with meaningful titles like "Discussion about project deadline with Nick"
- Helps with conversation discovery and context retrieval

### Vector Database Search for Transcriptions
- Store transcription text in a vector database for semantic search
- Query conversations by topic or content, not just speaker
- Each vector entry references conversation ID for easy retrieval
- Enables long-term memory and contextual conversation lookup
- Use cases:
  - "What did we discuss about the budget last month?"
  - "Find conversations where we talked about product features"
  - "Show me all discussions related to the new project"

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
