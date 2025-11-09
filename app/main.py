import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import gradio as gr
import os

from .database import init_db
from .api import router
from .conversation_api import router as conversation_router
from .mcp_api import router as mcp_router
from .gradio_ui import create_gradio_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and directories on startup"""
    print("Initializing database...")
    init_db()
    print("Database initialized!")

    # Create necessary directories (use relative paths for local dev, absolute for Docker)
    data_path = os.getenv("DATA_PATH", "./data")
    volumes_path = os.getenv("VOLUMES_PATH", "./volumes")

    os.makedirs(f"{data_path}/recordings", exist_ok=True)
    os.makedirs(f"{data_path}/temp", exist_ok=True)
    os.makedirs(volumes_path, exist_ok=True)

    yield
    # Cleanup on shutdown (if needed)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Speaker Diarization API",
    description="Speaker diarization and recognition service with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(router, prefix="/api/v1", tags=["Speaker Diarization"])
app.include_router(conversation_router, prefix="/api/v1")
app.include_router(mcp_router)  # MCP at /mcp


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Speaker Diarization API",
        "docs": "/docs",
        "gradio_ui": "/gradio",
        "api": "/api/v1",
        "mcp": "/mcp (AI agent interface - JSON-RPC)"
    }


# Create and mount Gradio app
print("Initializing Gradio interface...")
gradio_app = create_gradio_app()

# Mount with allowed_paths so Gradio can serve audio files
data_path = os.getenv("DATA_PATH", "./data")
app = gr.mount_gradio_app(
    app,
    gradio_app,
    path="/gradio",
    allowed_paths=[data_path, "./data"]  # Allow serving from data directory
)
print(f"Gradio interface mounted at /gradio with file access to {data_path}")


if __name__ == "__main__":
    # Start FastAPI server with Gradio mounted
    print("Starting server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
