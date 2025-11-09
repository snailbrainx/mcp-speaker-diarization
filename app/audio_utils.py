"""
Audio utilities for conversion and processing
"""
import os
from pydub import AudioSegment
from typing import Optional


def convert_to_mp3(
    input_path: str,
    output_path: Optional[str] = None,
    bitrate: str = "192k",
    delete_original: bool = False
) -> str:
    """
    Convert audio file to MP3 format

    Args:
        input_path: Path to input audio file (WAV, etc.)
        output_path: Path for output MP3 (default: replaces extension with .mp3)
        bitrate: MP3 bitrate (default: 192k for high quality)
        delete_original: Whether to delete original file after conversion

    Returns:
        Path to the created MP3 file
    """
    # Determine output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}.mp3"

    # Load audio file (supports WAV, FLAC, etc.)
    audio = AudioSegment.from_file(input_path)

    # Export as MP3
    audio.export(
        output_path,
        format="mp3",
        bitrate=bitrate,
        parameters=["-q:a", "2"]  # High quality VBR
    )

    # Delete original if requested
    if delete_original and os.path.exists(input_path):
        os.remove(input_path)
        print(f"Deleted original file: {input_path}")

    print(f"Converted to MP3: {output_path} (bitrate: {bitrate})")
    return output_path


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds


def extract_segment(
    audio_path: str,
    start_time: float,
    end_time: float,
    output_path: Optional[str] = None
) -> str:
    """
    Extract a segment from an audio file

    Args:
        audio_path: Path to source audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path for output file (default: temp file)

    Returns:
        Path to the extracted segment audio file
    """
    import tempfile

    # Load audio
    audio = AudioSegment.from_file(audio_path)

    # Convert seconds to milliseconds
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    # Extract segment
    segment = audio[start_ms:end_ms]

    # Determine output path
    if output_path is None:
        # Create temp file with same extension as input
        ext = os.path.splitext(audio_path)[1] or ".wav"
        fd, output_path = tempfile.mkstemp(suffix=ext, prefix="segment_")
        os.close(fd)

    # Export segment
    segment.export(output_path, format=os.path.splitext(output_path)[1][1:])

    return output_path


def batch_convert_to_mp3(
    input_paths: list[str],
    bitrate: str = "192k",
    delete_originals: bool = False
) -> list[str]:
    """
    Batch convert multiple audio files to MP3

    Args:
        input_paths: List of input file paths
        bitrate: MP3 bitrate
        delete_originals: Whether to delete original files

    Returns:
        List of output MP3 paths
    """
    output_paths = []

    for input_path in input_paths:
        try:
            output_path = convert_to_mp3(
                input_path,
                bitrate=bitrate,
                delete_original=delete_originals
            )
            output_paths.append(output_path)
        except Exception as e:
            print(f"Error converting {input_path}: {e}")

    return output_paths
