"""Voice transcription providers."""

import asyncio
import os
import time
from pathlib import Path

import httpx
from loguru import logger


class WhisperCppTranscriptionProvider:
    """Local transcription using whisper.cpp with ROCm acceleration."""

    def __init__(
        self,
        model_path: str = "/app/models/ggml-medium.bin",
        binary_path: str = "/usr/local/bin/whisper-cpp",
    ):
        self.model_path = Path(model_path)
        self.binary_path = Path(binary_path)

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using whisper.cpp.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        if not self.binary_path.exists():
            logger.warning("whisper.cpp binary not found: {}", self.binary_path)
            return ""

        if not self.model_path.exists():
            logger.warning("whisper.cpp model not found: {}", self.model_path)
            return ""

        # Convert to WAV if needed (Telegram sends OGG)
        wav_path = await self._ensure_wav(path)
        if not wav_path:
            logger.error("Failed to convert audio to WAV")
            return ""

        try:
            cmd = [
                str(self.binary_path),
                "-m", str(self.model_path),
                "-f", str(wav_path),
                "-t", "4",
                "--no-timestamps",
                "-otxt",
            ]

            audio_duration = await self._get_audio_duration(wav_path)
            start_time = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120.0)
            elapsed = time.perf_counter() - start_time

            if process.returncode != 0:
                logger.error("whisper.cpp failed: {}", stderr.decode())
                return ""

            # Read output file
            txt_path = wav_path.with_suffix(wav_path.suffix + ".txt")
            if txt_path.exists():
                text = txt_path.read_text().strip()
                txt_path.unlink()
                logger.debug("whisper.cpp transcribed {:.1f}s audio in {:.2f}s ({:.1f}x realtime)",
                            audio_duration, elapsed, audio_duration / elapsed if elapsed > 0 else 0)
                return text
            logger.debug("whisper.cpp transcribed {:.1f}s audio in {:.2f}s ({:.1f}x realtime)",
                        audio_duration, elapsed, audio_duration / elapsed if elapsed > 0 else 0)
            return stdout.decode().strip()
        except asyncio.TimeoutError:
            logger.error("whisper.cpp transcription timed out")
            return ""
        except Exception as e:
            logger.error("Transcription error: {}", e)
            return ""
        finally:
            if wav_path != path and wav_path.exists():
                wav_path.unlink()

    async def _ensure_wav(self, audio_path: Path) -> Path | None:
        """Convert audio to 16kHz mono WAV for whisper."""
        if audio_path.suffix.lower() == ".wav":
            return audio_path

        wav_path = audio_path.with_suffix(".wav")
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.wait()
        return wav_path if process.returncode == 0 else None

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            return float(stdout.decode().strip())
        except Exception:
            return 0.0


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""
