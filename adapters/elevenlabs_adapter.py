import asyncio
import base64
import io
import json
import os
import urllib.parse
from typing import Any, BinaryIO, Dict, Tuple

import aiohttp  # pip install aiohttp
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment  # pip install pydub (requires ffmpeg/libav)

STREAM_FORMATS: Dict[str, Tuple[int, int]] = {
    "pcm_8000": (8000, 2),
    "pcm_16000": (16000, 2),
    "pcm_22050": (22050, 2),
    "pcm_24000": (24000, 2),
    "pcm_44100": (44100, 2),
    "pcm_48000": (48000, 2),
    "ulaw_8000": (8000, 1),
}
REALTIME_WS = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
STREAM_CHUNK_SECONDS = 0.5  # stays inside the 0.1–1 s guideline


class ElenenlabsAdapter:
    def __init__(self):
        self.api_key = os.getenv("ELEVEN_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)

    def voice_to_text(
        self,
        audio_file: BinaryIO,
        audio_format: str,
        language: str = "eng",
        diarized: bool = True,
    ) -> Dict[str, Any]:
        stream, stream_format = self._prepare_stream(audio_file, audio_format)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._stream_voice_to_text(stream, stream_format, language, diarized)
            )
        finally:
            loop.close()

    def _prepare_stream(
        self, audio_file: BinaryIO, declared_format: str | None
    ) -> Tuple[BinaryIO, str]:
        fmt = self._infer_format(audio_file, declared_format)
        if fmt in STREAM_FORMATS:
            stream = self._ensure_seekable(audio_file)
            stream.seek(0)
            return stream, fmt

        pcm_bytes = self._transcode_to_pcm(self._ensure_seekable(audio_file), fmt)
        return io.BytesIO(pcm_bytes), "pcm_16000"

    def _infer_format(self, audio_file: BinaryIO, declared_format: str | None) -> str:
        fmt = (declared_format or "").strip().lower()
        if fmt:
            return fmt
        name = getattr(audio_file, "name", "")
        if name:
            return os.path.splitext(name)[1].lstrip(".").lower()
        return ""

    def _ensure_seekable(self, audio_file: BinaryIO) -> BinaryIO:
        if audio_file.seekable():
            return audio_file
        data = audio_file.read()
        if not data:
            raise ValueError("Audio stream is empty.")
        return io.BytesIO(data)

    def _transcode_to_pcm(
        self, audio_stream: BinaryIO, input_format: str | None
    ) -> bytes:
        audio_stream.seek(0)
        data = audio_stream.read()
        if not data:
            raise ValueError("Audio stream is empty.")
        buffer = io.BytesIO(data)

        fmt = input_format or None
        try:
            segment = AudioSegment.from_file(buffer, format=fmt)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Unable to decode audio format '{fmt or 'auto'}' via pydub/ffmpeg."
            ) from exc

        segment = (
            segment.set_channels(1)
            .set_frame_rate(16000)
            .set_sample_width(2)
        )

        output = io.BytesIO()
        segment.export(output, format="s16le")  # raw 16-bit PCM
        return output.getvalue()

    async def _stream_voice_to_text(
        self,
        audio_stream: BinaryIO,
        stream_format: str,
        language: str,
        diarized: bool,
    ) -> Dict[str, Any]:
        sample_rate, bytes_per_sample = STREAM_FORMATS[stream_format]
        params = {
            "model_id": "scribe_v2_realtime",
            "language_code": language,
            "audio_format": stream_format,
            "commit_strategy": "manual",
            "include_timestamps": "true" if diarized else "false",
        }

        audio_stream.seek(0)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"{REALTIME_WS}?{urllib.parse.urlencode(params)}",
                headers={"xi-api-key": self.api_key},
                heartbeat=30,
            ) as ws:
                await self._pump_audio(ws, audio_stream, sample_rate, bytes_per_sample)
                transcript, words = await self._consume_transcripts(ws, diarized)

        return {
            "text": transcript,
            "words": words if diarized else None,
            "language": language,
            "model": "scribe_v2_realtime",
        }

    async def _pump_audio(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        audio_stream: BinaryIO,
        sample_rate: int,
        bytes_per_sample: int,
    ) -> None:
        chunk_bytes = int(sample_rate * STREAM_CHUNK_SECONDS) * bytes_per_sample
        chunk = audio_stream.read(chunk_bytes)
        if not chunk:
            raise ValueError("Audio stream is empty.")

        while chunk:
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                "sample_rate": sample_rate,
            }
            next_chunk = audio_stream.read(chunk_bytes)
            if not next_chunk:
                message["commit"] = True
            await ws.send_str(json.dumps(message))
            chunk = next_chunk
            await asyncio.sleep(0)

    async def _consume_transcripts(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        expect_timestamps: bool,
    ) -> Tuple[str, list]:
        transcripts: list[str] = []
        words: list[Dict[str, Any]] = []
        committed = False
        timestamps_ready = not expect_timestamps

        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = json.loads(msg.data)
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"Realtime STT websocket error: {msg.data}")
            else:
                continue

            message_type = payload.get("message_type")
            if message_type == "committed_transcript":
                transcripts.append(payload.get("text", ""))
                committed = True
            elif (
                message_type == "committed_transcript_with_timestamps"
                and expect_timestamps
            ):
                words.extend(payload.get("words", []))
                timestamps_ready = True
            elif message_type in {
                "auth_error",
                "quota_exceeded",
                "transcriber_error",
                "input_error",
                "error",
            }:
                raise RuntimeError(f"Realtime STT error: {payload}")

            if committed and timestamps_ready:
                break

        text = " ".join(t.strip() for t in transcripts if t).strip()
        return text, words