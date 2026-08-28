"""Whisper transcription via faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import settings
from app.models.schemas import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model %s on %s (%s)…",
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
        )
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper model loaded.")
    return _model


def transcribe_audio(audio_path: Path, video_id: str, title: str | None = None) -> Transcript:
    """Transcribe an audio file and return a Transcript with timestamped segments."""
    model = _get_model()

    logger.info("Transcribing %s…", audio_path.name)
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=None,  # auto-detect
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        segments.append(
            TranscriptSegment(
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                text=seg.text.strip(),
            )
        )

    language = info.language
    logger.info(
        "Transcription complete: %d segments, language=%s", len(segments), language
    )

    return Transcript(
        video_id=video_id,
        title=title,
        language=language,
        segments=segments,
    )
