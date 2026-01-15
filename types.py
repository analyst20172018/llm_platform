from typing import Any, List, TypedDict


class ReasoningParameters(TypedDict, total=False):
    effort: str  # e.g. "none", "low", "medium", "high"
    summary: str  # OpenAI responses API summary mode (e.g. "auto")


class TextParameters(TypedDict, total=False):
    verbosity: str  # e.g. "low", "medium", "high"


class AdditionalParameters(TypedDict, total=False):
    response_modalities: List[str]  # e.g. ["text", "image", "audio"]
    web_search: bool  # allow integrated web search when supported
    url_context: bool  # enable URL context tool (Gemini)
    code_execution: bool  # allow code execution tool when supported
    citations_enabled: bool  # request citations for supported providers
    structured_output: Any  # pydantic model class for schema parsing
    aspect_ratio: str  # image output aspect ratio (e.g. "1:1")
    resolution: str  # image output resolution (e.g. "1K", "2K", "4K")
    temperature: float  # sampling temperature
    max_tokens: int  # hard cap on response tokens
    reasoning: ReasoningParameters  # reasoning/effort tuning
    text: TextParameters  # text verbosity tuning
    # Parameters for Elevenlabs STT
    language: str  # STT language code (e.g. "en")
    diarized: bool  # STT diarization on/off
    tag_audio_events: bool  # STT audio event tagging
    num_speakers: int  # STT speaker count hint
    # Parameters for OpenAI image models
    size: str  # image size (OpenAI image models)
    background: str  # image background (OpenAI image models)
    quality: str  # image quality (e.g. "low", "medium", "high")
    output_format: str  # image output format (e.g. "png", "jpg")
