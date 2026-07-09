from typing import Any, List, TypedDict


class ReasoningParameters(TypedDict, total=False):
    effort: str  # e.g. "none", "low", "medium", "high"
    mode: str  # OpenAI execution mode: "standard" or "pro" (GPT-5.6+)
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
    temperature: float  # sampling temperature
    max_tokens: int  # hard cap on response tokens
    reasoning: ReasoningParameters  # reasoning/effort tuning
    text: TextParameters  # text verbosity tuning
    agent_count: int # parameter for Grok Multi Agent model
