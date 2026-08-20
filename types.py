from typing import Any, List, TypedDict


class ReasoningParameters(TypedDict, total=False):
    effort: str  # e.g. "none", "low", "medium", "high"
    mode: str  # OpenAI execution mode: "standard" or "pro" (GPT-5.6+)
    summary: str  # OpenAI responses API summary mode (e.g. "auto")


class TextParameters(TypedDict, total=False):
    verbosity: str  # e.g. "low", "medium", "high"


class ThinkingParameters(TypedDict, total=False):
    type: str  # DeepSeek thinking mode: "enabled" or "disabled"


class AdditionalParameters(TypedDict, total=False):
    response_modalities: List[str]  # e.g. ["text", "image", "audio"]
    web_search: bool  # allow integrated web search when supported
    url_context: bool  # enable URL context tool (Gemini)
    code_execution: bool  # allow code execution tool when supported
    citations_enabled: bool  # request citations for supported providers
    structured_output: Any  # pydantic model class for schema parsing
    temperature: float  # sampling temperature
    max_tokens: int  # hard cap on response tokens
    reasoning_effort: str  # provider-native top-level effort (Kimi K3: "max"; DeepSeek: "low"/"high"/"max")
    thinking_mode: str  # DeepSeek thinking mode toggle: "enabled" or "disabled"
    tool_choice: str  # function selection policy, e.g. "auto", "none", "required"
    reasoning: ReasoningParameters  # reasoning/effort tuning
    text: TextParameters  # text verbosity tuning
    thinking: ThinkingParameters  # DeepSeek thinking mode (mapped from thinking_mode)
    agent_count: int # number of parallel agents (Grok Heavy; OpenAI Multi-agent subagents, 0 = off)
