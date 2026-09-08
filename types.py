from typing import Any, List, TypedDict


class ReasoningParameters(TypedDict, total=False):
    enabled: bool  # OpenRouter unified reasoning toggle
    effort: str  # e.g. "none", "low", "medium", "high"
    mode: str  # OpenAI execution mode: "standard" or "pro" (GPT-5.6+)
    summary: str  # OpenAI responses API summary mode (e.g. "auto")


class TextParameters(TypedDict, total=False):
    verbosity: str  # e.g. "low", "medium", "high"


class ThinkingParameters(TypedDict, total=False):
    type: str  # provider thinking mode: "enabled" or "disabled"
    clear_thinking: bool  # Z.AI: omit prior reasoning when true


class AdditionalParameters(TypedDict, total=False):
    response_modalities: List[str]  # e.g. ["text", "image", "audio"]
    web_search: bool  # allow integrated web search when supported
    url_context: bool  # enable URL context tool (Gemini)
    code_execution: bool  # allow code execution tool when supported
    citations_enabled: bool  # request citations for supported providers
    structured_output: Any  # pydantic model class for schema parsing
    temperature: float  # sampling temperature
    top_p: float  # nucleus-sampling probability threshold
    top_k: int  # sample from the K most likely tokens
    repetition_penalty: float  # discourage repeated tokens or phrases
    length_penalty: float  # provider-specific response-length preference
    max_tokens: int  # hard cap on response tokens
    min_tokens: int  # provider-specific minimum response-token count
    thinking_enabled: bool  # enable WiroAI thinking output
    reasoning_effort: str  # provider-native top-level effort (Kimi K3: "max"; DeepSeek: "low"/"high"/"max")
    thinking_mode: str  # DeepSeek thinking mode toggle: "enabled" or "disabled"
    clear_thinking: bool  # Z.AI preserved-thinking control
    tool_choice: str  # function selection policy, e.g. "auto", "none", "required"
    reasoning: ReasoningParameters  # reasoning/effort tuning
    text: TextParameters  # text verbosity tuning
    thinking: ThinkingParameters  # DeepSeek thinking mode (mapped from thinking_mode)
    agent_config: dict[str, Any]  # Google managed-agent configuration
    new_environment: bool  # Antigravity: reset remote state before this turn
    agent_count: int # number of parallel agents (Grok Heavy; OpenAI Multi-agent subagents, 0 = off)
