"""Normalize result metadata without changing provider-native replay payloads."""

from .serializers import provider_dump


_STATUSES = {
    "completed": "completed", "stop": "completed", "end_turn": "completed",
    "stop_sequence": "completed", "length": "incomplete", "max_tokens": "incomplete",
    "model_length": "incomplete", "incomplete": "incomplete",
    "refusal": "refused", "content_filter": "refused", "safety": "refused",
    "failed": "failed", "error": "failed", "cancelled": "cancelled",
    "tool_use": "requires_action", "tool_calls": "requires_action",
    "function_call": "requires_action", "requires_action": "requires_action",
    "pause_turn": "paused", "queued": "queued", "in_progress": "in_progress",
    "REASON_STOP": "completed", "REASON_TOOL_CALLS": "requires_action",
    "REASON_MAX_LEN": "incomplete", "REASON_MAX_CONTEXT": "incomplete",
    "REASON_TIME_LIMIT": "incomplete",
}


def result_status(reason=None, *, error=None, refused=False, has_calls=False):
    """Keep unknown reasons unknown; never turn a failed tool response into work."""
    status = _STATUSES.get(reason, "unknown")
    if error:
        return "failed"
    if status in {"failed", "cancelled", "incomplete"}:
        return status
    if refused:
        return "refused"
    if has_calls and (status == "completed" or reason is None):
        return "requires_action"
    return status


def citation_metadata(provider, citation, location=None):
    """Common URL/title/span fields plus the full native source and its location.

    Offsets retain the provider's units and are relative to the original block,
    not to the adapter's concatenated display text.
    """
    source = provider_dump(citation)
    data = {"url": source} if isinstance(source, str) else source
    data = data.get("url_citation") or data
    web = data.get("web_citation") or {}
    return {
        "provider": provider,
        "url": data.get("url") or data.get("uri") or web.get("url"),
        "title": data.get("title") or data.get("document_title"),
        "start_index": data.get("start_index"),
        "end_index": data.get("end_index"),
        "source": source,
        "location": location,
    }


def block_metadata(provider, blocks):
    """Extract citations and hosted records from ordered native response blocks."""
    citations, hosted = [], []
    for index, block in enumerate(blocks or []):
        kind = block.get("type", "")
        is_hosted = (
            kind.endswith(("_call", "_result", "_call_output"))
            and kind not in {"function_call", "function_result"}
        ) or kind in {"server_tool_use", "mcp_tool_use", "mcp_tool_result"}
        if is_hosted:
            hosted.append(block)
        if kind == "reference":
            citations.append(citation_metadata(provider, block, {"block_index": index}))
        content = block.get("content")
        parts = content if isinstance(content, list) else []
        for part_index, part in [(None, block), *enumerate(parts)]:
            if not isinstance(part, dict):
                continue
            for citation in (part.get("annotations") or []) + (part.get("citations") or []):
                citations.append(citation_metadata(provider, citation, {
                    "block_index": index, "content_index": part_index,
                }))
    return {"citations": citations, "hosted_tool_results": hosted}


def chat_metadata(response):
    choice = response.choices[0]
    assistant = choice.message
    reason = getattr(choice, "finish_reason", None)
    refusal = getattr(assistant, "refusal", None)
    return {
        "status": result_status(reason, error=getattr(response, "error", None), refused=bool(refusal),
                                has_calls=bool(getattr(assistant, "tool_calls", None))),
        "finish_reason": reason,
        "error": provider_dump(getattr(response, "error", None)),
    }


def grok_client_calls(response):
    # SDK ToolCall.type distinguishes local functions from already executed tools.
    return [call for call in (getattr(response, "tool_calls", None) or [])
            if getattr(call, "type", None) in (None, 1, "function", "TOOL_CALL_TYPE_CLIENT_SIDE_TOOL")]


def grok_metadata(response, native_messages=None):
    reason = getattr(response, "finish_reason", None)
    citations = [citation_metadata("grok", item)
                 for item in (getattr(response, "citations", None) or [])]
    citations.extend(citation_metadata("grok", item)
                     for item in (getattr(response, "inline_citations", None) or []))
    local_calls = grok_client_calls(response)
    hosted = [provider_dump(call) for call in (getattr(response, "tool_calls", None) or [])
              if call not in local_calls]
    hosted_ids = {call["id"] for call in hosted if call.get("id")}
    hosted.extend(message for message in native_messages or []
                  if message.get("tool_call_id") in hosted_ids)
    return {
        "status": result_status(reason, has_calls=bool(local_calls)),
        "finish_reason": reason, "citations": citations,
        "hosted_tool_results": hosted,
    }


def openai_metadata(response, is_internal=None):
    if response is None:
        return {}
    output = provider_dump(response.output)
    reason = getattr(response, "status", None)
    error = provider_dump(getattr(response, "error", None))
    visible_output = [block for item, block in zip(response.output, output)
                      if is_internal is None or not is_internal(item)]
    refused = any(part.get("type") == "refusal"
                  for item in visible_output for part in (item.get("content") or [])
                  if isinstance(part, dict))
    return {
        "status": result_status(reason, error=error, refused=refused,
                                has_calls=any(b.get("type") == "function_call" for b in output)),
        "finish_reason": reason,
        "error": error,
        "incomplete_details": provider_dump(getattr(response, "incomplete_details", None)),
        **block_metadata("openai", output),
    }


def anthropic_metadata(processor):
    return {
        "status": result_status(processor.stop_reason, error=processor.error),
        "finish_reason": processor.stop_reason,
        "error": processor.error,
        **block_metadata("anthropic", processor.content_blocks),
    }


def google_metadata(interaction):
    steps = provider_dump(getattr(interaction, "steps", []) or [])
    reason = getattr(interaction, "status", None)
    error = provider_dump(getattr(interaction, "error", None))
    return {
        "status": result_status(reason, error=error,
                                has_calls=any(b.get("type") == "function_call" for b in steps)),
        "finish_reason": reason,
        "error": error,
        "incomplete_details": provider_dump(getattr(interaction, "incomplete_details", None)),
        **block_metadata("google", steps),
    }
