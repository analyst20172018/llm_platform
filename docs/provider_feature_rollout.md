# Hosted tools, caching, and gateway rollout

Implementation date: 2026-09-08. Implements step 5 of the [provider audit](provider_api_audit_2026-09-07.md).

## Adopted features

- Claude search `web_search_20260318` and code execution `code_execution_20260521`. Search exposes bounded uses, domain restrictions, direct/dynamic callers, and response inclusion. Existing complete-block replay and pause/container handling apply to both versions. Contracts: [search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool), [code execution](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#tool-versions).
- OpenAI explicit cache policy and boundaries on registered GPT-5.6+ models. Cache read/write and reasoning usage remain available alongside native usage. The [prompt caching contract](https://developers.openai.com/api/docs/guides/prompt-caching) was searched and fetched through `openaiDeveloperDocs` from `.mcp.json`.
- Wiro Direct LLM Chat as a separate hidden route, with authenticated discovery before generation and the gateway's Bearer key/secret authentication. It supports local functions and discovered JSON modes through the shared Chat adapter. The original Run route remains available. [Gateway contract](https://wiro.ai/docs/#direct-llm-gateway).

Remote MCP integration for Gemini agents and DeepSeek's stateless hosted-search Responses route remain deferred: neither is needed by these three additions. Gemini's existing `agent_config` and environment continuation support remains available. This change does not introduce a generic hosted-tool passthrough or assume gateway models have the same capabilities.

## Usage

```python
from llm_platform.core.llm_handler import APIHandler

handler = APIHandler(system_prompt=long_reusable_instructions)
answer = handler.request(
    "gpt-5.6-luna",
    "Summarize this question.",
    additional_parameters={
        "prompt_cache_key": "shared-reference-v1",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_breakpoints": ["system"],
    },
)
print(answer.usage["cache_read_tokens"], answer.usage["cache_creation_tokens"])
```

`prompt_cache_breakpoints` can also contain the IDs of existing user messages in `handler.the_conversation.messages`. Each target marks that message's final content block. Targets are per-request, validated against the current conversation, and never stored in message content. Explicit targets trigger full history replay; omitting them preserves ordinary continuation except when returning from a system boundary, which requires one full replay to avoid duplicate instructions. Use at most four targets in explicit mode, or three with the default implicit mode. An explicit policy without targets disables new cache writes. Cache hits depend on prefix length, identity, routing, and provider availability; these controls do not guarantee a hit.

```python
answer = APIHandler().request(
    "claude-sonnet-5",
    "Search Python's official documentation for the asyncio entry point.",
    additional_parameters={
        "web_search": True,
        "web_search_options": {
            "max_uses": 1,
            "allowed_domains": ["docs.python.org"],
            "response_inclusion": "full",
        },
    },
)
```

Use `code_execution=True` to enable the newer code tool. Search uses dynamic filtering by default; `allowed_callers: ["direct"]` selects direct searches. `response_inclusion: "excluded"` permits the provider to omit completed nested search results; direct and paused results remain available for replay.

The Wiro opt-in model ID is `qwen/qwen3-8-27b-uncensored`; select it explicitly through `APIHandler`. It accepts text/extracted documents, `max_tokens`, local `functions`, `tool_choice`, and `structured_output`. The live contract must confirm the requested tools/JSON mode. Its price and context metadata are intentionally unspecified until authenticated discovery validates them for the deployment.

## Validation and deployment gate

Offline validation on 2026-09-08: the full suite passed (265 tests). After the final cache-policy transition fix, all 95 affected modernization, persistence, structured-output, and multi-agent tests passed again. The three baseline OrcaRouter/Wiro failures were stale catalog/default/timeout expectations; their tests now reflect the existing runtime configuration. Existing media deprecations and SDK/Pydantic parsed-response serialization warnings remain; the replay/persistence assertions pass. `git diff --check` passed.

Use the same Python/platform and dependency snapshot as the offline baseline:

```powershell
python -m pip install -r requirements-test.txt -c constraints-tested.txt
python -m pytest -q
```

`constraints-tested.txt` records the tested runtime and test dependency closure on Windows CPython 3.12. Regenerate and rerun tests when changing SDK versions or deployment platform. The current dependency set also passes `pip install --dry-run -r requirements-test.txt -c constraints-tested.txt`.

Before deploying the affected feature, run its explicit live scenario in the deployment account. These calls are billable. Set the normal `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `WIRO_API_KEY` (and optional `WIRO_API_SECRET`) in that environment. Run only the scenarios relevant to the deployment:

```powershell
python scripts/provider_smoke.py --run-live openai-cache --report smoke-reports/openai-cache.json
python scripts/provider_smoke.py --run-live claude-search --report smoke-reports/claude-search.json
python scripts/provider_smoke.py --run-live claude-code --report smoke-reports/claude-code.json
python scripts/provider_smoke.py --run-live wiro-gateway --report smoke-reports/wiro-gateway.json
```

Default targets are `gpt-5.6-luna`, `claude-sonnet-5`, and the hidden Wiro route. `--model` permits an explicit registered model in the same adapter; run again on the actual deployment model if different. The smoke runner requires the selected SDK to match the tested snapshot. Reports include the Python/SDK versions and a SHA-256 fingerprint of the constraints file. Keep passing reports with the deployment record; any nonzero exit blocks deployment of that feature. This is an explicit operational gate, not an automatic CI deployment workflow.

| Scenario | Bounded work | Required evidence |
| --- | --- | --- |
| OpenAI cache | Two requests, synthetic prefix above the cache threshold, 64 output tokens each | Completed text, cache-write count present, positive cache-read count on request two |
| Claude search | One initial request, at most one search per request, 2,048 output tokens per round | Completed text, actual search call and citations, no embedded tool error |
| Claude code | One initial request, 2,048 output tokens per round | Actual code call, completed answer containing 323 for 17 × 19, no embedded tool error |
| Wiro gateway | Authenticated model detail plus one Chat request, 512 output tokens | Eligible Chat/text contract and completed nonempty answer |

Claude may resume `pause_turn` within the existing bounded loop. Every scenario disables SDK retries and sets transport/orchestration timeouts to 60 seconds. These are not preemptive total wall-clock cancellation: a blocking SDK operation can finish after the orchestration deadline. No customer inputs, local function tools, attachments, or remote MCP endpoints are used. Reports exclude generated text, credentials, and provider error bodies. A cache miss remains a failed check after two requests; investigate it instead of generating an unbounded series of retries.

Live acceptance has **not** been established in this workspace: the required OpenAI, Anthropic, and Wiro credentials were absent. Offline transport success and missing-credential failure checks are not live passes. Deployment must wait for successful reports from the target account.
