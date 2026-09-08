**Provider API compatibility audit — 7 September 2026**

The application already uses several current APIs and recent model IDs. The main problems are incomplete conversation-state preservation, schema conversions inherited from older APIs, and capabilities that the adapters or model registry do not expose. A wholesale rewrite or replacement of all model IDs is not warranted.

Reviewed repository revision: `ab13d7e`, with a clean working tree before this audit. Scope: all eleven registered adapters, the 23 configured models, parameter normalization, conversation persistence, tool schemas, and relevant multimodal/response handling. OrcaRouter has an adapter but currently has no model entry.

The review used the OpenAI and Gemini documentation MCP servers configured in `.mcp.json`, current official provider websites, installed SDK source/signatures, the existing test suite, and offline reproductions with synthetic responses. Some search results and the Gemini SDK README still showed older documentation; current feature guides and API contracts were used to resolve those differences. No paid generation requests were made. Account-specific model access and actual production responses remain unverified. Runtime code and configuration were not changed.

**Fix first: confirmed failures and data loss**

**1. P1 — Continuation IDs are shared across providers, and unsent messages can disappear.**

Locations: [Conversation.last_assistant_id](<G:/My Drive/python/packages/llm_platform/llm_platform/services/conversation.py:177>), [OpenAI continuation](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/openai_adapter.py:249>), [Gemini continuation](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/google_adapter.py:942>); the Gemini async path repeats the problem.

`last_assistant_id` returns the most recent nonempty assistant ID, regardless of its origin. OpenAI consumes it as `previous_response_id`; Gemini consumes it as `previous_interaction_id`. Switching from Gemini, Grok, or Wiro to OpenAI can therefore send a foreign ID. If intervening assistants have no ID, the lookup can instead reuse an older response and omit intervening exchanges. Both adapters send only the latest user message, so multiple locally appended or previously unsuccessful user messages are not all transmitted.

Offline reproduction: a conversation ending with an assistant ID `google-interaction-id` and two new user messages generated an OpenAI payload containing that ID and only the second new message. The provider APIs define these IDs as references to their own stored responses/interactions. [OpenAI Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create), [Gemini state management](https://ai.google.dev/gemini-api/docs/interactions-overview).

Recommended fix: retain provider identity and a local history checkpoint with each continuation reference. Reuse it only when the stored prefix matches the local conversation, and send every message after that checkpoint. Otherwise rebuild compatible history, including a fallback for expired/deleted state.

**2. P1 — Saving and restoring a conversation loses response IDs and tool-result pairing.**

Locations: [serialization](<G:/My Drive/python/packages/llm_platform/llm_platform/services/conversation.py:192>) and [deserialization](<G:/My Drive/python/packages/llm_platform/llm_platform/services/conversation.py:258>).

Persistence omits `Message.id`, `FunctionCall.call_id`, `FunctionResponse.call_id`, and `additional_responses`. These are operational data, not merely display metadata. OpenAI response item IDs and function-call IDs are distinct, and several adapters construct function results with only `call_id` populated.

Offline round-trip: `resp_123` became `None`, a call's `call_123` became its item ID `fc_123`, and the matching result's call ID became `None`. Restored conversations cannot reliably resume provider state or pair tool outputs. Google fallback history also omits the original thought steps needed for a faithful function-calling replay. [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling).

Recommended fix: version the persistence format; preserve IDs, citations, provider identity, and opaque provider continuation data. Round-trip a conversation containing multiple tool calls and results, then validate the next provider payload. Keep model-specific reasoning signatures out of unrelated providers' payloads.

**3. P1 — OpenAI plain-callable tools can emit invalid strict schemas.**

Locations: [OpenAI callable conversion](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/openai_adapter.py:822>) and [shared schema generation](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/adapter_base.py:177>).

For `def weather(city: str, unit: str = "C")`, the converter emits `strict: true`, declares both properties, but marks only `city` required. OpenAI requires every property in a strict tool schema to be required, with nullable types used to express optional values. Nested object constraints and array item schemas also need proper generation; the current type lookup handles only a few bare Python types. This request can be rejected before the tool is used. [Official strict-mode requirements](https://developers.openai.com/api/docs/guides/function-calling#strict-mode).

Recommended fix: generate schemas from resolved Python annotations/Pydantic and apply a provider-appropriate strict conversion. Preserve Python default semantics when interpreting nullable values. Alternatively, explicitly use non-strict tools where that is the intended contract. Do not attach `strict: true` to an incompletely normalized schema.

**4. P1 — Gemini's old schema simplifier crashes on recursive schemas and changes nullable fields.**

Locations: [resolve_schema_for_google](<G:/My Drive/python/packages/llm_platform/llm_platform/tools/base.py:95>) and [structured-output conversion](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/google_adapter.py:299>).

The resolver eagerly expands `$ref` and removes the null branch of `anyOf`. A Pydantic `Employee` containing `reports: list[Employee]` reproduced `RecursionError`. A required `value: str | None` became a required non-null string. The current Gemini structured-output guide explicitly demonstrates recursive references and unions, including direct Pydantic schemas. These transformations are now both unnecessary for those supported features and destructive. [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output).

There is a separate shared bug in [clean_schema](<G:/My Drive/python/packages/llm_platform/llm_platform/tools/base.py:75>): it deletes `required` from nested object properties. A nested object's required `name` field became optional in the emitted schema. This affects other providers using the same helper.

Recommended fix: preserve schema semantics and distinguish each endpoint's actual supported subset. Keep supported references and nullability intact; never recursively expand a self-reference without cycle handling. The `extra_body` workaround also describes SDK 1.73.x, while the installed SDK is 2.22.0 and current documentation uses the named `response_format` parameter. Retire that workaround after a serialization check; its age alone does not establish a current failure.

**5. P1 — Antigravity starts a new sandbox on every user turn.**

Locations: [sync agent request](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/google_adapter.py:782>) and the corresponding async path.

The initial call within every `request_llm` uses `environment="remote"` and does not chain to the previous interaction. Only custom-tool follow-ups inside that one request reuse `environment_id`. The returned platform message does not retain the environment ID. A later request such as “edit the file you just created” therefore starts in a fresh sandbox. An offline two-turn check confirmed that the second request again sent `environment="remote"` with no `previous_interaction_id`.

Google documents `"remote"` as provisioning a fresh environment, and an environment ID as preserving existing files/state. [Antigravity environments and continuation](https://ai.google.dev/gemini-api/docs/antigravity-agent#environments).

Recommended fix: persist and reuse both interaction and environment IDs across related user turns, with an explicit new-environment operation. Expose the agent's supported token budget/configuration rather than discarding `agent_config`.

**6. P1 — Kimi's `tool_choice="required"` cannot naturally reach a final answer.**

Locations: [sync loop](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/kimi_adapter.py:385>) and [async loop](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/kimi_adapter.py:431>).

The adapter builds request parameters once and leaves `required` set on every follow-up. It also loops until a response contains no tool calls. Those conditions conflict: a model honoring the requested policy must keep calling tools. A synthetic provider honoring `required` ran 40 calls and hit the guard. Kimi documents requiring the initial call and subsequently returning the complete assistant/tool exchange for an answer. [Kimi K3 tool calling](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart).

Recommended fix: apply the forced choice to the initial request, then use `auto`, or expose an explicit per-round policy. If callers intentionally require tools forever, return control to them instead of promising a final-answer loop.

**7. P1 — Anthropic server-tool pauses are mistaken for completed answers.**

Locations: [response parser](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/anthropic_adapter.py:321>), [tool-loop termination](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/anthropic_adapter.py:399>), and equivalent streaming/async paths.

Only `stop_reason == "tool_use"` continues the loop. A server-side search/code operation returning `pause_turn` is returned to the caller as if complete. An offline response containing partial text, a server tool block, and `pause_turn` resulted in exactly one request. Moreover, the parsers preserve only thinking, text, and local tool calls, losing server-tool blocks and redacted thinking needed for faithful continuation.

Anthropic describes `pause_turn` as a paused long-running turn that can be resumed by returning its response content. [Code execution lifecycle](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#pause_turn-stop-reason).

Recommended fix: preserve the complete provider response blocks and relevant container state, handle pause continuation explicitly, and distinguish completed, truncated, refused, and failed results. Bound continuation by both rounds and elapsed time.

**Other concrete problems**

**8. P2 — Claude custom functions silently disable code execution.** [Tool-enabled request construction](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/anthropic_adapter.py:382>) adds custom functions and web search but never adds code execution; the streaming and async tool paths repeat the omission. The simple request paths do add it. An offline call with a custom function plus `code_execution=True` contained only the custom tool. Build hosted tools once and reuse that builder in every path.

**9. P2 — Gemini silently drops plain Python function tools.** [GoogleAdapter._build_tools](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/google_adapter.py:238>) skips every callable that is not a `BaseTool`, although `APIHandler.request` promises both forms. `_build_tools([weather], {})` returned `[]`. Convert ordinary callables to function declarations or reject them explicitly before generation.

**10. P2 — Gemini's PDF limits are wrong.** [PDF conversion](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/google_adapter.py:109>) permits up to 3,599 pages if under 20 MB, but the documented document limit is 1,000 pages/50 MB. Thus a small 1,200-page PDF is sent despite exceeding the page limit, while a larger valid PDF can lose its visual content through text fallback. Account for encoded HTTP request limits separately and use file upload where appropriate. [Gemini document limits](https://ai.google.dev/gemini-api/docs/document-processing#technical-details).

**11. P2 — OpenRouter reasoning is discarded.** [Shared response parser](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/openai_compatible_adapter.py:158>) reads only `reasoning_content`. OpenRouter documents `reasoning` and structured `reasoning_details`; a response containing those fields produced no `ThinkingResponse` in the offline check. Its request-side `extra_body.reasoning` mapping is already correct. Add provider-aware parsing and preserve opaque reasoning details for replay; do not reduce encrypted blocks to display text. [OpenRouter reasoning contract](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

**12. P2 — The registry blocks capabilities already available in the adapters or providers.** [Parameter filtering](<G:/My Drive/python/packages/llm_platform/llm_platform/core/parameter_normalizer.py:121>) removes fields absent from the model schema. Offline checks confirmed that it drops Deep Research `agent_config`, GLM-5.3 `structured_output` and `reasoning_effort`, Mistral `structured_output`, and OpenAI `store`/`service_tier`. The logging helps diagnosis but does not make the requested behavior happen. Add intentional, tested model capabilities; report unsupported behavior clearly instead of making an apparently successful request with different semantics. Relevant provider features are listed below. Also, the normalizer does not actually enforce declared enum values or numeric limits, so the YAML is not request validation.

**13. P2 — Mistral chunked responses break the next request.** [Response construction](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/mistral_adapter.py:146>) copies `message.content` directly into the domain message. [History conversion](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/mistral_adapter.py:48>) then wraps it as the `text` value of one text chunk. When the response is a list of chunks, the next request contains a list where a string is required. This reproduced a `ValidationError` against the installed Mistral `AssistantMessage` model. Split answer text, thinking, and references while preserving any provider data required for replay. Mistral explicitly permits string or chunk-list response content. [Mistral chat completions](https://docs.mistral.ai/studio/conversations/chat-completion), [reasoning chunks](https://docs.mistral.ai/studio/conversations/reasoning).

**14. P2 — Grok's configured context window is twice the documented value.** [The `grok-4.6` entry](<G:/My Drive/python/packages/llm_platform/llm_platform/models_config.yaml:495>) advertises 1,000,000 tokens; its current official model page says 500,000. Software relying on the catalog can admit oversized requests. Correct the metadata and separate context length, maximum output, and long-context pricing tiers. This is a documented mismatch, not a live oversized-request test. [Grok 4.6 model details](https://docs.x.ai/developers/models/grok-4.6).

**Provider-by-provider modernization opportunities**

These are feature gaps or improvements, not evidence that every existing request is invalid.

| Provider | Current implementation and worthwhile next step |
| --- | --- |
| OpenAI | Responses, structured-output parsing, hosted search/code execution, native async, and beta multi-agent routing are already implemented. Prioritize state/schema fixes. The app has no public support for explicit prompt-cache policy/breakpoints and discards cache read/write usage. Current GPT-5.6+ caching exposes these controls and distinct write/read charges. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching#how-caching-works). |
| Anthropic | Adaptive thinking, `output_config`, automatic caching, and async requests are already present. Search remains on `web_search_20250305`; `web_search_20260318` adds dynamic filtering and response-inclusion controls. Code execution remains on `20250825`; `20260521` is current and includes later programmatic-execution behavior. The older versions are still documented as supported, so upgrade after repairing response-block/lifecycle handling. [Search versions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool), [code versions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#tool-versions). |
| Google Gemini | Interactions is already the correct current foundation. Add current schema support and reliable history preservation. Deep Research can support collaborative planning and remote MCP, but its config is filtered and every research call currently starts from only the newest prompt. Antigravity also supports remote MCP and environment reuse. Deep Research still does not support local custom function tools, so that restriction should remain. [Deep Research](https://ai.google.dev/gemini-api/docs/deep-research), [Antigravity](https://ai.google.dev/gemini-api/docs/antigravity-agent). |
| xAI/Grok | The native SDK path already supports function calling, structured output, search, code execution, and effort control. Do not equate `xai_sdk.client.chat.create` with the legacy OpenAI-compatible REST endpoint merely because both use the word “chat.” The adapter still flattens responses and does not preserve citations or encrypted reasoning context. Current APIs expose encrypted reasoning for continuation. [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning). |
| DeepSeek | V4 thinking controls and the vision-model entry are current. The adapter explicitly rejects local tools and drops the platform structured-output switch, although all three configured V4 models support tool calls and JSON output. Implement these first. A Responses-compatible endpoint now also supports hosted web search, but is stateless: copying the OpenAI adapter's continuation logic would be incorrect. [Tool calls](https://api-docs.deepseek.com/guides/tool_calls/), [JSON output](https://api-docs.deepseek.com/guides/json_mode/), [Responses compatibility](https://api-docs.deepseek.com/guides/responses_api/). |
| OpenRouter | The Muse Spark 1.3 model ID and base input/output prices match the current catalog. Repair reasoning parsing and add supported function calling/structured output. Discover capabilities per model/provider route rather than treating all routed models as identical. [Muse Spark 1.3](https://openrouter.ai/meta/muse-spark-1.3), [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling). |
| OrcaRouter | The OpenAI-compatible base URL is current. However, no model routes to this adapter in the present YAML, and the old `obsidian/Qwen3.8-27B` test fails locally before any API request. Reconcile whether this provider is intentionally disabled; if retained, register an available catalog ID and add tools. Both Chat Completions and Responses are documented, as is tool calling. [HTTP compatibility](https://docs.orcarouter.ai/native-formats/openai-compat), [tools](https://docs.orcarouter.ai/advanced/tool-calling). |
| WiroAI | Run submission, Task/Detail polling, and terminal `pexit` checking match the documented asynchronous flow. Wiro now also offers a Direct LLM Gateway at `https://llm.wiro.ai/v1` with Chat Completions/Responses and model-specific contracts. Consider a separate gateway path after checking the chosen model's eligibility. It has different authentication from Run; changing only the base URL would be wrong. Run itself remains supported. [Wiro Run and Direct LLM Gateway](https://wiro.ai/docs/#direct-llm-gateway). |
| Mistral | The current `mistralai.client.Mistral` import and `chat.complete` surface are valid. Add a real `structured_output` mapping to `response_format`/`chat.parse`; it is currently filtered in YAML and discarded in adapter code. Preserve chunked responses. Reasoning-capable Mistral models now expose adjustable effort and thinking chunks; adding those models requires more than a catalog row. [Structured output](https://docs.mistral.ai/studio/conversations/structured-output/custom), [reasoning](https://docs.mistral.ai/studio/conversations/reasoning). |
| Z.AI | Preserved thinking and JSON-object mode are implemented appropriately. GLM-5.3's registry omits effort control and structured output despite provider support. GLM-5.3-Flash is incorrectly described as text-only: the provider now documents image/video/file input. The shared adapter already serializes images, but the metadata hides them and video needs implementation. [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3), [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash). Keep the distinction between JSON-object mode plus local validation and strict server-enforced schemas. [JSON output contract](https://docs.z.ai/guides/capabilities/struct-output). |
| Moonshot/Kimi | K3's endpoint, fixed-sampling omissions, max-completion-token mapping, and multimodal/tool support broadly match current guidance. Fix the forced-tool loop. The registry offers only `max` effort, whereas K3 now supports `low`, `high`, and `max`; direct parameter forwarding already makes the lower settings feasible. Expose them for latency/cost control. K3 permits larger completion limits than the configured default; the existing default is valid and need not be raised automatically. [Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart). |

**Response fidelity and operational gaps**

- Citations are not preserved consistently. OpenAI keeps container-file citations but drops URL citation metadata; Anthropic ignores citation events and text-block citations; Grok's message builder omits its citation data. Preserve URL/title/span/source metadata in a common result structure, with provider-specific data retained separately. This matters when the app displays search-backed answers or needs source attribution. [OpenAI parser](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/openai_adapter.py:377>), [Anthropic parser](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/anthropic_adapter.py:321>), [Grok result builder](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/grok_adapter.py:199>).
- Most result paths omit finish/status metadata. An OpenAI failed/incomplete background result can become an empty or partial normal message; Google standard/Antigravity parsing can turn an error into answer text. Handle statuses before interpreting text. OpenAI/Google polling has no overall deadline; Wiro does have one. [OpenAI polling](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/openai_adapter.py:466>), [Google parsing](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/google_adapter.py:486>).
- Usage is too lossy for accurate cost reporting. The shared helper retains only prompt/completion totals, discarding cache/reasoning and provider cost breakdowns. `usage_total.costs` returns zero unless a message supplies costs; Wiro does so, most adapters do not. DeepSeek's YAML prices currently match peak uncached rates, but omit cache-hit and off-peak rates. Do not label those rates simply “wrong” or treat a missing cost as a free request. [Shared usage](<G:/My Drive/python/packages/llm_platform/llm_platform/adapters/adapter_base.py:149>), [DeepSeek current pricing](https://api-docs.deepseek.com/quick_start/pricing/).
- The facade returns completed messages, not a unified stream of text/tool/status events. Anthropic streaming currently accumulates internally. A public event stream would improve long-running research/tool UX; it is an optional interface expansion.
- Requirements lack tested minimums for most SDK features and a reproducible dependency set. In this environment, provider SDKs are recent: OpenAI 3.8.0, Anthropic 1.3.0, google-genai 2.22.0, Mistral 2.9.4, xai-sdk 1.19.0, zai-sdk 0.2.3. The issue is deployment reproducibility rather than proof that these installed SDKs are obsolete. Establish a tested dependency baseline; the current Gemini guide requires google-genai 2.3.0+ for Interactions. [Gemini SDK guidance](https://ai.google.dev/gemini-api/docs/interactions-overview#sdks).

**Validation results and documentation drift**

`python -m pytest -q` completed in 28.79 seconds: **39 passed, 4 failed**. Existing tests predominantly replace SDK clients with mocks, so their success does not validate provider schemas or live acceptance.

| Failing test | Observed difference | Interpretation |
| --- | --- | --- |
| `test_orcarouter_model_is_registered_with_lazy_adapter` | `obsidian/Qwen3.8-27B` is absent from YAML | Adapter/test/catalog inconsistency; does not establish upstream model retirement. |
| `test_wiro_ai_model_is_registered_with_lazy_adapter` | Expected `min_tokens=0`; YAML has `500` | Default changed; decide which behavior is intended. |
| `test_wiro_ai_request_runs_polls_and_parses_structured_llm_output` | Expected HTTP timeout `30`; implementation uses `600` | Test/configuration drift; not a protocol failure. |
| `test_glm_flash_reasoning_defaults_are_normalized_from_model_config` | Expected `max`; YAML defaults to `high` | Test/default drift. `high` is a supported effort, so this does not justify changing it merely to satisfy the test. |

Additional offline checks reproduced the foreign continuation ID, dropped unsent turn, persistence ID loss, invalid strict schema, recursive-schema crash, changed nullability, nested `required` removal, fresh Antigravity sandbox, Kimi 40-round forced-tool loop, Claude code-tool omission/pause termination, Gemini callable omission, OpenRouter reasoning loss, filtered parameters, and a Mistral SDK validation error on chunked history. These checks used synthetic inputs and mocked clients, not a paid provider call.

The architecture document also needs a factual refresh when fixes are implemented: it says 20 models while YAML contains 23; refers to provider-specific continuation properties that no longer exist; describes an OrcaRouter model that is absent; and lists mutable `Message` list defaults that the current constructor has already fixed. Avoid carrying those stale observations into the implementation backlog.

**Suggested implementation order**

1. Repair provider-scoped continuation and lossless persistence, including Antigravity environments and exact provider reasoning/tool replay.
2. Correct schema generation and preserve schema meaning; cover optional, nested, recursive, nullable, and typed-container inputs.
3. Repair tool termination and hosted-tool combinations; expose statuses and preserve citations/results.
4. Align model capabilities and parameters: GLM visual inputs/effort, Kimi effort choices, Grok context, Gemini PDF limits, Mistral JSON output, and DeepSeek/OpenRouter tools.
5. Adopt newer hosted-tool/cache/gateway features where useful, with tested SDK versions and small explicit live smoke checks before deployment.

The first three steps address correctness. The later steps unlock provider improvements without confusing optional feature adoption with a necessary API migration.
