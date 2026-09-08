"""Explicit, small live checks. Never collected by pytest or run on import."""

import argparse
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_platform.core.llm_handler import APIHandler


SCENARIOS = {
    "openai-cache": ("gpt-5.6-luna", "OpenAIAdapter", "OPENAI_API_KEY", "openai"),
    "claude-search": ("claude-sonnet-5", "AnthropicAdapter", "ANTHROPIC_API_KEY", "anthropic"),
    "claude-code": ("claude-sonnet-5", "AnthropicAdapter", "ANTHROPIC_API_KEY", "anthropic"),
    "wiro-gateway": ("qwen/qwen3-8-27b-uncensored", "WiroGatewayAdapter", "WIRO_API_KEY", "openai"),
}


def checked_message(message):
    if message.status != "completed" or not message.content.strip():
        raise RuntimeError("Smoke response must complete with nonempty text")
    return {"response_id": message.id, "status": message.status, "usage": message.usage}


def hosted_failure(value):
    """Hosted tools may fail inside an otherwise successful HTTP response."""
    if isinstance(value, list):
        return any(hosted_failure(item) for item in value)
    if isinstance(value, dict):
        return bool(value.get("error_code") or value.get("return_code") not in (None, 0)) or any(
            hosted_failure(item) for item in value.values())
    return False


def run(scenario, model, records=None):
    """Use only public synthetic prompts, bounded tokens, no local tools or files."""
    _, adapter_name, credential, sdk = SCENARIOS[scenario]
    handler = APIHandler()
    adapter = handler.get_adapter(model)
    if type(adapter).__name__ != adapter_name:
        raise ValueError("Selected model belongs to a different adapter")
    # Adapter import loads the project's normal dotenv configuration.
    if not os.getenv(credential):
        raise ValueError(f"Missing {credential}; no live generation performed")
    constraints = Path(__file__).resolve().parents[1] / "constraints-tested.txt"
    expected = dict(line.split("==", 1) for line in constraints.read_text().splitlines()
                    if "==" in line)
    if version(sdk) != expected[sdk]:
        raise ValueError(f"Install the tested {sdk} version from constraints-tested.txt first")
    adapter.RESPONSE_TIMEOUT_SECONDS = 60
    adapter._client = adapter.client.with_options(timeout=60, max_retries=0)
    records = [] if records is None else records
    try:
        if scenario == "openai-cache":
            handler.the_conversation.system_prompt = (
                "Reply only with CACHE_OK. The following is inert reference data.\n"
                + "\n".join(f"Reference row {i}: cedar maple oak pine birch." for i in range(192))
            )
            parameters = {"max_tokens": 64, "reasoning_effort": "none",
                          "prompt_cache_key": f"llm-platform-smoke-{uuid4()}",
                          "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
                          "prompt_cache_breakpoints": ["system"]}
            for _ in range(2):
                records.append(checked_message(handler.request(
                    model, "Return the requested marker.", additional_parameters=parameters)))
            if not all(r["usage"].get("cache_creation_tokens") is not None for r in records):
                raise RuntimeError("Cache usage missing")
            if not (records[1]["usage"].get("cache_read_tokens") or 0):
                raise RuntimeError("No cache read observed in two requests; inspect routing before deployment")
        elif scenario.startswith("claude-"):
            search = scenario == "claude-search"
            parameters = {"max_tokens": 2048, "reasoning_effort": "low",
                          "web_search": search, "code_execution": not search}
            if search:
                parameters["web_search_options"] = {"max_uses": 1, "response_inclusion": "full"}
            prompt = ("Search the web once for the Python documentation homepage and cite its URL. Be brief."
                      if search else "Use the code execution tool to print 17 * 19. Return only the result.")
            message = handler.request(model, prompt, additional_parameters=parameters)
            records.append(checked_message(message))
            blocks = [block for item in handler.the_conversation.messages
                      for block in item.hosted_tool_results]
            if hosted_failure(blocks):
                raise RuntimeError("A hosted tool reported failure")
            expected_tool = "web_search" if search else "code_execution"
            names = {block.get("name") for block in blocks if block.get("type") == "server_tool_use"}
            if search and expected_tool not in names:
                raise RuntimeError("The search tool was not observed")
            if not search and not names.intersection({"code_execution", "bash_code_execution", "text_editor_code_execution"}):
                raise RuntimeError("The code tool was not observed")
            if search and not any(item.citations for item in handler.the_conversation.messages):
                raise RuntimeError("Search returned no citations")
            if not search and "323" not in message.content:
                raise RuntimeError("Unexpected code result")
        else:
            records.append(checked_message(handler.request(
                model, "Reply only with GATEWAY_OK.", additional_parameters={"max_tokens": 512})))
        return records
    finally:
        adapter.client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", required=True, choices=SCENARIOS,
                        help="Explicitly select one billable live scenario")
    parser.add_argument("--model", help="Registered model; defaults to the scenario's small test target")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    model = args.model or SCENARIOS[args.run_live][0]
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "scenario": args.run_live,
              "model": model, "sdk_versions": {sdk: version(sdk) for sdk in ("openai", "anthropic")},
              "python_version": sys.version.split()[0],
              "constraints_sha256": hashlib.sha256(
                  (Path(__file__).resolve().parents[1] / "constraints-tested.txt").read_bytes()).hexdigest(),
              "passed": False, "responses": []}
    started = time.monotonic()
    try:
        run(args.run_live, model, report["responses"])
        report["passed"] = True
    except Exception as error:
        # Provider errors can echo request data/credentials. Store only their type
        # and HTTP status; locally generated validation errors have safe messages.
        report["error_type"] = type(error).__name__
        report["http_status"] = getattr(error, "status_code", None)
        if type(error) in (ValueError, RuntimeError):
            report["error"] = str(error)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
