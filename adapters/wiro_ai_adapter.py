import hashlib
import hmac
import os
import time
from typing import Any, Callable, Dict, List
from urllib.parse import quote

from llm_platform.adapters.adapter_base import AdapterBase
from llm_platform.services.conversation import Conversation, Message, ThinkingResponse
from llm_platform.services.files import DocumentFile
from llm_platform.tools.base import BaseTool
from llm_platform.types import AdditionalParameters


class WiroAIAdapter(AdapterBase):
    """Adapter for WiroAI's asynchronous Run/Task API."""

    BASE_URL = "https://api.wiro.ai/v1"
    HTTP_TIMEOUT_SECONDS = 30
    TASK_TIMEOUT_SECONDS = 120
    INITIAL_POLL_INTERVAL_SECONDS = 2
    MAX_POLL_INTERVAL_SECONDS = 10
    TERMINAL_STATUSES = {"task_postprocess_end", "task_error", "task_cancel"}

    def _build_client(self):
        # ``requests`` is imported lazily to preserve adapter-level lazy loading.
        import requests

        return requests.Session()

    @property
    def base_url(self) -> str:
        return os.getenv("WIRO_API_BASE_URL", self.BASE_URL).rstrip("/")

    @staticmethod
    def _auth_headers() -> Dict[str, str]:
        api_key = os.getenv("WIRO_API_KEY")
        if not api_key:
            raise ValueError("WIRO_API_KEY is required to call WiroAI")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }
        api_secret = os.getenv("WIRO_API_SECRET")
        if api_secret:
            nonce = str(time.time_ns() // 1_000_000)
            signature = hmac.new(
                api_key.encode("utf-8"),
                f"{api_secret}{nonce}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers.update({"x-signature": signature, "x-nonce": nonce})

        return headers

    @staticmethod
    def _format_errors(errors: Any) -> str:
        if not errors:
            return "Unknown WiroAI error"
        if not isinstance(errors, list):
            return str(errors)

        messages = []
        for error in errors:
            if isinstance(error, dict):
                messages.append(str(error.get("message") or error.get("code") or error))
            else:
                messages.append(str(error))
        return "; ".join(messages)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._auth_headers(),
            timeout=self.HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("WiroAI returned an invalid JSON response")
        if data.get("result") is False:
            raise RuntimeError(f"WiroAI request failed: {self._format_errors(data.get('errors'))}")
        return data

    @staticmethod
    def _message_text(message: Message) -> str:
        parts = [message.content] if message.content else []
        for file in message.files:
            if not isinstance(file, DocumentFile):
                raise ValueError(
                    f"WiroAI text models do not support {type(file).__name__} input"
                )
            parts.append(f'<document name="{file.name}">{file.text}</document>')
        return "\n\n".join(part for part in parts if part)

    def convert_conversation_history_to_adapter_format(
        self,
        the_conversation: Conversation,
        *args,
        **kwargs,
    ) -> str:
        """Serialize the platform-owned conversation into one WiroAI prompt.

        WiroAI sessions are optional. Sending the complete transcript keeps this
        adapter consistent when conversations are cleared, restored, or switched
        between providers.
        """
        messages = [
            (message.role, self._message_text(message))
            for message in the_conversation.messages
        ]
        if len(messages) == 1 and messages[0][0] == "user":
            return messages[0][1]

        role_labels = {"user": "User", "assistant": "Assistant", "function": "Tool"}
        return "\n\n".join(
            f"{role_labels[role]}:\n{text}" for role, text in messages if text
        )

    def _model_route(self, model: str) -> str:
        model_object = self.model_config[model]
        if model_object is None:
            raise ValueError(f"Model '{model}' is not defined in models_config.yaml")

        owner = model_object["wiro_owner"]
        model_slug = model_object["wiro_model"]
        if not owner or not model_slug:
            raise ValueError(f"Model '{model}' is missing its WiroAI route metadata")
        return f"{quote(str(owner), safe='')}/{quote(str(model_slug), safe='')}"

    @staticmethod
    def _task_failure(task: Dict[str, Any]) -> RuntimeError:
        detail = task.get("debugerror") or task.get("debugoutput")
        if not detail:
            detail = f"status={task.get('status')}, pexit={task.get('pexit')}"
        return RuntimeError(f"WiroAI task failed: {detail}")

    def _poll_task(self, task_token: str | None, task_id: str | None) -> Dict[str, Any]:
        reference = {"tasktoken": task_token} if task_token else {"taskid": task_id}
        deadline = time.monotonic() + self.TASK_TIMEOUT_SECONDS
        poll_interval = self.INITIAL_POLL_INTERVAL_SECONDS

        while True:
            detail = self._post_json("/Task/Detail", reference)
            tasks = detail.get("tasklist") or []
            if not tasks:
                raise RuntimeError("WiroAI task response did not contain a task")

            task = tasks[0]
            status = task.get("status")
            if status in self.TERMINAL_STATUSES:
                if status == "task_postprocess_end" and str(task.get("pexit")) == "0":
                    return task
                raise self._task_failure(task)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                identifier = task_token or task_id
                raise TimeoutError(f"WiroAI task {identifier} did not finish within the timeout")

            time.sleep(min(poll_interval, remaining))
            poll_interval = min(poll_interval * 2, self.MAX_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _string_items(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item]

    def _message_from_task(self, model: str, task: Dict[str, Any]) -> Message:
        raw_content: Dict[str, Any] = {}
        for output in task.get("outputs") or []:
            if output.get("contenttype") == "raw" and isinstance(output.get("content"), dict):
                raw_content = output["content"]
                break

        answers = self._string_items(raw_content.get("answer"))
        thinking = self._string_items(raw_content.get("thinking"))
        content = "\n\n".join(answers)
        if not content:
            content = str(raw_content.get("raw") or task.get("debugoutput") or "")

        try:
            cost = float(task.get("totalcost") or 0)
        except (TypeError, ValueError):
            cost = 0.0

        usage = {
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "costs": cost,
        }
        return Message(
            role="assistant",
            content=content,
            thinking_responses=[
                ThinkingResponse(content=item, id=task.get("id")) for item in thinking
            ],
            usage=usage,
            id=task.get("id"),
        )

    def request_llm(
        self,
        model: str,
        the_conversation: Conversation,
        functions: List[BaseTool] = None,
        tool_output_callback: Callable = None,
        additional_parameters: AdditionalParameters | None = None,
        **kwargs,
    ) -> Message:
        if functions:
            raise NotImplementedError("WiroAIAdapter does not support tool calling")

        parameters = self._merge_additional_parameters(additional_parameters, kwargs)
        if isinstance(parameters.get("enableThinking"), bool):
            # The model schema exposes this select as the strings "true"/"false".
            parameters["enableThinking"] = str(parameters["enableThinking"]).lower()
        parameters["prompt"] = self.convert_conversation_history_to_adapter_format(
            the_conversation
        )
        if the_conversation.system_prompt:
            parameters["system_prompt"] = the_conversation.system_prompt

        run = self._post_json(f"/Run/{self._model_route(model)}", parameters)
        task_token = run.get("socketaccesstoken")
        task_id = run.get("taskid")
        if not task_token and not task_id:
            raise RuntimeError("WiroAI run response did not contain a task identifier")

        task = self._poll_task(task_token, task_id)
        message = self._message_from_task(model, task)
        the_conversation.messages.append(message)
        return message
