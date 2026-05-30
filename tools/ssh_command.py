from typing import Dict, Iterable

from pydantic import BaseModel, Field

from llm_platform.tools.base import BaseTool


class SSHCommandTool(BaseTool):
    """Executes a command on a remote host over SSH.

    Connection details are provided at construction time; subclasses only need to
    set a descriptive class name and docstring (the docstring is sent to the model
    as the tool description).

    Pass ``allowed_commands`` to restrict execution to an allow-list (matched on
    the command's first token); the default ``None`` applies no restriction.
    """

    class InputModel(BaseModel):
        command: str = Field(description="Command to execute")

    def __init__(self, host, username, password, allowed_commands: Iterable[str] | None = None):
        self.host = host
        self.username = username
        self.password = password
        self.allowed_commands = allowed_commands
        self.client = None

    def __call__(self, command: str) -> Dict:
        self._check_command_allowed(command, self.allowed_commands)

        if self.client is None:
            # Imported lazily so the tools layer does not hard-depend on paramiko.
            import paramiko

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.client.connect(self.host, username=self.username, password=self.password)
            except Exception as e:
                raise RuntimeError(f"Error connecting to {self.host}: {e}") from e

        stdin, stdout, stderr = self.client.exec_command(command)
        return {"output": stdout.read().decode(), "error": stderr.read().decode()}
