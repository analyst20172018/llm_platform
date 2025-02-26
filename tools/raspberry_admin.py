import paramiko
from llm_platform.tools.base import BaseTool
from typing import Dict
from pydantic import BaseModel, Field

class RaspberryAdmin(BaseTool):
    """Executes a command on a remote Raspberry Pi SSH server.

    Args:
        command: The command to execute on the remote server.

    Returns:
        A dictionary containing 'output' and 'error' keys with the command's standard output and standard error, respectively.
    """

    __name__ = "RaspberryAdmin"

    class InputModel(BaseModel):
        command: str = Field(description = "Command to execute", required = True)

    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        self.client = None

    def __call__(self, command: str) -> Dict:
        """Executes a command on a remote Raspberry Pi SSH server."""

        # Initialize the SSH client if it's not already initialized
        if self.client is None:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Connect to the remote server
            try:
                self.client.connect(self.host, 
                                    username=self.username, 
                                    password=self.password)
            except Exception as e:
                raise f"Error connecting to {self.host}: {e}"
        
        stdin, stdout, stderr = self.client.exec_command(command)
        
        output = stdout.read().decode()
        error = stderr.read().decode()
        return {"output": output, "error": error}