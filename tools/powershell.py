from llm_platform.tools.base import BaseTool
from typing import Dict
from pydantic import BaseModel, Field
import subprocess

from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.table import Table

class RunPowerShellCommand(BaseTool):
    """
    Runs a persistent PowerShell session on the user's Windows 11 machine.
    Once created, the same PowerShell process is reused for each command.
    """

    __name__ = "RunPowerShellCommand"

    class InputModel(BaseModel):
        command: str = Field(description="Command to execute", required=True)

    def __init__(self):
        """
        Initializes a single PowerShell process that remains active
        across multiple calls to the __call__() method.
        """
        super().__init__()
        self.console = Console()
        # Start a persistent PowerShell session
        self.process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NoExit",
                "-NonInteractive",
                "-Command", 
                "$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8;"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1  # Line-buffered
        )

    def __call__(self, command: str) -> Dict:
        """
        Executes a command in the existing PowerShell session. The current
        directory and other session state are preserved between calls.

        Args:
            command (str): The PowerShell command to execute.

        Returns:
            Dict: Contains "output" for standard output and "error" for
                  standard error.
        """
        # Show the command in a panel
        self.console.print(Panel(f"Command: [bold green]{command}", border_style="green"))

        # Prepare a marker to detect the end of the command's output
        output_marker = "END_OF_OUTPUT"
        error_marker = "END_OF_ERROR"

        # Write the command plus markers to both stdout and stderr
        # so we can distinguish when to stop reading each stream.
        full_command = (
            f"{command}\n"
            f'Write-Host "{output_marker}"\n'
            f'Write-Error "{error_marker}"\n'
        )

        # Send the command to PowerShell
        self.process.stdin.write(full_command)
        self.process.stdin.flush()

        # Read standard output until we reach the output marker
        output_lines = []
        while True:
            line = self.process.stdout.readline()
            # If process has closed or no more output
            if not line:
                break
            # Stop once we detect our marker
            if output_marker in line:
                break
            output_lines.append(line)

        # Read standard error until we reach the error marker
        error_lines = []
        while True:
            line = self.process.stderr.readline()
            # If process has closed or no more error output
            if not line:
                break
            # Stop once we detect our marker
            if error_marker in line:
                break
            error_lines.append(line)

        # Combine lines into strings
        output_str = "".join(output_lines).strip()
        error_str = "".join(error_lines).strip()

        # Display results
        self.console.print(Panel(f"Output:\n[bold yellow]{output_str}", border_style="yellow"))
        self.console.print(Panel(f"Error:\n[bold red]{error_str}", border_style="red"))

        return {
            "output": output_str,
            "error": error_str
        }