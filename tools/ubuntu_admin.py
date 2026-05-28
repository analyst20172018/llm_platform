from llm_platform.tools.ssh_command import SSHCommandTool


class UbuntuAdmin(SSHCommandTool):
    """Executes a command on a remote Ubuntu server via ssh.

    Args:
        command: The command to execute on the remote server.

    Returns:
        A dictionary containing 'output' and 'error' keys with the command's
        standard output and standard error, respectively.
    """
