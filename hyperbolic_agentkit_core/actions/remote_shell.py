import os
import json
import time
from typing import Optional
import logging

from collections.abc import Callable
from pydantic import BaseModel, Field
from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RemoteShell")

REMOTE_SHELL_PROMPT = """
Execute commands on a remote server via SSH. This tool requires an active SSH connection.

Input parameters:
- command: The shell command to execute on the remote server

Important notes:
- Use 'ssh_connect' first to establish an SSH connection
- Commands are executed in the same session, so environment variables and working directory are preserved between calls
- For long-running commands, consider adding '&' at the end to run in background
- Use 'cd' to change directories, 'pwd' to check current directory
- Standard output and error from the command will be returned
"""

class RemoteShellInput(BaseModel):
    """Input argument schema for remote shell execution."""
    command: str = Field(..., description="Shell command to execute on the remote server")

def execute_remote_command(command: str, instance_id: Optional[str] = None, max_retries: int = 3, retry_delay: int = 5) -> str:
    """
    Execute a command on a remote server via SSH.
    
    Args:
        command: Shell command to execute
        instance_id: Optional instance ID (not used directly but kept for backward compatibility)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        str: Command output or error message
    """
    # Check if SSH connection is active
    if not ssh_manager.is_connected:
        logger.warning("No active SSH connection")
        return "Error: No active SSH connection. Please connect first using ssh_connect."
    
    # Execute command with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Executing remote command (attempt {attempt+1}/{max_retries}): {command}")
            result = ssh_manager.execute(command, timeout=300)  # 5-minute timeout
            
            # Check if the command failed
            if result.startswith("Error:") or result.startswith("SSH Command Error:"):
                logger.warning(f"Command failed: {result}")
                
                # If this is not the last attempt, retry
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing remote command: {str(e)}")
            
            # If this is not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"Error executing remote command: {str(e)}"
    
    return "Error: Maximum retry attempts reached"

class RemoteShellAction(HyperbolicAction):
    """Remote shell execution action."""
    
    name: str = "remote_shell"
    description: str = REMOTE_SHELL_PROMPT
    args_schema: type[BaseModel] = RemoteShellInput
    func: Callable[..., str] = execute_remote_command