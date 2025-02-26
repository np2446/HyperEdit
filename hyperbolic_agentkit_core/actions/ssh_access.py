import os
from typing import Optional
from collections.abc import Callable
from pydantic import BaseModel, Field
from hyperbolic_agentkit_core.actions.hyperbolic_action import HyperbolicAction
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager

SSH_ACCESS_PROMPT = """
Connect to a remote server via SSH. Once connected, all shell commands will automatically run on this server.

Input parameters:
- host: The hostname or IP address of the remote server
- username: SSH username for authentication
- password: SSH password for authentication (optional if using key)
- private_key_path: Path to private key file (optional, uses SSH_PRIVATE_KEY_PATH from environment if not provided)
- port: SSH port number (default: 22)
- key_password: Password for encrypted private key (optional)

Important notes:
- After connecting, use the 'remote_shell' tool to execute commands on the server
- Use 'ssh_status' command to check connection status
- Connection remains active until explicitly disconnected or script ends
"""

class SSHAccessInput(BaseModel):
    """Input argument schema for SSH access."""
    host: str = Field(..., description="Hostname or IP address of the remote server")
    username: str = Field(..., description="SSH username for authentication")
    password: Optional[str] = Field(None, description="SSH password for authentication")
    private_key_path: Optional[str] = Field(None, description="Path to private key file")
    port: int = Field(22, description="SSH port number")
    key_password: Optional[str] = Field(None, description="Password for encrypted private key")

def connect_ssh(host: str, username: str, password: Optional[str] = None, 
                private_key_path: Optional[str] = None, port: int = 22,
                key_password: Optional[str] = None) -> str:
    """
    Establish SSH connection to remote server.
    
    Args:
        host: Hostname or IP address of the remote server
        username: SSH username for authentication
        password: Optional SSH password for authentication
        private_key_path: Optional path to private key file
        port: SSH port number (default: 22)
        key_password: Optional password for encrypted private key
    
    Returns:
        str: Connection status message
    """
    # If key_password is not provided, try to get it from environment
    if not key_password:
        key_password = os.environ.get('SSH_KEY_PASSWORD')
    
    return ssh_manager.connect(
        host=host,
        username=username,
        password=password,
        private_key_path=private_key_path,
        port=port,
        key_password=key_password
    )

class SSHAccessAction(HyperbolicAction):
    """SSH connection action."""
    
    name: str = "ssh_connect"
    description: str = SSH_ACCESS_PROMPT
    args_schema: type[BaseModel] = SSHAccessInput
    func: Callable[..., str] = connect_ssh