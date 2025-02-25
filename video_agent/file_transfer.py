"""
File transfer utilities for moving data between local machine and GPU instances.
Uses Hyperbolic's SSH access for secure file transfer.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager
from hyperbolic_agentkit_core.actions.get_gpu_status import get_gpu_status

class FileTransfer:
    """Handles file transfers between local machine and GPU instances."""
    
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self._ensure_ssh_access()
    
    def _ensure_ssh_access(self) -> None:
        """Ensure SSH access is configured for the instance."""
        # Try to get instance details from status
        try:
            response = execute_remote_command(f"ssh_status {self.instance_id}")
            if response.startswith("Error"):
                # If ssh_status command doesn't exist, we can still proceed if we have an active SSH connection
                if "command not found" in response and ssh_manager.is_connected:
                    print("ssh_status command not found, but SSH connection is active. Proceeding...")
                    return
                raise RuntimeError(f"Failed to get instance status: {response}")
            
            try:
                ssh_info = json.loads(response)
                if "error" in ssh_info:
                    raise RuntimeError(f"Failed to get SSH access: {ssh_info['error']}")
            except json.JSONDecodeError:
                # Try to extract useful information from non-JSON response
                if "host" in response and "username" in response:
                    print(f"SSH access appears to be configured despite JSON parse error: {response}")
                    return
                
                # If we have an active SSH connection, we can proceed
                if ssh_manager.is_connected:
                    print("SSH connection is active despite invalid response. Proceeding...")
                    return
                    
                raise RuntimeError(f"Invalid SSH status response: {response}")
        except Exception as e:
            # If we have an active SSH connection, we can proceed despite errors
            if ssh_manager.is_connected:
                print(f"SSH connection is active despite error: {str(e)}. Proceeding...")
                return
            raise RuntimeError(f"Failed to ensure SSH access: {str(e)}")
    
    def upload_file(self, local_path: str, remote_path: str, max_retries: int = 3) -> None:
        """Upload a file to the GPU instance with retry logic.
        
        Args:
            local_path: Path to local file
            remote_path: Destination path on GPU instance
            max_retries: Maximum number of retry attempts
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # Get SSH connection info
        ssh_info = self._get_ssh_info()
        
        # Create remote directory if needed
        remote_dir = str(Path(remote_path).parent)
        execute_remote_command(f"mkdir -p {remote_dir}")
        
        # Build scp command
        host = ssh_info["host"]
        port = ssh_info.get("port", 22)
        user = ssh_info["username"]
        
        # Use scp to upload file with retries
        for attempt in range(max_retries):
            try:
                # Build the scp command
                cmd = (
                    f"scp -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
                    f"-o ConnectTimeout=30 -o ServerAliveInterval=30 "
                    f"{local_path} {user}@{host}:{remote_path}"
                )
                
                print(f"Uploading file (attempt {attempt+1}/{max_retries}): {local_path} -> {remote_path}")
                
                # Execute scp locally
                result = os.system(cmd)
                if result == 0:
                    print(f"Successfully uploaded file: {local_path}")
                    
                    # Verify the file exists on the remote server
                    verify_cmd = f"test -f {remote_path} && echo 'exists'"
                    verify_result = execute_remote_command(verify_cmd)
                    if "exists" in verify_result:
                        return
                    else:
                        print(f"Warning: File upload succeeded but verification failed. Retrying...")
                else:
                    print(f"Upload failed with exit code {result}. Retrying...")
                
                # If we're not on the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                print(f"Upload attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))
        
        raise RuntimeError(f"Failed to upload file after {max_retries} attempts: {local_path}")
    
    def download_file(self, remote_path: str, local_path: str, max_retries: int = 3) -> None:
        """Download a file from the GPU instance with retry logic.
        
        Args:
            remote_path: Path to file on GPU instance
            local_path: Destination path on local machine
            max_retries: Maximum number of retry attempts
        """
        # Check if remote file exists
        result = execute_remote_command(f"test -f {remote_path} && echo 'exists'")
        if "exists" not in result:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        
        # Get SSH connection info
        ssh_info = self._get_ssh_info()
        
        # Create local directory if needed
        os.makedirs(str(Path(local_path).parent), exist_ok=True)
        
        # Build scp command
        host = ssh_info["host"]
        port = ssh_info.get("port", 22)
        user = ssh_info["username"]
        
        # Use scp to download file with retries
        for attempt in range(max_retries):
            try:
                # Build the scp command
                cmd = (
                    f"scp -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
                    f"-o ConnectTimeout=30 -o ServerAliveInterval=30 "
                    f"{user}@{host}:{remote_path} {local_path}"
                )
                
                print(f"Downloading file (attempt {attempt+1}/{max_retries}): {remote_path} -> {local_path}")
                
                # Execute scp locally
                result = os.system(cmd)
                if result == 0:
                    print(f"Successfully downloaded file: {remote_path}")
                    
                    # Verify the file exists locally
                    if os.path.exists(local_path):
                        return
                    else:
                        print(f"Warning: Download succeeded but file not found locally. Retrying...")
                else:
                    print(f"Download failed with exit code {result}. Retrying...")
                
                # If we're not on the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                print(f"Download attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))
        
        raise RuntimeError(f"Failed to download file after {max_retries} attempts: {remote_path}")
    
    def upload_directory(self, local_dir: str, remote_dir: str) -> None:
        """Upload a directory and its contents to the GPU instance.
        
        Args:
            local_dir: Path to local directory
            remote_dir: Destination directory on GPU instance
        """
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(f"Local directory not found: {local_dir}")
        
        # Get instance SSH details
        response = execute_remote_command(f"ssh_status {self.instance_id}")
        try:
            ssh_info = json.loads(response)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid SSH status response: {response}")
        
        # Create remote directory
        execute_remote_command(f"mkdir -p {remote_dir}")
        
        # Build scp command for recursive copy
        host = ssh_info["host"]
        port = ssh_info.get("port", 22)
        user = ssh_info["username"]
        
        cmd = (
            f"scp -r -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"{local_dir}/* {user}@{host}:{remote_dir}/"
        )
        
        # Execute scp locally
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Failed to upload directory: {local_dir}")
    
    def download_directory(self, remote_dir: str, local_dir: str) -> None:
        """Download a directory and its contents from the GPU instance.
        
        Args:
            remote_dir: Path to directory on GPU instance
            local_dir: Destination directory on local machine
        """
        # Check if remote directory exists
        result = execute_remote_command(f"test -d {remote_dir} && echo 'exists'")
        if "exists" not in result:
            raise NotADirectoryError(f"Remote directory not found: {remote_dir}")
        
        # Get instance SSH details
        response = execute_remote_command(f"ssh_status {self.instance_id}")
        try:
            ssh_info = json.loads(response)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid SSH status response: {response}")
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Build scp command for recursive copy
        host = ssh_info["host"]
        port = ssh_info.get("port", 22)
        user = ssh_info["username"]
        
        cmd = (
            f"scp -r -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"{user}@{host}:{remote_dir}/* {local_dir}/"
        )
        
        # Execute scp locally
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Failed to download directory: {remote_dir}")
    
    def list_remote_files(self, remote_path: str) -> List[str]:
        """List files in a remote directory.
        
        Args:
            remote_path: Path to directory on GPU instance
            
        Returns:
            List of file paths relative to remote_path
        """
        result = execute_remote_command(f"ls -R {remote_path}")
        if "No such file or directory" in result:
            raise FileNotFoundError(f"Remote path not found: {remote_path}")
        
        return [line.strip() for line in result.split('\n') if line.strip()]

    def _get_ssh_info(self) -> dict:
        """Get SSH connection information.
        
        Returns:
            dict: SSH connection info with host, username, and port
        """
        # First try to get info from ssh_status command
        try:
            response = execute_remote_command(f"ssh_status {self.instance_id}")
            if not response.startswith("Error") and not "command not found" in response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Try to extract host and username from the response
                    import re
                    host_match = re.search(r'host["\s:]+([^"\s,]+)', response, re.IGNORECASE)
                    user_match = re.search(r'username["\s:]+([^"\s,]+)', response, re.IGNORECASE)
                    
                    if host_match and user_match:
                        return {
                            "host": host_match.group(1),
                            "username": user_match.group(1),
                            "port": 22  # Default port
                        }
        except Exception as e:
            print(f"Warning: Failed to get SSH info from ssh_status: {str(e)}")
        
        # If ssh_status failed or returned invalid data, try to get info from ssh_manager
        if ssh_manager.is_connected:
            conn_info = ssh_manager.get_connection_info()
            if conn_info.get("status") == "connected":
                return {
                    "host": conn_info.get("host"),
                    "username": conn_info.get("username"),
                    "port": 22  # Default port
                }
        
        # If all else fails, try to get info from instance data
        try:
            status_data = get_gpu_status()
            if isinstance(status_data, str):
                status = json.loads(status_data)
            else:
                status = status_data
            
            # Find our instance
            for instance in status.get('instances', []):
                if instance.get('id') == self.instance_id:
                    # Extract from sshCommand
                    ssh_cmd = instance.get('sshCommand')
                    if ssh_cmd:
                        import re
                        host_match = re.search(r'@([\w.-]+)', ssh_cmd)
                        port_match = re.search(r'-p\s+(\d+)', ssh_cmd)
                        
                        if host_match:
                            return {
                                "host": host_match.group(1),
                                "username": "ubuntu",  # Default username
                                "port": int(port_match.group(1)) if port_match else 22
                            }
        except Exception as e:
            print(f"Warning: Failed to get SSH info from instance data: {str(e)}")
        
        raise RuntimeError("Could not determine SSH connection information") 