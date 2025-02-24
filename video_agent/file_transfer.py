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

class FileTransfer:
    """Handles file transfers between local machine and GPU instances."""
    
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self._ensure_ssh_access()
    
    def _ensure_ssh_access(self) -> None:
        """Ensure SSH access is configured for the instance."""
        # Get instance details from status
        response = execute_remote_command(f"ssh_status {self.instance_id}")
        if response.startswith("Error"):
            raise RuntimeError(f"Failed to get instance status: {response}")
        
        try:
            ssh_info = json.loads(response)
            if "error" in ssh_info:
                raise RuntimeError(f"Failed to get SSH access: {ssh_info['error']}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid SSH status response: {response}")
    
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the GPU instance.
        
        Args:
            local_path: Path to local file
            remote_path: Destination path on GPU instance
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # Get instance SSH details
        response = execute_remote_command(f"ssh_status {self.instance_id}")
        try:
            ssh_info = json.loads(response)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid SSH status response: {response}")
        
        # Create remote directory if needed
        remote_dir = str(Path(remote_path).parent)
        execute_remote_command(f"mkdir -p {remote_dir}")
        
        # Build scp command
        host = ssh_info["host"]
        port = ssh_info.get("port", 22)
        user = ssh_info["username"]
        
        # Use scp to upload file
        cmd = (
            f"scp -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"{local_path} {user}@{host}:{remote_path}"
        )
        
        # Execute scp locally
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Failed to upload file: {local_path}")
    
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the GPU instance.
        
        Args:
            remote_path: Path to file on GPU instance
            local_path: Destination path on local machine
        """
        # Check if remote file exists
        result = execute_remote_command(f"test -f {remote_path} && echo 'exists'")
        if "exists" not in result:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        
        # Get instance SSH details
        response = execute_remote_command(f"ssh_status {self.instance_id}")
        try:
            ssh_info = json.loads(response)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid SSH status response: {response}")
        
        # Create local directory if needed
        os.makedirs(str(Path(local_path).parent), exist_ok=True)
        
        # Build scp command
        host = ssh_info["host"]
        port = ssh_info.get("port", 22)
        user = ssh_info["username"]
        
        # Use scp to download file
        cmd = (
            f"scp -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"{user}@{host}:{remote_path} {local_path}"
        )
        
        # Execute scp locally
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Failed to download file: {remote_path}")
    
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