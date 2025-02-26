import paramiko
import os
import time
from typing import Optional, List, Dict, Any
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SSHManager")

class SSHManager:
    _instance = None
    _ssh_client = None
    _connected = False
    _host = None
    _username = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SSHManager, cls).__new__(cls)
        return cls._instance

    @property
    def is_connected(self) -> bool:
        """Check if there's an active SSH connection."""
        if self._ssh_client and self._connected:
            try:
                # Use a simple command with short timeout to check connection
                stdin, stdout, stderr = self._ssh_client.exec_command('echo 1', timeout=5)
                result = stdout.read().decode().strip()
                return result == '1'
            except Exception as e:
                logger.warning(f"SSH connection check failed: {str(e)}")
                self._connected = False
        return False

    def connect(self, host: str, username: str, password: Optional[str] = None, 
                private_key_path: Optional[str] = None, port: int = 22,
                key_password: Optional[str] = None) -> str:
        """Establish SSH connection."""
        try:
            # Close existing connection if any
            self.disconnect()
            
            # Initialize new client
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Get default key path from environment
            default_key_path = os.getenv('SSH_PRIVATE_KEY_PATH')
            if not default_key_path:
                default_key_path = os.path.expanduser('~/.ssh/id_rsa')
            
            # Use provided key_password or get from environment if available
            if not key_password:
                key_password = os.getenv('SSH_KEY_PASSWORD')
            
            if key_password:
                logger.info("Using SSH key password")

            # Try connecting with password if provided
            if password:
                logger.info(f"Attempting to connect to {host}:{port} as {username} using password")
                self._ssh_client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    password=password,
                    timeout=30,
                    allow_agent=False,
                    look_for_keys=False
                )
                self._connected = True
                self._host = host
                self._username = username
                return f"Successfully connected to {host} as {username} using password"
            
            # Try connecting with provided key or default key
            key_path = private_key_path if private_key_path else default_key_path
            
            # Check if key exists
            if not os.path.exists(key_path):
                logger.warning(f"SSH Key not found at {key_path}")
                # Try to find alternative keys
                alternative_keys = self._find_alternative_keys()
                if not alternative_keys:
                    return f"SSH Key Error: No valid key file found"
                
                # Try each alternative key
                for alt_key in alternative_keys:
                    try:
                        logger.info(f"Trying alternative key: {alt_key}")
                        self._try_connect_with_key(host, port, username, alt_key, key_password)
                        if self._connected:
                            self._host = host
                            self._username = username
                            return f"Successfully connected to {host} as {username} using key {alt_key}"
                    except Exception as e:
                        logger.warning(f"Failed with key {alt_key}: {str(e)}")
                
                # If we get here, none of the alternative keys worked
                return f"SSH Connection Error: Failed to connect with any available keys"
            
            # Try connecting with the specified key
            try:
                logger.info(f"Attempting to connect to {host}:{port} as {username} using key {key_path}")
                self._try_connect_with_key(host, port, username, key_path, key_password)
                if self._connected:
                    self._host = host
                    self._username = username
                    return f"Successfully connected to {host} as {username} using key {key_path}"
                else:
                    return f"SSH Connection Error: Failed to connect with key {key_path}"
            except Exception as e:
                logger.error(f"SSH connection error: {str(e)}")
                return f"SSH Connection Error: {str(e)}"

        except Exception as e:
            self._connected = False
            logger.error(f"SSH Connection Error: {str(e)}")
            return f"SSH Connection Error: {str(e)}"

    def _try_connect_with_key(self, host: str, port: int, username: str, 
                             key_path: str, key_password: Optional[str] = None) -> None:
        """Try to connect using a specific key with various methods."""
        # First try: Use key with password if available
        if key_password:
            try:
                logger.info(f"Trying key {key_path} with password")
                pkey = paramiko.RSAKey.from_private_key_file(key_path, password=key_password)
                self._ssh_client.connect(
                    hostname=host,
                    port=port,
                    username=username,
                    pkey=pkey,
                    timeout=30,
                    allow_agent=False,
                    look_for_keys=False
                )
                self._connected = True
                logger.info("Connected successfully with key and password")
                return
            except Exception as e:
                logger.warning(f"Failed to connect with key and password: {str(e)}")
        
        # Second try: Use key without password
        try:
            logger.info(f"Trying key {key_path} without password")
            self._ssh_client.connect(
                hostname=host,
                port=port,
                username=username,
                key_filename=key_path,
                timeout=30,
                allow_agent=False,
                look_for_keys=False
            )
            self._connected = True
            logger.info("Connected successfully with key without password")
            return
        except paramiko.ssh_exception.PasswordRequiredException:
            logger.warning(f"Key {key_path} requires a password but none provided")
            raise
        except Exception as e:
            logger.warning(f"Failed to connect with key without password: {str(e)}")
            raise

    def _find_alternative_keys(self) -> List[str]:
        """Find alternative SSH keys in the ~/.ssh directory."""
        alternative_keys = []
        ssh_dir = os.path.expanduser("~/.ssh")
        
        # Check if SSH directory exists
        if not os.path.isdir(ssh_dir):
            logger.warning(f"SSH directory not found: {ssh_dir}")
            return alternative_keys
        
        # Look for common key names
        key_names = [
            'hyperbolic', 'hyperbolic_key', 'hyperbolic_pem', 'hyperbolic_unencrypted',
            'id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa'
        ]
        
        for key_name in key_names:
            key_path = os.path.join(ssh_dir, key_name)
            if os.path.isfile(key_path):
                logger.info(f"Found potential SSH key: {key_path}")
                alternative_keys.append(key_path)
        
        return alternative_keys

    def execute(self, command: str, timeout: int = 60) -> str:
        """Execute a command on the remote server.
        
        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds
            
        Returns:
            str: Command output or error message
        """
        if not self._ssh_client or not self.is_connected:
            return "Error: No active SSH connection"
        
        try:
            logger.info(f"Executing command with timeout {timeout}s: {command}")
            
            # Create a channel for command execution
            transport = self._ssh_client.get_transport()
            if not transport:
                return "Error: SSH transport not available"
            
            channel = transport.open_session()
            channel.settimeout(timeout)
            
            # Execute the command
            channel.exec_command(command)
            
            # Read output with timeout handling
            stdout_data = b""
            stderr_data = b""
            
            # Set channel to non-blocking mode
            channel.setblocking(0)
            
            # Wait for command to complete with timeout
            start_time = time.time()
            while not channel.exit_status_ready():
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    channel.close()
                    return f"Error: Command timed out after {timeout} seconds"
                
                # Try to read data if available
                if channel.recv_ready():
                    chunk = channel.recv(4096)
                    if not chunk:
                        break
                    stdout_data += chunk
                
                if channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(4096)
                    if not chunk:
                        break
                    stderr_data += chunk
                
                # Sleep briefly to avoid CPU spinning
                time.sleep(0.1)
            
            # Read any remaining data
            while channel.recv_ready():
                chunk = channel.recv(4096)
                if not chunk:
                    break
                stdout_data += chunk
            
            while channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096)
                if not chunk:
                    break
                stderr_data += chunk
            
            # Get exit status
            exit_status = channel.recv_exit_status()
            
            # Close the channel
            channel.close()
            
            # Decode output
            stdout = stdout_data.decode('utf-8', errors='replace')
            stderr = stderr_data.decode('utf-8', errors='replace')
            
            # Return combined output or error
            if exit_status != 0:
                logger.warning(f"Command failed with exit status {exit_status}")
                if stderr:
                    return f"SSH Command Error (status {exit_status}): {stderr}\nOutput: {stdout}"
                else:
                    return f"SSH Command Error (status {exit_status}): {stdout}"
            
            return stdout
            
        except socket.timeout:
            logger.error(f"Command timed out after {timeout} seconds: {command}")
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return f"SSH Command Error: {str(e)}"

    def disconnect(self):
        """Close SSH connection."""
        if self._ssh_client:
            try:
                logger.info("Disconnecting SSH session")
                self._ssh_client.close()
                logger.info("SSH session disconnected")
            except Exception as e:
                logger.warning(f"Error during SSH disconnect: {str(e)}")
        self._connected = False
        self._host = None
        self._username = None

    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information."""
        if self.is_connected:
            return {
                "status": "connected",
                "host": self._host,
                "username": self._username,
                "transport_active": self._ssh_client.get_transport().is_active() if self._ssh_client else False
            }
        return {"status": "disconnected"}

# Global instance
ssh_manager = SSHManager() 