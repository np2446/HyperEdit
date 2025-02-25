#!/usr/bin/env python3
"""
Test script for SSH connection with Hyperbolic.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager
from hyperbolic_agentkit_core.actions.get_gpu_status import get_gpu_status
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command

def main():
    """Test SSH connection."""
    print("Testing SSH connection...")
    
    # Get SSH key path and password
    ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH', os.path.expanduser('~/.ssh/hyperbolic_key'))
    ssh_key_password = os.environ.get('SSH_KEY_PASSWORD')
    
    print(f"Using SSH key: {ssh_key_path}")
    print(f"SSH key password available: {'Yes' if ssh_key_password else 'No'}")
    
    # Get GPU status
    print("\nGetting GPU status...")
    status_data = get_gpu_status()
    
    if isinstance(status_data, str):
        try:
            status = json.loads(status_data)
        except json.JSONDecodeError:
            print(f"Error parsing status data as JSON: {status_data[:100]}...")
            return
    else:
        status = status_data
    
    # Extract instances
    instances = []
    if 'instances' in status and isinstance(status['instances'], list):
        instances = status['instances']
    elif 'data' in status and isinstance(status['data'], list):
        instances = status['data']
    elif 'data' in status and isinstance(status['data'], dict) and 'instances' in status['data']:
        instances = status['data']['instances']
    
    if not instances:
        print("No instances found.")
        return
    
    print(f"Found {len(instances)} instances.")
    
    # Try to connect to the first instance
    instance = instances[0]
    instance_id = instance.get('id')
    
    print(f"\nTrying to connect to instance: {instance_id}")
    print(f"Instance data: {json.dumps(instance, indent=2)}")
    
    # Extract SSH command if available
    ssh_command = instance.get('sshCommand')
    if ssh_command:
        print(f"SSH command: {ssh_command}")
        
        # Extract hostname and port
        import re
        host_match = re.search(r'@([\w.-]+)', ssh_command)
        port_match = re.search(r'-p\s+(\d+)', ssh_command)
        
        if host_match:
            hostname = host_match.group(1)
            port = int(port_match.group(1)) if port_match else 22
            
            print(f"Extracted hostname: {hostname}, port: {port}")
            
            # Try to connect
            print("\nConnecting to SSH...")
            result = connect_ssh(
                host=hostname,
                username="ubuntu",
                private_key_path=ssh_key_path,
                port=port,
                key_password=ssh_key_password
            )
            
            print(f"SSH connection result: {result}")
            
            # If connected, try a simple command
            if "Successfully connected" in result:
                print("\nExecuting a simple command...")
                cmd_result = execute_remote_command("echo 'Hello from Hyperbolic!'")
                print(f"Command result: {cmd_result}")
                
                # Disconnect
                ssh_manager.disconnect()
                print("SSH disconnected.")
            
        else:
            print("Could not extract hostname from SSH command.")
    else:
        print("No SSH command found in instance data.")

if __name__ == "__main__":
    main() 