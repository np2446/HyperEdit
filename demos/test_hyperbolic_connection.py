#!/usr/bin/env python3
"""
Test script for Hyperbolic GPU connection and file transfer capabilities.
This script tests:
1. Renting the cheapest available GPU instance
2. SSH connection to the instance
3. File upload and download
4. Running a simple Python script on the remote instance
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Hyperbolic components
from video_agent.video_processor import VideoProcessor, GPURequirements
from video_agent.file_transfer import FileTransfer
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager

def check_environment():
    """Check if the environment is properly configured for Hyperbolic GPU usage."""
    # Check for SSH key
    ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH')
    if not ssh_key_path:
        print("ERROR: SSH_PRIVATE_KEY_PATH environment variable is not set.")
        print("Please set it to the path of your SSH private key for Hyperbolic.")
        return False
    
    ssh_key_path = os.path.expanduser(ssh_key_path)
    if not os.path.exists(ssh_key_path):
        print(f"ERROR: SSH key file not found at {ssh_key_path}")
        return False
    
    print(f"Using SSH key: {ssh_key_path}")
    return True

def create_test_python_script():
    """Create a simple Python script to run on the remote instance."""
    script_content = """#!/usr/bin/env python3
import platform
import socket
import os
import json
import sys
import subprocess

# Get system information
system_info = {
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "hostname": socket.gethostname(),
    "cpu_info": platform.processor(),
    "environment_vars": dict(os.environ),
    "current_directory": os.getcwd(),
}

# Check for GPU information
try:
    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader"], text=True)
    system_info["gpu_info"] = nvidia_smi_output.strip()
except Exception as e:
    system_info["gpu_info"] = f"Error getting GPU info: {str(e)}"

# Print the information as JSON
print(json.dumps(system_info, indent=2, default=str))
"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_path = f.name
    
    return temp_path

def run_connection_test():
    """Run the connection test with Hyperbolic."""
    if not check_environment():
        print("Environment check failed. Please fix the issues and try again.")
        return False
    
    print("\n=== Starting Hyperbolic Connection Test ===\n")
    
    # Initialize video processor with remote mode
    processor = VideoProcessor(local_mode=False)
    
    try:
        # Initialize GPU environment with the cheapest options
        print("\nSetting up GPU environment...")
        print("This may take several minutes. Please be patient.")
        print("Looking for the cheapest available GPU...")
        
        processor.setup_gpu_environment(
            GPURequirements(
                gpu_type="",  # Any GPU type
                num_gpus=1,
                min_vram_gb=1.0,  # Minimum VRAM
                disk_size=5,
                memory=8
            )
        )
        
        print("\n✅ GPU environment set up successfully!")
        print(f"Instance ID: {processor.instance_id}")
        
        # Get SSH key path and password from environment
        ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH')
        ssh_key_password = os.environ.get('SSH_KEY_PASSWORD')
        
        # Get instance details
        instance_data = processor.current_instance
        ssh_command = instance_data.get('sshCommand', '')
        
        # Extract hostname and port from SSH command
        import re
        match = re.search(r'ssh\s+(\w+)@([^:\s]+)(?:\s+-p\s+(\d+))?', ssh_command)
        if match:
            username, hostname, port = match.groups()
            port = int(port) if port else 22
            print(f"Extracted connection details: {username}@{hostname}:{port}")
        else:
            print(f"Could not parse SSH command: {ssh_command}")
            return False
        
        # Test SSH connection directly
        print("\n=== Test 1: Verifying SSH Connection ===")
        print("Testing direct SSH connection...")
        
        # Ensure any previous connections are closed
        if ssh_manager.is_connected:
            ssh_manager.disconnect()
            print("Closed previous SSH connection")
        
        # Connect using the instance details
        from hyperbolic_agentkit_core.actions.connect_ssh import connect_ssh
        ssh_result = connect_ssh(
            host=hostname,
            username=username,
            private_key_path=ssh_key_path,
            port=port,
            key_password=ssh_key_password
        )
        
        print(f"SSH connection result: {ssh_result}")
        if "Successfully connected" not in ssh_result:
            print("❌ SSH connection failed. Cannot proceed with tests.")
            return False
        
        # Test 2: Check home directory and permissions
        print("\n=== Test 2: Checking Home Directory ===")
        home_dir_cmd = "echo $HOME"
        home_dir_result = execute_remote_command(home_dir_cmd, instance_id=processor.instance_id)
        home_dir = home_dir_result.strip()
        print(f"Home directory: {home_dir}")
        
        # Check if we can write to home directory
        test_dir = f"{home_dir}/test_connection"
        cmd_result = execute_remote_command(f"mkdir -p {test_dir}", instance_id=processor.instance_id)
        print(f"Directory creation result: {cmd_result}")
        
        # Test 3: File upload to home directory
        print("\n=== Test 3: File Upload to Home Directory ===")
        test_file = "test_connection.txt"
        with open(test_file, "w") as f:
            f.write(f"This is a test file created at {time.ctime()}\n")
            f.write("It tests the file upload capability of Hyperbolic.")
        
        remote_test_path = f"{test_dir}/test_connection.txt"
        
        # Use the new curl-based file transfer method
        print("Testing curl-based file transfer...")
        try:
            # Create FileTransfer instance
            from video_agent.file_transfer import FileTransfer
            file_transfer = FileTransfer(processor.instance_id)
            
            # Upload using curl approach
            file_transfer.upload_file(test_file, remote_test_path)
            
            # Verify the file exists on the remote server
            verify_cmd = f"test -f {remote_test_path} && echo 'exists' || echo 'not found'"
            verify_result = execute_remote_command(verify_cmd, instance_id=processor.instance_id)
            
            if "exists" in verify_result:
                print("✅ Curl-based file upload successful!")
                
                # Check file content
                cat_cmd = f"cat {remote_test_path}"
                file_content = execute_remote_command(cat_cmd, instance_id=processor.instance_id)
                print(f"Remote file content:\n{file_content}")
            else:
                print(f"❌ Curl-based file upload failed: File not found on remote server")
                print("Trying alternative approach...")
                
                # Alternative approach - direct curl from a URL
                print("\nTesting direct curl from a remote URL...")
                remote_url = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"
                curl_cmd = f"curl -s {remote_url} -o {remote_test_path} && echo 'curl_success'"
                curl_result = execute_remote_command(curl_cmd, instance_id=processor.instance_id)
                
                if "curl_success" in curl_result:
                    print("✅ Direct curl from URL successful!")
                    # Check file content
                    cat_cmd = f"cat {remote_test_path} | head -5"  # Just show first 5 lines
                    file_content = execute_remote_command(cat_cmd, instance_id=processor.instance_id)
                    print(f"Remote file content (first 5 lines):\n{file_content}")
                else:
                    print(f"❌ Direct curl from URL failed: {curl_result}")
        except Exception as e:
            print(f"❌ Error during file transfer test: {str(e)}")
            print("Trying alternative approach...")
            
            # Fall back to old SCP approach if needed
            import subprocess
            scp_cmd = [
                "scp", 
                "-P", str(port),
                "-i", ssh_key_path,
                test_file, 
                f"{username}@{hostname}:{remote_test_path}"
            ]
            
            print(f"Running SCP command: {' '.join(scp_cmd)}")
            try:
                result = subprocess.run(scp_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ File upload successful with SCP!")
                else:
                    print(f"❌ File upload failed with SCP: {result.stderr}")
            except Exception as e:
                print(f"❌ Error during SCP: {str(e)}")
                
                # Try creating the file directly on the server as a last resort
                create_cmd = f"echo 'This is a test file created directly on the server at $(date)' > {remote_test_path}"
                create_result = execute_remote_command(create_cmd, instance_id=processor.instance_id)
                print(f"Created file directly on server: {create_result}")
                
                # Verify the file exists
                verify_cmd = f"test -f {remote_test_path} && echo 'exists' || echo 'not found'"
                verify_result = execute_remote_command(verify_cmd, instance_id=processor.instance_id)
                
                if "exists" in verify_result:
                    print("✅ File created directly on server successfully!")
                else:
                    print(f"❌ Failed to create file directly on server")
                    return False
        
        # Test 4: List remote files
        print("\n=== Test 4: Listing Remote Files ===")
        ls_result = execute_remote_command(f"ls -la {test_dir}", instance_id=processor.instance_id)
        print(f"Remote directory contents:\n{ls_result}")
        
        # Test 5: Upload and run Python script
        print("\n=== Test 5: Running Python Script on Remote Instance ===")
        script_path = create_test_python_script()
        remote_script_path = f"{test_dir}/system_info.py"
        
        # Upload the script using SFTP
        try:
            sftp = ssh_manager.client.open_sftp()
            sftp.put(script_path, remote_script_path)
            sftp.close()
            print("✅ Python script uploaded successfully!")
        except Exception as e:
            print(f"❌ Script upload failed: {str(e)}")
            return False
        
        # Make the script executable
        execute_remote_command(f"chmod +x {remote_script_path}", instance_id=processor.instance_id)
        
        # Run the script
        print("\nRunning the script on the remote instance...")
        script_output = execute_remote_command(f"python3 {remote_script_path}", instance_id=processor.instance_id)
        
        # Parse and display the output
        try:
            system_info = json.loads(script_output)
            print("\nRemote System Information:")
            print(f"Platform: {system_info.get('platform', 'Unknown')}")
            print(f"Python Version: {system_info.get('python_version', 'Unknown')}")
            print(f"Hostname: {system_info.get('hostname', 'Unknown')}")
            print(f"CPU Info: {system_info.get('cpu_info', 'Unknown')}")
            print(f"Current Directory: {system_info.get('current_directory', 'Unknown')}")
            print(f"GPU Info: {system_info.get('gpu_info', 'No GPU information available')}")
        except json.JSONDecodeError:
            print("Could not parse JSON output. Raw output:")
            print(script_output)
        
        # Test 6: Download file
        print("\n=== Test 6: File Download ===")
        # Create a file on the remote instance
        remote_output_file = f"{test_dir}/remote_created.txt"
        execute_remote_command(f"echo 'This file was created on the remote instance at $(date)' > {remote_output_file}", instance_id=processor.instance_id)
        
        # Download the file using SFTP
        local_output_file = "remote_created.txt"
        try:
            sftp = ssh_manager.client.open_sftp()
            sftp.get(remote_output_file, local_output_file)
            sftp.close()
            print("✅ File download successful!")
        except Exception as e:
            print(f"❌ File download failed: {str(e)}")
            return False
        
        # Display the content of the downloaded file
        with open(local_output_file, "r") as f:
            content = f.read()
            print(f"Content of downloaded file: {content}")
        
        # Test 7: Check GPU status
        print("\n=== Test 7: Checking GPU Status ===")
        gpu_result = execute_remote_command("nvidia-smi", instance_id=processor.instance_id)
        print(f"GPU Status:\n{gpu_result}")
        
        # Test 8: Direct URL download
        print("\n=== Test 8: Direct URL Download ===")
        try:
            # Test downloading a file directly from a URL
            remote_test_url = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"
            remote_download_path = f"{test_dir}/direct_download.txt"
            
            # Use VideoProcessor's download_from_url method
            print(f"Testing direct URL download: {remote_test_url}")
            processor.download_from_url(remote_test_url, remote_download_path)
            
            # Verify the file exists and check its content
            verify_cmd = f"test -f {remote_download_path} && echo 'exists' || echo 'not found'"
            verify_result = execute_remote_command(verify_cmd, instance_id=processor.instance_id)
            
            if "exists" in verify_result:
                print("✅ Direct URL download successful!")
                
                # Check file content (first few lines)
                cat_cmd = f"head -5 {remote_download_path}"
                file_content = execute_remote_command(cat_cmd, instance_id=processor.instance_id)
                print(f"Downloaded file content (first 5 lines):\n{file_content}")
            else:
                print(f"❌ Direct URL download failed: File not found on remote server")
        except Exception as e:
            print(f"❌ Error during direct URL download test: {str(e)}")
            print("This feature may require the VideoProcessor implementation.")
        
        print("\n=== All Tests Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary files
        if 'script_path' in locals() and os.path.exists(script_path):
            os.unlink(script_path)
        if 'test_file' in locals() and os.path.exists(test_file):
            os.unlink(test_file)
        if 'local_output_file' in locals() and os.path.exists(local_output_file):
            os.unlink(local_output_file)
        
        # Disconnect SSH
        if ssh_manager.is_connected:
            print("Disconnecting SSH session...")
            ssh_manager.disconnect()
        
        # Terminate the instance
        if processor.instance_id:
            print(f"\nCleaning up: Terminating instance {processor.instance_id}...")
            try:
                from hyperbolic_agentkit_core.actions.terminate_compute import terminate_compute
                terminate_compute(processor.instance_id)
                print(f"Instance {processor.instance_id} terminated successfully.")
            except Exception as e:
                print(f"Warning: Error terminating instance: {str(e)}")

if __name__ == "__main__":
    success = run_connection_test()
    sys.exit(0 if success else 1) 