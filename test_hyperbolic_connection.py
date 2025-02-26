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
        
        # Test 1: Create remote directories
        print("\n=== Test 1: Creating Remote Directories ===")
        remote_test_dir = f"{processor.workspace_dir}/test_connection"
        cmd_result = execute_remote_command(f"mkdir -p {remote_test_dir}", instance_id=processor.instance_id)
        print(f"Directory creation result: {cmd_result}")
        
        # Test 2: File upload
        print("\n=== Test 2: File Upload ===")
        test_file = "test_connection.txt"
        with open(test_file, "w") as f:
            f.write(f"This is a test file created at {time.ctime()}\n")
            f.write("It tests the file upload capability of Hyperbolic.")
        
        remote_test_path = f"{remote_test_dir}/test_connection.txt"
        processor.file_transfer.upload_file(test_file, remote_test_path)
        print("✅ File upload successful!")
        
        # Test 3: List remote files
        print("\n=== Test 3: Listing Remote Files ===")
        remote_files = processor.file_transfer.list_remote_files(remote_test_dir)
        print(f"Remote files: {remote_files}")
        
        # Test 4: Upload and run Python script
        print("\n=== Test 4: Running Python Script on Remote Instance ===")
        script_path = create_test_python_script()
        remote_script_path = f"{remote_test_dir}/system_info.py"
        
        # Upload the script
        processor.file_transfer.upload_file(script_path, remote_script_path)
        print("✅ Python script uploaded successfully!")
        
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
        
        # Test 5: Download file
        print("\n=== Test 5: File Download ===")
        # Create a file on the remote instance
        remote_output_file = f"{remote_test_dir}/remote_created.txt"
        execute_remote_command(f"echo 'This file was created on the remote instance at $(date)' > {remote_output_file}", instance_id=processor.instance_id)
        
        # Download the file
        local_output_file = "remote_created.txt"
        processor.file_transfer.download_file(remote_output_file, local_output_file)
        print("✅ File download successful!")
        
        # Display the content of the downloaded file
        with open(local_output_file, "r") as f:
            content = f.read()
            print(f"Content of downloaded file: {content}")
        
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