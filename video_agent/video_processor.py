"""
GPU-accelerated video processing using either local ffmpeg or Hyperbolic's compute infrastructure.
This module provides high-level video processing capabilities with flexible execution options.
"""

import json
import os
import time
import tempfile
import signal
import sys
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dotenv import load_dotenv

from pydantic import BaseModel

from hyperbolic_agentkit_core.actions.get_available_gpus import get_available_gpus
from hyperbolic_agentkit_core.actions.rent_compute import rent_compute
from hyperbolic_agentkit_core.actions.get_gpu_status import get_gpu_status
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from hyperbolic_agentkit_core.actions.terminate_compute import terminate_compute
from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager

from .video_models import VideoEditPlan, VideoEditRequest, TransitionType, Scene
from .scene_processor import SceneProcessor
from .local_scene_processor import LocalSceneProcessor
from .file_transfer import FileTransfer

class GPURequirements(BaseModel):
    """GPU requirements for video processing."""
    gpu_type: str = "H100"  # Type of GPU (e.g., "H100", "A100", "RTX 4090")
    num_gpus: int = 1       # Number of GPUs to request
    min_vram_gb: float = 8.0  # Minimum VRAM in GB
    disk_size: int = 10     # Disk size in GB
    memory: int = 16        # RAM in GB
    preferred_gpu_model: Optional[str] = None  # For backward compatibility

class VideoProcessor:
    """Manages video processing using either local ffmpeg or Hyperbolic's infrastructure."""
    
    # Class variable to track active instances for cleanup
    active_instances: Set[str] = set()
    
    def __init__(self, local_mode: bool = False):
        """Initialize video processor.
        
        Args:
            local_mode: Whether to process videos locally using ffmpeg (True) or use Hyperbolic's GPUs (False)
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.local_mode = local_mode
        self.current_instance: Optional[Dict] = None
        self.instance_id: Optional[str] = None
        self.workspace_dir = "/workspace" if not local_mode else tempfile.mkdtemp(prefix="video_processor_")
        self.file_transfer: Optional[FileTransfer] = None
        self.scene_processor: Optional[SceneProcessor] = None
        self.local_processor: Optional[LocalSceneProcessor] = None
        
        # Register cleanup handlers for unexpected termination
        if not local_mode:
            # Only register once
            if not hasattr(VideoProcessor, '_cleanup_handlers_registered'):
                signal.signal(signal.SIGINT, self._cleanup_on_interrupt)
                signal.signal(signal.SIGTERM, self._cleanup_on_interrupt)
                atexit.register(self._cleanup_on_exit)
                VideoProcessor._cleanup_handlers_registered = True
    
    @classmethod
    def _cleanup_on_interrupt(cls, signum, frame):
        """Clean up GPU instances on interrupt signals."""
        print(f"\nReceived interrupt signal {signum}. Cleaning up GPU instances...")
        cls._terminate_all_instances()
        sys.exit(1)
    
    @classmethod
    def _cleanup_on_exit(cls):
        """Clean up GPU instances on normal exit."""
        cls._terminate_all_instances()
    
    @classmethod
    def _terminate_all_instances(cls):
        """Terminate all active GPU instances."""
        if cls.active_instances:
            print(f"Terminating {len(cls.active_instances)} active GPU instances...")
            for instance_id in list(cls.active_instances):
                try:
                    print(f"Terminating instance {instance_id}...")
                    terminate_compute(instance_id)
                    cls.active_instances.remove(instance_id)
                    print(f"Instance {instance_id} terminated successfully.")
                except Exception as e:
                    print(f"Error terminating instance {instance_id}: {str(e)}")
    
    def setup_gpu_environment(self, requirements: GPURequirements) -> None:
        """Set up processing environment based on mode and requirements.
        
        Args:
            requirements: GPU requirements for the video processing task
        """
        if self.local_mode:
            # For local mode, just set up the workspace
            self.local_processor = LocalSceneProcessor(self.workspace_dir)
            return
        
        # Get available GPUs
        gpu_info_str = get_available_gpus()
        print(f"\nAvailable GPUs:\n{gpu_info_str}")
        
        # Parse GPU info string to find suitable GPU
        selected_gpu = None
        current_cluster = None
        clusters = []
        
        for line in gpu_info_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('Available GPU Options:') or line.startswith('---'):
                continue
                
            if line.startswith('Cluster:'):
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = {'cluster_name': line.split(': ')[1]}
            elif line.startswith('Node ID:'):
                current_cluster['node_id'] = line.split(': ')[1]
            elif line.startswith('GPU Model:'):
                current_cluster['gpu_model'] = line.split(': ')[1]
            elif line.startswith('Available GPUs:'):
                available, total = line.split(': ')[1].split('/')
                current_cluster['gpus_available'] = int(available)
                current_cluster['total_gpus'] = int(total)
            elif line.startswith('Price:'):
                price_str = line.split('$')[1].split('/')[0]
                current_cluster['price'] = float(price_str)
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Sort clusters by price (cheapest first)
        clusters.sort(key=lambda x: x.get('price', float('inf')))
        
        # First try to find a cheap GPU (under $1/hour) that meets requirements
        for cluster in clusters:
            if (cluster.get('price', float('inf')) < 1.0 and 
                cluster['gpus_available'] > 0):
                # Check if it meets the GPU type requirement (if specified)
                if not requirements.gpu_type or requirements.gpu_type in cluster['gpu_model']:
                    selected_gpu = cluster
                    print(f"Selected affordable GPU: {cluster['cluster_name']} at ${cluster['price']}/hour")
                    break
        
        # If no cheap GPU found, try any GPU that meets the type requirement
        if not selected_gpu:
            for cluster in clusters:
                if requirements.gpu_type in cluster['gpu_model'] and cluster['gpus_available'] > 0:
                    selected_gpu = cluster
                    break
        
        # If still no GPU found, just take the cheapest available one
        if not selected_gpu:
            for cluster in clusters:
                if cluster['gpus_available'] > 0:
                    selected_gpu = cluster
                    break
        
        if not selected_gpu:
            raise RuntimeError(f"No available GPUs found matching requirements: {requirements}")
        
        print(f"\nSelected GPU cluster: {selected_gpu['cluster_name']}")
        print(f"Node ID: {selected_gpu['node_id']}")
        print(f"GPU Model: {selected_gpu['gpu_model']}")
        print(f"Price: ${selected_gpu['price']}/hour")
        
        # Remember the rental time to check for newly created instances
        self.rental_time = time.time()
        
        # Rent the compute instance
        response = rent_compute(
            cluster_name=selected_gpu['cluster_name'],
            node_name=selected_gpu['node_id'],
            gpu_count=str(requirements.num_gpus)
        )
        
        print(f"\nRent compute response:\n{json.dumps(response, indent=2)}")
        
        # Parse the response to get instance ID
        try:
            if isinstance(response, str):
                response = json.loads(response)
            
            # Check different possible response formats
            if 'instance' in response and 'id' in response['instance']:
                # Format: {"instance": {"id": "..."}}
                self.instance_id = response["instance"]["id"]
                self.current_instance = response["instance"]
            elif 'status' in response and response['status'] == 'success':
                # Format: {"status": "success"}
                # Need to find the instance ID from status
                print("Instance ID not found in response. Looking for recently created instances...")
                self._find_and_connect_to_instance(selected_gpu['cluster_name'])
                return
            else:
                # Try to find any instance ID in the response
                found = False
                for key, value in response.items():
                    if isinstance(value, dict) and 'id' in value:
                        self.instance_id = value['id']
                        self.current_instance = value
                        found = True
                        break
                
                if not found:
                    print("Instance ID not found in response. Looking for recently created instances...")
                    self._find_and_connect_to_instance(selected_gpu['cluster_name'])
                    return
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print("Looking for recently created instances...")
            self._find_and_connect_to_instance(selected_gpu['cluster_name'])
            return
        
        # Add to active instances for cleanup
        self.__class__.active_instances.add(self.instance_id)
        
        # Wait for instance to be ready
        self._wait_for_instance_ready()
    
    def _find_and_connect_to_instance(self, cluster_name, max_wait=600, check_interval=15):
        """Find a recently created instance and connect to it."""
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait:
                raise TimeoutError("Failed to find a running instance within timeout period")
            
            elapsed = int(time.time() - start_time)
            print(f"Looking for recently created instances... (elapsed time: {elapsed}s)")
            
            try:
                # Get instance status
                status_data = get_gpu_status()
                print(f"GPU status response type: {type(status_data)}")
                
                # Handle different response formats
                if isinstance(status_data, str):
                    try:
                        status = json.loads(status_data)
                    except json.JSONDecodeError:
                        print(f"Error parsing status data as JSON: {status_data[:100]}...")
                        time.sleep(check_interval)
                        continue
                elif isinstance(status_data, dict):
                    status = status_data
                else:
                    print(f"Unexpected status data type: {type(status_data)}")
                    time.sleep(check_interval)
                    continue
                
                # Debug output
                print(f"Status data keys: {list(status.keys())}")
                
                # Extract instances from different possible formats
                instances = []
                if 'instances' in status and isinstance(status['instances'], list):
                    instances = status['instances']
                elif 'data' in status and isinstance(status['data'], list):
                    instances = status['data']
                elif 'data' in status and isinstance(status['data'], dict) and 'instances' in status['data']:
                    instances = status['data']['instances']
                
                if not instances:
                    print("No instances found yet. Waiting...")
                else:
                    print(f"Found {len(instances)} instances. Looking for a suitable one...")
                    for i, instance in enumerate(instances):
                        print(f"Instance {i+1} data: {json.dumps(instance, indent=2)}")
                
                # Look for instances created after we initiated the rental
                # First try our cluster
                for instance in instances:
                    # Check if instance is in our cluster
                    instance_cluster = instance.get('cluster_name', '')
                    if instance_cluster and instance_cluster == cluster_name:
                        self.instance_id = instance.get('id')
                        self.current_instance = instance
                        # Add to active instances for cleanup
                        self.__class__.active_instances.add(self.instance_id)
                        print(f"Found instance {self.instance_id} in our cluster {cluster_name}")
                        self._wait_for_instance_ready()
                        return
                
                # If no instance in our cluster, try any recently created instance
                if instances:
                    # Sort by creation time if available
                    if all('created_at' in instance for instance in instances):
                        instances.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                    
                    # Take the first one (most recently created if sorted)
                    instance = instances[0]
                    self.instance_id = instance.get('id')
                    self.current_instance = instance
                    # Add to active instances for cleanup
                    self.__class__.active_instances.add(self.instance_id)
                    print(f"Found instance {self.instance_id} (may not be in our cluster)")
                    self._wait_for_instance_ready()
                    return
            
            except Exception as e:
                print(f"Error finding instance: {str(e)}")
                import traceback
                traceback.print_exc()
            
            print(f"No suitable instance found. Waiting {check_interval} seconds...")
            time.sleep(check_interval)
    
    def _wait_for_instance_ready(self, timeout: int = 600, check_interval: int = 15) -> None:
        """Wait for GPU instance to be ready for SSH connections."""
        if self.local_mode:
            return
            
        print(f"Waiting for GPU instance {self.instance_id} to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        # Wait full initial period with no connection attempts
        initial_wait = 30  # Increased from 20 to 30 seconds
        print(f"Waiting {initial_wait}s before attempting first SSH connection...")
        time.sleep(initial_wait)
        
        # Get SSH key path from environment variable or try multiple options
        ssh_key_path = self._find_ssh_key()
        print(f"Using SSH key: {ssh_key_path}")
        
        # Get SSH key password from environment
        ssh_key_password = os.environ.get('SSH_KEY_PASSWORD')
        if ssh_key_password:
            print("Found SSH key password in environment variables")
        
        # Now actively try to establish SSH connection
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"GPU instance {self.instance_id} failed to become SSH-ready within {timeout}s")
                
            elapsed = int(time.time() - start_time)
            print(f"Attempting SSH connection... (elapsed time: {elapsed}s)")
            
            try:
                # Get instance status
                status_data = get_gpu_status()
                if isinstance(status_data, str):
                    try:
                        status = json.loads(status_data)
                    except json.JSONDecodeError:
                        print(f"Error parsing status data as JSON: {status_data[:100]}...")
                        time.sleep(check_interval)
                        continue
                else:
                    status = status_data
                
                # Find our instance
                instance = None
                instances = []
                
                # Extract instances from different possible formats
                if 'instances' in status and isinstance(status['instances'], list):
                    instances = status['instances']
                elif 'data' in status and isinstance(status['data'], list):
                    instances = status['data']
                elif 'data' in status and isinstance(status['data'], dict) and 'instances' in status['data']:
                    instances = status['data']['instances']
                
                for inst in instances:
                    if inst.get('id') == self.instance_id:
                        instance = inst
                        break
                
                if not instance:
                    print(f"Instance {self.instance_id} not found in status data - retrying...")
                    time.sleep(check_interval)
                    continue
                
                # Print all available fields for debugging
                print(f"Instance data keys: {list(instance.keys())}")
                
                # Try different ways to extract IP address
                ip_info = self._extract_ip_address(instance)
                
                if not ip_info:
                    print(f"Instance {self.instance_id} has no IP address yet in any recognizable field - retrying...")
                    time.sleep(check_interval)
                    continue
                
                ip_address, port = ip_info
                print(f"Attempting to connect to instance at IP: {ip_address}, port: {port}")
                
                # First ensure any previous connections are closed
                from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager
                if ssh_manager.is_connected:
                    try:
                        ssh_manager.disconnect()
                        print("Closed previous SSH connection")
                    except Exception as e:
                        print(f"Warning: Error closing previous SSH connection: {e}")
                
                # Connect using the instance IP and username
                ssh_result = connect_ssh(
                    host=ip_address,
                    username="ubuntu",  # Default username for Hyperbolic instances
                    private_key_path=ssh_key_path,
                    port=port,
                    key_password=ssh_key_password
                )
                
                print(f"SSH connection result: {ssh_result}")
                
                # Check if connection was successful
                if "Successfully connected" in ssh_result:
                    print(f"SSH connection successful! Instance {self.instance_id} is ready.")
                    
                    # Initialize file transfer and scene processor
                    from .file_transfer import FileTransfer
                    from .scene_processor import SceneProcessor
                    
                    self.file_transfer = FileTransfer(self.instance_id)
                    self.scene_processor = SceneProcessor(self.instance_id, self.workspace_dir)
                    
                    # Set up the environment
                    self._setup_environment()
                    return
                else:
                    print(f"SSH connection failed: {ssh_result}")
                    
                    # If connection failed, try with alternative username
                    alt_username = "root"
                    print(f"Trying alternative username: {alt_username}")
                    ssh_result = connect_ssh(
                        host=ip_address,
                        username=alt_username,
                        private_key_path=ssh_key_path,
                        port=port,
                        key_password=ssh_key_password
                    )
                    
                    print(f"SSH connection result with {alt_username}: {ssh_result}")
                    
                    if "Successfully connected" in ssh_result:
                        print(f"SSH connection successful with {alt_username}! Instance {self.instance_id} is ready.")
                        
                        # Initialize file transfer and scene processor
                        from .file_transfer import FileTransfer
                        from .scene_processor import SceneProcessor
                        
                        self.file_transfer = FileTransfer(self.instance_id)
                        self.scene_processor = SceneProcessor(self.instance_id, self.workspace_dir)
                        
                        # Set up the environment
                        self._setup_environment()
                        return
            except Exception as e:
                print(f"SSH connection attempt failed: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"Retrying SSH connection in {check_interval} seconds...")
            time.sleep(check_interval)
    
    def _find_ssh_key(self) -> str:
        """Find a suitable SSH key for connecting to the instance."""
        # Get SSH key path from environment variable
        ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH')
        if not ssh_key_path:
            print("SSH_PRIVATE_KEY_PATH environment variable not found, using default path")
            # Use a configurable default path from environment or fall back to ~/.ssh/id_rsa
            default_ssh_path = os.environ.get('DEFAULT_SSH_KEY_PATH', '~/.ssh/id_rsa')
            ssh_key_path = os.path.expanduser(default_ssh_path)
        
        # Check if the key file exists
        if not os.path.exists(ssh_key_path):
            print(f"WARNING: SSH key file not found at {ssh_key_path}")
            # Try to find an alternative key
            # First check for Hyperbolic-specific keys
            hyperbolic_keys = ['hyperbolic', 'hyperbolic_key', 'hyperbolic_pem', 'hyperbolic_unencrypted']
            for key_name in hyperbolic_keys:
                alt_path = os.path.expanduser(f"~/.ssh/{key_name}")
                if os.path.exists(alt_path):
                    print(f"Found Hyperbolic SSH key at {alt_path}")
                    return alt_path
            
            # Then try standard keys
            standard_keys = ['id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa']
            for key_name in standard_keys:
                alt_path = os.path.expanduser(f"~/.ssh/{key_name}")
                if os.path.exists(alt_path):
                    print(f"Found standard SSH key at {alt_path}")
                    return alt_path
            
            raise FileNotFoundError(f"No valid SSH key found. Please set SSH_PRIVATE_KEY_PATH or create a key at ~/.ssh/id_rsa")
        
        return ssh_key_path
    
    def _extract_ip_address(self, instance: Dict) -> Optional[Tuple[str, int]]:
        """Extract IP address and port from instance data using multiple methods.
        
        Returns:
            Tuple of (ip_address, port) or None if not found
        """
        default_port = 22
        
        # 1. Check regular fields
        for field in ['public_ip', 'ip', 'ip_address', 'hostname', 'address']:
            if field in instance and instance[field]:
                ip_address = instance[field]
                print(f"Found IP address in field '{field}': {ip_address}")
                return (ip_address, default_port)
        
        # 2. Check sshCommand if available
        if 'sshCommand' in instance and instance['sshCommand']:
            ssh_cmd = instance['sshCommand']
            print(f"Found sshCommand: {ssh_cmd}")
            # Try to extract hostname from ssh command (format: "ssh user@hostname -p port")
            import re
            ip_match = re.search(r'@([\w.-]+)', ssh_cmd)
            port_match = re.search(r'-p\s+(\d+)', ssh_cmd)
            
            if ip_match:
                ip_address = ip_match.group(1)
                port = int(port_match.group(1)) if port_match else default_port
                print(f"Extracted IP address from sshCommand: {ip_address}, port: {port}")
                return (ip_address, port)
        
        # 3. Check nested 'instance' field if it exists
        if 'instance' in instance and isinstance(instance['instance'], dict):
            nested_instance = instance['instance']
            print(f"Found nested instance data, keys: {list(nested_instance.keys())}")
            for field in ['public_ip', 'ip', 'ip_address', 'hostname', 'address']:
                if field in nested_instance and nested_instance[field]:
                    ip_address = nested_instance[field]
                    print(f"Found IP address in nested instance.{field}: {ip_address}")
                    return (ip_address, default_port)
        
        # 4. Check for any field that looks like an IP address
        for key, value in instance.items():
            if isinstance(value, str) and re.match(r'^\d+\.\d+\.\d+\.\d+$', value):
                print(f"Found IP-like string in field '{key}': {value}")
                return (value, default_port)
        
        return None
    
    def _setup_environment(self) -> None:
        """Install required packages and setup workspace on GPU instance."""
        if self.local_mode:
            return
        
        # Check if we have sudo access
        sudo_check = execute_remote_command("sudo -n true && echo 'sudo_ok' || echo 'sudo_fail'")
        has_sudo = "sudo_ok" in sudo_check
        
        # If we don't have sudo access, use a user-writable directory instead of /workspace
        if not has_sudo and self.workspace_dir.startswith("/workspace"):
            user_home = execute_remote_command("echo $HOME").strip()
            self.workspace_dir = f"{user_home}/workspace"
            print(f"No sudo access. Using user workspace directory: {self.workspace_dir}")
        
        # Create workspace directory
        execute_remote_command(f"mkdir -p {self.workspace_dir}")
        
        # Only run system-level commands if we have sudo access
        if has_sudo:
            setup_commands = [
                # Update package lists
                "sudo apt-get update",
                
                # Install system dependencies
                "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg python3-pip imagemagick",
                
                # Configure ImageMagick policy to allow PDF operations (needed for some text effects)
                "sudo sed -i 's/rights=\"none\" pattern=\"PDF\"/rights=\"read|write\" pattern=\"PDF\"/' /etc/ImageMagick-6/policy.xml"
            ]
            
            for cmd in setup_commands:
                try:
                    result = execute_remote_command(cmd)
        setup_commands = [
            # Update package lists
            "apt-get update",
            
            # Install system dependencies
            "DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg python3-pip imagemagick",
            
            # Create workspace directory
            f"mkdir -p {self.workspace_dir}",
            
            # Install Python packages
            "pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip3 install --no-cache-dir opencv-python-headless moviepy numpy",
            
            # Configure ImageMagick policy to allow PDF operations (needed for some text effects)
            "sed -i 's/rights=\"none\" pattern=\"PDF\"/rights=\"read|write\" pattern=\"PDF\"/' /etc/ImageMagick-6/policy.xml"
        ]
        
        for cmd in setup_commands:
            try:
                result = execute_remote_command(self.instance_id, cmd)
                if "error" in result.lower():
                    raise RuntimeError(f"Failed to setup environment: {result}")
            except Exception as e:
                print(f"Warning: Command '{cmd}' failed: {e}")
                print("Continuing with setup process...")
    
    def process_video(self, edit_plan: VideoEditPlan, request: VideoEditRequest) -> str:
        """Execute video processing.
        
        Args:
            edit_plan: The video editing plan to execute
            request: The original video edit request
            
        Returns:
            str: Path to the processed output video
        """
        if self.local_mode:
            if not self.local_processor:
                raise RuntimeError("Local processor not initialized. Call setup_gpu_environment first.")
            
            # Process each scene
            scene_outputs = []
            for i, scene in enumerate(edit_plan.scenes):
                output_path = os.path.join(self.workspace_dir, f"scene_{i}.mp4")
                self.local_processor.process_scene(scene, request.video_paths, output_path)
                scene_outputs.append(output_path)
            
            # Apply transitions between scenes
            if len(scene_outputs) > 1:
                final_path = os.path.join(self.workspace_dir, "final.mp4")
                self._apply_transitions(scene_outputs, edit_plan.scenes, final_path)
                os.replace(final_path, request.output_path)
            else:
                os.replace(scene_outputs[0], request.output_path)
            
            return request.output_path
            
        else:
            if not self.current_instance or not self.file_transfer or not self.scene_processor:
                raise RuntimeError("GPU environment not set up. Call setup_gpu_environment first.")
            
            try:
                # Upload source videos
                remote_paths = []
                for i, path in enumerate(request.video_paths):
                    remote_path = f"{self.workspace_dir}/source_{i}{Path(path).suffix}"
                    self.file_transfer.upload_file(path, remote_path)
                    remote_paths.append(remote_path)
                
                # Process each scene
                scene_outputs = []
                for i, scene in enumerate(edit_plan.scenes):
                    output_path = f"{self.workspace_dir}/scene_{i}.mp4"
                    self.scene_processor.process_scene(scene, remote_paths, output_path)
                    scene_outputs.append(output_path)
                
                # Apply transitions between scenes
                if len(scene_outputs) > 1:
                    final_remote_path = f"{self.workspace_dir}/final.mp4"
                    self._apply_transitions(scene_outputs, edit_plan.scenes, final_remote_path)
                else:
                    final_remote_path = scene_outputs[0]
                
                # Download result
                self.file_transfer.download_file(final_remote_path, request.output_path)
                
                return request.output_path
                
            except Exception as e:
                raise RuntimeError(f"Video processing failed: {str(e)}")
    
    def _apply_transitions(self, scene_outputs: List[str], scenes: List[Scene], output_path: str) -> None:
        """Apply transitions between scenes.
        
        Args:
            scene_outputs: List of processed scene video paths
            scenes: List of scenes with transition information
            output_path: Path to save the final video
        """
        # Create filter complex for transitions
        filter_complex = []
        inputs = []
        
        # Add input files
        for i, scene_path in enumerate(scene_outputs):
            inputs.append(f"-i {scene_path}")
        
        # Create transitions
        for i in range(len(scenes)):
            if i == 0:
                # First scene
                filter_complex.append(f"[0]format=yuv420p[v0]")
                last_output = "v0"
            else:
                # Get transition info
                prev_scene = scenes[i-1]
                curr_scene = scenes[i]
                
                # Default to fade if no transition specified
                transition_type = (
                    prev_scene.transition_out.type if prev_scene.transition_out
                    else curr_scene.transition_in.type if curr_scene.transition_in
                    else TransitionType.FADE
                )
                
                # Get transition duration
                duration = (
                    prev_scene.transition_out.duration if prev_scene.transition_out
                    else curr_scene.transition_in.duration if curr_scene.transition_in
                    else 1.0
                )
                
                # Add transition
                if transition_type == TransitionType.FADE:
                    filter_complex.append(
                        f"[{i}]format=yuv420p[v{i}];"
                        f"[{last_output}][v{i}]xfade=transition=fade:duration={duration}:offset={i*5-duration}[vt{i}]"
                    )
                    last_output = f"vt{i}"
                else:
                    # For other transition types, just concatenate for now
                    filter_complex.append(
                        f"[{i}]format=yuv420p[v{i}];"
                        f"[{last_output}][v{i}]concat=n=2:v=1:a=0[vt{i}]"
                    )
                    last_output = f"vt{i}"
        
        # Build and execute ffmpeg command
        filter_str = ';'.join(filter_complex)
        cmd = (
            f"ffmpeg {' '.join(inputs)} "
            f"-filter_complex '{filter_str}' "
            f"-map '[{last_output}]' "
            f"-c:v libx264 -preset medium {output_path}"
        )
        
        if self.local_mode:
            self.local_processor._run_command(cmd)
        else:
            execute_remote_command(self.instance_id, cmd)
    
    def cleanup(self) -> None:
        """Release resources and clean up temporary files."""
        if self.local_mode:
            if os.path.exists(self.workspace_dir):
                import shutil
                shutil.rmtree(self.workspace_dir)
            self.local_processor = None
        else:
            # Close SSH connection using the ssh_manager
            try:
                if ssh_manager.is_connected:
                    try:
                        ssh_manager.disconnect()
                        print("SSH connection closed.")
                    except Exception as e:
                        print(f"Warning: Error disconnecting SSH: {str(e)}")
            except Exception as e:
                print(f"Warning: Error accessing SSH manager: {str(e)}")
            
            # Make sure to reset any stored SSH client references
            if hasattr(self, 'ssh_client'):
                self.ssh_client = None
            
            if self.instance_id:
                try:
                    print(f"Terminating instance {self.instance_id}...")
                    terminate_compute(self.instance_id)
                    print(f"Instance {self.instance_id} terminated successfully.")
                    
                    # Remove from active instances
                    if self.instance_id in self.__class__.active_instances:
                        self.__class__.active_instances.remove(self.instance_id)
                except Exception as e:
                    print(f"Warning: Failed to terminate instance {self.instance_id}: {str(e)}")
                finally:
                    self.instance_id = None
                    self.current_instance = None
                    self.file_transfer = None
                    self.scene_processor = None 