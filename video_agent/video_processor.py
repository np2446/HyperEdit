"""
GPU-accelerated video processing using either local ffmpeg or Hyperbolic's compute infrastructure.
This module provides high-level video processing capabilities with flexible execution options.
"""

import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from hyperbolic_agentkit_core.actions.get_available_gpus import get_available_gpus
from hyperbolic_agentkit_core.actions.rent_compute import rent_compute
from hyperbolic_agentkit_core.actions.get_gpu_status import get_gpu_status
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from hyperbolic_agentkit_core.actions.terminate_compute import terminate_compute
from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh

from .video_models import VideoEditPlan, VideoEditRequest, TransitionType, Scene
from .scene_processor import SceneProcessor
from .local_scene_processor import LocalSceneProcessor
from .file_transfer import FileTransfer

class GPURequirements(BaseModel):
    """GPU requirements for video processing."""
    min_vram_gb: float = 8.0
    gpu_count: int = 1
    preferred_gpu_model: Optional[str] = None

class VideoProcessor:
    """Manages video processing using either local ffmpeg or Hyperbolic's infrastructure."""
    
    def __init__(self, local_mode: bool = False):
        """Initialize video processor.
        
        Args:
            local_mode: Whether to process videos locally using ffmpeg (True) or use Hyperbolic's GPUs (False)
        """
        self.local_mode = local_mode
        self.current_instance: Optional[Dict] = None
        self.instance_id: Optional[str] = None
        self.workspace_dir = "/workspace" if not local_mode else tempfile.mkdtemp(prefix="video_processor_")
        self.file_transfer: Optional[FileTransfer] = None
        self.scene_processor: Optional[SceneProcessor] = None
        self.local_processor: Optional[LocalSceneProcessor] = None
    
    def setup_gpu_environment(self, requirements: GPURequirements) -> None:
        """Set up processing environment based on mode and requirements.
        
        Args:
            requirements: GPU requirements for the video processing task
        """
        if self.local_mode:
            # For local mode, just set up the workspace
            self.local_processor = LocalSceneProcessor(self.workspace_dir)
            return
        
        # Remote mode - set up GPU environment
        gpu_info = json.loads(get_available_gpus())
        
        # Select optimal GPU based on requirements
        selected_gpu = self._select_gpu(gpu_info, requirements)
        if not selected_gpu:
            raise RuntimeError("No suitable GPU found matching requirements")
        
        # Rent the compute instance
        response = json.loads(rent_compute(
            cluster_name=selected_gpu["cluster_name"],
            node_name=selected_gpu["node_id"],
            gpu_count=str(requirements.gpu_count)
        ))
        
        self.instance_id = response["instance"]["id"]
        self.current_instance = response["instance"]
        
        # Wait for instance to be ready
        self._wait_for_instance_ready()
        
        # Initialize file transfer and scene processor
        self.file_transfer = FileTransfer(self.instance_id)
        self.scene_processor = SceneProcessor(self.instance_id, self.workspace_dir)
        
        # Set up the environment
        self._setup_environment()
    
    def _select_gpu(self, available_gpus: Dict, requirements: GPURequirements) -> Optional[Dict]:
        """Select the optimal GPU instance based on requirements."""
        best_match = None
        min_price = float('inf')
        
        for instance in available_gpus.get("instances", []):
            # Skip if not enough GPUs available
            if instance.get("gpus_available", 0) < requirements.gpu_count:
                continue
            
            # Check GPU model if specified
            gpu_model = instance.get("gpu_model", "")
            
            # Estimate VRAM based on GPU model
            gpu_vram = self._estimate_gpu_vram(gpu_model)
            
            if gpu_vram < requirements.min_vram_gb:
                continue
            
            if requirements.preferred_gpu_model and requirements.preferred_gpu_model not in gpu_model:
                continue
            
            # Check price
            price = instance.get("price", float('inf'))
            
            if price < min_price:
                min_price = price
                best_match = {
                    "cluster_name": instance["cluster_name"],
                    "node_id": instance["node_id"],
                    "gpu_model": gpu_model,
                    "gpu_vram": gpu_vram,
                    "price": price,
                    "gpus_available": instance["gpus_available"]
                }
        
        return best_match
    
    def _estimate_gpu_vram(self, gpu_model: str) -> float:
        """Estimate GPU VRAM in GB based on model name."""
        gpu_model = gpu_model.upper()
        if "A100-80GB" in gpu_model:
            return 80.0
        elif "A100" in gpu_model:
            return 40.0
        elif "H100" in gpu_model:
            return 80.0
        elif "A6000" in gpu_model:
            return 48.0
        elif "A5000" in gpu_model:
            return 24.0
        elif "A4000" in gpu_model:
            return 16.0
        elif "V100-32GB" in gpu_model:
            return 32.0
        elif "V100" in gpu_model:
            return 16.0
        else:
            return 8.0  # Conservative default
    
    def _wait_for_instance_ready(self, timeout: int = 300, check_interval: int = 5) -> None:
        """Wait for GPU instance to be ready."""
        if self.local_mode:
            return
            
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("GPU instance failed to start within timeout period")
                
            status = json.loads(get_gpu_status(self.instance_id))
            for instance in status:
                if instance["id"] == self.instance_id:
                    if instance["status"] == "running":
                        return
                    elif instance["status"] == "failed":
                        raise RuntimeError(f"Instance failed to start: {instance.get('error', 'Unknown error')}")
            
            time.sleep(check_interval)
    
    def _setup_environment(self) -> None:
        """Install required packages and setup workspace on GPU instance."""
        if self.local_mode:
            return
            
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
            result = execute_remote_command(self.instance_id, cmd)
            if "error" in result.lower():
                raise RuntimeError(f"Failed to setup environment: {result}")
    
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
            if self.instance_id:
                try:
                    terminate_compute(self.instance_id)
                except Exception as e:
                    print(f"Warning: Failed to terminate instance {self.instance_id}: {str(e)}")
                finally:
                    self.instance_id = None
                    self.current_instance = None
                    self.file_transfer = None
                    self.scene_processor = None 