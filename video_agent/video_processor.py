"""
GPU-accelerated video processing using Hyperbolic's compute infrastructure.
This module provides high-level video processing capabilities while efficiently
managing GPU resources through Hyperbolic's marketplace.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from hyperbolic_agentkit_core.actions.get_available_gpus import get_available_gpus
from hyperbolic_agentkit_core.actions.rent_compute import rent_compute
from hyperbolic_agentkit_core.actions.get_gpu_status import get_gpu_status
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from hyperbolic_agentkit_core.actions.terminate_compute import terminate_compute
from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh

from .video_models import VideoEditPlan, VideoEditRequest
from .scene_processor import SceneProcessor
from .file_transfer import FileTransfer

class GPURequirements(BaseModel):
    """GPU requirements for video processing."""
    min_vram_gb: float = 8.0
    gpu_count: int = 1
    preferred_gpu_model: Optional[str] = None

class VideoProcessor:
    """Manages GPU-accelerated video processing using Hyperbolic's infrastructure."""
    
    def __init__(self):
        self.current_instance: Optional[Dict] = None
        self.instance_id: Optional[str] = None
        self.workspace_dir = "/workspace"
        self.file_transfer: Optional[FileTransfer] = None
        self.scene_processor: Optional[SceneProcessor] = None
        
    def setup_gpu_environment(self, requirements: GPURequirements) -> None:
        """Set up GPU environment based on video requirements.
        
        Args:
            requirements: GPU requirements for the video processing task
        """
        # Get available GPUs
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
        """Execute video processing on GPU.
        
        Args:
            edit_plan: The video editing plan to execute
            request: The original video edit request
            
        Returns:
            str: Path to the processed output video
        """
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
            
            # Concatenate scenes if multiple
            if len(scene_outputs) > 1:
                final_remote_path = f"{self.workspace_dir}/final.mp4"
                self.scene_processor._concatenate_scenes(scene_outputs, final_remote_path)
            else:
                final_remote_path = scene_outputs[0]
            
            # Download result
            self.file_transfer.download_file(final_remote_path, request.output_path)
            
            return request.output_path
            
        except Exception as e:
            raise RuntimeError(f"Video processing failed: {str(e)}")
    
    def cleanup(self) -> None:
        """Release GPU resources and clean up temporary files."""
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