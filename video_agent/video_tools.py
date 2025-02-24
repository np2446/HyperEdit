"""
Video Editing Toolkit for Hyperbolic AgentKit

This module provides tools for AI-powered video editing using Hyperbolic's GPU infrastructure.
It enables automated video processing with capabilities including:
- Multi-clip video assembly
- AI-powered enhancements
- Automated captioning
- GPU resource management
- Intelligent workload distribution

Key Components:
1. VideoEditRequest - Schema for video editing requests
2. VideoEditPlan - Schema for planned editing operations
3. GPU requirement estimation
4. Optimal GPU selection
5. Video processing execution

Dependencies:
    - ffmpeg: For video processing and metadata extraction
    - Hyperbolic AgentKit: For GPU resource management
    - Pydantic: For data validation
    - moviepy: For video editing operations
    - torch: For GPU-accelerated processing
    - PIL: For image processing
    - OpenCV: For video processing
"""

# Standard library imports
import json
import os
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import moviepy.editor as mp
import numpy as np
import torch
import requests
from langchain.tools import Tool
from moviepy.video.fx.all import colorx, resize, speedx
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
import cv2

# Local imports
from hyperbolic_agentkit_core.actions.get_available_gpus import get_available_gpus
from hyperbolic_agentkit_core.actions.rent_compute import rent_compute
from hyperbolic_agentkit_core.actions.terminate_compute import terminate_compute
from hyperbolic_agentkit_core.actions.utils import get_api_key
from hyperbolic_agentkit_core.actions.get_current_balance import get_current_balance
from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh

# Constants
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_VRAM_GB = 8  # Default VRAM requirement in GB
MIN_4K_WIDTH = 3840  # Minimum width for 4K resolution

class Position(BaseModel):
    """Defines position and size of a clip in the composition."""
    x: Union[float, str] = Field(..., description="X position (0-1 or 'left', 'center', 'right')")
    y: Union[float, str] = Field(..., description="Y position (0-1 or 'top', 'center', 'bottom')")
    width: float = Field(..., description="Width as fraction of composition (0-1)")
    height: float = Field(..., description="Height as fraction of composition (0-1)")
    z_index: int = Field(default=0, description="Stack order")

class TransitionEffect(BaseModel):
    """Defines a transition between clips or scenes."""
    type: str = Field(..., description="Type of transition (fade, wipe, dissolve, etc.)")
    duration: float = Field(..., description="Duration in seconds")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class VideoEffect(BaseModel):
    """Defines a video effect to apply."""
    type: str = Field(..., description="Type of effect")
    params: Dict[str, Any] = Field(default_factory=dict, description="Effect parameters")
    start_time: float = Field(..., description="When effect starts (seconds)")
    end_time: Optional[float] = Field(None, description="When effect ends (seconds)")

class AudioEffect(BaseModel):
    """Defines an audio effect or modification."""
    type: str = Field(..., description="Type of audio effect")
    params: Dict[str, Any] = Field(default_factory=dict, description="Effect parameters")
    start_time: float = Field(..., description="When effect starts (seconds)")
    end_time: Optional[float] = Field(None, description="When effect ends (seconds)")

class Caption(BaseModel):
    """Defines a caption or text overlay."""
    text: str = Field(..., description="Caption text")
    start_time: float = Field(..., description="Start time (seconds)")
    end_time: float = Field(..., description="End time (seconds)")
    position: Position = Field(..., description="Caption position and size")
    style: Dict[str, Any] = Field(default_factory=dict, description="Text styling")

class ClipSegment(BaseModel):
    """Defines a segment of a source clip to use."""
    source_index: int = Field(..., description="Index of source video in video_paths")
    start_time: float = Field(..., description="Start time in source video (seconds)")
    end_time: float = Field(..., description="End time in source video (seconds)")
    position: Position = Field(..., description="Position in composition")
    effects: List[VideoEffect] = Field(default_factory=list, description="Effects to apply")
    audio_effects: List[AudioEffect] = Field(default_factory=list, description="Audio effects")

class Scene(BaseModel):
    """Defines a complete scene composition."""
    duration: float = Field(..., description="Scene duration in seconds")
    clips: List[ClipSegment] = Field(..., description="Clips in this scene")
    captions: List[Caption] = Field(default_factory=list, description="Scene captions")
    background_color: Optional[str] = Field(None, description="Background color if needed")
    transition_in: Optional[TransitionEffect] = Field(None, description="Transition from previous scene")
    transition_out: Optional[TransitionEffect] = Field(None, description="Transition to next scene")

class VideoEditRequest(BaseModel):
    """Schema for video editing request parameters."""
    video_paths: List[str] = Field(
        ..., 
        description="List of paths to input videos",
        example=["/path/to/video1.mp4", "/path/to/video2.mp4"]
    )
    edit_prompt: str = Field(
        ..., 
        description="Natural language description of desired edits",
        example="Create a split-screen comparison of both gameplay videos, with captions highlighting key moments"
    )
    output_path: str = Field(
        ..., 
        description="Desired output path for the final video",
        example="/path/to/output.mp4"
    )
    target_duration: Optional[float] = Field(
        None,
        description="Desired duration of final video in seconds"
    )
    style_reference: Optional[str] = Field(
        None,
        description="URL or description of style reference"
    )

class VideoEditPlan(BaseModel):
    """Schema for the video editing execution plan."""
    scenes: List[Scene] = Field(
        ..., 
        description="Ordered list of scenes to create"
    )
    estimated_gpu_requirements: Dict[str, float] = Field(
        ..., 
        description="Required GPU specifications"
    )
    estimated_duration: float = Field(
        ..., 
        description="Estimated processing time in minutes"
    )

def estimate_gpu_requirements(video_paths: List[str], steps: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Estimate GPU requirements based on video specifications and editing operations.
    
    Args:
        video_paths (List[str]): Paths to input video files
        steps (List[Dict[str, str]]): List of planned editing operations
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - vram_gb: Required GPU VRAM in gigabytes
            - gpu_count: Number of GPUs needed
    
    Raises:
        FileNotFoundError: If any video file does not exist
        subprocess.CalledProcessError: If ffprobe fails to analyze a video
        json.JSONDecodeError: If ffprobe output is not valid JSON
    """
    if not video_paths:
        raise ValueError("No video paths provided")

    total_duration = 0
    max_resolution = (0, 0)
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video metadata using ffprobe
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            
            # Calculate total duration
            duration = float(metadata['format']['duration'])
            total_duration += duration
            
            # Find maximum resolution
            for stream in metadata['streams']:
                if stream['codec_type'] == 'video':
                    width = int(stream.get('width', 0))
                    height = int(stream.get('height', 0))
                    max_resolution = max(max_resolution, (width, height))
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(
                e.returncode,
                e.cmd,
                f"Failed to analyze video: {video_path}"
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid video metadata for {video_path}: {str(e)}")
    
    # Base requirements
    requirements = {
        "vram_gb": DEFAULT_VRAM_GB,
        "gpu_count": 1
    }
    
    # Adjust based on resolution
    if max_resolution[0] >= MIN_4K_WIDTH:  # 4K or higher
        requirements["vram_gb"] = 16
        requirements["gpu_count"] = 2
    
    # Adjust based on editing steps
    for step in steps:
        step_type = step.get("type", "").lower()
        if "ai_enhancement" in step_type or "upscale" in step_type:
            requirements["vram_gb"] = max(requirements["vram_gb"], 24)
            requirements["gpu_count"] = max(requirements["gpu_count"], 2)
        elif "stabilization" in step_type:
            requirements["vram_gb"] = max(requirements["vram_gb"], 12)
    
    return requirements

def create_video_edit_plan(request: VideoEditRequest) -> VideoEditPlan:
    """
    Create a detailed plan for video editing based on the request.
    
    Args:
        request (VideoEditRequest): The video editing request containing input videos,
            edit prompt, and output specifications.
    
    Returns:
        VideoEditPlan: A structured plan containing scenes, GPU requirements,
            and estimated processing duration.
    
    Raises:
        ValueError: If the request is invalid or videos cannot be processed
        FileNotFoundError: If any input video file does not exist
        subprocess.CalledProcessError: If ffprobe fails to analyze a video
    """
    if not request.video_paths:
        raise ValueError("No input videos provided")

    # Get video metadata for all input videos
    video_metadata = []
    for video_path in request.video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            video_metadata.append(metadata)
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(
                e.returncode,
                e.cmd,
                f"Failed to analyze video: {video_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid video metadata for {video_path}: {str(e)}")

    # Initialize scenes list
    scenes = []
    prompt_lower = request.edit_prompt.lower()

    # Parse the edit prompt to create scenes
    try:
        if "split-screen" in prompt_lower:
            if len(request.video_paths) < 2:
                raise ValueError("Split-screen effect requires at least 2 videos")

            # Get duration from the first video
            duration = float(video_metadata[0]['format']['duration'])
            
            # Create a split-screen scene
            scene = Scene(
                duration=duration,
                clips=[
                    ClipSegment(
                        source_index=0,
                        start_time=0,
                        end_time=duration,
                        position=Position(x=0, y=0, width=0.5, height=1, z_index=0)
                    ),
                    ClipSegment(
                        source_index=1,
                        start_time=0,
                        end_time=min(duration, float(video_metadata[1]['format']['duration'])),
                        position=Position(x=0.5, y=0, width=0.5, height=1, z_index=0)
                    )
                ],
                captions=[],
                transition_in=TransitionEffect(type="fade", duration=1.0, params={})
            )
            scenes.append(scene)
        else:
            # Default to single video scene if no specific effect requested
            for idx, metadata in enumerate(video_metadata):
                duration = float(metadata['format']['duration'])
                scene = Scene(
                    duration=duration,
                    clips=[
                        ClipSegment(
                            source_index=idx,
                            start_time=0,
                            end_time=duration,
                            position=Position(x=0, y=0, width=1, height=1, z_index=0)
                        )
                    ],
                    captions=[],
                    transition_in=TransitionEffect(type="fade", duration=1.0, params={})
                    if idx > 0 else None
                )
                scenes.append(scene)

    except (KeyError, ValueError) as e:
        raise ValueError(f"Failed to create edit plan: {str(e)}")

    # Calculate GPU requirements based on video specs and editing complexity
    steps = [{"type": "basic_edit"}]  # Base step
    if "enhance" in prompt_lower or "upscale" in prompt_lower:
        steps.append({"type": "ai_enhancement"})
    if "stabilize" in prompt_lower:
        steps.append({"type": "stabilization"})

    gpu_requirements = estimate_gpu_requirements(request.video_paths, steps)
    
    # Estimate processing duration (rough estimate)
    total_duration = sum(scene.duration for scene in scenes)
    processing_factor = 1.5  # Base processing time multiplier
    if len(steps) > 1:
        processing_factor *= len(steps)
    
    return VideoEditPlan(
        scenes=scenes,
        estimated_gpu_requirements=gpu_requirements,
        estimated_duration=total_duration * processing_factor
    )

def select_optimal_gpu(requirements):
    """
    Select the optimal GPU instance based on requirements.
    
    Args:
        requirements (Dict[str, float]): Dictionary containing GPU requirements
            - vram_gb (float): Required VRAM in GB
            - gpu_count (float): Number of GPUs needed
        
    Returns:
        dict: Selected GPU instance details
        
    Raises:
        ValueError: If no suitable GPUs are found or if balance is insufficient
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get current balance
    api_key = get_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        # Get current balance
        balance_url = "https://api.hyperbolic.xyz/billing/get_current_balance"
        balance_response = requests.get(balance_url, headers=headers)
        balance_response.raise_for_status()
        balance_data = balance_response.json()
        
        # Credits are already in dollars (e.g. 100 credits = $1.00)
        balance_usd = balance_data.get("credits", 0) / 100
        logger.debug(f"Current balance: ${balance_usd:.2f}")
        
    except Exception as e:
        logger.warning(f"Could not parse balance from response: {str(e)}")
        balance_usd = 0
    
    # Get purchase history to check for any pending transactions
    try:
        history_url = "https://api.hyperbolic.xyz/billing/purchase_history"
        history_response = requests.get(history_url, headers=headers)
        history_response.raise_for_status()
    except Exception as e:
        logger.warning(f"Could not get purchase history: {str(e)}")

    url = "https://api.hyperbolic.xyz/v1/marketplace"
    data = {"filters": {}}
    response = requests.post(url, headers=headers, json=data)
    data = response.json()
    
    if "instances" not in data:
        raise ValueError("No instances found in marketplace response")
        
    suitable_nodes = []
    lowest_price = float('inf')
    
    for node in data["instances"]:
        logger.debug(f"\nEvaluating node: {node.get('id')}")
        
        # Check node status
        status = node.get("status")
        logger.debug(f"Node status: {status}")
        if status != "node_ready":
            continue
            
        # Check if node is reserved
        is_reserved = node.get("reserved", True)
        logger.debug(f"Node reserved: {is_reserved}")
        if is_reserved:
            logger.debug("Skipping reserved node")
            continue
            
        # Get GPU information
        gpus = node.get("hardware", {}).get("gpus", [])
        if not gpus:
            continue
            
        logger.debug(f"Found {len(gpus)} GPUs in hardware specs")
        
        # Check if GPUs meet memory requirements
        gpus_meeting_requirements = 0
        for gpu in gpus:
            memory_mb = gpu.get("ram", 0)
            model = gpu.get("model", "Unknown")
            logger.debug(f"Evaluating GPU {model} - Memory: {memory_mb}MB, Required: {requirements['vram_gb'] * 1024}MB")
            if memory_mb >= requirements['vram_gb'] * 1024:
                gpus_meeting_requirements += 1
                
        logger.debug(f"Found {gpus_meeting_requirements} GPUs meeting memory requirements")
        
        # Get total and reserved GPU counts
        total_gpus = node.get("gpus_total", 0)
        reserved_gpus = node.get("gpus_reserved", 0)
        available_gpus = total_gpus - reserved_gpus
        
        logger.debug(f"Total GPUs: {total_gpus}, Reserved: {reserved_gpus}, Available: {available_gpus}")
        
        if available_gpus < requirements['gpu_count']:
            logger.debug(f"Not enough available GPUs (need {requirements['gpu_count']}, have {available_gpus})")
            continue
            
        # Get price
        price_per_hour = node.get("pricing", {}).get("price", {}).get("amount", 0) / 100
        logger.debug(f"Node meets requirements, price: ${price_per_hour}/hour")
        
        # Update lowest price seen
        lowest_price = min(lowest_price, price_per_hour)
        
        # Check if price is within our balance
        if price_per_hour > balance_usd:
            logger.debug(f"Node price ${price_per_hour}/hour exceeds balance ${balance_usd}")
            continue
        
        suitable_nodes.append({
            "cluster_name": node.get("cluster_name"),
            "node_name": node.get("id"),  # Use node ID as the node name
            "gpu_count": str(int(requirements['gpu_count'])),  # Convert to string for API
            "available_gpus": available_gpus,
            "price": price_per_hour,
            "gpu_model": gpus[0].get("model"),
            "status": status
        })
    
    if not suitable_nodes:
        error_msg = "No suitable GPUs found"
        if lowest_price != float('inf'):
            error_msg = f"Insufficient balance (${balance_usd}) for available nodes (minimum ${lowest_price}/hour required)"
        raise ValueError(error_msg)
        
    # Sort by price and available GPUs
    suitable_nodes.sort(key=lambda x: (x["price"], -x["available_gpus"]))
    selected_node = suitable_nodes[0]
    logger.debug(f"Selected node: {selected_node}")
    
    return {
        "cluster_name": selected_node["cluster_name"],
        "node_name": selected_node["node_name"],
        "gpu_count": selected_node["gpu_count"]
    }

def execute_video_edit(plan: VideoEditPlan, request: VideoEditRequest, local_mode: bool = False) -> str:
    """
    Execute the video editing plan using available GPU resources.
    
    Args:
        plan (VideoEditPlan): The editing plan to execute
        request (VideoEditRequest): Original edit request
        local_mode (bool): Whether to run locally without GPU selection
        
    Returns:
        str: Path to the output video file
        
    Raises:
        RuntimeError: If GPU allocation fails or editing encounters an error
    """
    import logging
    import time
    import tempfile
    import os
    import json
    import subprocess
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    instance = None
    last_error = None
    max_retries = 3
    base_delay = 2  # Base delay in seconds
    
    try:
        if not local_mode:
            # Get list of suitable GPUs
            gpu_selection = select_optimal_gpu(plan.estimated_gpu_requirements)
            
            # Try each suitable node until one works or we run out of retries
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to rent compute, attempt {attempt + 1}/{max_retries}")
                    logger.info(f"Trying node: {gpu_selection}")
                    
                    instance_response = rent_compute(
                        cluster_name=gpu_selection['cluster_name'],
                        node_name=gpu_selection['node_name'],
                        gpu_count=str(int(plan.estimated_gpu_requirements['gpu_count']))
                    )
                    
                    # Parse instance details
                    instance = json.loads(instance_response)
                    instance_id = instance.get('instance_id')
                    if not instance_id:
                        raise ValueError("No instance_id in response")
                    
                    logger.info(f"Successfully created instance: {instance_id}")
                    
                    # Wait for instance to be ready
                    max_wait = 300  # 5 minutes
                    wait_start = time.time()
                    while True:
                        status_response = get_gpu_status()
                        for gpu_instance in status_response.get('instances', []):
                            if gpu_instance.get('id') == instance_id:
                                if gpu_instance.get('status') == 'ready':
                                    # Get SSH details
                                    ssh_host = gpu_instance.get('ssh_host')
                                    ssh_user = gpu_instance.get('ssh_user')
                                    ssh_port = gpu_instance.get('ssh_port', 22)
                                    if not all([ssh_host, ssh_user]):
                                        raise ValueError("Missing SSH connection details")
                                    break
                        else:
                            if time.time() - wait_start > max_wait:
                                raise TimeoutError("Instance failed to become ready")
                            time.sleep(5)
                            continue
                        break
                    
                    # Connect to instance via SSH
                    ssh_result = connect_ssh(
                        host=ssh_host,
                        username=ssh_user,
                        port=ssh_port
                    )
                    if "Error" in ssh_result:
                        raise RuntimeError(f"SSH connection failed: {ssh_result}")
                    
                    # Install dependencies
                    deps_commands = [
                        "sudo apt-get update",
                        "sudo apt-get install -y python3-pip ffmpeg",
                        "pip3 install moviepy numpy torch opencv-python pillow",
                    ]
                    for cmd in deps_commands:
                        result = execute_remote_command(cmd)
                        if "Error" in result:
                            raise RuntimeError(f"Failed to install dependencies: {result}")
                    
                    # Create work directory
                    work_dir = "/tmp/video_edit"
                    execute_remote_command(f"mkdir -p {work_dir}")
                    
                    # Transfer input videos
                    for idx, video_path in enumerate(request.video_paths):
                        remote_path = f"{work_dir}/input_{idx}.mp4"
                        scp_cmd = f"scp -P {ssh_port} {video_path} {ssh_user}@{ssh_host}:{remote_path}"
                        subprocess.run(scp_cmd, shell=True, check=True)
                    
                    # Create remote processing script
                    script_content = """
import moviepy.editor as mp
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import sys

# Constants
DEFAULT_RESOLUTION = (1920, 1080)

def apply_video_effect(clip, effect):
    effect_functions = {
        "blur": lambda clip, params: clip.fl_image(
            lambda frame: cv2.GaussianBlur(
                frame,
                (int(params.get("radius", 1)) * 2 + 1,) * 2,
                params.get("radius", 1)
            )
        ),
        "color_adjust": lambda clip, params: clip.fl_image(
            lambda frame: np.clip(
                frame * np.array([
                    params.get("r", 1.0),
                    params.get("g", 1.0),
                    params.get("b", 1.0)
                ])[None, None, :],
                0, 255
            ).astype('uint8')
        ),
        "speed": lambda clip, params: clip.fx(speedx, params.get("factor", 1)),
    }
    
    if effect.type not in effect_functions:
        raise ValueError(f"Unsupported video effect: {effect.type}")
    
    try:
        return effect_functions[effect.type](clip, effect.params)
    except Exception as e:
        raise RuntimeError(f"Failed to apply video effect {effect.type}: {str(e)}")

def apply_audio_effect(clip, effect):
    audio_effect_functions = {
        "volume": lambda clip, params: clip.volumex(params.get("factor", 1)),
        "fadeout": lambda clip, params: clip.audio_fadeout(params.get("duration", 1)),
        "fadein": lambda clip, params: clip.audio_fadein(params.get("duration", 1)),
    }
    
    if effect.type not in audio_effect_functions:
        raise ValueError(f"Unsupported audio effect: {effect.type}")
    
    try:
        return audio_effect_functions[effect.type](clip, effect.params)
    except Exception as e:
        raise RuntimeError(f"Failed to apply audio effect {effect.type}: {str(e)}")

def create_caption_clip(caption):
    try:
        txt_clip = mp.TextClip(
            caption.text,
            fontsize=caption.style.get("size", 40),
            color=caption.style.get("color", "white"),
            method='label'
        )
        
        pos = caption.position
        if isinstance(pos.x, str):
            x_pos = {"left": 0, "center": 0.5, "right": 1}[pos.x]
        else:
            x_pos = pos.x * DEFAULT_RESOLUTION[0]
            
        if isinstance(pos.y, str):
            y_pos = {"top": 0, "center": 0.5, "bottom": 1}[pos.y]
        else:
            y_pos = pos.y * DEFAULT_RESOLUTION[1]
        
        txt_clip = txt_clip.set_position((x_pos, y_pos))
        return txt_clip.set_duration(caption.end_time - caption.start_time)
    
    except Exception as e:
        raise RuntimeError(f"Failed to create caption: {str(e)}")

def apply_transition_effect(clip, transition):
    transition_functions = {
        "fade": lambda clip, params: clip.crossfadein(params.get("duration", 1)),
        "wipe": lambda clip, params: create_wipe_transition(clip, params),
    }
    
    if transition.type not in transition_functions:
        raise ValueError(f"Unsupported transition effect: {transition.type}")
    
    try:
        return transition_functions[transition.type](clip, transition.params)
    except Exception as e:
        raise RuntimeError(f"Failed to apply transition {transition.type}: {str(e)}")

def create_wipe_transition(clip, params):
    direction = params.get("direction", "left")
    duration = params.get("duration", 1.0)
    
    if direction not in ["left", "right", "up", "down"]:
        raise ValueError(f"Unsupported wipe direction: {direction}")
    
    try:
        return clip.crossfadein(duration)
    except Exception as e:
        raise RuntimeError(f"Failed to create wipe transition: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--plan_json', required=True)
    parser.add_argument('--video_paths_json', required=True)
    args = parser.parse_args()

    # Load plan and video paths
    with open(args.plan_json) as f:
        plan = json.load(f)
    with open(args.video_paths_json) as f:
        video_paths = json.load(f)

    # Load videos
    source_clips = []
    for i, _ in enumerate(video_paths):
        clip = mp.VideoFileClip(f"{args.work_dir}/input_{i}.mp4")
        source_clips.append(clip)

    # Process scenes
    final_clips = []
    for scene in plan['scenes']:
        scene_clips = []
        for clip_segment in scene['clips']:
            source_clip = source_clips[clip_segment['source_index']]
            clip = source_clip.subclip(clip_segment['start_time'], clip_segment['end_time'])
            
            for effect in clip_segment.get('effects', []):
                clip = apply_video_effect(clip, effect)
            
            for effect in clip_segment.get('audio_effects', []):
                clip = apply_audio_effect(clip, effect)
            
            pos = clip_segment['position']
            clip = clip.resize(width=pos['width'] * DEFAULT_RESOLUTION[0],
                             height=pos['height'] * DEFAULT_RESOLUTION[1])
            clip = clip.set_position((pos['x'] * DEFAULT_RESOLUTION[0],
                                    pos['y'] * DEFAULT_RESOLUTION[1]))
            scene_clips.append(clip)
        
        if len(scene_clips) > 1:
            scene_clip = mp.CompositeVideoClip(scene_clips, size=DEFAULT_RESOLUTION)
        else:
            scene_clip = scene_clips[0]
        
        scene_clip = scene_clip.set_duration(scene['duration'])
        
        if scene.get('transition_in'):
            scene_clip = apply_transition_effect(scene_clip, scene['transition_in'])
        if scene.get('transition_out'):
            scene_clip = apply_transition_effect(scene_clip, scene['transition_out'])
        
        caption_clips = []
        for caption in scene.get('captions', []):
            txt_clip = create_caption_clip(caption)
            caption_clips.append(txt_clip)
        
        if caption_clips:
            scene_clip = mp.CompositeVideoClip([scene_clip] + caption_clips)
        
        final_clips.append(scene_clip)

    # Concatenate all scenes
    final_video = mp.concatenate_videoclips(final_clips)

    # Write output file
    final_video.write_videofile(args.output_path)

    # Clean up
    final_video.close()
    for clip in source_clips:
        clip.close()
"""
                    remote_script = f"{work_dir}/process.py"
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
                        f.write(script_content)
                        f.flush()
                        scp_cmd = f"scp -P {ssh_port} {f.name} {ssh_user}@{ssh_host}:{remote_script}"
                        subprocess.run(scp_cmd, shell=True, check=True)
                    
                    # Create plan and video paths JSON files
                    plan_json = f"{work_dir}/plan.json"
                    video_paths_json = f"{work_dir}/video_paths.json"
                    
                    # Write plan JSON
                    plan_dict = {
                        "scenes": [scene.dict() for scene in plan.scenes]
                    }
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                        json.dump(plan_dict, f)
                        f.flush()
                        scp_cmd = f"scp -P {ssh_port} {f.name} {ssh_user}@{ssh_host}:{plan_json}"
                        subprocess.run(scp_cmd, shell=True, check=True)
                    
                    # Write video paths JSON
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                        json.dump(request.video_paths, f)
                        f.flush()
                        scp_cmd = f"scp -P {ssh_port} {f.name} {ssh_user}@{ssh_host}:{video_paths_json}"
                        subprocess.run(scp_cmd, shell=True, check=True)
                    
                    # Run processing
                    remote_output = f"{work_dir}/output.mp4"
                    cmd = f"python3 {remote_script} --work_dir {work_dir} --output_path {remote_output} --plan_json {plan_json} --video_paths_json {video_paths_json}"
                    result = execute_remote_command(cmd)
                    if "Error" in result:
                        raise RuntimeError(f"Video processing failed: {result}")
                    
                    # Transfer output back
                    scp_cmd = f"scp -P {ssh_port} {ssh_user}@{ssh_host}:{remote_output} {request.output_path}"
                    subprocess.run(scp_cmd, shell=True, check=True)
                    
                    # Clean up remote files
                    execute_remote_command(f"rm -rf {work_dir}")
                    
                    return request.output_path
                    
                except Exception as e:
                    last_error = e
                    logger.error(f"Failed to process video on attempt {attempt + 1}: {str(e)}")
                    
                    # Exponential backoff before next attempt
                    if attempt < max_retries - 1:  # Don't sleep after last attempt
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Waiting {delay} seconds before next attempt...")
                        time.sleep(delay)
            
            if not local_mode:
                error_msg = f"Failed to process video after {max_retries} attempts."
                if last_error:
                    error_msg += f" Last error: {str(last_error)}"
                raise RuntimeError(error_msg)

        # Local mode processing
        source_clips = []
        for video_path in request.video_paths:
            try:
                clip = mp.VideoFileClip(video_path)
                source_clips.append(clip)
            except Exception as e:
                raise RuntimeError(f"Failed to load video {video_path}: {str(e)}")

        # Process each scene
        final_clips = []
        for scene in plan.scenes:
            # Process clips in the scene
            scene_clips = []
            for clip_segment in scene.clips:
                # Get source clip
                source_clip = source_clips[clip_segment.source_index]
                
                # Cut the segment
                clip = source_clip.subclip(clip_segment.start_time, clip_segment.end_time)
                
                # Apply video effects
                for effect in clip_segment.effects:
                    clip = apply_video_effect(clip, effect)
                
                # Apply audio effects
                for effect in clip_segment.audio_effects:
                    clip = apply_audio_effect(clip, effect)
                
                # Position the clip
                pos = clip_segment.position
                clip = clip.resize(width=pos.width * DEFAULT_RESOLUTION[0], 
                                 height=pos.height * DEFAULT_RESOLUTION[1])
                clip = clip.set_position((pos.x * DEFAULT_RESOLUTION[0], 
                                        pos.y * DEFAULT_RESOLUTION[1]))
                
                scene_clips.append(clip)
            
            # Composite scene clips
            if len(scene_clips) > 1:
                scene_clip = mp.CompositeVideoClip(scene_clips, 
                                                 size=DEFAULT_RESOLUTION)
            else:
                scene_clip = scene_clips[0]
            
            # Apply scene duration
            scene_clip = scene_clip.set_duration(scene.duration)
            
            # Apply transitions
            if scene.transition_in:
                scene_clip = apply_transition_effect(scene_clip, scene.transition_in)
            if scene.transition_out:
                scene_clip = apply_transition_effect(scene_clip, scene.transition_out)
            
            # Add captions
            caption_clips = []
            for caption in scene.captions:
                try:
                    txt_clip = create_caption_clip(caption)
                    caption_clips.append(txt_clip)
                except Exception as e:
                    logger.warning(f"Failed to create caption: {str(e)}")
            
            if caption_clips:
                scene_clip = mp.CompositeVideoClip([scene_clip] + caption_clips)
            
            final_clips.append(scene_clip)
        
        # Concatenate all scenes
        final_video = mp.concatenate_videoclips(final_clips)
        
        # Write output file
        final_video.write_videofile(request.output_path)
        
        # Clean up
        final_video.close()
        for clip in source_clips:
            clip.close()
        
        return request.output_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to execute video edit: {str(e)}")
    finally:
        # Clean up instance if it was created
        if instance and not local_mode:
            try:
                terminate_compute(instance.get('instance_id'))
            except Exception as e:
                logger.error(f"Failed to clean up instance: {str(e)}")

def apply_video_effect(clip: mp.VideoClip, effect: VideoEffect) -> mp.VideoClip:
    """
    Apply a video effect to a clip.
    
    Args:
        clip (mp.VideoClip): The video clip to modify
        effect (VideoEffect): The effect to apply
    
    Returns:
        mp.VideoClip: The modified video clip
    
    Raises:
        ValueError: If the effect type is not supported
    """
    effect_functions = {
        "blur": lambda clip, params: clip.fl_image(
            lambda frame: cv2.GaussianBlur(
                frame,
                (int(params.get("radius", 1)) * 2 + 1,) * 2,
                params.get("radius", 1)
            )
        ),
        "color_adjust": lambda clip, params: clip.fl_image(
            lambda frame: np.clip(
                frame * np.array([
                    params.get("r", 1.0),
                    params.get("g", 1.0),
                    params.get("b", 1.0)
                ])[None, None, :],
                0, 255
            ).astype('uint8')
        ),
        "speed": lambda clip, params: clip.fx(speedx, params.get("factor", 1)),
    }
    
    if effect.type not in effect_functions:
        raise ValueError(f"Unsupported video effect: {effect.type}")
    
    try:
        return effect_functions[effect.type](clip, effect.params)
    except Exception as e:
        raise RuntimeError(f"Failed to apply video effect {effect.type}: {str(e)}")

def apply_audio_effect(clip: mp.VideoClip, effect: AudioEffect) -> mp.VideoClip:
    """
    Apply an audio effect to a clip.
    
    Args:
        clip (mp.VideoClip): The video clip to modify
        effect (AudioEffect): The audio effect to apply
    
    Returns:
        mp.VideoClip: The modified video clip
    
    Raises:
        ValueError: If the effect type is not supported
    """
    audio_effect_functions = {
        "volume": lambda clip, params: clip.volumex(params.get("factor", 1)),
        "fadeout": lambda clip, params: clip.audio_fadeout(params.get("duration", 1)),
        "fadein": lambda clip, params: clip.audio_fadein(params.get("duration", 1)),
    }
    
    if effect.type not in audio_effect_functions:
        raise ValueError(f"Unsupported audio effect: {effect.type}")
    
    try:
        return audio_effect_functions[effect.type](clip, effect.params)
    except Exception as e:
        raise RuntimeError(f"Failed to apply audio effect {effect.type}: {str(e)}")

def create_caption_clip(caption: Caption) -> mp.TextClip:
    """
    Create a styled caption clip.
    
    Args:
        caption (Caption): The caption configuration
    
    Returns:
        mp.TextClip: The created text clip
    
    Raises:
        RuntimeError: If the caption creation fails
    """
    try:
        # Create a simple text clip without ImageMagick
        txt_clip = mp.TextClip(
            caption.text,
            fontsize=caption.style.get("size", 40),
            color=caption.style.get("color", "white"),
            method='label'  # Use label method instead of caption
        )
        
        # Position the caption
        pos = caption.position
        if isinstance(pos.x, str):
            x_pos = {"left": 0, "center": 0.5, "right": 1}[pos.x]
        else:
            x_pos = pos.x * DEFAULT_RESOLUTION[0]
            
        if isinstance(pos.y, str):
            y_pos = {"top": 0, "center": 0.5, "bottom": 1}[pos.y]
        else:
            y_pos = pos.y * DEFAULT_RESOLUTION[1]
        
        txt_clip = txt_clip.set_position((x_pos, y_pos))
        return txt_clip.set_duration(caption.end_time - caption.start_time)
    
    except Exception as e:
        raise RuntimeError(f"Failed to create caption: {str(e)}")

def apply_transition_effect(clip: mp.VideoClip, transition: TransitionEffect) -> mp.VideoClip:
    """
    Apply a transition effect to a clip.
    
    Args:
        clip (mp.VideoClip): The video clip to modify
        transition (TransitionEffect): The transition to apply
    
    Returns:
        mp.VideoClip: The modified video clip
    
    Raises:
        ValueError: If the transition type is not supported
    """
    transition_functions = {
        "fade": lambda clip, params: clip.crossfadein(params.get("duration", 1)),
        "wipe": lambda clip, params: create_wipe_transition(clip, params),
    }
    
    if transition.type not in transition_functions:
        raise ValueError(f"Unsupported transition effect: {transition.type}")
    
    try:
        return transition_functions[transition.type](clip, transition.params)
    except Exception as e:
        raise RuntimeError(f"Failed to apply transition {transition.type}: {str(e)}")

def create_wipe_transition(clip: mp.VideoClip, params: Dict[str, Any]) -> mp.VideoClip:
    """
    Create a wipe transition effect.
    
    Args:
        clip (mp.VideoClip): The video clip to modify
        params (Dict[str, Any]): Transition parameters
    
    Returns:
        mp.VideoClip: The modified video clip
    
    Raises:
        ValueError: If required parameters are missing
    """
    direction = params.get("direction", "left")
    duration = params.get("duration", 1.0)
    
    if direction not in ["left", "right", "up", "down"]:
        raise ValueError(f"Unsupported wipe direction: {direction}")
    
    try:
        # Implementation of wipe transition
        # This is a placeholder - actual implementation would depend on moviepy capabilities
        return clip.crossfadein(duration)  # Fallback to crossfade for now
    except Exception as e:
        raise RuntimeError(f"Failed to create wipe transition: {str(e)}")

def create_video_editing_tools() -> List[Tool]:
    """
    Create and return video editing tools for the agent.
    
    Returns:
        List[Tool]: List of LangChain tools for video editing:
            1. plan_video_edit: Creates editing plans
            2. execute_video_edit: Executes editing plans
    
    Raises:
        ImportError: If required dependencies are not available
    """
    try:
        return [
            Tool(
                name="plan_video_edit",
                description="Create a detailed plan for video editing based on input videos and editing requirements",
                func=lambda x: create_video_edit_plan(VideoEditRequest(**json.loads(x))),
                args_schema=VideoEditRequest
            ),
            Tool(
                name="execute_video_edit",
                description="Execute a video editing plan using Hyperbolic GPU resources",
                func=lambda x: execute_video_edit(
                    VideoEditPlan(**json.loads(x["plan"])), 
                    VideoEditRequest(**json.loads(x["request"]))
                ),
                args_schema=dict
            )
        ]
    except ImportError as e:
        raise ImportError(f"Failed to create video editing tools: {str(e)}") 