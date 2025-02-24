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
from langchain.tools import Tool
from moviepy.video.fx.all import colorx, resize, speedx
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

# Local imports
from hyperbolic_agentkit_core.actions import get_available_gpus, rent_compute, terminate_compute

# Constants
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_VRAM_GB = 8
MIN_4K_WIDTH = 3840

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

def select_optimal_gpu(requirements: Dict[str, float]) -> Dict[str, str]:
    """
    Select the optimal GPU configuration from available resources.
    
    This function matches processing requirements with available GPU resources
    on the Hyperbolic platform to find the most cost-effective solution.
    
    Args:
        requirements (Dict[str, float]): Required GPU specifications
            - vram_gb: Minimum VRAM in gigabytes
            - gpu_count: Minimum number of GPUs
    
    Returns:
        Dict[str, str]: Selected GPU configuration:
            - cluster_name: Name of the selected cluster
            - node_name: Name of the selected node
            - gpu_count: Number of GPUs to rent
    
    Selection Criteria:
        - VRAM capacity
        - GPU availability
        - Cost efficiency
        - Processing capabilities
    
    Raises:
        ValueError: If no suitable GPUs are available
    """
    available_gpus = get_available_gpus()
    
    # Parse available GPUs
    gpu_options = []
    current_gpu = {}
    
    for line in available_gpus.split('\n'):
        if line.startswith('Cluster:'):
            if current_gpu:
                gpu_options.append(current_gpu)
            current_gpu = {'cluster_name': line.split(': ')[1]}
        elif line.startswith('Node ID:'):
            current_gpu['node_name'] = line.split(': ')[1]
        elif line.startswith('GPU Model:'):
            current_gpu['model'] = line.split(': ')[1]
        elif line.startswith('Available GPUs:'):
            current_gpu['available'] = int(line.split(': ')[1].split('/')[0])
    
    if current_gpu:
        gpu_options.append(current_gpu)
    
    # Filter suitable GPUs
    suitable_gpus = [
        gpu for gpu in gpu_options
        if gpu['available'] >= requirements['gpu_count']
        and any(vram_str in gpu['model'].lower() for vram_str in [
            '24gb', '32gb', '48gb'
        ] if requirements['vram_gb'] <= int(vram_str[:2]))
    ]
    
    if not suitable_gpus:
        raise ValueError("No suitable GPUs available for the required specifications")
    
    # Select most cost-effective option
    selected_gpu = suitable_gpus[0]  # Could implement more sophisticated selection logic
    
    return {
        'cluster_name': selected_gpu['cluster_name'],
        'node_name': selected_gpu['node_name'],
        'gpu_count': str(requirements['gpu_count'])
    }

def execute_video_edit(plan: VideoEditPlan, request: VideoEditRequest) -> str:
    """
    Execute the video editing plan using selected GPU resources.
    
    Args:
        plan (VideoEditPlan): The video editing plan to execute
        request (VideoEditRequest): The original video editing request
    
    Returns:
        str: Path to the output video file
    
    Raises:
        ValueError: If the plan or request is invalid
        RuntimeError: If GPU resources cannot be allocated
        FileNotFoundError: If input videos are not found
        Exception: For other processing errors
    """
    # Validate inputs
    if not plan.scenes:
        raise ValueError("No scenes defined in the editing plan")
    
    for video_path in request.video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(request.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Select and rent GPU
    try:
        gpu_selection = select_optimal_gpu(plan.estimated_gpu_requirements)
        instance = rent_compute(
            cluster_name=gpu_selection['cluster_name'],
            node_name=gpu_selection['node_name'],
            gpu_count=gpu_selection['gpu_count']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to allocate GPU resources: {str(e)}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load all source videos
            source_videos = []
            for path in request.video_paths:
                try:
                    video = mp.VideoFileClip(path)
                    source_videos.append(video)
                except Exception as e:
                    raise RuntimeError(f"Failed to load video {path}: {str(e)}")
            
            # Process each scene
            scene_clips = []
            for scene in plan.scenes:
                try:
                    # Create clip segments for this scene
                    clip_segments = []
                    for clip_spec in scene.clips:
                        # Extract and process the clip segment
                        source = source_videos[clip_spec.source_index]
                        segment = source.subclip(clip_spec.start_time, clip_spec.end_time)
                        
                        # Apply position and size
                        pos = clip_spec.position
                        segment = segment.resize(width=pos.width, height=pos.height)
                        
                        # Apply all video effects
                        for effect in clip_spec.effects:
                            segment = apply_video_effect(segment, effect)
                        
                        # Apply all audio effects
                        for effect in clip_spec.audio_effects:
                            segment = apply_audio_effect(segment, effect)
                        
                        clip_segments.append(segment)
                    
                    # Create scene composition
                    scene_clip = mp.CompositeVideoClip(
                        clip_segments,
                        size=DEFAULT_RESOLUTION,
                        bg_color=scene.background_color or "black"
                    )
                    
                    # Add captions
                    for caption in scene.captions:
                        txt_clip = create_caption_clip(caption)
                        scene_clip = mp.CompositeVideoClip([scene_clip, txt_clip])
                    
                    # Add transitions
                    if scene.transition_in:
                        scene_clip = apply_transition_effect(scene_clip, scene.transition_in)
                    if scene.transition_out:
                        scene_clip = apply_transition_effect(scene_clip, scene.transition_out)
                    
                    scene_clips.append(scene_clip)
                
                except Exception as e:
                    raise RuntimeError(f"Failed to process scene: {str(e)}")
            
            try:
                # Concatenate all scenes
                final_clip = mp.concatenate_videoclips(scene_clips)
                
                # Write final output
                final_clip.write_videofile(
                    request.output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(temp_dir, 'temp_audio.m4a'),
                    remove_temp=True,
                    threads=4,
                    preset='medium'
                )
            except Exception as e:
                raise RuntimeError(f"Failed to write output video: {str(e)}")
            
            finally:
                # Clean up video clips
                for clip in source_videos + scene_clips:
                    try:
                        clip.close()
                    except:
                        pass  # Ignore cleanup errors

    except Exception as e:
        raise Exception(f"Video processing failed: {str(e)}")

    finally:
        # Always cleanup GPU resources
        if instance:
            try:
                terminate_compute(instance["instance_id"])
            except Exception as e:
                print(f"Warning: Failed to cleanup GPU resources: {str(e)}")

    return f"Video processing completed. Output saved to: {request.output_path}"

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
        "blur": lambda clip, params: clip.fx(mp.vfx.blur, params.get("radius", 1)),
        "color_adjust": lambda clip, params: clip.fx(colorx, 
            params.get("r", 1), params.get("g", 1), params.get("b", 1)),
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
        txt_clip = mp.TextClip(
            caption.text,
            font=caption.style.get("font", "Arial"),
            fontsize=caption.style.get("size", 40),
            color=caption.style.get("color", "white"),
            bg_color=caption.style.get("bg_color", "rgba(0,0,0,0.5)"),
            method='caption'
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
                func=lambda x: execute_video_edit(VideoEditPlan(**json.loads(x["plan"])), 
                                                VideoEditRequest(**json.loads(x["request"]))),
                args_schema=dict
            )
        ]
    except ImportError as e:
        raise ImportError(f"Failed to create video editing tools: {str(e)}") 