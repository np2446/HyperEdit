"""
Tools for integrating video processing capabilities with the AI agent.
This module provides LangChain tools that allow the agent to plan and execute video edits.
"""

import json
from typing import Dict, List, Optional
from langchain.tools import Tool
from pydantic import BaseModel, Field

from .video_processor import VideoProcessor, GPURequirements
from .video_models import (
    VideoEditRequest, VideoEditPlan, Scene, ClipSegment,
    Position, VideoEffect, AudioEffect, Caption,
    VideoEffectType, AudioEffectType, TransitionEffect,
    TransitionType, TextStyle
)

class VideoEditInput(BaseModel):
    """Input schema for video editing requests."""
    video_paths: List[str] = Field(
        ..., 
        description="List of paths to input videos"
    )
    edit_prompt: str = Field(
        ..., 
        description="Natural language description of desired edits"
    )
    output_path: str = Field(
        ..., 
        description="Desired output path for the final video"
    )
    target_duration: Optional[float] = Field(
        None,
        description="Desired duration of final video in seconds"
    )
    style_reference: Optional[str] = Field(
        None,
        description="URL or path to style reference video/image"
    )

def create_video_edit_plan(input_dict: Dict) -> Dict:
    """Create a video editing plan based on the input request.
    
    Args:
        input_dict: Dictionary containing VideoEditInput fields
        
    Returns:
        Dict containing the VideoEditPlan
    """
    # Convert input to request
    request = VideoEditRequest(**input_dict)
    
    # Parse the edit prompt to create scenes
    scenes = []
    prompt_lower = request.edit_prompt.lower()
    
    if "split-screen" in prompt_lower or "comparison" in prompt_lower:
        if len(request.video_paths) < 2:
            raise ValueError("Split-screen effect requires at least 2 videos")
        
        # Create a split-screen scene
        scene = Scene(
            duration=float('inf'),  # Will be determined by video duration
            clips=[
                # Left video
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=float('inf'),
                    position=Position(
                        x=0,
                        y=0,
                        width=0.5,
                        height=1.0
                    )
                ),
                # Right video
                ClipSegment(
                    source_index=1,
                    start_time=0,
                    end_time=float('inf'),
                    position=Position(
                        x=0.5,
                        y=0,
                        width=0.5,
                        height=1.0
                    )
                )
            ]
        )
        scenes.append(scene)
        
    elif "highlight" in prompt_lower or "compilation" in prompt_lower:
        # Create scenes for highlights
        for i, video_path in enumerate(request.video_paths):
            scene = Scene(
                duration=float('inf'),  # Will be determined by video duration
                clips=[
                    ClipSegment(
                        source_index=i,
                        start_time=0,
                        end_time=float('inf'),
                        position=Position(
                            x=0,
                            y=0,
                            width=1.0,
                            height=1.0
                        ),
                        effects=[
                            # Add subtle enhancement
                            VideoEffect(
                                type=VideoEffectType.COLOR_ADJUST,
                                params={"contrast": 1.1, "saturation": 1.1},
                                start_time=0
                            )
                        ],
                        audio_effects=[
                            # Add fade in/out
                            AudioEffect(
                                type=AudioEffectType.FADE,
                                params={"duration": 0.5},
                                start_time=0
                            )
                        ]
                    )
                ],
                transition_in=TransitionEffect(
                    type=TransitionType.FADE,
                    duration=0.5
                ) if i > 0 else None
            )
            scenes.append(scene)
            
    else:
        # Default to simple concatenation
        for i, video_path in enumerate(request.video_paths):
            scene = Scene(
                duration=float('inf'),
                clips=[
                    ClipSegment(
                        source_index=i,
                        start_time=0,
                        end_time=float('inf'),
                        position=Position(
                            x=0,
                            y=0,
                            width=1.0,
                            height=1.0
                        )
                    )
                ],
                transition_in=TransitionEffect(
                    type=TransitionType.FADE,
                    duration=0.5
                ) if i > 0 else None
            )
            scenes.append(scene)
    
    # Create the plan
    plan = VideoEditPlan(
        scenes=scenes,
        estimated_gpu_requirements={
            "min_vram_gb": 8.0,
            "gpu_count": 1
        },
        estimated_duration=5.0  # minutes
    )
    
    return plan.dict()

def execute_video_edit(input_dict: Dict) -> str:
    """Execute a video editing plan.
    
    Args:
        input_dict: Dictionary containing:
            - plan: VideoEditPlan dictionary
            - request: VideoEditRequest dictionary
            
    Returns:
        str: Path to the output video
    """
    # Convert inputs to proper types
    plan = VideoEditPlan(**input_dict["plan"])
    request = VideoEditRequest(**input_dict["request"])
    
    # Initialize video processor
    processor = VideoProcessor()
    
    try:
        # Setup GPU environment
        processor.setup_gpu_environment(GPURequirements(
            min_vram_gb=plan.estimated_gpu_requirements["min_vram_gb"],
            gpu_count=int(plan.estimated_gpu_requirements["gpu_count"])
        ))
        
        # Process video
        output_path = processor.process_video(plan, request)
        return f"Video processing completed successfully. Output saved to: {output_path}"
        
    finally:
        # Ensure cleanup
        processor.cleanup()

def create_video_editing_tools() -> List[Tool]:
    """Create and return video editing tools for the agent.
    
    Returns:
        List[Tool]: List of LangChain tools for video editing
    """
    return [
        Tool(
            name="plan_video_edit",
            description="""
            Create a detailed plan for video editing based on input videos and editing requirements.
            Input should be a JSON string containing:
            - video_paths: List of paths to input videos
            - edit_prompt: Natural language description of desired edits
            - output_path: Desired output path for the final video
            - target_duration: (optional) Desired duration in seconds
            - style_reference: (optional) URL or path to style reference
            """,
            func=lambda x: json.dumps(create_video_edit_plan(json.loads(x)), indent=2)
        ),
        Tool(
            name="execute_video_edit",
            description="""
            Execute a video editing plan using Hyperbolic GPU resources.
            Input should be a JSON string containing:
            - plan: The video editing plan (from plan_video_edit)
            - request: The original video edit request
            """,
            func=lambda x: execute_video_edit(json.loads(x))
        )
    ] 