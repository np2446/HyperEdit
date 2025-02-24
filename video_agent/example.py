"""
Example usage of the GPU-accelerated video processing system.
This script demonstrates how to create and execute video editing tasks.
"""

from pathlib import Path
from typing import List

from .video_processor import VideoProcessor, GPURequirements
from .video_models import (
    VideoEditRequest, VideoEditPlan, Scene, ClipSegment,
    Position, VideoEffect, AudioEffect, Caption,
    VideoEffectType, AudioEffectType, TransitionEffect,
    TransitionType, TextStyle
)

def create_split_screen_comparison(
    video1_path: str,
    video2_path: str,
    output_path: str,
    title: str = "Video Comparison"
) -> None:
    """Create a split-screen comparison of two videos.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path for output video
        title: Optional title for the comparison
    """
    # Create video edit request
    request = VideoEditRequest(
        video_paths=[video1_path, video2_path],
        edit_prompt="Create a split-screen comparison",
        output_path=output_path
    )
    
    # Create scene with split-screen layout
    scene = Scene(
        duration=float('inf'),  # Will be limited by video duration
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
        ],
        captions=[
            # Title
            Caption(
                text=title,
                start_time=0,
                end_time=float('inf'),
                position=Position(
                    x='center',
                    y=0.05,
                    width=0.8,
                    height=0.1
                ),
                style=TextStyle(
                    font_size=48,
                    bold=True
                )
            )
        ]
    )
    
    # Create edit plan
    plan = VideoEditPlan(
        scenes=[scene],
        estimated_gpu_requirements={
            "min_vram_gb": 8.0,
            "gpu_count": 1
        },
        estimated_duration=5.0  # minutes
    )
    
    # Initialize video processor
    processor = VideoProcessor()
    
    try:
        # Setup GPU environment
        processor.setup_gpu_environment(GPURequirements(
            min_vram_gb=8.0,
            gpu_count=1
        ))
        
        # Process video
        output_path = processor.process_video(plan, request)
        print(f"Video comparison created successfully: {output_path}")
        
    finally:
        # Ensure cleanup
        processor.cleanup()

def create_highlight_reel(
    video_paths: List[str],
    highlights: List[tuple[int, float, float]],  # (video_index, start_time, end_time)
    output_path: str,
    title: str = "Highlight Reel"
) -> None:
    """Create a highlight reel from multiple video clips.
    
    Args:
        video_paths: List of source video paths
        highlights: List of (video_index, start_time, end_time) for each highlight
        output_path: Path for output video
        title: Optional title for the highlight reel
    """
    # Create video edit request
    request = VideoEditRequest(
        video_paths=video_paths,
        edit_prompt="Create a highlight reel",
        output_path=output_path
    )
    
    # Create scenes for each highlight
    scenes = []
    for i, (video_idx, start, end) in enumerate(highlights):
        scene = Scene(
            duration=end - start,
            clips=[
                ClipSegment(
                    source_index=video_idx,
                    start_time=start,
                    end_time=end,
                    position=Position(
                        x=0,
                        y=0,
                        width=1.0,
                        height=1.0
                    ),
                    effects=[
                        # Add subtle color enhancement
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
            # Add transition between highlights
            transition_in=TransitionEffect(
                type=TransitionType.FADE,
                duration=0.5
            ) if i > 0 else None
        )
        scenes.append(scene)
    
    # Add title to first scene
    if scenes:
        scenes[0].captions.append(
            Caption(
                text=title,
                start_time=0,
                end_time=3.0,
                position=Position(
                    x='center',
                    y=0.1,
                    width=0.8,
                    height=0.1
                ),
                style=TextStyle(
                    font_size=48,
                    bold=True,
                    background_color="#00000080"  # Semi-transparent black
                )
            )
        )
    
    # Create edit plan
    plan = VideoEditPlan(
        scenes=scenes,
        estimated_gpu_requirements={
            "min_vram_gb": 8.0,
            "gpu_count": 1
        },
        estimated_duration=10.0  # minutes
    )
    
    # Initialize video processor
    processor = VideoProcessor()
    
    try:
        # Setup GPU environment
        processor.setup_gpu_environment(GPURequirements(
            min_vram_gb=8.0,
            gpu_count=1
        ))
        
        # Process video
        output_path = processor.process_video(plan, request)
        print(f"Highlight reel created successfully: {output_path}")
        
    finally:
        # Ensure cleanup
        processor.cleanup()

if __name__ == "__main__":
    # Example: Create split-screen comparison
    create_split_screen_comparison(
        video1_path="gameplay1.mp4",
        video2_path="gameplay2.mp4",
        output_path="comparison.mp4",
        title="Gameplay Comparison"
    )
    
    # Example: Create highlight reel
    create_highlight_reel(
        video_paths=["match1.mp4", "match2.mp4", "match3.mp4"],
        highlights=[
            (0, 120, 135),  # First highlight from match1
            (1, 45, 60),    # Second highlight from match2
            (2, 90, 100)    # Third highlight from match3
        ],
        output_path="highlights.mp4",
        title="Tournament Highlights"
    ) 