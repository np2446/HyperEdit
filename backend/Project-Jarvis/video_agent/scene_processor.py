"""
Scene processing implementation using ffmpeg and other video processing tools.
This module handles the actual video processing operations on the GPU instance.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
from .video_models import (
    Scene, ClipSegment, VideoEffect, AudioEffect,
    Caption, Position, VideoEffectType, AudioEffectType
)

class SceneProcessor:
    """Handles video processing operations for scenes."""
    
    def __init__(self, instance_id: str, workspace_dir: str):
        """Initialize scene processor.
        
        Args:
            instance_id: ID of the GPU instance
            workspace_dir: Base workspace directory on GPU instance
        """
        self.instance_id = instance_id
        self.workspace_dir = workspace_dir
        self._setup_workspace()
    
    def _setup_workspace(self) -> None:
        """Create required workspace directories."""
        dirs = ['clips', 'effects', 'captions', 'output']
        for dir_name in dirs:
            execute_remote_command(f"mkdir -p {self.workspace_dir}/{dir_name}")
    
    def process_scene(self, scene: Scene, source_videos: List[str], output_path: str) -> None:
        """Process a scene with multiple clips.
        
        Args:
            scene: Scene to process
            source_videos: List of source video paths
            output_path: Path to save the processed scene
        """
        # Process each clip
        clip_outputs = []
        for i, clip in enumerate(scene.clips):
            clip_output = f"{self.workspace_dir}/clips/clip_{i}.mp4"
            self._process_clip(clip, source_videos, clip_output)
            clip_outputs.append(clip_output)
        
        # Combine clips if multiple
        if len(clip_outputs) > 1:
            self._combine_clips(clip_outputs, scene, output_path)
        else:
            execute_remote_command(f"mv {clip_outputs[0]} {output_path}")
    
    def _process_clip(self, clip: ClipSegment, source_videos: List[str], output_path: str) -> None:
        """Process a single clip with effects.
        
        Args:
            clip: Clip segment to process
            source_videos: List of source video paths
            output_path: Path to save the processed clip
        """
        input_path = source_videos[clip.source_index]
        
        # Build ffmpeg filter chain
        filters = []
        
        # Add trim filter if needed
        if clip.start_time > 0 or clip.end_time > 0:
            filters.append(f"trim=start={clip.start_time}:end={clip.end_time}")
        
        # Add video effects
        for effect in clip.effects:
            if effect.type == VideoEffectType.COLOR_ADJUST:
                params = effect.params
                filters.append(
                    f"eq=contrast={params.get('contrast', 1.0)}:"
                    f"brightness={params.get('brightness', 0.0)}:"
                    f"saturation={params.get('saturation', 1.0)}"
                )
            elif effect.type == VideoEffectType.BLUR:
                strength = effect.params.get('strength', 5)
                filters.append(f"boxblur={strength}")
            elif effect.type == VideoEffectType.SHARPEN:
                strength = effect.params.get('strength', 1)
                filters.append(f"unsharp={strength}:5:0")
            elif effect.type == VideoEffectType.SPEED:
                speed = effect.params.get('factor', 1.0)
                filters.append(f"setpts={1/speed}*PTS")
            elif effect.type == VideoEffectType.STABILIZE:
                filters.append("vidstabdetect=shakiness=10:accuracy=15")
                filters.append("vidstabtransform=smoothing=30")
        
        # Add scale and position filters
        width = int(clip.position.width * 100)
        height = int(clip.position.height * 100)
        filters.append(f"scale=iw*{width}/100:ih*{height}/100")
        
        # Build and execute ffmpeg command
        filter_str = ','.join(filters) if filters else 'copy'
        cmd = f"ffmpeg -i {input_path} -vf '{filter_str}' -c:a copy {output_path}"
        execute_remote_command(cmd)
    
    def _combine_clips(self, clip_paths: List[str], scene: Scene, output_path: str) -> None:
        """Combine multiple clips into a scene.
        
        Args:
            clip_paths: List of processed clip paths
            scene: Scene containing clip layout information
            output_path: Path to save the combined scene
        """
        # Create filter complex for combining clips
        filter_complex = []
        inputs = []
        overlays = []
        
        for i, (clip_path, clip) in enumerate(zip(clip_paths, scene.clips)):
            # Add input
            inputs.append(f"-i {clip_path}")
            
            # Calculate position
            x = int(clip.position.x * 100)
            y = int(clip.position.y * 100)
            
            if i == 0:
                # First clip is the base
                filter_complex.append(f"[0]scale=1920:1080[base]")
            else:
                # Overlay subsequent clips
                filter_complex.append(
                    f"[{i}]scale=1920*{clip.position.width}:1080*{clip.position.height}"
                    f"[clip{i}]"
                )
                overlays.append(
                    f"[tmp{i-1}][clip{i}]overlay=x={x}*W/100:y={y}*H/100"
                    f"[tmp{i}]"
                )
        
        # Combine all filters
        filter_str = ';'.join(filter_complex + overlays)
        
        # Build and execute ffmpeg command
        cmd = (
            f"ffmpeg {' '.join(inputs)} "
            f"-filter_complex '{filter_str}' "
            f"-map '[tmp{len(clip_paths)-1}]' "
            f"-c:v libx264 -preset medium {output_path}"
        )
        execute_remote_command(cmd)
    
    def _create_caption_image(self, caption: Caption) -> str:
        """Create an image file containing the caption text."""
        style = caption.style
        output_path = f"{self.workspace_dir}/captions/caption_{hash(caption.text)}.png"
        
        # Create text file
        text_path = f"{self.workspace_dir}/captions/text_{hash(caption.text)}.txt"
        execute_remote_command(f"echo '{caption.text}' > {text_path}")
        
        # Create image with text
        cmd = (
            f"convert -size 1920x1080 xc:transparent -font {style.font_family} "
            f"-pointsize {style.font_size} -fill '{style.color}' "
            f"-stroke '{style.stroke_color}' -strokewidth {style.stroke_width} "
            f"-gravity center -annotate 0 @{text_path} {output_path}"
        )
        execute_remote_command(cmd)
        
        return output_path 