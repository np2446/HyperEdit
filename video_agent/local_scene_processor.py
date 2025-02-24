"""
Local scene processing implementation using ffmpeg and other video processing tools.
This module handles video processing operations on the local machine.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .video_models import (
    Scene, ClipSegment, VideoEffect, AudioEffect,
    Caption, Position, VideoEffectType, AudioEffectType
)

class LocalSceneProcessor:
    """Handles video processing operations for scenes locally."""
    
    def __init__(self, workspace_dir: str):
        """Initialize scene processor.
        
        Args:
            workspace_dir: Base workspace directory for temporary files
        """
        self.workspace_dir = workspace_dir
        self._setup_workspace()
    
    def _setup_workspace(self) -> None:
        """Create required workspace directories."""
        dirs = ['clips', 'effects', 'captions', 'output']
        for dir_name in dirs:
            os.makedirs(os.path.join(self.workspace_dir, dir_name), exist_ok=True)
    
    def _run_command(self, cmd: str) -> str:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {e.stderr}")
    
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
            clip_output = os.path.join(self.workspace_dir, "clips", f"clip_{i}.mp4")
            self._process_clip(clip, source_videos, clip_output)
            clip_outputs.append(clip_output)
        
        # Combine clips if multiple
        if len(clip_outputs) > 1:
            self._combine_clips(clip_outputs, scene, output_path)
        else:
            os.replace(clip_outputs[0], output_path)
    
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
        width = int(clip.position.width * 1920)  # Scale to 1920x1080
        height = int(clip.position.height * 1080)
        filters.append(f"scale={width}:{height}")
        
        # Build and execute ffmpeg command
        filter_str = ','.join(filters) if filters else 'copy'
        cmd = f"ffmpeg -i {input_path} -vf '{filter_str}' -c:a copy {output_path}"
        self._run_command(cmd)
    
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
        
        # Add input files
        for i, clip_path in enumerate(clip_paths):
            inputs.append(f"-i {clip_path}")
        
        # Create overlays
        for i, clip in enumerate(scene.clips):
            # Calculate position and size
            x = int(clip.position.x * 1920)  # Scale to 1920x1080
            y = int(clip.position.y * 1080)
            width = int(clip.position.width * 1920)
            height = int(clip.position.height * 1080)
            
            if i == 0:
                # First clip - scale and format
                filter_complex.append(
                    f"[0]scale={width}:{height},format=yuv420p,pad=1920:1080:{x}:{y}:black[v0]"
                )
                last_output = "v0"
            else:
                # Scale subsequent clips and overlay
                filter_complex.append(
                    f"[{i}]scale={width}:{height},format=yuv420p[fmt{i}];"
                    f"[{last_output}][fmt{i}]overlay=x={x}:y={y}[v{i}]"
                )
                last_output = f"v{i}"
        
        # Build and execute ffmpeg command
        filter_str = ';'.join(filter_complex)
        cmd = (
            f"ffmpeg {' '.join(inputs)} "
            f"-filter_complex '{filter_str}' "
            f"-map '[{last_output}]' "
            f"-c:v libx264 -preset medium {output_path}"
        )
        self._run_command(cmd)
    
    def _create_caption_image(self, caption: Caption) -> str:
        """Create an image file containing the caption text."""
        style = caption.style
        output_path = os.path.join(self.workspace_dir, "captions", f"caption_{hash(caption.text)}.png")
        
        # Create text file
        text_path = os.path.join(self.workspace_dir, "captions", f"text_{hash(caption.text)}.txt")
        with open(text_path, 'w') as f:
            f.write(caption.text)
        
        # Create image with text
        cmd = (
            f"convert -size 1920x1080 xc:transparent -font {style.font_family} "
            f"-pointsize {style.font_size} -fill '{style.color}' "
            f"-stroke '{style.stroke_color}' -strokewidth {style.stroke_width} "
            f"-gravity center -annotate 0 @{text_path} {output_path}"
        )
        self._run_command(cmd)
        
        return output_path 