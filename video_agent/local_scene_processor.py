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
        complex_filters = []
        using_complex = False
        
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
                using_complex = True
                strength = effect.params.get('strength', 5)
                # Create a gradient blur effect that transitions from left to right
                complex_filters.extend([
                    "[0:v]split[main][mask]",
                    f"[main]boxblur={strength}[blurred]",
                    "[mask]geq=lum='if(lt(X/W,0.5),1,if(gt(X/W,0.7),0,(0.7-X/W)/0.2))':a=1[gmask]",
                    "[blurred][main][gmask]maskedmerge[blended]"
                ])
            elif effect.type == VideoEffectType.SHARPEN:
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
        
        if using_complex:
            complex_filters.append(f"[blended]scale={width}:{height}[out]")
            # Build and execute ffmpeg command with complex filtering
            filter_complex = ';'.join(complex_filters)
            cmd = f'ffmpeg -i "{input_path}" -filter_complex "{filter_complex}" -map "[out]" -c:a copy "{output_path}"'
        else:
            filters.append(f"scale={width}:{height}")
            # Build and execute ffmpeg command with simple filtering
            filter_str = ','.join(filters) if filters else 'copy'
            cmd = f'ffmpeg -i "{input_path}" -vf "{filter_str}" -c:a copy "{output_path}"'
        
        print(f"Executing ffmpeg command: {cmd}")  # Debug print
        self._run_command(cmd)
    
    def _combine_clips(self, clips: List[str], scene: Scene, output_path: str) -> None:
        """Combine multiple clips into a single scene.
        
        Args:
            clips: List of processed clip paths
            scene: Scene object containing captions and other metadata
            output_path: Path to save the combined output
        """
        # Create filter complex for combining clips and adding overlays
        inputs = []
        filter_complex = []
        
        # Add input clips
        for i, clip in enumerate(clips):
            inputs.extend(["-i", clip])
        
        # Start with first video (left side)
        filter_complex.append(f"[0]scale=960:1080,format=yuv420p[v0]")
        
        # Add second video (right side)
        if len(clips) > 1:
            filter_complex.append(f"[1]scale=960:1080,format=yuv420p[v1]")
            filter_complex.append(f"[v0][v1]hstack=inputs=2[base]")
            last_output = "base"
        else:
            last_output = "v0"
        
        # Add captions using drawtext filter
        for i, caption in enumerate(scene.captions):
            # Calculate fade times and alpha expression
            fade_in_duration = 0.5
            fade_out_duration = 0.5
            fade_out_start = float(caption.end_time) - fade_out_duration
            
            # Create alpha expression for fade in/out
            alpha_expr = (
                f"if(lt(t,{fade_in_duration}),"  # Fade in
                f"t/{fade_in_duration},"
                f"if(gt(t,{fade_out_start}),"  # Fade out
                f"(1-(t-{fade_out_start})/{fade_out_duration}),"
                f"1))"  # Full opacity between fade in/out
            )
            
            # Add drawtext filter with fading
            font_size = getattr(caption.style, 'font_size', 72)
            filter_complex.append(
                f"[{last_output}]drawtext="
                f"text='{caption.text}':"
                f"fontsize={font_size}:"
                f"fontcolor=white:"
                f"fontfile=/System/Library/Fonts/Helvetica.ttc:"  # Use system font
                f"x=(w-text_w)/2:"  # Center horizontally
                f"y=100:"  # Position from top
                f"alpha='{alpha_expr}':"  # Fade in/out
                f"box=1:"  # Add background box
                f"boxcolor=black@0.5:"  # Semi-transparent black background
                f"boxborderw=10:"  # Box padding
                f"enable='between(t,0,{caption.end_time})'[v{i+2}]"  # Only show during caption duration
            )
            last_output = f"v{i+2}"
        
        # Build final command
        filter_str = ";".join(filter_complex)
        cmd = [
            "ffmpeg", *inputs,
            "-filter_complex", filter_str,
            "-map", f"[{last_output}]",
            "-c:v", "libx264",
            "-preset", "medium",
            output_path
        ]
        
        print(f"Executing ffmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    def _create_caption_image(self, caption: Caption) -> str:
        """Create an image file containing the caption text."""
        style = caption.style
        output_path = os.path.join(self.workspace_dir, "captions", f"caption_{hash(caption.text)}.png")
        
        # Ensure captions directory exists
        os.makedirs(os.path.join(self.workspace_dir, "captions"), exist_ok=True)
        
        # Create text file
        text_path = os.path.join(self.workspace_dir, "captions", f"text_{hash(caption.text)}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(caption.text)
        
        # Set default style values if not provided
        font_size = getattr(style, 'font_size', 72)
        
        # Create image with text - white text on black background
        cmd = (
            f'convert -size 1920x200 xc:black -alpha set -background none '
            f'-fill white -font Arial -pointsize {font_size} '
            f'-gravity center -annotate 0 @"{text_path}" '
            f'-channel A -evaluate set 50% '  # Make background 50% transparent
            f'"{output_path}"'
        )
        print(f"Creating caption image with command: {cmd}")  # Debug print
        self._run_command(cmd)
        
        # Clean up text file
        if os.path.exists(text_path):
            os.remove(text_path)
        
        return output_path 