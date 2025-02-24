"""
Unit tests for the video agent components.
Tests each module independently and their integration.
Generates test videos and processes them with real ffmpeg commands.
"""

import unittest
import os
import json
import numpy as np
import cv2
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

from video_agent.video_models import (
    VideoEditRequest, VideoEditPlan, Scene, ClipSegment,
    Position, VideoEffect, AudioEffect, Caption,
    VideoEffectType, AudioEffectType, TransitionEffect,
    TransitionType, TextStyle
)
from video_agent.video_processor import VideoProcessor, GPURequirements

def create_test_pattern_video(output_path: str, pattern_type: str = "circles", duration: int = 5, fps: int = 30, resolution: tuple = (1280, 720)) -> None:
    """Create a test video with moving patterns and text.
    
    Args:
        output_path: Path to save the video
        pattern_type: Type of pattern to generate ("circles" or "rectangles")
        duration: Duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
    """
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # Create patterns
    patterns = []
    
    if pattern_type == "circles":
        # Create moving circles
        for i in range(5):
            patterns.append({
                'type': 'circle',
                'radius': np.random.randint(30, 80),
                'color': tuple(np.random.randint(0, 255, 3).tolist()),
                'pos': [np.random.randint(100, width-100), np.random.randint(100, height-100)],
                'vel': [np.random.randint(-7, 7), np.random.randint(-7, 7)]
            })
    else:
        # Create rotating rectangles
        for i in range(3):
            patterns.append({
                'type': 'rect',
                'size': (np.random.randint(80, 150), np.random.randint(50, 100)),
                'color': tuple(np.random.randint(0, 255, 3).tolist()),
                'pos': [np.random.randint(100, width-100), np.random.randint(100, height-100)],
                'angle': 0,
                'angle_vel': np.random.uniform(-3, 3)
            })
    
    # Add text overlay
    patterns.append({
        'type': 'text',
        'text': f'Test Pattern - {pattern_type.title()}',
        'color': (255, 255, 255),
        'pos': [width//2, 50],
        'font_scale': 2,
        'alpha': 0,
        'alpha_vel': 0.02
    })
    
    # Generate frames
    for frame_idx in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw grid pattern in background
        grid_size = 50
        for x in range(0, width, grid_size):
            cv2.line(frame, (x, 0), (x, height), (32, 32, 32), 1)
        for y in range(0, height, grid_size):
            cv2.line(frame, (0, y), (width, y), (32, 32, 32), 1)
        
        # Update and draw patterns
        for pattern in patterns:
            if pattern['type'] == 'circle':
                # Update position
                pattern['pos'][0] += pattern['vel'][0]
                pattern['pos'][1] += pattern['vel'][1]
                
                # Bounce off walls
                for i in range(2):
                    if pattern['pos'][i] - pattern['radius'] < 0:
                        pattern['pos'][i] = pattern['radius']
                        pattern['vel'][i] *= -1
                    elif pattern['pos'][i] + pattern['radius'] > (resolution[i]):
                        pattern['pos'][i] = resolution[i] - pattern['radius']
                        pattern['vel'][i] *= -1
                
                # Draw circle
                cv2.circle(
                    frame,
                    (int(pattern['pos'][0]), int(pattern['pos'][1])),
                    pattern['radius'],
                    pattern['color'],
                    -1
                )
                
            elif pattern['type'] == 'rect':
                # Update rotation
                pattern['angle'] += pattern['angle_vel']
                
                # Create rotated rectangle
                rect = cv2.boxPoints((
                    (pattern['pos'][0], pattern['pos'][1]),
                    pattern['size'],
                    pattern['angle']
                ))
                rect = np.int0(rect)
                
                # Draw rectangle
                cv2.drawContours(frame, [rect], 0, pattern['color'], -1)
                
            elif pattern['type'] == 'text':
                # Update alpha
                pattern['alpha'] += pattern['alpha_vel']
                if pattern['alpha'] > 1 or pattern['alpha'] < 0:
                    pattern['alpha_vel'] *= -1
                pattern['alpha'] = np.clip(pattern['alpha'], 0, 1)
                
                # Draw text with alpha
                text_size = cv2.getTextSize(
                    pattern['text'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    pattern['font_scale'],
                    2
                )[0]
                text_x = int(pattern['pos'][0] - text_size[0]/2)
                text_y = int(pattern['pos'][1] + text_size[1]/2)
                
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    pattern['text'],
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    pattern['font_scale'],
                    pattern['color'],
                    2
                )
                cv2.addWeighted(
                    overlay,
                    pattern['alpha'],
                    frame,
                    1 - pattern['alpha'],
                    0,
                    frame
                )
        
        # Add frame counter
        cv2.putText(
            frame,
            f'Frame: {frame_idx}/{duration*fps}',
            (10, height-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1
        )
        
        out.write(frame)
    
    out.release()

class TestLocalVideoProcessing(unittest.TestCase):
    """Test video processing with real effects locally."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test videos and output directory."""
        # Create output directory
        cls.output_dir = Path("test_outputs")
        cls.output_dir.mkdir(exist_ok=True)
        
        # Create test videos with different patterns
        cls.video1_path = cls.output_dir / "input_circles.mp4"
        cls.video2_path = cls.output_dir / "input_rectangles.mp4"
        
        create_test_pattern_video(str(cls.video1_path), pattern_type="circles", duration=5)
        create_test_pattern_video(str(cls.video2_path), pattern_type="rectangles", duration=5)
    
    def setUp(self):
        """Initialize video processor for each test."""
        # Create processor in local mode
        self.processor = VideoProcessor(local_mode=True)
        self.processor.setup_gpu_environment(GPURequirements(
            min_vram_gb=4.0,
            gpu_count=1
        ))
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
    
    def test_basic_effects(self):
        """Test applying basic video effects."""
        output_path = str(self.output_dir / "basic_effects.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path)],
            edit_prompt="Apply multiple effects",
            output_path=output_path
        )
        
        scene = Scene(
            duration=5.0,
            clips=[
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0, y=0, width=1.0, height=1.0),
                    effects=[
                        VideoEffect(
                            type=VideoEffectType.BLUR,
                            params={"strength": 10},
                            start_time=0,
                            end_time=2
                        ),
                        VideoEffect(
                            type=VideoEffectType.SHARPEN,
                            params={"strength": 5},
                            start_time=2,
                            end_time=4
                        ),
                        VideoEffect(
                            type=VideoEffectType.COLOR_ADJUST,
                            params={"contrast": 1.3, "saturation": 1.5},
                            start_time=4,
                            end_time=5
                        )
                    ]
                )
            ]
        )
        
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={
                "min_vram_gb": 4.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nBasic effects output saved to: {output_path}")
    
    def test_split_screen(self):
        """Test split-screen comparison."""
        output_path = str(self.output_dir / "split_screen.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path), str(self.video2_path)],
            edit_prompt="Create split-screen comparison",
            output_path=output_path
        )
        
        scene = Scene(
            duration=5.0,
            clips=[
                # Left video - circles
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0, y=0, width=0.5, height=1.0)
                ),
                # Right video - rectangles
                ClipSegment(
                    source_index=1,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0.5, y=0, width=0.5, height=1.0)
                )
            ],
            captions=[
                Caption(
                    text="Pattern Comparison",
                    start_time=0,
                    end_time=5,
                    position=Position(x='center', y=0.1, width=0.8, height=0.1),
                    style=TextStyle(font_size=48, bold=True)
                )
            ]
        )
        
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={
                "min_vram_gb": 4.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nSplit screen output saved to: {output_path}")
    
    def test_picture_in_picture(self):
        """Test picture-in-picture effect."""
        output_path = str(self.output_dir / "pip.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path), str(self.video2_path)],
            edit_prompt="Create picture-in-picture effect",
            output_path=output_path
        )
        
        scene = Scene(
            duration=5.0,
            clips=[
                # Main video - circles
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0, y=0, width=1.0, height=1.0)
                ),
                # PiP video - rectangles
                ClipSegment(
                    source_index=1,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0.7, y=0.7, width=0.25, height=0.25)
                )
            ]
        )
        
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={
                "min_vram_gb": 4.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nPicture-in-picture output saved to: {output_path}")
    
    def test_transitions(self):
        """Test video transitions."""
        output_path = str(self.output_dir / "transitions.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path), str(self.video2_path)],
            edit_prompt="Create video with transitions",
            output_path=output_path
        )
        
        scenes = [
            # First scene - circles
            Scene(
                duration=2.5,
                clips=[
                    ClipSegment(
                        source_index=0,
                        start_time=0,
                        end_time=2.5,
                        position=Position(x=0, y=0, width=1.0, height=1.0)
                    )
                ],
                transition_out=TransitionEffect(
                    type=TransitionType.FADE,
                    duration=1.0
                )
            ),
            # Second scene - rectangles
            Scene(
                duration=2.5,
                clips=[
                    ClipSegment(
                        source_index=1,
                        start_time=0,
                        end_time=2.5,
                        position=Position(x=0, y=0, width=1.0, height=1.0)
                    )
                ],
                transition_in=TransitionEffect(
                    type=TransitionType.FADE,
                    duration=1.0
                )
            )
        ]
        
        plan = VideoEditPlan(
            scenes=scenes,
            estimated_gpu_requirements={
                "min_vram_gb": 4.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nTransitions output saved to: {output_path}")

class TestRemoteVideoProcessing(TestLocalVideoProcessing):
    """Test video processing with real effects using Hyperbolic's infrastructure."""
    
    def setUp(self):
        """Initialize video processor for each test."""
        # Create processor in remote mode
        self.processor = VideoProcessor(local_mode=False)
        self.processor.setup_gpu_environment(GPURequirements(
            min_vram_gb=8.0,  # Use higher VRAM for remote processing
            gpu_count=1,
            preferred_gpu_model="A4000"  # Request a specific GPU model
        ))
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
    
    def test_basic_effects(self):
        """Test applying basic video effects."""
        output_path = str(self.output_dir / "remote_basic_effects.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path)],
            edit_prompt="Apply multiple effects",
            output_path=output_path
        )
        
        scene = Scene(
            duration=5.0,
            clips=[
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0, y=0, width=1.0, height=1.0),
                    effects=[
                        VideoEffect(
                            type=VideoEffectType.BLUR,
                            params={"strength": 10},
                            start_time=0,
                            end_time=2
                        ),
                        VideoEffect(
                            type=VideoEffectType.SHARPEN,
                            params={"strength": 5},
                            start_time=2,
                            end_time=4
                        ),
                        VideoEffect(
                            type=VideoEffectType.COLOR_ADJUST,
                            params={"contrast": 1.3, "saturation": 1.5},
                            start_time=4,
                            end_time=5
                        )
                    ]
                )
            ]
        )
        
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={
                "min_vram_gb": 8.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nRemote basic effects output saved to: {output_path}")
    
    def test_split_screen(self):
        """Test split-screen comparison."""
        output_path = str(self.output_dir / "remote_split_screen.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path), str(self.video2_path)],
            edit_prompt="Create split-screen comparison",
            output_path=output_path
        )
        
        scene = Scene(
            duration=5.0,
            clips=[
                # Left video - circles
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0, y=0, width=0.5, height=1.0)
                ),
                # Right video - rectangles
                ClipSegment(
                    source_index=1,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0.5, y=0, width=0.5, height=1.0)
                )
            ],
            captions=[
                Caption(
                    text="Pattern Comparison",
                    start_time=0,
                    end_time=5,
                    position=Position(x='center', y=0.1, width=0.8, height=0.1),
                    style=TextStyle(font_size=48, bold=True)
                )
            ]
        )
        
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={
                "min_vram_gb": 8.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nRemote split screen output saved to: {output_path}")
    
    def test_picture_in_picture(self):
        """Test picture-in-picture effect."""
        output_path = str(self.output_dir / "remote_pip.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path), str(self.video2_path)],
            edit_prompt="Create picture-in-picture effect",
            output_path=output_path
        )
        
        scene = Scene(
            duration=5.0,
            clips=[
                # Main video - circles
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0, y=0, width=1.0, height=1.0)
                ),
                # PiP video - rectangles
                ClipSegment(
                    source_index=1,
                    start_time=0,
                    end_time=5,
                    position=Position(x=0.7, y=0.7, width=0.25, height=0.25)
                )
            ]
        )
        
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={
                "min_vram_gb": 8.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nRemote picture-in-picture output saved to: {output_path}")
    
    def test_transitions(self):
        """Test video transitions."""
        output_path = str(self.output_dir / "remote_transitions.mp4")
        
        request = VideoEditRequest(
            video_paths=[str(self.video1_path), str(self.video2_path)],
            edit_prompt="Create video with transitions",
            output_path=output_path
        )
        
        scenes = [
            # First scene - circles
            Scene(
                duration=2.5,
                clips=[
                    ClipSegment(
                        source_index=0,
                        start_time=0,
                        end_time=2.5,
                        position=Position(x=0, y=0, width=1.0, height=1.0)
                    )
                ],
                transition_out=TransitionEffect(
                    type=TransitionType.FADE,
                    duration=1.0
                )
            ),
            # Second scene - rectangles
            Scene(
                duration=2.5,
                clips=[
                    ClipSegment(
                        source_index=1,
                        start_time=0,
                        end_time=2.5,
                        position=Position(x=0, y=0, width=1.0, height=1.0)
                    )
                ],
                transition_in=TransitionEffect(
                    type=TransitionType.FADE,
                    duration=1.0
                )
            )
        ]
        
        plan = VideoEditPlan(
            scenes=scenes,
            estimated_gpu_requirements={
                "min_vram_gb": 8.0,
                "gpu_count": 1
            },
            estimated_duration=1.0
        )
        
        output_path = self.processor.process_video(plan, request)
        self.assertTrue(os.path.exists(output_path))
        print(f"\nRemote transitions output saved to: {output_path}")

if __name__ == '__main__':
    unittest.main() 