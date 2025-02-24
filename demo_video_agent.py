"""
Demonstration script for the video agent's capabilities.
Generates test videos and shows how the AI can analyze and edit them.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import warnings
import json
# import api key from .env
from dotenv import load_dotenv
load_dotenv()

# Disable LangSmith warnings
warnings.filterwarnings("ignore", ".*Failed to get info from your_langchain_endpoint.*")
warnings.filterwarnings("ignore", ".*Failed to batch ingest runs.*")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from video_agent import VideoTool, VideoProcessor
from video_agent.video_processor import GPURequirements

def create_test_pattern_video(
    output_path: str,
    pattern_type: str = "circles",
    duration: int = 5,
    fps: int = 30,
    resolution: tuple = (1280, 720),
    color_scheme: str = "random"
) -> None:
    """Create a test video with moving patterns and text.
    
    Args:
        output_path: Path to save the video
        pattern_type: Type of pattern ("circles", "rectangles", "waves", "particles")
        duration: Duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        color_scheme: Color scheme ("random", "warm", "cool", "grayscale")
    """
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    def get_color():
        if color_scheme == "random":
            return tuple(np.random.randint(0, 255, 3).tolist())
        elif color_scheme == "warm":
            return (
                np.random.randint(150, 255),  # Red
                np.random.randint(50, 150),   # Green
                np.random.randint(0, 100)     # Blue
            )
        elif color_scheme == "cool":
            return (
                np.random.randint(0, 100),    # Red
                np.random.randint(50, 150),   # Green
                np.random.randint(150, 255)   # Blue
            )
        else:  # grayscale
            v = np.random.randint(0, 255)
            return (v, v, v)
    
    # Create patterns
    patterns = []
    
    if pattern_type == "circles":
        # Create moving circles
        for i in range(5):
            patterns.append({
                'type': 'circle',
                'radius': np.random.randint(30, 80),
                'color': get_color(),
                'pos': [np.random.randint(100, width-100), np.random.randint(100, height-100)],
                'vel': [np.random.randint(-7, 7), np.random.randint(-7, 7)]
            })
    
    elif pattern_type == "rectangles":
        # Create rotating rectangles
        for i in range(3):
            patterns.append({
                'type': 'rect',
                'size': (np.random.randint(80, 150), np.random.randint(50, 100)),
                'color': get_color(),
                'pos': [np.random.randint(100, width-100), np.random.randint(100, height-100)],
                'angle': 0,
                'angle_vel': np.random.uniform(-3, 3)
            })
    
    elif pattern_type == "waves":
        # Create moving sine waves
        patterns.append({
            'type': 'wave',
            'amplitude': height/4,
            'frequency': np.random.uniform(1, 3),
            'color': get_color(),
            'phase': 0,
            'phase_vel': np.random.uniform(0.1, 0.3)
        })
    
    elif pattern_type == "particles":
        # Create particle system
        for i in range(20):
            patterns.append({
                'type': 'particle',
                'radius': np.random.randint(2, 8),
                'color': get_color(),
                'pos': [np.random.randint(0, width), np.random.randint(0, height)],
                'vel': [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
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
            
            elif pattern['type'] == 'wave':
                # Update phase
                pattern['phase'] += pattern['phase_vel']
                
                # Draw wave
                points = []
                for x in range(0, width, 2):
                    y = int(height/2 + pattern['amplitude'] * 
                          np.sin(2*np.pi*pattern['frequency']*x/width + pattern['phase']))
                    points.append([x, y])
                
                if len(points) > 1:
                    points = np.array(points, np.int32)
                    cv2.polylines(frame, [points], False, pattern['color'], 2)
            
            elif pattern['type'] == 'particle':
                # Update position
                pattern['pos'][0] += pattern['vel'][0]
                pattern['pos'][1] += pattern['vel'][1]
                
                # Wrap around edges
                pattern['pos'][0] = pattern['pos'][0] % width
                pattern['pos'][1] = pattern['pos'][1] % height
                
                # Draw particle
                cv2.circle(
                    frame,
                    (int(pattern['pos'][0]), int(pattern['pos'][1])),
                    pattern['radius'],
                    pattern['color'],
                    -1
                )
            
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

def generate_test_videos():
    """Generate a set of test videos with different patterns and styles."""
    input_dir = Path("input_videos")
    input_dir.mkdir(exist_ok=True)
    
    # List of videos to generate
    videos = [
        {
            "name": "circles_warm",
            "pattern": "circles",
            "duration": 10,
            "color_scheme": "warm",
            "resolution": (1280, 720)
        },
        {
            "name": "rectangles_cool",
            "pattern": "rectangles",
            "duration": 8,
            "color_scheme": "cool",
            "resolution": (1920, 1080)
        },
        {
            "name": "waves_grayscale",
            "pattern": "waves",
            "duration": 12,
            "color_scheme": "grayscale",
            "resolution": (1280, 720)
        },
        {
            "name": "particles_random",
            "pattern": "particles",
            "duration": 15,
            "color_scheme": "random",
            "resolution": (1920, 1080)
        }
    ]
    
    print("Generating test videos...")
    for video in videos:
        output_path = input_dir / f"{video['name']}.mp4"
        print(f"Creating {output_path}...")
        create_test_pattern_video(
            str(output_path),
            pattern_type=video['pattern'],
            duration=video['duration'],
            resolution=video['resolution'],
            color_scheme=video['color_scheme']
        )
    print("Video generation complete!")

async def run_video_agent_demo():
    """Run the video agent on the generated test videos."""
    print("\nInitializing video agent...")
    
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        callbacks=None,  # Disable LangSmith callbacks
        tags=None  # Disable LangSmith tags
    )
    
    # Initialize video tool with local mode
    video_tool = VideoTool(
        llm=llm,
        processor=VideoProcessor(local_mode=True)
    )
    
    # Initialize GPU environment
    video_tool.processor.setup_gpu_environment(GPURequirements(min_vram_gb=4.0))
    
    # Create runnable config with callbacks disabled
    runnable_config = RunnableConfig(
        callbacks=None,  # Disable LangSmith callbacks
        tags=None  # Disable LangSmith tags
    )
    
    # Example editing tasks for the LLM agent
    tasks = [
        """Create a split-screen comparison of the warm and cool colored patterns.
        Input videos:
        - Left side: input_videos/circles_warm.mp4
        - Right side: input_videos/rectangles_cool.mp4
        
        Requirements:
        - Position videos side by side (left video at x=0, right video at x=0.5)
        - Add title caption "Warm vs Cool Patterns"
        - Add subtle blur effect (strength 3) to both videos
        - Add fade transition at start and end (1 second)"""
    ]
    
    print("\nProcessing video editing requests...")
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}:")
        print("-" * 80)
        print(f"Request: {task}")
        
        try:
            # Log video analysis
            print("\nAnalyzing input videos...")
            for video_path, info in video_tool._analyze_videos().items():
                print(f"\nVideo: {video_path}")
                print(f"- Resolution: {info.width}x{info.height}")
                print(f"- Duration: {info.duration:.2f} seconds")
                print(f"- FPS: {info.fps}")
            
            # Let the LLM agent process the natural language request
            print("\nSending request to LLM for parsing...")
            parsed_request = video_tool._parse_edit_request(task)
            print(f"\nParsed request from LLM:\n{json.dumps(parsed_request, indent=2)}")
            
            print("\nCreating edit plan...")
            request, edit_plan = video_tool._create_edit_plan(parsed_request)
            
            print("\nEdit plan details:")
            print(f"Output path: {request.output_path}")
            for scene_idx, scene in enumerate(edit_plan.scenes):
                print(f"\nScene {scene_idx + 1}:")
                print(f"- Duration: {scene.duration} seconds")
                print("- Clips:")
                for clip_idx, clip in enumerate(scene.clips):
                    print(f"  Clip {clip_idx + 1}:")
                    print(f"  - Source: {request.video_paths[clip.source_index]}")
                    print(f"  - Position: x={clip.position.x}, y={clip.position.y}, width={clip.position.width}, height={clip.position.height}")
                    print(f"  - Effects: {[e.type.name for e in clip.effects]}")
                print("- Captions:", [c.text for c in scene.captions])
                if scene.transition_out:
                    print(f"- Transition: {scene.transition_out.type.name} ({scene.transition_out.duration}s)")
            
            print("\nProcessing video with edit plan...")
            result = await video_tool._arun(task, runnable_config)
            print(f"\nResult: {result}")
            
            # Verify output file
            output_path = Path(request.output_path)
            if output_path.exists():
                print(f"\nOutput file created successfully: {output_path}")
                # Get output video info
                cap = cv2.VideoCapture(str(output_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                print(f"- Resolution: {width}x{height}")
                print(f"- Duration: {duration} seconds")
            else:
                print(f"\nWarning: Output file not found at {output_path}")
            
        except Exception as e:
            print(f"\nError processing video: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
        
        print("-" * 80)

async def main():
    """Run the complete demonstration."""
    print("Starting Video Agent Demonstration")
    print("=" * 80)
    
    # Generate test videos
    generate_test_videos()
    
    # Run video agent demo
    await run_video_agent_demo()
    
    print("\nDemonstration complete!")
    print("Check the 'test_outputs' directory for the processed videos.")

if __name__ == "__main__":
    asyncio.run(main()) 