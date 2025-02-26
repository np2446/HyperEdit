"""
Demonstration script for the video agent's capabilities using Hyperbolic GPUs.
Generates test videos and shows how the AI can analyze and edit them using remote GPU acceleration.
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
import time
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
    
    # List of videos to generate - using shorter durations for faster processing
    videos = [
        {
            "name": "circles_warm",
            "pattern": "circles",
            "duration": 5,  # Reduced from 10
            "color_scheme": "warm",
            "resolution": (1280, 720)
        },
        {
            "name": "rectangles_cool",
            "pattern": "rectangles",
            "duration": 5,  # Reduced from 8
            "color_scheme": "cool",
            "resolution": (1280, 720)  # Reduced resolution for faster processing
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

def check_environment():
    """Check if the environment is properly configured for Hyperbolic GPU usage."""
    # Check for SSH key
    ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH')
    if not ssh_key_path:
        print("ERROR: SSH_PRIVATE_KEY_PATH environment variable is not set.")
        print("Please set it to the path of your SSH private key for Hyperbolic.")
        return False
    
    ssh_key_path = os.path.expanduser(ssh_key_path)
    if not os.path.exists(ssh_key_path):
        print(f"ERROR: SSH key file not found at {ssh_key_path}")
        return False
    
    print(f"Using SSH key: {ssh_key_path}")
    
    # Check for Anthropic API key if using LLM
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("WARNING: ANTHROPIC_API_KEY environment variable is not set.")
        print("LLM-based video editing will not be available.")
    
    return True

async def run_hyperbolic_video_demo():
    """Run the video agent on Hyperbolic GPUs."""
    print("\nInitializing video processor with Hyperbolic GPUs...")
    
    # Initialize video processor with remote mode
    processor = VideoProcessor(local_mode=False)
    
    # Create output directory
    output_dir = Path("hyperbolic_outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize GPU environment with affordable options
        print("\nSetting up GPU environment...")
        print("This may take several minutes. Please be patient and do not interrupt the process.")
        print("If you need to cancel, press Ctrl+C once and wait for graceful cleanup.")
        
        processor.setup_gpu_environment(
            GPURequirements(
                gpu_type="RTX",  # Prefer RTX GPUs which are typically cheaper
                num_gpus=1,
                min_vram_gb=4.0,
                disk_size=10,
                memory=16
            )
        )
        
        print("\nGPU environment set up successfully!")
        print(f"Instance ID: {processor.instance_id}")
        
        # Test file transfer
        print("\nTesting file transfer...")
        test_file = "test_file.txt"
        with open(test_file, "w") as f:
            f.write("This is a test file for Hyperbolic GPU integration.")
        
        remote_test_path = f"{processor.workspace_dir}/test_file.txt"
        processor.file_transfer.upload_file(test_file, remote_test_path)
        print("File upload successful!")
        
        # List remote files
        remote_files = processor.file_transfer.list_remote_files(processor.workspace_dir)
        print(f"Remote files: {remote_files}")
        
        # Upload test videos
        print("\nUploading test videos to Hyperbolic instance...")
        input_dir = Path("input_videos")
        remote_input_dir = f"{processor.workspace_dir}/input_videos"
        
        # Create remote input directory
        from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
        execute_remote_command(f"mkdir -p {remote_input_dir}", instance_id=processor.instance_id, timeout=30)
        
        # Upload each video
        video_files = list(input_dir.glob("*.mp4"))
        if not video_files:
            print("No test videos found in input_videos directory. Generating test videos...")
            generate_test_videos()
            video_files = list(input_dir.glob("*.mp4"))
        
        for i, video_file in enumerate(video_files):
            print(f"Uploading {video_file} ({i+1}/{len(video_files)})...")
            remote_path = f"{remote_input_dir}/{video_file.name}"
            processor.file_transfer.upload_file(str(video_file), remote_path)
        
        # Process a simple video edit
        print("\nProcessing a simple video edit...")
        
        # Define a simple edit: create a split-screen of two videos
        from video_agent.video_models import VideoEditRequest, VideoEditPlan, Scene, Clip, Position, Effect, EffectType, Caption, Transition, TransitionType
        
        # Create a simple edit plan
        request = VideoEditRequest(
            video_paths=[f"{remote_input_dir}/circles_warm.mp4", f"{remote_input_dir}/rectangles_cool.mp4"],
            output_path=f"{processor.workspace_dir}/split_screen_output.mp4"
        )
        
        # Create a split-screen scene
        scene = Scene(
            duration=5.0,
            clips=[
                Clip(
                    source_index=0,
                    start_time=0.0,
                    duration=5.0,
                    position=Position(x=0.0, y=0.0, width=0.5, height=1.0),
                    effects=[]
                ),
                Clip(
                    source_index=1,
                    start_time=0.0,
                    duration=5.0,
                    position=Position(x=0.5, y=0.0, width=0.5, height=1.0),
                    effects=[]
                )
            ],
            captions=[
                Caption(
                    text="Split Screen Demo",
                    position=Position(x=0.5, y=0.1, width=0.8, height=0.1),
                    start_time=0.5,
                    duration=4.0
                )
            ],
            transition_out=None
        )
        
        # Create the edit plan
        edit_plan = VideoEditPlan(
            scenes=[scene]
        )
        
        # Process the video
        try:
            print("\nExecuting video processing...")
            processor.process_video(edit_plan, request)
            
            # Download the result
            local_output_path = str(output_dir / "split_screen_output.mp4")
            print(f"\nDownloading result to {local_output_path}...")
            processor.file_transfer.download_file(request.output_path, local_output_path)
            
            print(f"\nVideo processing complete! Output saved to {local_output_path}")
            
        except Exception as e:
            print(f"\nError processing video: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
        
        # If Anthropic API key is available, try LLM-based video editing
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            try:
                print("\nTrying LLM-based video editing...")
                
                # Import necessary modules
                from langchain.tools import BaseTool
                from langchain.agents import AgentExecutor
                from langchain_anthropic import ChatAnthropic
                from langchain.agents.format_scratchpad import format_to_openai_function_messages
                from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
                from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
                
                # Create a video editing tool
                class VideoEditingTool(BaseTool):
                    name = "video_editor"
                    description = "Edit videos based on a natural language description"
                    
                    def _run(self, task: str):
                        # This is a synchronous implementation that will be called by the agent
                        raise NotImplementedError("This tool only supports async execution")
                    
                    async def _arun(self, task: str, runnable_config=None):
                        # Create a simple request based on the task
                        request = VideoEditRequest(
                            video_paths=[f"{remote_input_dir}/circles_warm.mp4", f"{remote_input_dir}/rectangles_cool.mp4"],
                            output_path=f"{processor.workspace_dir}/llm_output.mp4"
                        )
                        
                        # Use a simple edit plan for demonstration
                        # In a real implementation, this would parse the task and create a more complex edit plan
                        scene = Scene(
                            duration=5.0,
                            clips=[
                                Clip(
                                    source_index=0,
                                    start_time=0.0,
                                    duration=2.5,
                                    position=Position(x=0.0, y=0.0, width=1.0, height=1.0),
                                    effects=[Effect(type=EffectType.GRAYSCALE, start_time=0.0, duration=2.5)]
                                ),
                                Clip(
                                    source_index=1,
                                    start_time=0.0,
                                    duration=2.5,
                                    position=Position(x=0.0, y=0.0, width=1.0, height=1.0),
                                    effects=[]
                                )
                            ],
                            captions=[
                                Caption(
                                    text=f"LLM Generated: {task}",
                                    position=Position(x=0.5, y=0.1, width=0.8, height=0.1),
                                    start_time=0.5,
                                    duration=4.0
                                )
                            ],
                            transition_out=Transition(type=TransitionType.FADE, duration=0.5)
                        )
                        
                        edit_plan = VideoEditPlan(
                            scenes=[scene, scene]  # Use the same scene twice with a transition
                        )
                        
                        # Process the video
                        processor.process_video(edit_plan, request)
                        
                        # Download the result
                        local_output_path = str(output_dir / "llm_generated_output.mp4")
                        processor.file_transfer.download_file(request.output_path, local_output_path)
                        
                        return f"Video edited and saved to {local_output_path}"
                
                # Create the LLM
                llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
                
                # Create the video editing tool
                video_tool = VideoEditingTool()
                
                # Create a simple agent
                tools = [video_tool]
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant that can edit videos."),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                
                agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: format_to_openai_function_messages(
                            x["intermediate_steps"]
                        ),
                    }
                    | prompt
                    | llm.bind(functions=[tool.to_openai_function() for tool in tools])
                    | OpenAIFunctionsAgentOutputParser()
                )
                
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                
                # Run the agent
                task = "Create a video that transitions from grayscale to color"
                runnable_config = {"run_name": "Video Editing Agent"}
                
                # Print the edit plan for demonstration
                print("\nExample edit plan:")
                for i, scene in enumerate(edit_plan.scenes):
                    print(f"Scene {i+1}:")
                    for j, clip in enumerate(scene.clips):
                        print(f"  Clip {j+1}:")
                        print(f"  - Source: {request.video_paths[clip.source_index]}")
                        print(f"  - Position: x={clip.position.x}, y={clip.position.y}, width={clip.position.width}, height={clip.position.height}")
                        print(f"  - Effects: {[e.type.name for e in clip.effects]}")
                    print("- Captions:", [c.text for c in scene.captions])
                    if scene.transition_out:
                        print(f"- Transition: {scene.transition_out.type.name} ({scene.transition_out.duration}s)")
                
                print("\nProcessing video with LLM-generated edit plan...")
                result = await video_tool._arun(task, runnable_config)
                print(f"\nResult: {result}")
                
                # Verify output file
                llm_output_path = Path(request.output_path)
                local_llm_output = str(output_dir / "llm_generated_output.mp4")
                processor.file_transfer.download_file(str(llm_output_path), local_llm_output)
                
                print(f"\nLLM-generated video saved to {local_llm_output}")
                
            except Exception as e:
                print(f"\nError processing LLM-based video editing: {str(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError in Hyperbolic GPU demo: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
    finally:
        # Clean up
        print("\nCleaning up resources...")
        processor.cleanup()
        
        # Remove test file
        if os.path.exists(test_file):
            os.remove(test_file)

async def main():
    """Run the complete demonstration."""
    print("Starting Hyperbolic Video Agent Demonstration")
    print("=" * 80)
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please fix the issues and try again.")
        return
    
    # Generate test videos
    generate_test_videos()
    
    # Run Hyperbolic video demo
    await run_hyperbolic_video_demo()
    
    print("\nDemonstration complete!")
    print("Check the 'hyperbolic_outputs' directory for the processed videos.")

if __name__ == "__main__":
    asyncio.run(main()) 