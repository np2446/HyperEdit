"""
VideoAgentProcessor: A comprehensive class for video processing with LLM integration.
This class encapsulates the functionality of the demo scripts with LLM integration,
allowing for processing videos with arbitrary prompts using Hyperbolic GPUs.
"""

import os
import cv2
import numpy as np
import asyncio
import tempfile
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv

# LLM and agent imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, tool
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages

# Video processing imports
from .video_processor import VideoProcessor, GPURequirements
from .video_models import (
    VideoEditRequest, 
    VideoEditPlan, 
    Scene, 
    ClipSegment, 
    Position, 
    VideoEffect, 
    VideoEffectType,
    Caption, 
    TransitionEffect, 
    TransitionType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VideoAgentProcessor")

class VideoAgentProcessor:
    """
    A comprehensive class for video processing with LLM integration.
    This class encapsulates the functionality of the demo scripts with LLM integration,
    allowing for processing videos with arbitrary prompts using Hyperbolic GPUs.
    """
    
    def __init__(
        self, 
        local_mode: bool = False,
        llm_provider: str = "anthropic",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        gpu_requirements: Optional[GPURequirements] = None,
        workspace_dir: Optional[str] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize the VideoAgentProcessor.
        
        Args:
            local_mode: Whether to process videos locally (True) or use Hyperbolic GPUs (False)
            llm_provider: LLM provider to use ("anthropic" or "openai")
            llm_model: Specific model to use (defaults to claude-3-opus for Anthropic, gpt-4 for OpenAI)
            api_key: API key for the LLM provider (if None, will try to load from environment)
            gpu_requirements: GPU requirements for Hyperbolic (if None, will use defaults)
            workspace_dir: Directory to use for workspace (if None, will use default)
            log_file: Path to log file (if None, will log to console only)
        """
        # Load environment variables
        load_dotenv()
        
        # Set up logging
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        
        # Initialize properties
        self.local_mode = local_mode
        self.llm_provider = llm_provider
        self.workspace_dir = workspace_dir
        
        # Set up LLM
        self._setup_llm(llm_provider, llm_model, api_key)
        
        # Initialize video processor
        self.processor = VideoProcessor(local_mode=local_mode)
        
        # Set up default GPU requirements if not provided
        if not gpu_requirements and not local_mode:
            self.gpu_requirements = GPURequirements(
                gpu_type="RTX",  # Prefer RTX GPUs which are typically cheaper
                num_gpus=1,
                min_vram_gb=4.0,
                disk_size=10,
                memory=16
            )
        else:
            self.gpu_requirements = gpu_requirements
        
        # Create output directories
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Track active instances
        self.instance_id = None
    
    def _setup_llm(self, provider: str, model: Optional[str], api_key: Optional[str]):
        """
        Set up the LLM based on the provider.
        
        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Specific model to use
            api_key: API key for the provider
        """
        if provider.lower() == "anthropic":
            # Use Anthropic Claude
            if not model:
                model = "claude-3-opus-20240229"
            
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            
            self.llm = ChatAnthropic(model=model, temperature=0, anthropic_api_key=api_key)
            logger.info(f"Using Anthropic LLM: {model}")
            
        elif provider.lower() == "openai":
            # Use OpenAI GPT
            if not model:
                model = "gpt-4"
            
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found in environment variables")
            
            self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)
            logger.info(f"Using OpenAI LLM: {model}")
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'anthropic' or 'openai'.")
    
    def check_environment(self) -> bool:
        """
        Check if the environment is properly configured.
        
        Returns:
            bool: True if environment is properly configured, False otherwise
        """
        # Check for SSH key if using remote mode
        if not self.local_mode:
            ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH')
            if not ssh_key_path:
                logger.error("SSH_PRIVATE_KEY_PATH environment variable is not set")
                return False
            
            ssh_key_path = os.path.expanduser(ssh_key_path)
            if not os.path.exists(ssh_key_path):
                logger.error(f"SSH key file not found at {ssh_key_path}")
                return False
            
            logger.info(f"Using SSH key: {ssh_key_path}")
        
        # Check for LLM API keys
        if self.llm_provider.lower() == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                logger.warning("ANTHROPIC_API_KEY environment variable is not set")
                return False
        elif self.llm_provider.lower() == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY environment variable is not set")
                return False
        
        return True
    
    def create_test_pattern_video(
        self,
        output_path: str,
        pattern_type: str = "circles",
        duration: int = 5,
        fps: int = 30,
        resolution: tuple = (1280, 720),
        color_scheme: str = "random"
    ) -> None:
        """
        Create a test video with moving patterns and text.
        
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
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Test Pattern: {pattern_type.capitalize()} ({color_scheme})"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        # Generate frames
        for frame_idx in range(duration * fps):
            # Create a black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Update and draw patterns
            for pattern in patterns:
                if pattern['type'] == 'circle':
                    # Update position
                    pattern['pos'][0] += pattern['vel'][0]
                    pattern['pos'][1] += pattern['vel'][1]
                    
                    # Bounce off walls
                    if pattern['pos'][0] <= pattern['radius'] or pattern['pos'][0] >= width - pattern['radius']:
                        pattern['vel'][0] *= -1
                    if pattern['pos'][1] <= pattern['radius'] or pattern['pos'][1] >= height - pattern['radius']:
                        pattern['vel'][1] *= -1
                    
                    # Draw circle
                    cv2.circle(frame, tuple(map(int, pattern['pos'])), pattern['radius'], pattern['color'], -1)
                
                elif pattern['type'] == 'rect':
                    # Update angle
                    pattern['angle'] += pattern['angle_vel']
                    
                    # Create rotation matrix
                    rect_center = tuple(map(int, pattern['pos']))
                    rect_size = pattern['size']
                    angle = pattern['angle']
                    
                    # Calculate rectangle corners
                    half_width, half_height = rect_size[0] // 2, rect_size[1] // 2
                    corners = [
                        [-half_width, -half_height],
                        [half_width, -half_height],
                        [half_width, half_height],
                        [-half_width, half_height]
                    ]
                    
                    # Rotate corners
                    import math
                    rotated_corners = []
                    for x, y in corners:
                        new_x = x * math.cos(angle) - y * math.sin(angle) + rect_center[0]
                        new_y = x * math.sin(angle) + y * math.cos(angle) + rect_center[1]
                        rotated_corners.append([int(new_x), int(new_y)])
                    
                    # Draw rotated rectangle
                    pts = np.array(rotated_corners, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(frame, [pts], pattern['color'])
            
            # Add timestamp
            timestamp = f"Frame: {frame_idx}/{duration * fps} ({frame_idx / fps:.1f}s)"
            cv2.putText(frame, timestamp, (10, 30), font, 0.7, (255, 255, 255), 2)
            
            # Add pattern info
            cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
        
        # Release resources
        out.release()
        logger.info(f"Created test video: {output_path}")
    
    def generate_test_videos(self, output_dir: str = "input_videos") -> List[str]:
        """
        Generate test videos for demonstration.
        
        Args:
            output_dir: Directory to save test videos
            
        Returns:
            List[str]: Paths to generated test videos
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate test videos
        video_paths = []
        
        # Circles with warm colors
        circles_path = str(output_path / "circles_warm.mp4")
        self.create_test_pattern_video(
            circles_path,
            pattern_type="circles",
            color_scheme="warm",
            duration=5
        )
        video_paths.append(circles_path)
        
        # Rectangles with cool colors
        rectangles_path = str(output_path / "rectangles_cool.mp4")
        self.create_test_pattern_video(
            rectangles_path,
            pattern_type="rectangles",
            color_scheme="cool",
            duration=5
        )
        video_paths.append(rectangles_path)
        
        logger.info(f"Generated {len(video_paths)} test videos in {output_dir}")
        return video_paths
    
    async def setup_environment(self) -> bool:
        """
        Set up the processing environment (local or remote).
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if self.local_mode:
            logger.info("Using local processing mode")
            # Initialize local processor
            try:
                self.processor.setup_local_environment()
                return True
            except Exception as e:
                logger.error(f"Error setting up local environment: {str(e)}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                return False
        
        try:
            # Initialize GPU environment
            logger.info("Setting up GPU environment...")
            logger.info("This may take several minutes. Please be patient.")
            
            self.processor.setup_gpu_environment(self.gpu_requirements)
            
            self.instance_id = self.processor.instance_id
            logger.info(f"GPU environment set up successfully! Instance ID: {self.instance_id}")
            
            # Test file transfer
            logger.info("Testing file transfer...")
            test_file = "test_file.txt"
            with open(test_file, "w") as f:
                f.write("This is a test file for Hyperbolic GPU integration.")
            
            remote_test_path = f"{self.processor.workspace_dir}/test_file.txt"
            self.processor.file_transfer.upload_file(test_file, remote_test_path)
            logger.info("File upload successful!")
            
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    async def upload_videos(self, video_paths: List[str]) -> List[str]:
        """
        Upload videos to the processing environment.
        
        Args:
            video_paths: List of local video paths
            
        Returns:
            List[str]: List of remote video paths
        """
        if self.local_mode:
            # In local mode, just return the original paths
            return video_paths
        
        try:
            # Create remote input directory
            remote_input_dir = f"{self.processor.workspace_dir}/input_videos"
            from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
            execute_remote_command(f"mkdir -p {remote_input_dir}", instance_id=self.instance_id, timeout=30)
            
            # Upload each video
            remote_paths = []
            for i, video_path in enumerate(video_paths):
                logger.info(f"Uploading {video_path} ({i+1}/{len(video_paths)})...")
                video_file = Path(video_path)
                remote_path = f"{remote_input_dir}/{video_file.name}"
                self.processor.file_transfer.upload_file(str(video_path), remote_path)
                remote_paths.append(remote_path)
            
            logger.info(f"Uploaded {len(remote_paths)} videos")
            return remote_paths
            
        except Exception as e:
            logger.error(f"Error uploading videos: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return []
    
    def create_video_editing_tool(self):
        """
        Create a video editing tool for the LLM agent.
        
        Returns:
            The video editing tool
        """
        processor = self.processor
        
        @tool
        async def edit_video(task: str) -> str:
            """
            Edit videos based on the given task description.
            
            Args:
                task: Natural language description of the video editing task
                
            Returns:
                str: Path to the output video
            """
            try:
                # Get the list of available videos
                if self.local_mode:
                    video_dir = Path("input_videos")
                    video_paths = list(video_dir.glob("*.mp4"))
                    video_paths = [str(p) for p in video_paths]
                else:
                    remote_input_dir = f"{processor.workspace_dir}/input_videos"
                    video_paths = processor.file_transfer.list_remote_files(remote_input_dir)
                    video_paths = [p for p in video_paths if p.endswith(".mp4")]
                
                if not video_paths:
                    return "No videos available for editing."
                
                # Create a simple edit plan based on the task
                # This is a simplified version - in a real implementation,
                # you would use the LLM to generate a more complex edit plan
                
                # Define output path
                if self.local_mode:
                    output_path = str(self.output_dir / "output.mp4")
                else:
                    output_path = f"{processor.workspace_dir}/output/output.mp4"
                    # Create output directory
                    from hyperbolic_agentkit_core.actions.remote_shell import execute_remote_command
                    execute_remote_command(f"mkdir -p {processor.workspace_dir}/output", 
                                          instance_id=processor.instance_id, 
                                          timeout=30)
                
                # Create a simple scene with the first video
                scene = Scene(
                    duration=5.0,
                    clips=[
                        ClipSegment(
                            source_index=0,
                            start_time=0.0,
                            end_time=5.0,
                            position=Position(x=0.0, y=0.0, width=1.0, height=1.0),
                            effects=[]
                        )
                    ],
                    captions=[
                        Caption(
                            text=f"LLM Generated: {task}",
                            position=Position(x=0.5, y=0.1, width=0.8, height=0.1),
                            start_time=0.5,
                            end_time=4.5,
                            duration=4.0
                        )
                    ],
                    transition_out=TransitionEffect(type=TransitionType.FADE, duration=0.5)
                )
                
                # Create the edit plan
                edit_plan = VideoEditPlan(
                    scenes=[scene],
                    estimated_duration=5.0,
                    estimated_gpu_requirements={"vram_gb": 4.0, "compute_units": 1.0}
                )
                
                # Create the edit request
                request = VideoEditRequest(
                    video_paths=video_paths,
                    edit_prompt=task,
                    output_path=output_path
                )
                
                # Process the video
                processor.process_video(edit_plan, request)
                
                # Download the result if in remote mode
                if not self.local_mode:
                    local_output_path = str(self.output_dir / "output.mp4")
                    processor.file_transfer.download_file(output_path, local_output_path)
                    output_path = local_output_path
                
                return f"Video edited and saved to {output_path}"
                
            except Exception as e:
                logger.error(f"Error in edit_video tool: {str(e)}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                return f"Error editing video: {str(e)}"
        
        return edit_video
    
    async def process_with_llm(self, prompt: str) -> str:
        """
        Process videos using the LLM agent.
        
        Args:
            prompt: Natural language prompt describing the video editing task
            
        Returns:
            str: Result of the processing
        """
        try:
            # Create the video editing tool
            video_tool = self.create_video_editing_tool()
            
            # For simplicity, let's just use the tool directly without a complex agent
            logger.info(f"Processing prompt: {prompt}")
            
            # Call the video editing tool directly using the invoke method
            result = await video_tool.ainvoke(prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return f"Error processing with LLM: {str(e)}"
    
    async def process_videos(self, prompt: str, video_paths: Optional[List[str]] = None) -> str:
        """
        Process videos with the given prompt.
        
        Args:
            prompt: Natural language prompt describing the video editing task
            video_paths: List of video paths to process (if None, will generate test videos)
            
        Returns:
            str: Result of the processing
        """
        try:
            # Check environment
            if not self.check_environment():
                return "Environment check failed. Please fix the issues and try again."
            
            # Generate test videos if not provided
            if not video_paths:
                video_paths = self.generate_test_videos()
            
            # Set up environment
            if not await self.setup_environment():
                return "Failed to set up environment."
            
            # Upload videos
            remote_paths = await self.upload_videos(video_paths)
            if not remote_paths:
                return "Failed to upload videos."
            
            # Process with LLM
            result = await self.process_with_llm(prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing videos: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return f"Error processing videos: {str(e)}"
        finally:
            # Clean up
            if not self.local_mode and self.processor:
                logger.info("Cleaning up resources...")
                self.processor.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if not self.local_mode and self.processor:
            logger.info("Cleaning up resources...")
            self.processor.cleanup() 