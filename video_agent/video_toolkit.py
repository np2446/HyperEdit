"""
Video processing toolkit for AgentKit.
Provides tools for video editing, effects, and GPU-accelerated processing.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import Field
import asyncio
from langchain.schema import HumanMessage
import json

from .video_processor import VideoProcessor, GPURequirements
from .video_models import (
    VideoEditRequest, VideoEditPlan, Scene, ClipSegment,
    Position, VideoEffect, AudioEffect, Caption,
    VideoEffectType, AudioEffectType, TransitionEffect,
    TransitionType, TextStyle
)
from .video_knowledge_base import VideoKnowledgeBase

class VideoInfo:
    """Class to store video analysis information."""
    def __init__(self, video_path: str):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Basic metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate properties
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid video dimensions: {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"Invalid video framerate: {self.fps}")
        if self.frame_count <= 0:
            raise ValueError(f"Invalid frame count: {self.frame_count}")
            
        self.duration = self.frame_count / self.fps
        
        try:
            # Sample frames for analysis
            self.samples = self._analyze_samples()
        finally:
            # Always release the video capture
            self.cap.release()
    
    def _analyze_samples(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Analyze sample frames from the video."""
        samples = []
        frame_indices = np.linspace(0, self.frame_count - 1, num_samples, dtype=int)
        
        prev_frame = None
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                print(f"Warning: Could not read frame at index {idx}")
                continue
                
            try:
                # Calculate average brightness
                brightness = np.mean(frame)
                
                # Calculate motion (if not first frame)
                motion = 0
                if prev_frame is not None:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    motion = np.mean(cv2.absdiff(prev_gray, curr_gray))
                
                samples.append({
                    'frame': frame,
                    'timestamp': idx / self.fps,
                    'brightness': brightness,
                    'motion': motion
                })
                
                prev_frame = frame
            except Exception as e:
                print(f"Warning: Error analyzing frame at index {idx}: {e}")
                continue
        
        if not samples:
            raise ValueError("Could not analyze any frames from the video")
        
        return samples
    
    def __str__(self) -> str:
        return f"Video: {self.path} ({self.width}x{self.height} @ {self.fps}fps, {self.duration:.1f}s)"

class VideoTool(BaseTool):
    """Tool for processing videos using natural language instructions."""
    name: str = "video_processor"
    description: str = "Process and edit videos based on natural language instructions"
    llm: BaseChatModel = Field(description="Language model to use for processing requests")
    processor: VideoProcessor = Field(default_factory=lambda: VideoProcessor(local_mode=True))
    knowledge_base: VideoKnowledgeBase = Field(default_factory=VideoKnowledgeBase)
    analyzed_videos: Dict[str, VideoInfo] = Field(default_factory=dict)
    input_dir: Path = Field(default_factory=lambda: Path("input_videos"))
    output_dir: Path = Field(default_factory=lambda: Path("test_outputs"))
    
    def __init__(self, **data):
        super().__init__(**data)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        print(f"VideoTool initialized with input_dir={self.input_dir}, output_dir={self.output_dir}")
    
    def _get_input_videos(self) -> List[str]:
        """Get list of available input videos."""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        videos = []
        for file in self.input_dir.iterdir():
            if file.suffix.lower() in video_extensions:
                videos.append(str(file))
        print(f"Found {len(videos)} input videos: {videos}")
        return sorted(videos)
    
    def _analyze_videos(self) -> Dict[str, VideoInfo]:
        """Analyze all input videos and cache the results."""
        print("\nAnalyzing videos...")
        videos = self._get_input_videos()
        
        if not videos:
            print("No videos found in input directory")
            return {}
        
        # Clear cache of non-existent videos
        self.analyzed_videos = {
            path: info for path, info in self.analyzed_videos.items()
            if path in videos
        }
        
        # Analyze new videos
        for video_path in videos:
            if video_path not in self.analyzed_videos:
                try:
                    print(f"\nAnalyzing {video_path}...")
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        raise ValueError(f"Could not open video: {video_path}")
                    
                    # Basic metadata
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    if width == 0 or height == 0 or fps == 0:
                        raise ValueError(f"Invalid video properties: width={width}, height={height}, fps={fps}")
                    
                    print(f"Video properties: {width}x{height} @ {fps}fps, {duration:.1f}s")
                    
                    # Create VideoInfo object with basic metadata
                    self.analyzed_videos[video_path] = VideoInfo(video_path)
                    print(f"Analysis complete: {self.analyzed_videos[video_path]}")
                    
                    cap.release()
                except Exception as e:
                    print(f"Error analyzing video {video_path}: {e}")
                    # Don't add failed videos to analyzed_videos
                    continue
        
        if not self.analyzed_videos:
            raise ValueError("No valid videos could be analyzed")
        
        return self.analyzed_videos
    
    def _parse_edit_request(self, query: str) -> Dict[str, Any]:
        """Use LLM to parse natural language query into structured edit request."""
        print("\nParsing edit request...")
        
        # Extract video name from the query if specified
        target_video = None
        if "Process only the video '" in query:
            target_video = query.split("Process only the video '")[1].split("'")[0]
            query = query.split(" with the following instructions: ")[1]
        
        # Analyze videos first
        video_info = self._analyze_videos()
        
        # Filter to only the target video if specified
        if target_video:
            video_info = {
                path: info for path, info in video_info.items()
                if Path(path).name == target_video
            }
            if not video_info:
                raise ValueError(f"Target video '{target_video}' not found")
        
        # Create video information summary
        video_summaries = []
        for path, info in video_info.items():
            summary = f"""
Video: {info.path}
- Duration: {info.duration:.2f} seconds
- Resolution: {info.width}x{info.height}
- FPS: {info.fps}
- Content Analysis:
  * Average brightness varies from {min(s['brightness'] for s in info.samples):.1f} to {max(s['brightness'] for s in info.samples):.1f}
  * Motion levels vary from {min(s['motion'] for s in info.samples):.1f} to {max(s['motion'] for s in info.samples):.1f}
  * Key timestamps: {', '.join(f"{s['timestamp']:.1f}s" for s in info.samples)}
"""
            video_summaries.append(summary)
            print(f"Video summary:\n{summary}")
        
        print("\nSending request to LLM...")
        prompt = f"""Parse the following video editing request into a structured format.

Available input videos:
{chr(10).join(video_summaries)}

Request: {query}

You must respond with ONLY a valid JSON object in the following format:

{{
    "output_name": "videoplayback_captioned",
    "scenes": [
        {{
            "duration": {list(video_info.values())[0].duration},
            "clips": [
                {{
                    "source_video": "{list(video_info.keys())[0]}",
                    "start_time": 0,
                    "end_time": {list(video_info.values())[0].duration},
                    "position": {{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}},
                    "effects": []
                }}
            ],
            "captions": [
                {{
                    "text": "Add your caption text here",
                    "start_time": 0,
                    "end_time": 5,
                    "position": {{"x": 0.5, "y": 0.1, "width": 0.8, "height": 0.1}},
                    "style": {{"font_size": 32, "bold": true}}
                }}
            ]
        }}
    ]
}}

Rules:
1. Use actual video duration for scene and clip durations (shown in video summaries)
2. Keep effects array empty unless specifically requested
3. Adjust caption text, timing, and style based on the request
4. Use the exact video path as shown in the available videos list
5. All position values must be between 0.0 and 1.0
6. output_name should be descriptive of the edit being made

Your response must be ONLY the JSON object, with no other text."""

        response = self.llm.invoke(prompt).content
        print(f"\nLLM response received:\n{response}")
        
        try:
            # Parse the JSON response
            parsed = json.loads(response)
            print("Successfully parsed response as JSON")
            
            # Ensure all required fields are present with defaults
            for scene in parsed.get("scenes", []):
                scene.setdefault("duration", -1)
                scene.setdefault("clips", [])
                scene.setdefault("captions", [])
                
                for clip in scene["clips"]:
                    clip.setdefault("effects", [])
                    clip.setdefault("start_time", 0)
                    clip.setdefault("end_time", -1)
                    clip.setdefault("position", {
                        "x": 0.0,
                        "y": 0.0,
                        "width": 1.0,
                        "height": 1.0
                    })
                
                for caption in scene["captions"]:
                    caption.setdefault("style", {"font_size": 32, "bold": True})
                    caption.setdefault("start_time", 0)
                    caption.setdefault("end_time", 5)
                    caption.setdefault("position", {
                        "x": 0.5,
                        "y": 0.1,
                        "width": 0.8,
                        "height": 0.1
                    })
            
            return parsed
            
        except Exception as e:
            print(f"Failed to parse LLM response: {str(e)}")
            raise ValueError(f"Failed to parse LLM response into edit request: {str(e)}")
    
    def _create_edit_plan(self, parsed_request: Dict[str, Any]) -> Tuple[VideoEditRequest, VideoEditPlan]:
        """Convert parsed request into VideoEditRequest and VideoEditPlan."""
        print("\nCreating edit plan from parsed request...")
        output_path = str(self.output_dir / f"{parsed_request['output_name']}.mp4")
        print(f"Output path: {output_path}")
        
        # Get absolute paths for input videos and analyze them
        input_videos = self._get_input_videos()
        print(f"Input videos: {input_videos}")
        
        if not input_videos:
            raise ValueError("No input videos found")
        
        # Analyze videos first
        try:
            analyzed_videos = self._analyze_videos()
            if not analyzed_videos:
                raise ValueError("No videos could be analyzed successfully")
        except Exception as e:
            raise ValueError(f"Failed to analyze videos: {str(e)}")
        
        # Create VideoEditRequest with absolute paths
        request = VideoEditRequest(
            video_paths=input_videos,
            edit_prompt=parsed_request.get('description', ''),
            output_path=output_path
        )
        
        # Create a simple scene with all videos
        clips = []
        for scene_data in parsed_request['scenes']:
            for clip_data in scene_data['clips']:
                # Get the video path
                source_video = clip_data['source_video']
                print(f"Processing source video: {source_video}")
                
                # Find the matching video in input_videos
                source_name = Path(source_video).name
                matching_paths = [p for p in input_videos if Path(p).name == source_name]
                if not matching_paths:
                    raise ValueError(f"Video not found: {source_video}")
                source_video = matching_paths[0]
                
                # Verify the video was analyzed successfully
                if source_video not in analyzed_videos:
                    raise ValueError(f"Video could not be analyzed: {source_video}")
                
                # Get source index from absolute path
                source_index = input_videos.index(source_video)
                print(f"Using video at index {source_index}: {source_video}")
                
                # Get video info for duration
                video_info = analyzed_videos[source_video]
                if video_info.duration <= 0:
                    raise ValueError(f"Invalid video duration for {source_video}: {video_info.duration}")
                
                # Create clip with position
                clip = ClipSegment(
                    source_index=source_index,
                    start_time=clip_data.get('start_time', 0),
                    end_time=video_info.duration if clip_data.get('end_time', -1) < 0 else clip_data['end_time'],
                    position=Position(
                        x=float(clip_data['position']['x']),
                        y=float(clip_data['position']['y']),
                        width=float(clip_data['position']['width']),
                        height=float(clip_data['position']['height'])
                    ),
                    effects=[]  # Keep effects empty for now
                )
                print(f"Created clip segment: {clip}")
                clips.append(clip)
        
        if not clips:
            raise ValueError("No valid clips could be created")
        
        # Calculate total duration from clips
        max_duration = max((clip.end_time for clip in clips), default=0)
        if max_duration <= 0:
            raise ValueError("Invalid scene duration")
        
        # Create scene with clips
        scene = Scene(
            duration=max_duration,
            clips=clips,
            transition_out=None,  # No transitions for now
            captions=[]  # No captions for now
        )
        print(f"Created scene with duration {scene.duration}s and {len(clips)} clips")
        
        # Create edit plan with minimal GPU requirements
        plan = VideoEditPlan(
            scenes=[scene],
            estimated_gpu_requirements={"min_vram_gb": 4.0, "gpu_count": 1},
            estimated_duration=max_duration
        )
        print(f"\nCreated edit plan with {len(plan.scenes)} scenes")
        
        return request, plan
    
    async def _arun(self, query: str, runnable_config: Optional[RunnableConfig] = None) -> str:
        """Process a video editing request."""
        try:
            # Get list of available videos
            videos = self._get_input_videos()
            if not videos:
                return "No input videos found in the input_videos directory."
            
            # Process the request directly if it's a dictionary
            if isinstance(query, dict):
                try:
                    print("\nProcessing direct dictionary request...")
                    request, edit_plan = self._create_edit_plan(query)
                    output_path = self.processor.process_video(edit_plan, request)
                    return f"Video processing complete! Output saved to: {output_path}"
                except Exception as e:
                    return f"Error processing request: {str(e)}"
            
            # Otherwise, parse the request from text
            try:
                print("\nProcessing natural language request...")
                parsed_request = self._parse_edit_request(query)
                print(f"\nParsed request:\n{json.dumps(parsed_request, indent=2)}")
                
                print("\nCreating edit plan...")
                request, edit_plan = self._create_edit_plan(parsed_request)
                
                print("\nProcessing video with edit plan...")
                output_path = self.processor.process_video(edit_plan, request)
                return f"Video processing complete! Output saved to: {output_path}"
            except ValueError as e:
                return f"Error parsing LLM response: {str(e)}"
            except Exception as e:
                return f"Error processing request: {str(e)}"
            
        except Exception as e:
            return f"Error processing video: {str(e)}"
    
    def _run(self, query: str, runnable_config: Optional[RunnableConfig] = None) -> str:
        """Synchronous version of video processing."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._arun(query, runnable_config))

class VideoToolkit:
    """Toolkit for video processing capabilities."""
    
    def __init__(self, llm: ChatOpenAI = None):
        """Initialize the video toolkit."""
        self.llm = llm
    
    def get_tools(self) -> List[BaseTool]:
        """Get the list of tools in the toolkit."""
        return [VideoTool(llm=self.llm)]
    
    @classmethod
    def from_llm(cls, llm: ChatOpenAI = None) -> "VideoToolkit":
        """Create a VideoToolkit from an LLM."""
        return cls(llm=llm) 