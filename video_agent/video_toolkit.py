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
import aiohttp
import logging
from datetime import datetime
import traceback

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
        
        # Basic metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        # Sample frames for analysis
        self.samples = self._analyze_samples()
        self.cap.release()
    
    def _analyze_samples(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Analyze sample frames from the video."""
        samples = []
        frame_indices = np.linspace(0, self.frame_count - 1, num_samples, dtype=int)
        
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                # Calculate average brightness
                brightness = np.mean(frame)
                
                # Calculate motion (if not first frame)
                motion = 0
                if len(samples) > 0:
                    prev_frame = cv2.cvtColor(samples[-1]['frame'], cv2.COLOR_BGR2GRAY)
                    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    motion = np.mean(cv2.absdiff(prev_frame, curr_frame))
                
                samples.append({
                    'frame': frame,
                    'timestamp': idx / self.fps,
                    'brightness': brightness,
                    'motion': motion
                })
        
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
    execution_service_url: str = Field(default="http://localhost:4003")
    
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
                    self.analyzed_videos[video_path] = VideoInfo(video_path)
                    print(f"Analysis complete: {self.analyzed_videos[video_path]}")
                except Exception as e:
                    print(f"Error analyzing video {video_path}: {e}")
        
        return self.analyzed_videos
    
    def _verify_llm_response_sync(self, prompt: str, response: str) -> Dict[str, Any]:
        """Synchronous version of LLM response verification."""
        try:
            print("\n=== Starting LLM Response Verification ===")
            print(f"Execution Service URL: {self.execution_service_url}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            
            import requests
            
            payload = {
                "query": prompt,
                "response": response,
                "taskDefinitionId": 0
            }
            
            verification_url = f"{self.execution_service_url}/task/execute"
            print(f"\nSending verification request to: {verification_url}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            headers = {"Content-Type": "application/json"}
            resp = requests.post(
                verification_url,
                json=payload,
                headers=headers
            )
            
            print(f"\nResponse status: {resp.status_code}")
            if resp.status_code == 200:
                result = resp.json()
                print(f"Verification result: {json.dumps(result, indent=2)}")
                
                tx_hash = result.get("data", {}).get("proofOfTask")
                chain_url = f"https://sepolia.etherscan.io/tx/{tx_hash}" if tx_hash else None
                
                return {
                    "is_valid": True,
                    "verification_id": tx_hash,
                    "chain_url": chain_url,
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": result
                }
            else:
                error_text = resp.text
                print(f"Verification failed: {error_text}")
                return {
                    "is_valid": False,
                    "error": f"Verification request failed with status {resp.status_code}: {error_text}"
                }
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            traceback.print_exc()
            return {
                "is_valid": False,
                "error": str(e)
            }

    def _get_verified_llm_response_sync(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Synchronous version of getting verified LLM response."""
        response = self.llm.invoke(prompt).content
        print(f"\nLLM response received:\n{response}")
        
        verification_result = self._verify_llm_response_sync(prompt, response)
        if not verification_result["is_valid"]:
            error_msg = verification_result.get("error", "Response verification failed")
            raise ValueError(f"Invalid LLM response: {error_msg}")
        
        return response, verification_result

    def _parse_edit_request_sync(self, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Synchronous version of parsing edit request."""
        print("\n=== Starting Edit Request Parsing ===")
        print(f"Input query: {query}")
        
        # Extract video name from the query if specified
        target_video = None
        if "Process only the video '" in query:
            target_video = query.split("Process only the video '")[1].split("'")[0]
            query = query.split(" with the following instructions: ")[1]
            print(f"Extracted target video: {target_video}")
        
        # Analyze videos first
        video_info = self._analyze_videos()
        print(f"\nAnalyzed {len(video_info)} videos")
        
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
        
        print("\n=== Sending request to LLM for verification ===")
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

        # Get verified response from LLM
        response, verification_details = self._get_verified_llm_response_sync(prompt)
        print("\nVerification complete")
        print(f"Verification details: {json.dumps(verification_details, indent=2)}")
        
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
            
            return parsed, verification_details
            
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
                try:
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
                    
                    # Calculate clip timing
                    start_time = clip_data.get('start_time', 0)
                    end_time = video_info.duration if clip_data.get('end_time', -1) < 0 else clip_data['end_time']
                    
                    # Create clip with position and audio effects
                    clip = ClipSegment(
                        source_index=source_index,
                        start_time=start_time,
                        end_time=end_time,
                        position=Position(
                            x=float(clip_data['position']['x']),
                            y=float(clip_data['position']['y']),
                            width=float(clip_data['position']['width']),
                            height=float(clip_data['position']['height'])
                        ),
                        effects=[],  # Keep video effects empty for now
                        audio_effects=[
                            AudioEffect(
                                type=AudioEffectType.VOLUME,
                                params={"volume": 1.0 / len(scene_data['clips'])},  # Normalize volume based on number of clips
                                start_time=start_time,  # Add start time for audio effect
                                end_time=end_time,  # Add end time for audio effect
                                fade_in=0.5,  # Add a small fade in
                                fade_out=0.5  # Add a small fade out
                            )
                        ]
                    )
                    print(f"Created clip segment: {clip}")
                    clips.append(clip)
                except Exception as e:
                    print(f"Error processing clip: {str(e)}")
                    continue
        
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
    
    def _run(self, query: str, runnable_config: Optional[RunnableConfig] = None) -> str:
        """Synchronous version of video processing."""
        try:
            # Get list of available videos
            videos = self._get_input_videos()
            if not videos:
                return {
                    "status": "error",
                    "message": "No input videos found in the input_videos directory."
                }
            
            try:
                print("\nProcessing request...")
                # Convert dictionary to string if needed
                if isinstance(query, dict):
                    query_str = json.dumps(query)
                else:
                    query_str = query
                
                # Always parse and verify the request
                parsed_request, verification_details = self._parse_edit_request_sync(query_str)
                print(f"\nParsed request:\n{json.dumps(parsed_request, indent=2)}")
                
                print("\nCreating edit plan...")
                request, edit_plan = self._create_edit_plan(parsed_request)
                
                print("\nProcessing video with edit plan...")
                output_path = self.processor.process_video(edit_plan, request)
                
                return {
                    "status": "success",
                    "message": "Video processing complete!",
                    "output_path": str(output_path),
                    "verification": verification_details
                }
                
            except ValueError as e:
                return {
                    "status": "error",
                    "message": f"Error parsing request: {str(e)}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error processing request: {str(e)}"
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing video: {str(e)}"
            }

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