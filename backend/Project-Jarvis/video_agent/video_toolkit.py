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
    
    def _parse_edit_request(self, query: str) -> Dict[str, Any]:
        """Use LLM to parse natural language query into structured edit request."""
        print("\nParsing edit request...")
        # Analyze videos first
        video_info = self._analyze_videos()
        
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

Available effects: {self.knowledge_base.get_supported_effects()}
Available transitions: {self.knowledge_base.get_supported_transitions()}
Available compositions: {self.knowledge_base.get_supported_compositions()}

Request: {query}

You must respond with ONLY a valid JSON object in the following format, with no additional text or explanation:

{{
    "output_name": "descriptive_name",
    "scenes": [
        {{
            "duration": 10.0,
            "clips": [
                {{
                    "source_video": "input_videos/video1.mp4",
                    "start_time": 0,
                    "end_time": 10.0,
                    "position": {{"x": 0.0, "y": 0.5, "width": 0.5, "height": 1.0}},
                    "effects": [
                        {{
                            "type": "blur",
                            "params": {{"strength": 5}},
                            "start_time": 0,
                            "end_time": 2
                        }}
                    ]
                }}
            ],
            "transitions": {{
                "type": "fade",
                "duration": 1.0
            }},
            "captions": [
                {{
                    "text": "Example Caption",
                    "start_time": 0,
                    "end_time": 5,
                    "position": {{"x": 0.5, "y": 0.0, "width": 0.8, "height": 0.1}},
                    "style": {{"font_size": 32, "bold": true}}
                }}
            ]
        }}
    ]
}}

Consider the content analysis when planning:
- Use brightness levels to determine where effects might be most effective
- Use motion analysis to plan transitions and effects
- Consider video durations when planning scene lengths
- Match resolutions appropriately for compositions
- For split-screen layouts, use x=0.0 for left side and x=0.5 for right side

Your response must be ONLY the JSON object, with no other text."""

        response = self.llm.invoke(prompt).content
        print(f"\nLLM response received:\n{response}")
        
        try:
            # First try json.loads for safer parsing
            try:
                parsed = json.loads(response)
                print("Successfully parsed response as JSON")
                return parsed
            except json.JSONDecodeError:
                # Fall back to eval if the response has Python bool values (True/False)
                print("JSON parsing failed, trying eval for Python bool values...")
                parsed = eval(response)
                print("Successfully parsed response using eval")
                return parsed
        except Exception as e:
            print(f"Failed to parse LLM response: {str(e)}")
            raise ValueError(f"Failed to parse LLM response into edit request: {str(e)}")
    
    def _create_edit_plan(self, parsed_request: Dict[str, Any]) -> Tuple[VideoEditRequest, VideoEditPlan]:
        """Convert parsed request into VideoEditRequest and VideoEditPlan."""
        print("\nCreating edit plan from parsed request...")
        output_path = str(self.output_dir / f"{parsed_request['output_name']}.mp4")
        print(f"Output path: {output_path}")
        
        # Create VideoEditRequest
        request = VideoEditRequest(
            video_paths=self._get_input_videos(),
            edit_prompt=parsed_request.get('description', ''),
            output_path=output_path
        )
        
        def convert_position(pos_data: Dict[str, Any]) -> Position:
            """Convert position data to numeric values."""
            x = pos_data['x']
            y = pos_data['y']
            
            # Convert string positions to numeric values
            if isinstance(x, str):
                x = {
                    'left': 0.0,
                    'center': 0.5,
                    'right': 1.0
                }.get(x.lower(), 0.5)
            
            if isinstance(y, str):
                y = {
                    'top': 0.0,
                    'center': 0.5,
                    'bottom': 1.0
                }.get(y.lower(), 0.5)
            
            pos = Position(
                x=float(x),
                y=float(y),
                width=float(pos_data['width']),
                height=float(pos_data['height'])
            )
            print(f"Converted position: {pos_data} -> x={pos.x}, y={pos.y}, width={pos.width}, height={pos.height}")
            return pos
        
        # Create scenes
        scenes = []
        for scene_idx, scene_data in enumerate(parsed_request['scenes']):
            print(f"\nProcessing scene {scene_idx + 1}...")
            clips = []
            for clip_idx, clip_data in enumerate(scene_data['clips']):
                print(f"\nProcessing clip {clip_idx + 1}:")
                print(f"Source video: {clip_data['source_video']}")
                
                # Get source index from path
                source_index = self._get_input_videos().index(clip_data['source_video'])
                print(f"Source index: {source_index}")
                
                # Create clip effects
                effects = []
                for effect_data in clip_data.get('effects', []):
                    effect = VideoEffect(
                        type=VideoEffectType[effect_data['type'].upper()],
                        params=effect_data.get('params', {}),
                        start_time=effect_data.get('start_time', 0),
                        end_time=effect_data.get('end_time', -1)
                    )
                    print(f"Added effect: {effect.type.name} with params {effect.params}")
                    effects.append(effect)
                
                # Create clip with converted position
                clip = ClipSegment(
                    source_index=source_index,
                    start_time=clip_data.get('start_time', 0),
                    end_time=clip_data.get('end_time', -1),
                    position=convert_position(clip_data['position']),
                    effects=effects
                )
                print(f"Created clip segment: {clip}")
                clips.append(clip)
            
            # Create scene transitions
            transition = None
            if 'transitions' in scene_data:
                transition = TransitionEffect(
                    type=TransitionType[scene_data['transitions']['type'].upper()],
                    duration=scene_data['transitions'].get('duration', 1.0)
                )
                print(f"Added transition: {transition.type.name} ({transition.duration}s)")
            
            # Create scene captions
            captions = []
            for caption_data in scene_data.get('captions', []):
                caption = Caption(
                    text=caption_data['text'],
                    start_time=caption_data.get('start_time', 0),
                    end_time=caption_data.get('end_time', -1),
                    position=convert_position(caption_data['position']),
                    style=TextStyle(**caption_data.get('style', {}))
                )
                print(f"Added caption: {caption.text} at position {caption.position}")
                captions.append(caption)
            
            # Create scene
            scene = Scene(
                duration=scene_data['duration'],
                clips=clips,
                transition_out=transition,
                captions=captions
            )
            print(f"Created scene with duration {scene.duration}s and {len(clips)} clips")
            scenes.append(scene)
        
        # Estimate GPU requirements
        effects = []
        transitions = []
        compositions = []
        for scene in scenes:
            for clip in scene.clips:
                effects.extend([e.type.name.lower() for e in clip.effects])
            if scene.transition_out:
                transitions.append(scene.transition_out.type.name.lower())
            if len(scene.clips) > 1:
                compositions.append('split_screen')
        
        gpu_reqs = self.knowledge_base.estimate_gpu_requirements(
            effects=list(set(effects)),
            transitions=list(set(transitions)),
            compositions=list(set(compositions))
        )
        print(f"\nEstimated GPU requirements: {gpu_reqs}")
        
        # Create edit plan
        plan = VideoEditPlan(
            scenes=scenes,
            estimated_gpu_requirements=gpu_reqs,
            estimated_duration=sum(s.duration for s in scenes)
        )
        print(f"\nCreated edit plan with {len(scenes)} scenes, estimated duration: {plan.estimated_duration}s")
        
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