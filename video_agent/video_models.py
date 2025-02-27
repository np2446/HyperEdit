"""
Data models for video editing operations.
These models define the structure of video editing requests and plans.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

class Position(BaseModel):
    """Defines position and size of a clip in the composition."""
    x: Union[float, str] = Field(
        ..., 
        description="X position (0-1 or 'left', 'center', 'right')"
    )
    y: Union[float, str] = Field(
        ..., 
        description="Y position (0-1 or 'top', 'center', 'bottom')"
    )
    width: float = Field(
        ..., 
        description="Width as fraction of composition (0-1)",
        ge=0,
        le=1
    )
    height: float = Field(
        ..., 
        description="Height as fraction of composition (0-1)",
        ge=0,
        le=1
    )
    z_index: int = Field(
        default=0, 
        description="Stack order"
    )

class TransitionType(str, Enum):
    """Types of transitions between clips or scenes."""
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    SLIDE = "slide"
    NONE = "none"

class TransitionEffect(BaseModel):
    """Defines a transition between clips or scenes."""
    type: TransitionType = Field(
        ..., 
        description="Type of transition"
    )
    duration: float = Field(
        ..., 
        description="Duration in seconds",
        gt=0
    )
    params: Dict[str, Union[str, float, int]] = Field(
        default_factory=dict, 
        description="Additional parameters for the transition"
    )

class VideoEffectType(str, Enum):
    """Types of video effects."""
    COLOR_ADJUST = "color_adjust"
    BLUR = "blur"
    SHARPEN = "sharpen"
    SPEED = "speed"
    STABILIZE = "stabilize"
    DENOISE = "denoise"
    OVERLAY = "overlay"
    CROP = "crop"
    SCALE = "scale"

class VideoEffect(BaseModel):
    """Defines a video effect to apply."""
    type: VideoEffectType = Field(
        ..., 
        description="Type of effect"
    )
    params: Dict[str, Union[str, float, int]] = Field(
        default_factory=dict, 
        description="Effect parameters"
    )
    start_time: float = Field(
        ..., 
        description="When effect starts (seconds)",
        ge=0
    )
    end_time: Optional[float] = Field(
        None, 
        description="When effect ends (seconds)"
    )

class AudioEffectType(str, Enum):
    """Types of audio effects."""
    VOLUME = "volume"
    FADE = "fade"
    EQUALIZER = "equalizer"
    NORMALIZE = "normalize"
    NOISE_REDUCTION = "noise_reduction"
    COMPRESSION = "compression"

class AudioEffect(BaseModel):
    """Defines an audio effect or modification."""
    type: AudioEffectType = Field(
        ..., 
        description="Type of audio effect"
    )
    params: Dict[str, Union[str, float, int]] = Field(
        default_factory=dict, 
        description="Effect parameters"
    )
    start_time: float = Field(
        ..., 
        description="When effect starts (seconds)",
        ge=0
    )
    end_time: Optional[float] = Field(
        None, 
        description="When effect ends (seconds)"
    )

class TextStyle(BaseModel):
    """Text styling options for captions."""
    font_family: str = Field(
        default="Arial", 
        description="Font family name"
    )
    font_size: int = Field(
        default=32, 
        description="Font size in pixels"
    )
    color: str = Field(
        default="#FFFFFF", 
        description="Text color in hex format"
    )
    background_color: Optional[str] = Field(
        None, 
        description="Background color in hex format"
    )
    bold: bool = Field(
        default=False, 
        description="Bold text"
    )
    italic: bool = Field(
        default=False, 
        description="Italic text"
    )
    stroke_width: int = Field(
        default=0, 
        description="Text outline width in pixels"
    )
    stroke_color: str = Field(
        default="#000000", 
        description="Text outline color in hex format"
    )

class Caption(BaseModel):
    """Defines a caption or text overlay."""
    text: str = Field(
        ..., 
        description="Caption text"
    )
    start_time: float = Field(
        ..., 
        description="Start time (seconds)",
        ge=0
    )
    end_time: float = Field(
        ..., 
        description="End time (seconds)",
        ge=0
    )
    position: Position = Field(
        ..., 
        description="Caption position and size"
    )
    style: TextStyle = Field(
        default_factory=TextStyle, 
        description="Text styling"
    )

class ClipSegment(BaseModel):
    """Defines a segment of a source clip to use."""
    source_index: int = Field(
        ..., 
        description="Index of source video in video_paths",
        ge=0
    )
    start_time: float = Field(
        ..., 
        description="Start time in source video (seconds)",
        ge=0
    )
    end_time: float = Field(
        ..., 
        description="End time in source video (seconds)",
        ge=0
    )
    position: Position = Field(
        ..., 
        description="Position in composition"
    )
    effects: List[VideoEffect] = Field(
        default_factory=list, 
        description="Effects to apply"
    )
    audio_effects: List[AudioEffect] = Field(
        default_factory=list, 
        description="Audio effects"
    )

class Scene(BaseModel):
    """Defines a complete scene composition."""
    duration: float = Field(
        ..., 
        description="Scene duration in seconds",
        gt=0
    )
    clips: List[ClipSegment] = Field(
        ..., 
        description="Clips in this scene"
    )
    captions: List[Caption] = Field(
        default_factory=list, 
        description="Scene captions"
    )
    background_color: Optional[str] = Field(
        None, 
        description="Background color if needed"
    )
    transition_in: Optional[TransitionEffect] = Field(
        None, 
        description="Transition from previous scene"
    )
    transition_out: Optional[TransitionEffect] = Field(
        None, 
        description="Transition to next scene"
    )

class VideoEditRequest(BaseModel):
    """Schema for video editing request parameters."""
    video_paths: List[str] = Field(
        ..., 
        description="List of paths to input videos or URLs (http/https) for remote videos"
    )
    edit_prompt: str = Field(
        ..., 
        description="Natural language description of desired edits"
    )
    output_path: str = Field(
        ..., 
        description="Desired output path for the final video"
    )
    target_duration: Optional[float] = Field(
        None,
        description="Desired duration of final video in seconds",
        gt=0
    )
    style_reference: Optional[str] = Field(
        None,
        description="URL or path to style reference video/image"
    )
    output_format: str = Field(
        default="mp4",
        description="Output video format"
    )
    output_quality: int = Field(
        default=23,
        description="Output video quality (lower is better, typical range: 17-28)",
        ge=0,
        le=51
    )
    resolution: Optional[Dict[str, int]] = Field(
        None,
        description="Output resolution (e.g., {'width': 1920, 'height': 1080})"
    )

class VideoEditPlan(BaseModel):
    """Schema for the video editing execution plan."""
    scenes: List[Scene] = Field(
        ..., 
        description="Ordered list of scenes to create"
    )
    estimated_gpu_requirements: Dict[str, float] = Field(
        ..., 
        description="Required GPU specifications"
    )
    estimated_duration: float = Field(
        ..., 
        description="Estimated processing time in minutes",
        gt=0
    ) 