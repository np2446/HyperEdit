"""
Video processing agent for AgentKit.
Provides GPU-accelerated video editing capabilities.
"""

from .video_toolkit import VideoToolkit, VideoTool
from .video_knowledge_base import VideoKnowledgeBase
from .video_processor import VideoProcessor, GPURequirements
from .video_models import (
    VideoEditRequest, VideoEditPlan, Scene, ClipSegment,
    Position, VideoEffect, AudioEffect, Caption,
    VideoEffectType, AudioEffectType, TransitionEffect,
    TransitionType, TextStyle
)

__all__ = [
    'VideoToolkit',
    'VideoTool',
    'VideoKnowledgeBase',
    'VideoProcessor',
    'GPURequirements',
    'VideoEditRequest',
    'VideoEditPlan',
    'Scene',
    'ClipSegment',
    'Position',
    'VideoEffect',
    'AudioEffect',
    'Caption',
    'VideoEffectType',
    'AudioEffectType',
    'TransitionEffect',
    'TransitionType',
    'TextStyle'
] 