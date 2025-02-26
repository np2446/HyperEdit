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
# import dotenv
import dotenv
import time
import signal
import sys
import atexit

from hyperbolic_agentkit_core.actions.get_available_gpus import get_available_gpus
from hyperbolic_agentkit_core.actions.rent_compute import rent_compute
from hyperbolic_agentkit_core.actions.get_gpu_status import get_gpu_status
from hyperbolic_agentkit_core.actions.terminate_compute import terminate_compute
from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh
from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager

dotenv.load_dotenv()

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

def parse_gpu_info(gpu_info_str: str) -> dict:
    """Parse the GPU info string into a dictionary format."""
    clusters = []
    current_cluster = None
    
    for line in gpu_info_str.split('\n'):
        line = line.strip()
        if not line or line.startswith('Available GPU Options:') or line.startswith('---'):
            continue
            
        if line.startswith('Cluster:'):
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = {'cluster_name': line.split(': ')[1]}
        elif line.startswith('Node ID:'):
            current_cluster['node_id'] = line.split(': ')[1]
        elif line.startswith('GPU Model:'):
            current_cluster['gpu_model'] = line.split(': ')[1]
        elif line.startswith('Available GPUs:'):
            available, total = line.split(': ')[1].split('/')
            current_cluster['gpus_available'] = int(available)
            current_cluster['total_gpus'] = int(total)
        elif line.startswith('Price:'):
            price_str = line.split('$')[1].split('/')[0]
            current_cluster['price'] = float(price_str)
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return {'instances': clusters}

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
    
    # Class variable to track active instances for cleanup
    active_instances: set = set()
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources and register cleanup handlers."""
        super().setUpClass()
        
        # Register cleanup handlers for unexpected termination
        signal.signal(signal.SIGINT, cls._cleanup_on_interrupt)
        signal.signal(signal.SIGTERM, cls._cleanup_on_interrupt)
        atexit.register(cls._cleanup_on_exit)
    
    @classmethod
    def _cleanup_on_interrupt(cls, signum, frame):
        """Clean up GPU instances on interrupt signals."""
        print(f"\nReceived interrupt signal {signum}. Cleaning up GPU instances...")
        cls._terminate_all_instances()
        sys.exit(1)
    
    @classmethod
    def _cleanup_on_exit(cls):
        """Clean up GPU instances on normal exit."""
        cls._terminate_all_instances()
    
    @classmethod
    def _terminate_all_instances(cls):
        """Terminate all active GPU instances."""
        if cls.active_instances:
            print(f"Terminating {len(cls.active_instances)} active GPU instances...")
            for instance_id in list(cls.active_instances):
                try:
                    print(f"Terminating instance {instance_id}...")
                    terminate_compute(instance_id)
                    cls.active_instances.remove(instance_id)
                    print(f"Instance {instance_id} terminated successfully.")
                except Exception as e:
                    print(f"Error terminating instance {instance_id}: {str(e)}")
    
    def setUp(self):
        """Initialize video processor for each test."""
        # Get SSH key path from environment variable
        ssh_key_path = os.environ.get('SSH_PRIVATE_KEY_PATH')
        if not ssh_key_path:
            print("SSH_PRIVATE_KEY_PATH environment variable not found, using default path")
            # Use a configurable default path from environment or fall back to ~/.ssh/id_rsa
            default_ssh_path = os.environ.get('DEFAULT_SSH_KEY_PATH', '~/.ssh/id_rsa')
            ssh_key_path = os.path.expanduser(default_ssh_path)
        
        # Also check for key password
        key_password = os.environ.get('SSH_KEY_PASSWORD')
        if key_password:
            print("Found SSH key password in environment variables")
            
            # If we have a password, prioritize the key specified in env var that matches the password
            password_key_path = os.environ.get('SSH_PASSWORD_KEY_PATH')
            if password_key_path and os.path.exists(os.path.expanduser(password_key_path)):
                print(f"Using password-protected key from environment variable: {password_key_path}")
                ssh_key_path = os.path.expanduser(password_key_path)
        
        # Check if the key file exists
        if not os.path.exists(ssh_key_path):
            print(f"WARNING: SSH key file not found at {ssh_key_path}")
            # Try to find an alternative key from environment variable
            alternative_keys = os.environ.get('ALTERNATIVE_SSH_KEYS', 'id_rsa,id_ed25519').split(',')
            for key_name in alternative_keys:
                alt_path = os.path.expanduser(f"~/.ssh/{key_name.strip()}")
                if os.path.exists(alt_path):
                    print(f"Found alternative SSH key at {alt_path}")
                    ssh_key_path = alt_path
                    break
        
        print(f"Using SSH key: {ssh_key_path}")
        
        # Create a VideoProcessor instance
        self.processor = VideoProcessor(local_mode=False)
        
        # Set up GPU environment with affordable GPU requirements
        # Prefer RTX models which are typically cheaper than H100s
        requirements = GPURequirements(
            gpu_type="RTX",  # Changed from H100 to RTX for more affordable options
            num_gpus=1,
            disk_size=10,
            memory=16
        )
        
        self.processor.setup_gpu_environment(requirements)
        
        # Add to active instances for cleanup
        if self.processor.instance_id:
            self.__class__.active_instances.add(self.processor.instance_id)
        
        # Wait for SSH to be ready
        self._wait_for_ssh_ready(self.processor.instance_id, private_key_path=ssh_key_path)
    
    def _wait_for_ssh_ready(self, instance_id, timeout=300, check_interval=15, private_key_path=None):
        """Wait for SSH to be ready on the instance."""
        print(f"Waiting for SSH to be ready on instance {instance_id}...")
        start_time = time.time()
        
        # Wait full initial period with no connection attempts
        initial_wait = 20
        print(f"Waiting {initial_wait}s before first SSH attempt...")
        time.sleep(initial_wait)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"SSH connection to instance {instance_id} failed within {timeout}s timeout")
                
            elapsed = int(time.time() - start_time)
            print(f"Attempting SSH connection... (elapsed time: {elapsed}s)")
            
            try:
                # Get instance status 
                status_data = get_gpu_status()
                if isinstance(status_data, str):
                    status = json.loads(status_data)
                else:
                    status = status_data
                
                # Find our instance
                instance = None
                for inst in status.get('instances', []):
                    if inst.get('id') == instance_id:
                        instance = inst
                        break
                
                if not instance:
                    print(f"Instance {instance_id} not found in status data - retrying...")
                    time.sleep(check_interval)
                    continue
                
                # Print all available fields for debugging
                print(f"Instance data keys: {list(instance.keys())}")
                
                # Try different ways to extract IP address
                ip_address = None
                
                # 1. Check regular fields
                for field in ['public_ip', 'ip', 'ip_address', 'hostname', 'address']:
                    if field in instance and instance[field]:
                        ip_address = instance[field]
                        print(f"Found IP address in field '{field}': {ip_address}")
                        break
                
                # 2. Check sshCommand if available
                if not ip_address and 'sshCommand' in instance and instance['sshCommand']:
                    ssh_cmd = instance['sshCommand']
                    print(f"Found sshCommand: {ssh_cmd}")
                    # Try to extract hostname from ssh command (format: "ssh user@hostname")
                    import re
                    ip_match = re.search(r'@([\w.-]+)', ssh_cmd)
                    if ip_match:
                        ip_address = ip_match.group(1)
                        print(f"Extracted IP address from sshCommand: {ip_address}")
                
                # 3. Check nested 'instance' field if it exists
                if not ip_address and 'instance' in instance and isinstance(instance['instance'], dict):
                    nested_instance = instance['instance']
                    print(f"Found nested instance data, keys: {list(nested_instance.keys())}")
                    for field in ['public_ip', 'ip', 'ip_address', 'hostname', 'address']:
                        if field in nested_instance and nested_instance[field]:
                            ip_address = nested_instance[field]
                            print(f"Found IP address in nested instance.{field}: {ip_address}")
                            break
                
                if not ip_address:
                    print(f"Instance {instance_id} has no IP address yet in any recognizable field - retrying...")
                    time.sleep(check_interval)
                    continue
                
                print(f"Attempting to connect to instance at IP: {ip_address}")
                
                # Use the ssh_connect function to establish a connection
                from hyperbolic_agentkit_core.actions.ssh_access import connect_ssh
                from hyperbolic_agentkit_core.actions.ssh_manager import ssh_manager
                
                # First ensure any previous connections are closed
                if ssh_manager.is_connected:
                    try:
                        ssh_manager.disconnect()
                        print("Closed previous SSH connection")
                    except Exception as e:
                        print(f"Warning: Error closing previous SSH connection: {e}")
                
                # Connect using the instance IP and username
                ssh_result = connect_ssh(
                    host=ip_address,
                    username="ubuntu",
                    private_key_path=private_key_path  # Use provided key
                )
                
                print(f"SSH connection result: {ssh_result}")
                
                # Check if connection was successful
                if "Successfully connected" in ssh_result:
                    print(f"SSH connection successful to instance {instance_id}!")
                    print("\nGPU instance is ready with SSH connectivity!")
                    
                    # Set up the processor with our instance
                    self._setup_processor()
                    return
                else:
                    print(f"SSH connection failed: {ssh_result}")
            except Exception as e:
                print(f"SSH connection attempt failed: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"Retrying in {check_interval} seconds...")
            time.sleep(check_interval)

    def _setup_processor(self):
        """Set up the processor with our instance ID after SSH is ready."""
        if not hasattr(self, 'instance_id') or not self.instance_id:
            raise ValueError("No instance ID available to set up processor")
            
        # Set the processor's instance ID
        self.processor.instance_id = self.instance_id
        self.processor.current_instance = {"id": self.instance_id}
        
        # Initialize file transfer and scene processor
        from video_agent.file_transfer import FileTransfer
        from video_agent.scene_processor import SceneProcessor
        
        # Using the global SSH connection that was established in _wait_for_ssh_ready
        self.processor.file_transfer = FileTransfer(self.instance_id)
        self.processor.scene_processor = SceneProcessor(self.instance_id, self.processor.workspace_dir)
        
        # Set up the environment
        try:
            self.processor._setup_environment()
        except Exception as e:
            print(f"Error setting up environment: {e}")
            print("Trying again with a fresh connection...")
            # If _setup_environment fails, wait a moment and try again
            time.sleep(5)
            # We don't need to re-establish the SSH connection, we just need to retry the command
            self.processor._setup_environment()

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
            
        # Terminate the compute instance if it exists
        if hasattr(self, 'instance_id') and self.instance_id:
            try:
                terminate_compute(self.instance_id)
                print(f"\nTerminated instance {self.instance_id}")
                # Remove from active instances
                if self.instance_id in self.__class__.active_instances:
                    self.__class__.active_instances.remove(self.instance_id)
            except Exception as e:
                print(f"Error terminating instance {self.instance_id}: {str(e)}")
    
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