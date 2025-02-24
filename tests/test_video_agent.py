"""
Unit tests for the video agent components.
Tests each module independently and their integration.
"""

import unittest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, call

from video_agent.video_models import (
    VideoEditRequest, VideoEditPlan, Scene, ClipSegment,
    Position, VideoEffect, AudioEffect, Caption,
    VideoEffectType, AudioEffectType, TransitionEffect,
    TransitionType, TextStyle
)
from video_agent.video_processor import VideoProcessor, GPURequirements
from video_agent.scene_processor import SceneProcessor
from video_agent.file_transfer import FileTransfer

class TestVideoModels(unittest.TestCase):
    """Test the data models for video editing."""
    
    def test_video_edit_request(self):
        """Test VideoEditRequest model creation and validation."""
        request = VideoEditRequest(
            video_paths=["test1.mp4", "test2.mp4"],
            edit_prompt="Create a split-screen comparison",
            output_path="output.mp4"
        )
        self.assertEqual(len(request.video_paths), 2)
        self.assertEqual(request.output_format, "mp4")  # Test default value
        
    def test_scene_creation(self):
        """Test Scene model with clips and captions."""
        scene = Scene(
            duration=10.0,
            clips=[
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=10,
                    position=Position(x=0, y=0, width=1.0, height=1.0)
                )
            ],
            captions=[
                Caption(
                    text="Test Caption",
                    start_time=0,
                    end_time=5,
                    position=Position(x='center', y='top', width=0.8, height=0.1)
                )
            ]
        )
        self.assertEqual(len(scene.clips), 1)
        self.assertEqual(len(scene.captions), 1)
        self.assertEqual(scene.duration, 10.0)

class TestFileTransfer(unittest.TestCase):
    """Test cases for FileTransfer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.instance_id = "test-instance-1"
        self.patcher = patch('video_agent.file_transfer.execute_remote_command')
        self.mock_execute = self.patcher.start()
        
        def mock_execute_side_effect(*args):
            if len(args) == 1:
                command = args[0]
            else:
                instance_id, command = args
                
            if command.startswith("ssh_status"):
                return json.dumps({
                    "host": "test.example.com",
                    "username": "ubuntu",
                    "port": 22
                })
            elif command.startswith("test -f") and "echo 'exists'" in command:
                return "exists"  # File exists
            elif command.startswith("test -d") and "echo 'exists'" in command:
                return "exists"  # Directory exists
            elif command.startswith("apt-get") or command.startswith("pip3") or command.startswith("mkdir"):
                return ""  # Success
            elif command.startswith("ffmpeg"):
                return "frame=100 fps=30 q=28.0 size=1024kB time=00:00:10.00 bitrate=836.6kbits/s"
            return ""
            
        self.mock_execute.side_effect = mock_execute_side_effect
        self.file_transfer = FileTransfer(self.instance_id)

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()

    @patch('os.path.exists')
    @patch('os.system')
    def test_upload_file(self, mock_system, mock_exists):
        """Test uploading a file to remote instance."""
        mock_exists.return_value = True
        mock_system.return_value = 0
        
        local_path = "/local/file.mp4"
        remote_path = "/remote/file.mp4"
        
        self.file_transfer.upload_file(local_path, remote_path)
        
        # Verify SSH status was checked
        self.mock_execute.assert_any_call(f"ssh_status {self.instance_id}")
        
        # Verify remote directory was created
        self.mock_execute.assert_any_call("mkdir -p /remote")
        
        # Verify scp command was executed
        mock_system.assert_called_once()
        scp_cmd = mock_system.call_args[0][0]
        self.assertIn("scp", scp_cmd)
        self.assertIn(local_path, scp_cmd)
        self.assertIn("test.example.com", scp_cmd)

    @patch('os.makedirs')
    @patch('os.system')
    def test_download_file(self, mock_system, mock_makedirs):
        """Test downloading a file from remote instance."""
        mock_system.return_value = 0
        
        remote_path = "/remote/file.mp4"
        local_path = "/local/file.mp4"
        
        self.file_transfer.download_file(remote_path, local_path)
        
        # Verify remote file existence was checked
        self.mock_execute.assert_any_call(f"test -f {remote_path} && echo 'exists'")
        
        # Verify SSH status was checked
        self.mock_execute.assert_any_call(f"ssh_status {self.instance_id}")
        
        # Verify local directory was created
        mock_makedirs.assert_called_once()
        
        # Verify scp command was executed
        mock_system.assert_called_once()
        scp_cmd = mock_system.call_args[0][0]
        self.assertIn("scp", scp_cmd)
        self.assertIn(remote_path, scp_cmd)
        self.assertIn("test.example.com", scp_cmd)

    @patch('os.path.isdir')
    @patch('os.system')
    def test_upload_directory(self, mock_system, mock_isdir):
        """Test uploading a directory to remote instance."""
        mock_isdir.return_value = True
        mock_system.return_value = 0
        
        local_dir = "/local/videos"
        remote_dir = "/remote/videos"
        
        self.file_transfer.upload_directory(local_dir, remote_dir)
        
        # Verify SSH status was checked
        self.mock_execute.assert_any_call(f"ssh_status {self.instance_id}")
        
        # Verify remote directory was created
        self.mock_execute.assert_any_call(f"mkdir -p {remote_dir}")
        
        # Verify scp command was executed
        mock_system.assert_called_once()
        scp_cmd = mock_system.call_args[0][0]
        self.assertIn("scp -r", scp_cmd)
        self.assertIn(local_dir, scp_cmd)
        self.assertIn("test.example.com", scp_cmd)

    @patch('os.makedirs')
    @patch('os.system')
    def test_download_directory(self, mock_system, mock_makedirs):
        """Test downloading a directory from remote instance."""
        mock_system.return_value = 0
        
        remote_dir = "/remote/videos"
        local_dir = "/local/videos"
        
        self.file_transfer.download_directory(remote_dir, local_dir)
        
        # Verify remote directory existence was checked
        self.mock_execute.assert_any_call(f"test -d {remote_dir} && echo 'exists'")
        
        # Verify SSH status was checked
        self.mock_execute.assert_any_call(f"ssh_status {self.instance_id}")
        
        # Verify local directory was created
        mock_makedirs.assert_called_once_with(local_dir, exist_ok=True)
        
        # Verify scp command was executed
        mock_system.assert_called_once()
        scp_cmd = mock_system.call_args[0][0]
        self.assertIn("scp -r", scp_cmd)
        self.assertIn(remote_dir, scp_cmd)
        self.assertIn("test.example.com", scp_cmd)

class TestSceneProcessor(unittest.TestCase):
    """Test cases for SceneProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher = patch('video_agent.scene_processor.execute_remote_command')
        self.mock_execute = self.patcher.start()
        
        def mock_execute_side_effect(*args):
            if len(args) == 1:
                command = args[0]
            else:
                instance_id, command = args
                
            if command.startswith("mkdir"):
                return ""
            elif command.startswith("ffmpeg"):
                return "frame=100 fps=30 q=28.0 size=1024kB time=00:00:10.00 bitrate=836.6kbits/s"
            return ""
            
        self.mock_execute.side_effect = mock_execute_side_effect
        self.processor = SceneProcessor(instance_id="test-instance-1", workspace_dir="/workspace")

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()

    def test_setup_workspace(self):
        """Test workspace directory setup."""
        self.processor._setup_workspace()
        
        # Verify workspace directories were created
        expected_dirs = [
            "/workspace/clips",
            "/workspace/effects",
            "/workspace/output",
            "/workspace/captions"
        ]
        
        for directory in expected_dirs:
            self.mock_execute.assert_any_call(f"mkdir -p {directory}")

    def test_process_clip(self):
        """Test processing a clip with effects."""
        clip = ClipSegment(
            source_index=0,
            start_time=0,
            end_time=10,
            position=Position(x=0, y=0, width=1.0, height=1.0),
            effects=[
                VideoEffect(
                    type=VideoEffectType.BLUR,
                    params={"strength": 5},
                    start_time=0
                ),
                VideoEffect(
                    type=VideoEffectType.SPEED,
                    params={"factor": 0.5},
                    start_time=0
                )
            ]
        )
        
        source_videos = ["/input/clip.mp4"]
        output_path = "/output/processed.mp4"
        
        self.processor._process_clip(clip, source_videos, output_path)
        
        # Verify ffmpeg command was executed
        self.mock_execute.assert_called()
        ffmpeg_cmd = self.mock_execute.call_args[0][0]
        
        # Verify command contains input file
        self.assertIn(f"-i {source_videos[0]}", ffmpeg_cmd)
        
        # Verify effects were applied
        self.assertIn("boxblur=5", ffmpeg_cmd)
        self.assertIn("setpts=2.0*PTS", ffmpeg_cmd)  # 0.5x speed = 2.0x PTS
        
        # Verify output path
        self.assertIn(output_path, ffmpeg_cmd)

    def test_process_scene(self):
        """Test processing a complete scene."""
        scene = Scene(
            duration=10.0,
            clips=[
                ClipSegment(
                    source_index=0,
                    start_time=0,
                    end_time=10,
                    position=Position(x=0, y=0, width=0.5, height=1.0),
                    effects=[VideoEffect(
                        type=VideoEffectType.BLUR,
                        params={"strength": 5},
                        start_time=0
                    )]
                ),
                ClipSegment(
                    source_index=1,
                    start_time=0,
                    end_time=10,
                    position=Position(x=0.5, y=0, width=0.5, height=1.0),
                    effects=[VideoEffect(
                        type=VideoEffectType.SPEED,
                        params={"factor": 0.5},
                        start_time=0
                    )]
                )
            ]
        )
        
        source_videos = ["/input/clip1.mp4", "/input/clip2.mp4"]
        output_path = "/output/scene.mp4"
        
        self.processor.process_scene(scene, source_videos, output_path)
        
        # Verify workspace was set up
        self.mock_execute.assert_any_call("mkdir -p /workspace/clips")
        self.mock_execute.assert_any_call("mkdir -p /workspace/effects")
        self.mock_execute.assert_any_call("mkdir -p /workspace/output")
        
        # Verify each clip was processed
        for i, source_video in enumerate(source_videos):
            ffmpeg_calls = [call for call in self.mock_execute.call_args_list 
                          if "ffmpeg" in call[0][0] and source_video in call[0][0]]
            self.assertGreater(len(ffmpeg_calls), 0)
            
            # Verify effects were applied
            if i == 0:
                self.assertIn("boxblur=5", ffmpeg_calls[0][0][0])
            else:
                self.assertIn("setpts=2.0*PTS", ffmpeg_calls[0][0][0])

class TestVideoProcessor(unittest.TestCase):
    """Test cases for VideoProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher_gpu = patch('video_agent.video_processor.get_available_gpus')
        self.patcher_rent = patch('video_agent.video_processor.rent_compute')
        self.patcher_status = patch('video_agent.video_processor.get_gpu_status')
        self.patcher_execute = patch('video_agent.video_processor.execute_remote_command')
        self.patcher_file_execute = patch('video_agent.file_transfer.execute_remote_command')
        
        self.mock_gpus = self.patcher_gpu.start()
        self.mock_rent = self.patcher_rent.start()
        self.mock_status = self.patcher_status.start()
        self.mock_execute = self.patcher_execute.start()
        self.mock_file_execute = self.patcher_file_execute.start()
        
        def mock_gpus_side_effect():
            return json.dumps({
                "instances": [{
                    "cluster_name": "test-cluster",
                    "node_id": "node-1",
                    "gpu_model": "RTX 4090",
                    "gpus_available": 2,
                    "gpus_total": 4,
                    "price": 1.50
                }]
            })
            
        def mock_rent_side_effect(*args, **kwargs):
            cluster_name = kwargs.get("cluster_name")
            node_name = kwargs.get("node_name")
            gpu_count = kwargs.get("gpu_count")
            return json.dumps({
                "instance": {
                    "id": "test-instance-1",
                    "status": "running",
                    "gpu_info": {
                        "model": "RTX 4090",
                        "count": int(gpu_count) if gpu_count else 1
                    }
                }
            })
            
        def mock_status_side_effect(instance_id):
            return json.dumps([{
                "id": instance_id,
                "status": "running",
                "gpu_utilization": 85,
                "memory_used": 8192,
                "power_draw": 250
            }])
            
        def mock_execute_side_effect(*args):
            if len(args) == 1:
                command = args[0]
            else:
                instance_id, command = args
                
            if command.startswith("ssh_status"):
                return json.dumps({
                    "host": "test.example.com",
                    "username": "ubuntu",
                    "port": 22
                })
            elif command.startswith("test -f") and "echo 'exists'" in command:
                return "exists"  # File exists
            elif command.startswith("test -d") and "echo 'exists'" in command:
                return "exists"  # Directory exists
            elif command.startswith("apt-get") or command.startswith("pip3") or command.startswith("mkdir"):
                return ""  # Success
            elif command.startswith("ffmpeg"):
                return "frame=100 fps=30 q=28.0 size=1024kB time=00:00:10.00 bitrate=836.6kbits/s"
            return ""
            
        self.mock_gpus.side_effect = mock_gpus_side_effect
        self.mock_rent.side_effect = mock_rent_side_effect
        self.mock_status.side_effect = mock_status_side_effect
        self.mock_execute.side_effect = mock_execute_side_effect
        self.mock_file_execute.side_effect = mock_execute_side_effect
        
        self.processor = VideoProcessor()

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher_gpu.stop()
        self.patcher_rent.stop()
        self.patcher_status.stop()
        self.patcher_execute.stop()
        self.patcher_file_execute.stop()

    def test_select_gpu(self):
        """Test GPU selection based on requirements."""
        gpu_info = json.loads(self.mock_gpus())
        requirements = GPURequirements(
            min_vram_gb=8.0,
            gpu_count=1.0
        )
        
        selected = self.processor._select_gpu(gpu_info, requirements)
        
        self.assertEqual(selected["cluster_name"], "test-cluster")
        self.assertEqual(selected["node_id"], "node-1")
        self.assertEqual(selected["gpu_model"], "RTX 4090")
        self.assertEqual(selected["gpus_available"], 2)

    @patch('os.path.exists')
    @patch('os.system')
    @patch('os.makedirs')
    def test_process_video(self, mock_makedirs, mock_system, mock_exists):
        """Test complete video processing workflow."""
        # Mock local file existence and system commands
        mock_exists.return_value = True
        mock_system.return_value = 0  # Success for scp commands
        mock_makedirs.return_value = None  # Success for directory creation
        
        request = VideoEditRequest(
            video_paths=["/input/video.mp4"],
            edit_prompt="Add blur effect",
            output_path="/output/processed.mp4"
        )
        
        plan = VideoEditPlan(
            scenes=[
                Scene(
                    duration=10.0,
                    clips=[
                        ClipSegment(
                            source_index=0,
                            start_time=0,
                            end_time=10,
                            position=Position(x=0, y=0, width=1.0, height=1.0),
                            effects=[
                                VideoEffect(
                                    type=VideoEffectType.BLUR,
                                    params={"strength": 20},
                                    start_time=0
                                )
                            ]
                        )
                    ]
                )
            ],
            estimated_gpu_requirements={
                "min_vram_gb": 8.0,
                "gpu_count": 1.0
            },
            estimated_duration=10.0
        )
        
        # Setup GPU environment first
        self.processor.setup_gpu_environment(GPURequirements(
            min_vram_gb=8.0,
            gpu_count=1.0
        ))
        
        # Process video
        output_path = self.processor.process_video(plan, request)
        
        # Verify GPU selection
        self.mock_gpus.assert_called_once()
        
        # Verify compute rental
        self.mock_rent.assert_called_once()
        rent_kwargs = self.mock_rent.call_args.kwargs
        self.assertEqual(rent_kwargs["cluster_name"], "test-cluster")
        self.assertEqual(rent_kwargs["node_name"], "node-1")
        self.assertEqual(rent_kwargs["gpu_count"], "1")
        
        # Verify status check
        self.mock_status.assert_called()
        self.assertEqual(self.mock_status.call_args[0][0], "test-instance-1")
        
        # Verify SCP commands were executed
        mock_system.assert_called()
        scp_calls = [call for call in mock_system.call_args_list if "scp" in call[0][0]]
        self.assertGreater(len(scp_calls), 0)
        
        # Verify directory creation
        mock_makedirs.assert_called()
        
        # Verify output path
        self.assertEqual(output_path, "/output/processed.mp4")

if __name__ == '__main__':
    unittest.main() 