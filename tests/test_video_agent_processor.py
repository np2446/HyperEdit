"""
Unit tests for the VideoAgentProcessor class.
"""

import os
import sys
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_agent.video_agent_processor import VideoAgentProcessor
from video_agent.video_processor import GPURequirements

class TestVideoAgentProcessor(unittest.TestCase):
    """Test cases for the VideoAgentProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set environment variables for testing
        os.environ["SSH_PRIVATE_KEY_PATH"] = str(Path(self.test_dir) / "test_key")
        os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key"
        os.environ["OPENAI_API_KEY"] = "test_openai_key"
        
        # Create a dummy SSH key file
        with open(os.environ["SSH_PRIVATE_KEY_PATH"], "w") as f:
            f.write("TEST SSH KEY")
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    def test_init_local_mode(self, mock_chat_anthropic, mock_video_processor):
        """Test initialization in local mode."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor in local mode
        processor = VideoAgentProcessor(local_mode=True)
        
        # Check that the processor was initialized correctly
        self.assertTrue(processor.local_mode)
        self.assertEqual(processor.llm_provider, "anthropic")
        mock_video_processor.assert_called_once_with(local_mode=True)
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatOpenAI')
    def test_init_openai(self, mock_chat_openai, mock_video_processor):
        """Test initialization with OpenAI."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor with OpenAI
        processor = VideoAgentProcessor(llm_provider="openai")
        
        # Check that the processor was initialized correctly
        self.assertEqual(processor.llm_provider, "openai")
        mock_chat_openai.assert_called_once()
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    def test_check_environment_local(self, mock_chat_anthropic, mock_video_processor):
        """Test environment check in local mode."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor in local mode
        processor = VideoAgentProcessor(local_mode=True)
        
        # Check the environment
        result = processor.check_environment()
        
        # Should be True in local mode with API keys set
        self.assertTrue(result)
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    @patch('video_agent.video_agent_processor.cv2')
    @patch('video_agent.video_agent_processor.np')
    def test_create_test_pattern_video(self, mock_np, mock_cv2, mock_chat_anthropic, mock_video_processor):
        """Test creating a test pattern video."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Mock video writer
        mock_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 'mp4v'
        
        # Mock numpy random
        mock_np.random.randint.return_value = 100
        mock_np.random.uniform.return_value = 1.0
        mock_np.zeros.return_value = "mock_frame"
        
        # Initialize the processor
        processor = VideoAgentProcessor(local_mode=True)
        
        # Create a test video
        output_path = str(Path(self.test_dir) / "test_video.mp4")
        processor.create_test_pattern_video(output_path)
        
        # Check that the video writer was created and used
        mock_cv2.VideoWriter.assert_called_once()
        mock_writer.write.assert_called()
        mock_writer.release.assert_called_once()
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    def test_generate_test_videos(self, mock_chat_anthropic, mock_video_processor):
        """Test generating test videos."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor
        processor = VideoAgentProcessor(local_mode=True)
        
        # Mock the create_test_pattern_video method
        processor.create_test_pattern_video = MagicMock()
        
        # Generate test videos
        output_dir = str(Path(self.test_dir) / "test_videos")
        video_paths = processor.generate_test_videos(output_dir)
        
        # Check that the videos were generated
        self.assertEqual(len(video_paths), 2)
        processor.create_test_pattern_video.assert_called()
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    async def test_setup_environment_local(self, mock_chat_anthropic, mock_video_processor):
        """Test setting up the environment in local mode."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor in local mode
        processor = VideoAgentProcessor(local_mode=True)
        
        # Set up the environment
        result = await processor.setup_environment()
        
        # Should be True in local mode
        self.assertTrue(result)
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    async def test_upload_videos_local(self, mock_chat_anthropic, mock_video_processor):
        """Test uploading videos in local mode."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor in local mode
        processor = VideoAgentProcessor(local_mode=True)
        
        # Upload videos
        video_paths = ["video1.mp4", "video2.mp4"]
        result = await processor.upload_videos(video_paths)
        
        # In local mode, should return the original paths
        self.assertEqual(result, video_paths)
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    @patch('video_agent.video_agent_processor.tool')
    def test_create_video_editing_tool(self, mock_tool, mock_chat_anthropic, mock_video_processor):
        """Test creating a video editing tool."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Mock the tool decorator
        mock_tool.return_value = lambda f: f
        
        # Initialize the processor
        processor = VideoAgentProcessor(local_mode=True)
        
        # Create the video editing tool
        tool = processor.create_video_editing_tool()
        
        # Check that the tool was created
        self.assertIsNotNone(tool)
        self.assertTrue(callable(tool))
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    @patch('video_agent.video_agent_processor.AgentExecutor')
    async def test_process_with_llm(self, mock_agent_executor, mock_chat_anthropic, mock_video_processor):
        """Test processing with LLM."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Mock the agent executor
        mock_executor = MagicMock()
        mock_agent_executor.return_value = mock_executor
        mock_executor.ainvoke.return_value = {"output": "Test result"}
        
        # Initialize the processor
        processor = VideoAgentProcessor(local_mode=True)
        
        # Mock the create_video_editing_tool method
        processor.create_video_editing_tool = MagicMock(return_value="mock_tool")
        
        # Process with LLM
        result = await processor.process_with_llm("Test prompt")
        
        # Check the result
        self.assertEqual(result, "Test result")
        mock_executor.ainvoke.assert_called_once()
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    async def test_process_videos(self, mock_chat_anthropic, mock_video_processor):
        """Test processing videos."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor
        processor = VideoAgentProcessor(local_mode=True)
        
        # Mock the methods
        processor.check_environment = MagicMock(return_value=True)
        processor.generate_test_videos = MagicMock(return_value=["video1.mp4", "video2.mp4"])
        processor.setup_environment = MagicMock(return_value=asyncio.Future())
        processor.setup_environment.return_value.set_result(True)
        processor.upload_videos = MagicMock(return_value=asyncio.Future())
        processor.upload_videos.return_value.set_result(["video1.mp4", "video2.mp4"])
        processor.process_with_llm = MagicMock(return_value=asyncio.Future())
        processor.process_with_llm.return_value.set_result("Test result")
        
        # Process videos
        result = await processor.process_videos("Test prompt")
        
        # Check the result
        self.assertEqual(result, "Test result")
        processor.check_environment.assert_called_once()
        processor.generate_test_videos.assert_called_once()
        processor.setup_environment.assert_called_once()
        processor.upload_videos.assert_called_once()
        processor.process_with_llm.assert_called_once_with("Test prompt")
    
    @patch('video_agent.video_agent_processor.VideoProcessor')
    @patch('video_agent.video_agent_processor.ChatAnthropic')
    def test_cleanup(self, mock_chat_anthropic, mock_video_processor):
        """Test cleanup."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_anthropic.return_value = mock_llm
        
        # Create a mock video processor
        mock_processor = MagicMock()
        mock_video_processor.return_value = mock_processor
        
        # Initialize the processor
        processor = VideoAgentProcessor(local_mode=False)
        
        # Clean up
        processor.cleanup()
        
        # Check that the processor was cleaned up
        mock_processor.cleanup.assert_called_once()

def run_tests():
    """Run the tests."""
    unittest.main()

if __name__ == "__main__":
    run_tests() 