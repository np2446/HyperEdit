import os
import logging
from video_agent.video_tools import (
    VideoEditRequest, 
    create_video_edit_plan, 
    execute_video_edit,
    VideoEffect,
    AudioEffect,
    TransitionEffect,
    Scene,
    ClipSegment,
    Position,
    VideoEditPlan,
    Caption
)
import unittest
import json
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestVideoEditing(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("test_videos")
        self.output_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def tearDown(self):
        """Clean up after tests."""
        try:
            # Clean up output files
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.logger.info("Cleaned up output directory")
        except Exception as e:
            self.logger.error(f"Error cleaning up: {str(e)}")
    
    def test_remote_gpu_video_editing(self):
        """Test video editing using remote GPU."""
        self.logger.info("Starting remote GPU video editing test")
        
        # Input video paths
        input1 = self.test_dir / "input1.mp4"
        input2 = self.test_dir / "input2.mp4"
        output_path = self.output_dir / "remote_gpu_output.mp4"
        
        self.logger.info(f"Using input videos: {input1}, {input2}")
        self.logger.info(f"Output will be saved to: {output_path}")
        
        # Verify input files exist
        self.assertTrue(os.path.exists(input1), f"Input file missing: {input1}")
        self.assertTrue(os.path.exists(input2), f"Input file missing: {input2}")
        
        # Create a simple edit request
        request = VideoEditRequest(
            video_paths=[str(input1), str(input2)],
            output_path=str(output_path),
            edit_prompt="Create a simple split-screen comparison"
        )
        
        try:
            self.logger.info("Creating edit plan")
            plan = create_video_edit_plan(request)
            
            # Reduce GPU requirements for testing
            plan.estimated_gpu_requirements = {
                "vram_gb": 4.0,
                "gpu_count": 1.0
            }
            self.logger.debug(f"Created plan with {len(plan.scenes)} scenes")
            self.logger.debug(f"GPU requirements: {plan.estimated_gpu_requirements}")
            
            self.logger.info("Executing edit plan on remote GPU")
            result_path = execute_video_edit(plan, request, local_mode=False)
            
            # Verify the result
            self.assertTrue(Path(result_path).exists())
            self.assertTrue(Path(result_path).stat().st_size > 0)
            self.logger.info(f"Successfully created output video at {result_path}")
            
        except RuntimeError as e:
            if "500 Server Error: Internal Server Error" in str(e):
                self.logger.error("Marketplace API returned 500 error - this may be a temporary issue")
                self.logger.error(f"Full error: {str(e)}")
                self.skipTest("Marketplace API is temporarily unavailable")
            else:
                self.logger.error("Remote GPU video editing failed")
                raise

if __name__ == "__main__":
    unittest.main(verbosity=2) 