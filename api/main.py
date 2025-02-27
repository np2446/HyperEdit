import os
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime

from video_agent import VideoTool, VideoProcessor
from video_agent.video_processor import GPURequirements
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig

app = FastAPI(title="Hyperbolic Video Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if we should use local processing
use_local = os.environ.get("USE_LOCAL_PROCESSING", "false").lower() == "true"
logger.info(f"Using {'local' if use_local else 'remote'} processing mode")

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",  # Match the model from demo
    temperature=0,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    callbacks=None,  # Disable LangSmith callbacks
    tags=None  # Disable LangSmith tags
)

# Ensure test_outputs directory exists
output_dir = project_root / "test_outputs"
output_dir.mkdir(exist_ok=True)

# Mount the test_outputs directory for static file serving
app.mount("/test_outputs", StaticFiles(directory=str(output_dir)), name="test_outputs")

# poetry run python -m uvicorn api.main:app --reload

# Sample request: 
# curl -X POST "http://localhost:8000/process-video/circles_warm.mp4?prompt=Add%20a%20title%20caption%20%27Warm%20Patterns%27%20at%20the%20start"
@app.post("/process-video/{video_name}")
async def process_video(
    video_name: str,
    prompt: str
):
    """Process a video using the Hyperbolic agent."""
    try:
        logger.info(f"Processing video: {video_name}")
        logger.info(f"Prompt: {prompt}")
        
        # Set up paths
        input_path = project_root / "input_videos" / video_name
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_name} not found")
        
        logger.info(f"Input path: {input_path}")

        # Initialize video tool with processor
        logger.info("Initializing video tool...")
        video_tool = VideoTool(
            llm=llm,
            processor=VideoProcessor(local_mode=use_local)
        )
        
        # Set up GPU requirements
        logger.info(f"Setting up GPU environment (local_mode={use_local})...")
        if not use_local:
            video_tool.processor.setup_gpu_environment(
                GPURequirements(
                    gpu_type="RTX",  # Prefer RTX GPUs which are typically cheaper
                    num_gpus=1,
                    min_vram_gb=4.0,
                    disk_size=10,
                    memory=16
                )
            )
        else:
            # For local mode, just set up basic requirements
            video_tool.processor.setup_gpu_environment(
                GPURequirements(min_vram_gb=4.0)
            )

        # Create runnable config with callbacks disabled
        runnable_config = RunnableConfig(
            callbacks=None,  # Disable LangSmith callbacks
            tags=None  # Disable LangSmith tags
        )

        # Process the video with user's prompt
        logger.info("Starting video processing...")
        task = prompt  # Use the user's prompt directly
        
        # Log video analysis
        logger.info("\nAnalyzing input videos...")
        for video_path, info in video_tool._analyze_videos().items():
            logger.info(f"\nVideo: {video_path}")
            logger.info(f"- Resolution: {info.width}x{info.height}")
            logger.info(f"- Duration: {info.duration:.2f} seconds")
            logger.info(f"- FPS: {info.fps}")
        
        # Let the LLM agent process the request
        logger.info("\nSending request to LLM for parsing...")
        result = await video_tool._arun(task, runnable_config)
        logger.info(f"Processing complete. Result: {result}")

        # Extract the output path from the result
        if "Output saved to:" in result:
            actual_output_path = result.split("Output saved to:")[1].strip()
            actual_output_path = Path(actual_output_path)
            
            if not actual_output_path.exists():
                logger.error(f"Output file not found at {actual_output_path}")
                raise HTTPException(status_code=500, detail="Video processing failed - output file not created")
            
            return {
                "status": "success",
                "message": "Video processed successfully",
                "input_video": video_name,
                "output_path": str(actual_output_path),
                "details": result
            }
        else:
            logger.error("Could not find output path in result")
            raise HTTPException(status_code=500, detail="Video processing failed - could not determine output path")

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 