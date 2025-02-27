import os
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import logging
import traceback
import json
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from typing import List

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
use_local = os.environ.get("USE_LOCAL_PROCESSING", "true").lower() == "true"  # Default to true
logger.info(f"Using {'local' if use_local else 'remote'} processing mode")

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    callbacks=None,
    tags=None
)

# Ensure directories exist
input_dir = project_root / "input_videos"
output_dir = project_root / "test_outputs"
input_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Mount the test_outputs directory for static file serving
app.mount("/test_outputs", StaticFiles(directory=str(output_dir)), name="test_outputs")

@app.post("/process-videos")
async def process_videos(
    prompt: str = Form(...),
    videos: List[str] = Form(...)
):
    """Process one or more videos using the Hyperbolic agent."""
    try:
        logger.info("\n=== Starting Video Processing ===")
        logger.info(f"Processing videos: {videos}")
        logger.info(f"Prompt: {prompt}")
        
        # Verify all videos exist and get absolute paths
        video_paths = []
        for video_name in videos:
            input_path = input_dir / video_name
            logger.info(f"Checking video path: {input_path} (exists: {input_path.exists()})")
            if not input_path.exists():
                error_msg = f"Video {video_name} not found at {input_path}"
                logger.error(error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            logger.info(f"Found input video: {input_path}")
            video_paths.append(str(input_path))

        # Initialize video tool with processor
        logger.info("Initializing video tool...")
        try:
            video_tool = VideoTool(
                llm=llm,
                processor=VideoProcessor(local_mode=use_local),
                input_dir=input_dir,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to initialize VideoTool: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to initialize video processor: {str(e)}")
        
        # Set up GPU requirements
        logger.info(f"Setting up GPU environment (local_mode={use_local})...")
        try:
            if not use_local:
                video_tool.processor.setup_gpu_environment(
                    GPURequirements(
                        gpu_type="RTX",
                        num_gpus=1,
                        min_vram_gb=4.0,
                        disk_size=10,
                        memory=16
                    )
                )
            else:
                video_tool.processor.setup_gpu_environment(
                    GPURequirements(min_vram_gb=4.0)
                )
        except Exception as e:
            logger.error(f"Failed to setup GPU environment: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to setup GPU environment: {str(e)}")

        # Create runnable config with callbacks disabled
        runnable_config = RunnableConfig(
            callbacks=None,
            tags=None
        )

        # Create a timestamp-based output name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"combined_{timestamp}"
        
        # Create a task that references all input videos
        task = {
            "output_name": output_name,
            "description": prompt,
            "scenes": [
                {
                    "clips": [
                        {
                            "source_video": str(input_dir / video_name),
                            "start_time": 0,
                            "end_time": -1,
                            "position": {
                                "x": 0.0, 
                                "y": float(i) * (1.0 / len(videos)),
                                "width": 1.0,
                                "height": 1.0 / len(videos)
                            },
                            "effects": []
                        } for i, video_name in enumerate(videos)
                    ],
                    "captions": []
                }
            ]
        }
        
        logger.info(f"Created task structure: {json.dumps(task, indent=2)}")
        
        try:
            # Use the synchronous version of the video tool
            result = video_tool._run(task, runnable_config)
            logger.info(f"Processing complete. Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error during video processing: {str(e)}")

        if result["status"] == "error":
            error_msg = result["message"]
            logger.error(f"Processing failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        # Extract the output path and verification details from the result
        output_path = Path(result["output_path"])
        if not output_path.exists():
            error_msg = f"Output file not found at {output_path}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        response_data = {
            "status": "success",
            "message": "Videos processed successfully",
            "input_videos": videos,
            "output_path": str(output_path),
            "details": result.get("message", "")
        }

        # Add verification details if present
        if "verification" in result:
            logger.info(f"Verification details: {json.dumps(result['verification'], indent=2)}")
            response_data["verification"] = result["verification"]
        else:
            logger.warning("No verification details in result")
        
        logger.info(f"Returning response: {json.dumps(response_data, indent=2)}")
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Video processing failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 