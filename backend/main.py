import os
import sys
import logging
import asyncio
import uuid
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add the parent directory to the path so we can import the video agent module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the video agent processor and config
from video_agent.video_agent_processor import VideoAgentProcessor
from config import (
    UPLOADS_DIR, OUTPUTS_DIR, CORS_ORIGINS, 
    API_HOST, API_PORT, MAX_UPLOAD_SIZE,
    DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL,
    TASK_RETENTION_HOURS, DEBUG
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("video-agent-backend")

app = FastAPI(
    title="Video Agent Backend",
    description="API for processing videos with AI",
    version="0.1.0",
    debug=DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Store processing tasks
processing_tasks: Dict[str, Dict[str, Any]] = {}

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Completed in {process_time:.4f}s - Status: {response.status_code}")
    return response

# Periodic task cleanup
async def cleanup_old_tasks():
    """Clean up old tasks and their files."""
    try:
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task_data in processing_tasks.items():
            # Check if the task has a timestamp
            if 'timestamp' in task_data:
                task_time = task_data['timestamp']
                age = current_time - task_time
                
                # If task is older than retention period, schedule for removal
                if age.total_seconds() > TASK_RETENTION_HOURS * 3600:
                    tasks_to_remove.append(task_id)
        
        # Remove old tasks
        for task_id in tasks_to_remove:
            # Clean up files
            upload_dir = UPLOADS_DIR / task_id
            output_dir = OUTPUTS_DIR / task_id
            
            if upload_dir.exists():
                import shutil
                shutil.rmtree(upload_dir)
            
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
            
            # Remove from tasks dict
            del processing_tasks[task_id]
            
            logger.info(f"Cleaned up task {task_id} due to age")
    
    except Exception as e:
        logger.error(f"Error in cleanup_old_tasks: {str(e)}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting video agent backend")
    
    # Schedule periodic cleanup task
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Run cleanup periodically."""
    while True:
        await cleanup_old_tasks()
        # Run every hour
        await asyncio.sleep(3600)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Video Agent Backend API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "process_video": "/process-video",
            "task_status": "/task-status/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

class VideoProcessingResult(BaseModel):
    task_id: str
    status: str
    message: str
    status_url: str

@app.post("/process-video", response_model=VideoProcessingResult)
async def process_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    local_mode: bool = Form(False),
    videos: List[UploadFile] = File(...)
):
    try:
        # Generate a unique ID for this processing task
        task_id = str(uuid.uuid4())
        logger.info(f"New processing task {task_id}: prompt='{prompt}', local_mode={local_mode}, videos={len(videos)}")
        
        # Validate file sizes
        for video in videos:
            # Read a chunk to force fastapi to calculate the content length
            await video.read(1024)
            await video.seek(0)  # Reset position
            
            content_length = video.size
            if content_length and content_length > MAX_UPLOAD_SIZE * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE}MB."
                )
        
        # Save uploaded videos to disk
        video_paths = []
        for i, video in enumerate(videos):
            # Create a temporary file with the original extension
            orig_filename = video.filename or f"video_{i}.mp4"
            file_extension = os.path.splitext(orig_filename)[1]
            upload_dir = UPLOADS_DIR / task_id
            upload_dir.mkdir(exist_ok=True)
            
            video_path = upload_dir / f"video_{i}{file_extension}"
            
            # Save the video file
            with open(video_path, "wb") as f:
                f.write(await video.read())
            
            video_paths.append(str(video_path))
            logger.info(f"Saved video {i+1}/{len(videos)} to {video_path}")
        
        # Set up the processing task in the background
        background_tasks.add_task(
            process_videos_task,
            task_id=task_id,
            prompt=prompt,
            local_mode=local_mode,
            video_paths=video_paths,
            llm_provider=DEFAULT_LLM_PROVIDER,
            llm_model=DEFAULT_LLM_MODEL
        )
        
        # Return task ID and status url
        processing_tasks[task_id] = {
            "status": "processing",
            "timestamp": datetime.now(),
            "prompt": prompt,
            "local_mode": local_mode,
            "num_videos": len(videos),
            "video_paths": video_paths
        }
        
        return VideoProcessingResult(
            task_id=task_id,
            status="processing",
            message="Video processing has started",
            status_url=f"/task-status/{task_id}"
        )
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/task-status/{task_id}")
async def task_status(task_id: str):
    """Get the status of a processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return processing_tasks[task_id]

async def process_videos_task(
    task_id: str, 
    prompt: str, 
    local_mode: bool, 
    video_paths: List[str],
    llm_provider: str,
    llm_model: str
):
    """Process videos in a background task."""
    processor = None
    try:
        # Update task status
        processing_tasks[task_id].update({
            "status": "processing",
            "message": "Initializing video processor..."
        })
        
        # Initialize the VideoAgentProcessor
        processor = VideoAgentProcessor(
            local_mode=local_mode,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        
        # Create output directory
        output_dir = OUTPUTS_DIR / task_id
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "output.mp4"
        
        # Update task status
        processing_tasks[task_id].update({
            "status": "processing",
            "message": "Processing videos..."
        })
        
        # Process the videos
        result = await processor.process_videos(prompt, video_paths)
        
        # Copy the output to our output directory
        if os.path.exists("outputs/output.mp4"):
            # Move the output to our task-specific output directory
            os.rename("outputs/output.mp4", output_path)
        
        # Determine the output URL
        output_url = f"/outputs/{task_id}/output.mp4"
        
        # Update task status
        processing_tasks[task_id].update({
            "status": "completed",
            "message": "Video processing completed",
            "video_url": output_url,
            "result": result,
            "completed_at": datetime.now()
        })
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
        processing_tasks[task_id].update({
            "status": "failed",
            "message": f"Video processing failed: {str(e)}",
            "error": str(e),
            "failed_at": datetime.now()
        })
    finally:
        # Clean up
        if processor:
            processor.cleanup()

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=DEBUG) 