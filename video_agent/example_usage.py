"""
Example usage of the VideoAgent for video editing tasks.
"""

from pathlib import Path
from video_tools import VideoAgent

async def main():
    # Initialize the video agent
    agent = VideoAgent()
    
    # Example 1: Simple video editing
    video_paths = [
        Path("input_videos/clip1.mp4"),
        Path("input_videos/clip2.mp4")
    ]
    edit_prompt = """
    Create a promotional video that:
    1. Combines both clips with smooth transitions
    2. Add engaging captions
    3. Make it look more vibrant and professional
    4. Keep the total length under 2 minutes
    """
    
    try:
        # Process the video editing request
        output_path = await agent.process_video_request(
            video_paths=video_paths,
            edit_prompt=edit_prompt
        )
        print(f"Video successfully rendered at: {output_path}")
        
    except ValueError as e:
        print(f"Error processing video: {e}")
        # Handle specific errors (e.g., no suitable GPU found)
    except Exception as e:
        print(f"Unexpected error: {e}")

# Example 2: Advanced effects
async def apply_advanced_effects():
    agent = VideoAgent()
    
    video_path = Path("input_videos/original.mp4")
    effects_prompt = """
    Transform this video to:
    1. Make it look like it was shot in Paris at sunset
    2. Add cinematic color grading
    3. Stabilize any shaky footage
    4. Add subtle background music
    """
    
    try:
        output_path = await agent.process_video_request(
            video_paths=[video_path],
            edit_prompt=effects_prompt
        )
        print(f"Video with effects rendered at: {output_path}")
        
    except Exception as e:
        print(f"Error applying effects: {e}")

if __name__ == "__main__":
    import asyncio
    
    # Run the examples
    asyncio.run(main())
    # Uncomment to run advanced effects example
    # asyncio.run(apply_advanced_effects()) 