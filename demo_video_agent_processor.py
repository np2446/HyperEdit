#!/usr/bin/env python3
"""
Demo script for the VideoAgentProcessor class.
This script demonstrates how to use the VideoAgentProcessor class to process videos
with arbitrary prompts using Hyperbolic GPUs.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

from video_agent.video_agent_processor import VideoAgentProcessor
from video_agent.video_processor import GPURequirements

async def main():
    """Run the demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo for VideoAgentProcessor")
    parser.add_argument("--local", action="store_true", help="Run in local mode (no Hyperbolic GPUs)")
    parser.add_argument("--llm", choices=["anthropic", "openai"], default="anthropic", help="LLM provider to use")
    parser.add_argument("--model", help="Specific LLM model to use")
    parser.add_argument("--prompt", default="Create a video that transitions from grayscale to color", 
                        help="Prompt for the video editing task")
    parser.add_argument("--videos", nargs="*", help="Paths to input videos (if not provided, will generate test videos)")
    parser.add_argument("--log", help="Path to log file")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Print banner
    print("\n" + "=" * 80)
    print("VideoAgentProcessor Demo")
    print("=" * 80)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"- Local mode: {args.local}")
    print(f"- LLM provider: {args.llm}")
    print(f"- LLM model: {args.model or 'default'}")
    print(f"- Prompt: {args.prompt}")
    print(f"- Input videos: {args.videos or 'Will generate test videos'}")
    print(f"- Log file: {args.log or 'None (console only)'}")
    
    # Check for required environment variables
    if not args.local and not os.environ.get("SSH_PRIVATE_KEY_PATH"):
        print("\nERROR: SSH_PRIVATE_KEY_PATH environment variable is not set.")
        print("Please set it to the path of your SSH private key for Hyperbolic.")
        return 1
    
    if args.llm == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set it to your Anthropic API key.")
        return 1
    
    if args.llm == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to your OpenAI API key.")
        return 1
    
    # Create GPU requirements if not in local mode
    gpu_requirements = None
    if not args.local:
        gpu_requirements = GPURequirements(
            gpu_type="RTX",  # Prefer RTX GPUs which are typically cheaper
            num_gpus=1,
            min_vram_gb=4.0,
            disk_size=10,
            memory=16
        )
    
    # Initialize the processor
    print("\nInitializing VideoAgentProcessor...")
    processor = VideoAgentProcessor(
        local_mode=args.local,
        llm_provider=args.llm,
        llm_model=args.model,
        gpu_requirements=gpu_requirements,
        log_file=args.log
    )
    
    try:
        # Process videos
        print("\nProcessing videos...")
        result = await processor.process_videos(args.prompt, args.videos)
        
        # Print result
        print("\nResult:")
        print(result)
        
        return 0
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Cleaning up...")
        processor.cleanup()
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return 1
    finally:
        # Clean up
        print("\nCleaning up...")
        processor.cleanup()

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 