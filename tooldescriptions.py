"""
Contains all tool descriptions used by the chatbot agent.
"""

# Twitter state management tool descriptions
TWITTER_REPLY_CHECK_DESCRIPTION = """Check if we have already replied to a tweet. MUST be used before replying to any tweet.
Input: tweet ID string.
Rules:
1. Always check this before replying to any tweet
2. If returns True, do NOT reply and select a different tweet
3. If returns False, proceed with reply_to_tweet then add_replied_to"""

TWITTER_ADD_REPLIED_DESCRIPTION = """Add a tweet ID to the database of replied tweets. 
MUST be used after successfully replying to a tweet.
Input: tweet ID string.
Rules:
1. Only use after successful reply_to_tweet
2. Must verify with has_replied_to first
3. Stores tweet ID permanently to prevent duplicate replies"""

TWITTER_REPOST_CHECK_DESCRIPTION = "Check if we have already reposted a tweet. Input should be a tweet ID string."

TWITTER_ADD_REPOSTED_DESCRIPTION = "Add a tweet ID to the database of reposted tweets."

# Knowledge base tool descriptions
TWITTER_KNOWLEDGE_BASE_DESCRIPTION = """Query the Twitter knowledge base for relevant tweets about crypto/AI/tech trends.
Input should be a search query string.
Example: query_twitter_knowledge_base("latest developments in AI")"""

PODCAST_KNOWLEDGE_BASE_DESCRIPTION = "Query the podcast knowledge base for relevant podcast segments about crypto/Web3/gaming. Input should be a search query string."

# Query enhancement tool description
ENHANCE_QUERY_DESCRIPTION = "Analyze the initial query and its results to generate an enhanced follow-up query. Takes two parameters: initial_query (the original query string) and query_result (the results obtained from that query)."

# Web search tool description
WEB_SEARCH_DESCRIPTION = "Search the internet for current information."

# Video editing tool descriptions
VIDEO_EDIT_PLAN_DESCRIPTION = """Create a detailed plan for video editing based on input videos and editing requirements.
Input should be a JSON object with:
- video_paths: List of paths to input videos
- edit_prompt: Description of desired edits and final output
- output_path: Desired output path for the final video

Example: {
    "video_paths": ["/path/to/video1.mp4", "/path/to/video2.mp4"],
    "edit_prompt": "Create a promotional video with captions and enhance the quality",
    "output_path": "/path/to/output.mp4"
}

Rules:
1. All input videos must exist and be accessible
2. Edit prompt should clearly describe desired modifications
3. Output path must be writable"""

VIDEO_EDIT_EXECUTE_DESCRIPTION = """Execute a video editing plan using Hyperbolic GPU resources.
Input should be a JSON object with:
- plan: The VideoEditPlan object returned from plan_video_edit
- request: The original VideoEditRequest object

Important notes:
- Automatically selects and rents appropriate GPU resources based on video requirements
- GPU selection considers resolution, duration, and editing complexity
- Uses Hyperbolic platform for GPU-accelerated video processing
- Supports AI-powered enhancements, captions, and video assembly"""

# Add to existing exports if needed
__all__ = [
    # ... existing exports ...
    'VIDEO_EDIT_PLAN_DESCRIPTION',
    'VIDEO_EDIT_EXECUTE_DESCRIPTION'
]