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
VIDEO_ANALYZE_DESCRIPTION = """Analyze input video clips to extract metadata and create VideoClip objects.
Input: List of video file paths.
Output: List of VideoClip objects with metadata."""

VIDEO_EDIT_PLAN_DESCRIPTION = """Create an editing plan based on input clips and user prompt.
Input: List of VideoClip objects and a text prompt describing desired edits.
Output: Dictionary containing structured editing plan.
Example: create_edit_plan(clips, "edit these clips into a promotional video with captions")"""

VIDEO_RENDER_REQUIREMENTS_DESCRIPTION = """Estimate GPU memory and compute requirements for video editing job.
Input: Dictionary containing edit plan.
Output: RenderJob object with memory and compute requirements."""

VIDEO_GPU_SELECT_DESCRIPTION = """Select optimal GPU for video rendering based on job requirements.
Input: RenderJob object.
Output: Dictionary containing selected GPU specifications.
Rules:
1. Checks available GPU memory against job requirements
2. Verifies GPU compatibility with required effects
3. Selects GPU with best specs that meets all requirements"""

VIDEO_EFFECTS_DESCRIPTION = """Apply AI-generated effects to video clips based on prompts.
Input: VideoClip object and effect prompt string.
Output: Modified VideoClip with applied effects.
Example: apply_ai_effects(clip, "make this video look like it was shot in Paris")"""

VIDEO_RENDER_DESCRIPTION = """Render final video using selected GPU.
Input: RenderJob object and selected GPU specifications.
Output: Path to rendered video file."""

VIDEO_PROCESS_REQUEST_DESCRIPTION = """Main entry point for processing video editing requests.
Input: List of video paths and edit prompt string.
Output: Path to final rendered video.
Example: process_video_request([video1.mp4, video2.mp4], "create a promotional video with smooth transitions")""" 