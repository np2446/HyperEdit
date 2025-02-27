# Hyperbolic's Agent Framework - Video Editing Extension

This repository is a fork of the [Hyperbolic AgentKit](https://github.com/HyperbolicLabs/Hyperbolic-AgentKit), extended with GPU-accelerated video editing capabilities for the Hyperbolic x Eigenlayer Hackathon. Our extension demonstrates how to build hyperintelligent AI agents that can perform complex video processing tasks using Hyperbolic's decentralized GPU marketplace.

## Video Editing Agent Extension

We've extended the base AgentKit with an intelligent video editing agent that leverages Hyperbolic's GPU marketplace for efficient video processing. Our agent can:

- Create dynamic split-screen comparisons using parallel GPU processing
- Apply complex video and audio effects with GPU acceleration
- Manage multi-scene compositions with intelligent resource allocation
- Generate and overlay captions using AI models
- Automatically scale processing across available GPU resources
- Adapt processing strategies based on video complexity and available compute

### Video Processing Setup

After following the main AgentKit installation steps with Poetry (see [Installation Steps](#installation-steps)), you'll need:

1. **Configure Hyperbolic Access** (required for GPU processing)
   ```bash
   # Add to your .env file:
   HYPERBOLIC_API_KEY=your_api_key_here
   ```
   Get your API key from the [Hyperbolic Portal](https://app.hyperbolic.xyz)

2. **Install FFmpeg** (required for local processing fallback)
   ```bash
   # On macOS
   brew install ffmpeg

   # On Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg
   ```

3. **Install OpenCV Dependencies** (required for test video generation)
   ```bash
   # On macOS
   brew install opencv

   # On Ubuntu/Debian
   sudo apt install python3-opencv
   ```

4. **Run Demo**
   ```bash
   # Run the video agent demo
   poetry run python demo_video_agent.py
   ```

The demo will:
1. Generate test videos with different patterns and color schemes in the `input_videos` directory:
   - `circles_warm.mp4`: Warm-colored moving circles
   - `rectangles_cool.mp4`: Cool-colored rotating rectangles
   - `waves_grayscale.mp4`: Grayscale wave patterns
   - `particles_random.mp4`: Random colored particle effects

2. Process a split-screen comparison video that demonstrates:
   - Side-by-side video layout
   - Title caption overlay
   - Blur effects
   - Fade transitions
   - Proper scaling and positioning

3. Output the processed video to `test_outputs/warm_cool_comparison.mp4`

### Using the Video Agent

The video agent can be used in two ways:

1. **Direct API Usage**
   ```python
   from video_agent import VideoTool, VideoProcessor
   from langchain_anthropic import ChatAnthropic

   # Initialize the video tool with Hyperbolic GPU processing
   llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
   video_tool = VideoTool(
       llm=llm,
       processor=VideoProcessor()  # Uses Hyperbolic by default
   )

   # For local processing (fallback mode)
   video_tool = VideoTool(
       llm=llm,
       processor=VideoProcessor(local_mode=True)
   )

   # Process a video editing request
   result = await video_tool._arun(
       "Create a split-screen comparison of video1.mp4 and video2.mp4"
   )
   ```

### To run the frontend

```bash
cd frontend
npm run dev 
```

#### To run the backend

```bash
poetry run python -m uvicorn api.main:app --reload
```

2. **Through AgentKit**
   ```python
   # In your agent configuration
   from video_agent import VideoToolkit
   video_toolkit = VideoToolkit.from_llm(llm)  # Uses Hyperbolic by default
   tools.extend(video_toolkit.get_tools())
   ```

### Supported Features

1. **Video Effects** (GPU-accelerated)
   - Blur (adjustable strength)
   - Sharpen
   - Color adjustments (contrast, saturation, brightness)
   - Speed modification
   - Video stabilization

2. **Transitions** (GPU-accelerated)
   - Fade
   - Dissolve
   - Custom duration control

3. **Compositions** (GPU-accelerated)
   - Split-screen layouts
   - Picture-in-picture
   - Multi-scene sequences
   - Custom positioning and scaling

4. **Text and Captions**
   - Title overlays
   - Custom font styles
   - Position control
   - Fade animations

5. **GPU Acceleration**
   - Automatic GPU requirement estimation
   - Dynamic GPU selection based on task complexity
   - Multi-GPU support for parallel processing
   - Automatic resource scaling
   - Fallback to local processing if needed

### Environment Configuration

Add these variables to your `.env` file:
```bash
# Required: Hyperbolic API key for GPU processing
HYPERBOLIC_API_KEY=your_api_key_here

# Optional: Force local processing mode (not recommended for complex tasks)
USE_LOCAL_PROCESSING=false

# Optional: Configure GPU preferences
PREFERRED_GPU_MODEL=A4000  # Optional: Specify preferred GPU model
MIN_VRAM_GB=8.0           # Optional: Minimum VRAM requirement
GPU_COUNT=1               # Optional: Number of GPUs to use
```

The video agent will automatically select and manage GPU resources through Hyperbolic's marketplace based on your task requirements. Local processing mode is available as a fallback but is not recommended for complex video processing tasks.

## Original AgentKit Features

This repository is inspired by and modified from Coinbase's [CDP Agentkit](https://github.com/coinbase/cdp-agentkit). We extend our gratitude to the Coinbase Developer Platform team for their original work.
For the voice agent, we extend the work of [langchain-ai/react-voice-agent](https://github.com/langchain-ai/react-voice-agent).

We recommend reading this entire README before running the application or developing new tools, as many of your questions will be answered here.

## Features

This template demonstrates a chatbot with the following capabilities:

### Compute Operations (via Hyperbolic):

- Connect Ethereum wallet address to Hyperbolic account
- Rent GPU compute resources
- Terminate GPU instances
- Check GPU availability
- Monitor GPU status
- Query billing history
- SSH access to GPU machines
- Run command line tools on remote GPU machines

### Blockchain Operations (via CDP):

- Deploy tokens (ERC-20 & NFTs)
- Manage wallets
- Execute transactions
- Interact with smart contracts

### Twitter Operations:

- Get X account info
- Get User ID from username
- Get an account's recent tweets
- Post tweet
- Delete tweet
- Reply to tweet and check reply status
- Retweet a tweet and check retweet status

### Additional Tools:

- **Podcast Agent**: Tools for video processing and transcription
  - `podcast_agent/aiagenteditor.py`: Trim video files using Gemini and ffmpeg
  - `podcast_agent/geminivideo.py`: Transcribe video files using Gemini

### Knowledge Base Integrations:

- Twitter Knowledge Base: Scrapes tweets from KOLs for informed X posting
- Podcast Knowledge Base: Uses podcast transcripts for accurate Q&A

## Prerequisites

### 1. System Requirements

- Operating System: macOS or Linux (Windows has not been tested)
- Python 3.12 (required)
- Node.js 18+ (for web interface)
- Git

### 2. API Keys and Configuration

- **Core API Keys (Required)**

  - **Anthropic**
    - Get API key from [Anthropic Portal](https://console.anthropic.com/dashboard)
  - **OpenAI** (Required only for voice agent)
    - Get API key from [OpenAI Portal](https://platform.openai.com/api-keys)
  - **CDP**
    - Sign up at [CDP Portal](https://portal.cdp.coinbase.com/access/api)
  - **Hyperbolic** (Required for compute tools)
    - Sign up at [Hyperbolic Portal](https://app.hyperbolic.xyz)
    - Navigate to Settings to generate API key, this is also where you configure ssh access with your RSA public key

- **Optional Integrations**
  - **X (Twitter) API Access**
    - Create a developer account at [Twitter Developer Portal](https://developer.twitter.com)
    - Required credentials: API Key/Secret, Access Token/Secret, Bearer Token, Client ID/Secret
  - **Web Search**: Tavily API key
  - **Google Cloud** (for Podcast Agent/Gemini)
    - Create a service account and download key as `eaccservicekey.json` into the project root
  - **LangChain**: Endpoint, API key, and project name

### 3. Crypto Setup for GPU Compute

To pay for Hyperbolic's GPU compute using crypto:

1. Have an Ethereum wallet with funds on Base network
2. Connect your wallet:
   ```
   Prompt the agent: "connect my wallet 0xYOUR_WALLET_ADDRESS to Hyperbolic"
   ```
3. Send funds:
   - Supported tokens: USDC, USDT, or DAI on Base network
   - Send to: `0xd3cB24E0Ba20865C530831C85Bd6EbC25f6f3B60`
4. Start computing:
   - Funds will be available immediately
   - Use the agent to rent and manage GPU resources

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Hyperbolic-AgentKit.git
cd Hyperbolic-AgentKit
```

### 2. Python Environment Setup

**Using Poetry (Recommended)**:

```bash
# Install Poetry if you haven't
curl -sSL https://install.python-poetry.org | python3 -

# Set up the environment
poetry env use python3.12
poetry install
```

**Browser Automation**

- Install Playwright browsers after installing dependencies:

```bash
poetry run playwright install
```

### 3. Environment Configuration

```bash
# Copy and edit the environment file
cp .env.example .env
nano .env  # or use any text editor
```

**API Keys**
The `.env.example` file contains all possible configurations. Required fields depend on which features you want to use and are specified in the file.

### 4. Character Configuration

The `template.json` file allows you to customize your AI agent's personality and communication style. Duplicate the file and edit the fields to define:

- Agent's name, twitter account info, and description
- Personality traits
- Communication style, tone, and examples
- Background lore and expertise
- KOL list for automated interaction

### 5. Additional Setup

- **Browser Automation** (if using browser tools):
  ```bash
  poetry run playwright install  # or: playwright install
  ```
- **SSH Key** (for GPU compute):
  - Ensure you have an RSA key at `~/.ssh/id_rsa` or configure `SSH_PRIVATE_KEY_PATH`
    - Only RSA keys are supported for now
    - In order to generate an RSA key, run `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`
  - Additional SSH configuration options via environment variables:
    - `DEFAULT_SSH_KEY_PATH`: Default path to use if `SSH_PRIVATE_KEY_PATH` is not set
    - `SSH_KEY_PASSWORD`: Password for your SSH key if it's password-protected
    - `SSH_PASSWORD_KEY_PATH`: Path to a specific password-protected key that matches `SSH_KEY_PASSWORD` 
    - `ALTERNATIVE_SSH_KEYS`: Comma-separated list of alternative key names to try if primary key isn't found (e.g., `id_rsa,id_ed25519,hyperbolic_key`)

## Running the Application

### 1. Voice Agent (Web Interface)

```bash
# Start the server
PYTHONPATH=$PWD/server/src poetry run python server/src/server/app.py

# Access the interface at http://localhost:3000
```

### 2. Terminal Interface

```bash
poetry run python chatbot.py
```

### 3. Gradio Web Interface

```bash
poetry run python gradio_ui.py
# Access the interface at http://localhost:7860
```

## Troubleshooting

### Common Issues:

1. **API Key Errors**

   - Verify all API keys are correctly set in `.env`
   - Check API key permissions and quotas

2. **Python Version Issues**

   ```bash
   # Check Python version
   python --version

   # If needed, install Python 3.12
   # On macOS:
   brew install python@3.12
   # On Ubuntu:
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.12
   ```

3. **Dependencies Issues**

   ```bash
   # Clean Poetry environment
   poetry env remove python3.12
   poetry env use python3.12
   poetry install --no-cache
   ```

4. **Browser Automation Issues**

   ```bash
   # Reinstall Playwright browsers
   poetry run playwright install --force
   ```

5. **Chrome Browser Setup Issues**
   - Ensure Google Chrome is installed on your system
   - Configure a default Chrome profile:
     1. Open Chrome
     2. Make sure a profile is already selected/active
     3. Remove all pinned tabs from the active profile (they can cause browser automation issues)
     4. Ensure Chrome doesn't show a profile selector on startup
   - If using browser automation tools, the agent assumes:
     - Chrome is your default browser
     - A default profile exists and is automatically selected
     - No pinned tabs are present in the active profile

## Adding New Tools

The agent framework supports two main interfaces, each with its own tool registration point:

### 1. Project Structure

New agentic capabilities should be organized in dedicated folders at the root level. For example:

- `twitter_agent/` - Twitter API integration and knowledge base
- `browser_agent/` - Browser automation capabilities
- `podcast_agent/` - Podcast processing and transcription

Each agent folder typically contains:

- `__init__.py` - Exports and initialization
- Core functionality modules (e.g., `twitter_state.py`, `browser_tool.py`)
- Knowledge base implementations if applicable

### 2. Repository Organization

```
Hyperbolic-AgentKit/
├── characters/              # Character configurations
│   └── default.json        # Default character profile
├── *_agent/                # Agent-specific capabilities
│   ├── __init__.py
│   └── core modules
├── server/                 # Voice agent interface
│   └── src/
│       └── server/
│           └── tools.py   # Voice agent tools
└── chatbot.py             # Main agent initialization
```

### 3. Agent Initialization Flow

The agent is initialized through several key functions in `chatbot.py`:

1. `loadCharacters()`:

   - Loads character configurations from JSON files
   - Supports multiple characters with fallback to default
   - Handles character file path resolution

2. `process_character_config()`:

   - Transforms character JSON into agent personality
   - Processes bio, lore, knowledge, style guidelines
   - Formats examples and KOL lists

3. `create_agent_tools()`:

   - Registers tools based on environment configuration
   - Supports multiple tool categories (browser, Twitter, podcast, etc.)
   - Handles tool dependencies and state management

4. `initialize_agent()`:
   - Orchestrates the entire setup process
   - Initializes LLM, character, and knowledge bases
   - Configures tools and agent state

### 4. Voice Agent Structure

The voice agent is implemented in `server/src/server/app.py` using WebSocket communication:

```
server/src/server/
├── app.py              # Main server implementation
├── tools.py            # Voice agent tools
├── prompt.py           # Voice agent instructions
└── static/             # Web interface files
    └── index.html
```

Key components:

1. Server Setup:

   ```python
   app = Starlette(
       routes=[
           Route("/", homepage),
           WebSocketRoute("/ws", websocket_endpoint)
       ]
   )
   ```

2. WebSocket Communication:

   - Browser ↔️ Server real-time communication
   - Handles voice input/output streams
   - Maintains persistent connection for conversation

3. Agent Configuration:

   ```python
   agent = OpenAIVoiceReactAgent(
       model="gpt-4o-realtime-preview", # gpt-4o-realtime-preview and gpt-4o-mini-realtime-preview are the only models that support the voice agent
       tools=TOOLS,
       instructions=full_instructions,
       voice="verse"  # Available: alloy, ash, ballad, coral, echo, sage, shimmer, verse
   )
   ```

4. Character Integration:
   - Reuses `loadCharacters()` and `process_character_config()`
   - Combines base instructions with character personality
   - Maintains consistent persona across interfaces

### 5. Tool Registration

Tools are registered in two places:

1. Main chatbot interface (`chatbot.py`) via `create_agent_tools()`
2. Voice agent interface (`server/src/server/tools.py`) via `create_tools()`

Look at the existing action implementations in `/hyperbolic_agentkit_core` for examples of:

- Adding individual tools and toolkits
- Configuring via environment variables
- Managing dependencies and state

### 6. Tool Categories

The framework includes several categories of pre-built tools you can reference:

- Browser automation tools
- Knowledge base tools
- Social media tools (Twitter/X)
- Blockchain tools (CDP)
- Compute tools (Hyperbolic)
- Web search tools
- HTTP request tools

When adding a new capability, examine similar implementations in existing agent folders for patterns and best practices.

## Support and Resources

- [Hyperbolic Documentation](https://docs.hyperbolic.xyz/docs/getting-started) 
- [CDP Documentation](https://docs.cdp.coinbase.com/agentkit/docs/welcome)
- [X API Documentation](https://docs.x.com/x-api/introduction)
- [Report Issues](https://github.com/yourusername/Hyperbolic-AgentKit/issues)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

This project incorporates work from:

- [CDP Agentkit](https://github.com/coinbase/cdp-agentkit) (Apache License 2.0)
- [langchain-ai/react-voice-agent](https://github.com/langchain-ai/react-voice-agent) (MIT License)
