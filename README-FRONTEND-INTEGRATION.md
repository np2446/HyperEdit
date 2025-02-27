# AI Video Editor: Frontend Integration

This project integrates the AI Video Agent with a Next.js frontend, allowing users to upload videos, describe their desired edits, and have them processed by the AI.

## Project Structure

- `frontend/`: Next.js frontend application
- `backend/`: FastAPI backend that interfaces with the VideoAgentProcessor
- `video_agent/`: Core video agent code

## Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- OpenAI API key or Anthropic API key
- (Optional) GPU for video processing

## Setup Instructions

### 1. Set up environment variables

Create or update your `.env` file with the required API keys:

```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 2. Install and run the FastAPI backend

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt
pip install -e ..  # Install the main project in development mode

# Run the backend server
python main.py
```

The FastAPI backend will run on http://localhost:8000.

### 3. Install and run the Next.js frontend

```bash
# Install frontend dependencies
cd frontend
npm install

# Run the development server
npm run dev
```

The Next.js frontend will run on http://localhost:3000.

## How to Use

1. Open http://localhost:3000 in your browser
2. Upload one or more video clips using the file upload interface
3. Enter a prompt describing how you want the videos to be edited
4. (Optional) Check the "Local Mode" checkbox to run in local mode
5. Click "Process Video" to start the AI video editing process
6. Wait for the processing to complete (this may take several minutes)
7. Once complete, the edited video will appear in the preview area
8. You can download the edited video by clicking the "Download" button

## Local Mode vs. Remote Mode

- **Local Mode**: Processes videos on your local machine. Faster to set up but may be limited by your hardware.
- **Remote Mode**: Processes videos on remote GPU servers. More powerful but requires SSH keys and GPU access.

## Troubleshooting

- If you encounter issues with video processing, check the backend logs for detailed error information.
- Ensure your API keys are correctly set in the `.env` file.
- If using remote mode, ensure your SSH keys are properly configured.
- For file upload issues, check that your videos are in a supported format (MP4, MOV, AVI) and under the size limit.

## Development Notes

- The frontend proxies API requests to the backend through Next.js API routes to avoid CORS issues.
- Video processing occurs asynchronously in the background on the backend.
- Task status is polled every 2 seconds to update the UI with progress. 