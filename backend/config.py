import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root (parent of backend directory)
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# Directories
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
INPUT_VIDEOS_DIR = PROJECT_ROOT / "input_videos"

# Create required directories
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
INPUT_VIDEOS_DIR.mkdir(exist_ok=True)

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# VideoAgentProcessor configuration
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-3-opus-20240229")

# File upload limits (in MB)
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "500"))

# Task cleanup configuration
TASK_RETENTION_HOURS = int(os.getenv("TASK_RETENTION_HOURS", "24"))

# Debug mode
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t") 