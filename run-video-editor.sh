#!/bin/bash

# AI Video Editor Runner Script
# This script helps set up and run both the frontend and backend components

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required commands are available
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is required but not installed. Please install $1 and try again."
        exit 1
    fi
}

# Check required dependencies
check_command python3
check_command poetry
check_command node
check_command npm

print_message "=== AI Video Editor Setup ==="

# Create directories if they don't exist
mkdir -p backend/uploads backend/outputs input_videos
print_message "Created required directories"

# Check for environment file
if [ ! -f .env ]; then
    print_message "Creating .env file. Please edit it to add your API keys."
    cp .env.example .env
    if [ $? -ne 0 ]; then
        print_warning "No .env.example found. Creating a basic .env file."
        cat > .env << EOL
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LLM Configuration
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-3-opus-20240229

# Backend API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,*

# File Upload Limits
MAX_UPLOAD_SIZE=500

# Environment Mode
DEBUG=true
NODE_ENV=development

# Task Cleanup Configuration
TASK_RETENTION_HOURS=24
EOL
    fi
    print_message "Created .env file - please edit it with your API keys"
fi

# Function to set up Python environment using Poetry
setup_python_env() {
    print_message "Setting up Python environment using Poetry..."
    
    # Check if poetry is installed
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is required but not installed. Please install Poetry and try again."
        print_message "You can install Poetry by following instructions at: https://python-poetry.org/docs/#installation"
        exit 1
    fi
    
    # Install Python dependencies using Poetry
    print_message "Installing Python dependencies..."
    poetry install
    
    # Add backend dependencies if needed
    print_message "Ensuring required packages are installed..."
    poetry add opencv-python-headless fastapi uvicorn python-multipart aiofiles pydantic python-dotenv --group backend
    
    print_message "Python setup complete!"
}

# Function to set up Node.js environment
setup_node_env() {
    print_message "Setting up Node.js frontend environment..."
    
    # Navigate to frontend directory
    cd frontend
    
    # Install dependencies if node_modules doesn't exist or force flag is set
    if [ ! -d "node_modules" ] || [ "$1" == "--force" ]; then
        print_message "Installing Node.js dependencies (this may take a while)..."
        npm install
    else
        print_message "Node modules already installed. Use --force-install to reinstall."
    fi
    
    cd ..
    
    print_message "Node.js setup complete!"
}

# Function to run the backend
run_backend() {
    print_message "Starting backend server..."
    cd backend
    poetry run python main.py
}

# Function to run the frontend
run_frontend() {
    print_message "Starting frontend server..."
    cd frontend
    npm run dev
}

# Parse command line arguments
FORCE_INSTALL=false

for arg in "$@"
do
    case $arg in
        --force-install)
        FORCE_INSTALL=true
        shift
        ;;
    esac
done

# Setup environments
if [ "$FORCE_INSTALL" = true ]; then
    print_message "Forcing reinstallation of dependencies..."
    setup_python_env --force
    setup_node_env --force
else
    setup_python_env
    setup_node_env
fi

# Ask user which components to run
echo ""
echo "What would you like to run?"
echo "1. Backend only"
echo "2. Frontend only"
echo "3. Both (in separate terminals)"
echo "4. Exit without running anything"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        run_backend
        ;;
    2)
        run_frontend
        ;;
    3)
        # Start backend in background
        print_message "Starting backend server in background..."
        (run_backend) &
        backend_pid=$!
        
        # Wait a bit for backend to start
        print_message "Waiting for backend to initialize..."
        sleep 3
        
        # Start frontend
        print_message "Starting frontend server..."
        run_frontend
        
        # Kill backend when frontend exits
        print_message "Shutting down backend server..."
        kill $backend_pid
        ;;
    4)
        print_message "Setup complete. Exiting without running servers."
        exit 0
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac 