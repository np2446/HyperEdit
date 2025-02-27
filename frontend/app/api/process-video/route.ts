import { NextRequest, NextResponse } from 'next/server';

export const maxDuration = 300; // Set timeout to 5 minutes (300 seconds)

export async function POST(req: NextRequest) {
  try {
    // Create a FormData object to forward to the backend
    const formData = await req.formData();
    
    // Extract the prompt and local_mode flag
    const prompt = formData.get('prompt') as string;
    const localModeStr = formData.get('local_mode') as string;
    const localMode = localModeStr === 'true';
    
    // Extract video files
    const videos = formData.getAll('videos');
    
    if (!prompt) {
      return NextResponse.json(
        { detail: 'Prompt is required' },
        { status: 400 }
      );
    }
    
    if (!videos || videos.length === 0) {
      return NextResponse.json(
        { detail: 'At least one video file is required' },
        { status: 400 }
      );
    }

    // Forward the request to our FastAPI backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    // Create a new FormData to send to the backend
    const backendFormData = new FormData();
    backendFormData.append('prompt', prompt);
    backendFormData.append('local_mode', localMode.toString());
    
    // Add each video file to the form data
    videos.forEach((video, index) => {
      if (video instanceof File) {
        backendFormData.append('videos', video);
      }
    });
    
    // Send the request to the backend
    const response = await fetch(`${backendUrl}/process-video`, {
      method: 'POST',
      body: backendFormData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { detail: errorData.detail || 'Failed to process video' },
        { status: response.status }
      );
    }
    
    // Return the response from the backend
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
} 