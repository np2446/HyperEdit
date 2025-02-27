import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';

export const maxDuration = 300; // Set timeout to 5 minutes (300 seconds)

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const prompt = formData.get('prompt') as string;
    const videos = formData.getAll('videos') as File[];

    if (!videos || videos.length === 0) {
      return NextResponse.json(
        { error: 'No video files provided' },
        { status: 400 }
      );
    }

    if (!prompt || prompt.trim() === '') {
      return NextResponse.json(
        { error: 'No prompt provided' },
        { status: 400 }
      );
    }

    // Create input_videos directory if it doesn't exist
    const inputDir = path.join(process.cwd(), '..', 'input_videos');
    await mkdir(inputDir, { recursive: true });
    
    // Save all videos
    const videoNames = [];
    console.log('Saving videos to input directory:', inputDir);
    for (const video of videos) {
      const filePath = path.join(inputDir, video.name);
      console.log('Saving video:', filePath);
      await writeFile(filePath, Buffer.from(await video.arrayBuffer()));
      videoNames.push(video.name);
    }
    console.log('Video names to process:', videoNames);

    // Create form data for the backend request
    const backendFormData = new FormData();
    backendFormData.append('prompt', prompt);
    videoNames.forEach(name => {
      backendFormData.append('videos', name);
    });

    // Log the request details
    console.log('=== Sending request to backend ===');
    console.log('Prompt:', prompt);
    console.log('Videos:', videoNames);

    // Call the FastAPI backend
    const response = await fetch('http://localhost:8000/process-videos', {
      method: 'POST',
      body: backendFormData
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Backend processing failed:', error);
      return NextResponse.json(
        { error: `Backend processing failed: ${error}` },
        { status: response.status }
      );
    }

    const result = await response.json();
    console.log('Backend response:', result);

    // Return the complete response including verification details
    return NextResponse.json({
      status: result.status,
      message: result.message,
      output_path: result.output_path,
      verification: result.verification
    });
  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 