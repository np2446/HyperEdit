import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import path from 'path';

export const maxDuration = 300; // Set timeout to 5 minutes (300 seconds)

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const prompt = formData.get('prompt') as string;
    const video = formData.get('videos') as File;

    if (!video) {
      return NextResponse.json(
        { error: 'No video file provided' },
        { status: 400 }
      );
    }

    // Create input_videos directory if it doesn't exist
    const inputDir = path.join(process.cwd(), '..', 'input_videos');
    await writeFile(path.join(inputDir, video.name), Buffer.from(await video.arrayBuffer()));

    // Call the FastAPI backend
    const backendUrl = `http://localhost:8000/process-video/${encodeURIComponent(video.name)}?prompt=${encodeURIComponent(prompt)}`;
    const response = await fetch(backendUrl, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: `Backend processing failed: ${error}` },
        { status: response.status }
      );
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 