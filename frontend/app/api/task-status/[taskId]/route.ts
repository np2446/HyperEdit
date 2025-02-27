import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  req: NextRequest,
  { params }: { params: { taskId: string } }
) {
  try {
    const taskId = params.taskId;
    
    if (!taskId) {
      return NextResponse.json(
        { detail: 'Task ID is required' },
        { status: 400 }
      );
    }
    
    // Forward the request to our FastAPI backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    const response = await fetch(`${backendUrl}/task-status/${taskId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { detail: errorData.detail || 'Failed to get task status' },
        { status: response.status }
      );
    }
    
    // Return the task status from the backend
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('Error getting task status:', error);
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
} 