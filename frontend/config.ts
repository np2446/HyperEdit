// Configuration for the frontend application

export const config = {
  // API configuration
  api: {
    // Base URL for the backend API
    backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
    
    // Timeout for API requests (in milliseconds)
    timeout: 30000,
    
    // Polling interval for checking task status (in milliseconds)
    pollingInterval: 2000,
  },
  
  // Upload configuration
  upload: {
    // Maximum file size in MB
    maxFileSize: 500,
    
    // Allowed file types
    allowedFileTypes: ['video/mp4', 'video/quicktime', 'video/x-msvideo'],
    
    // Max number of files
    maxFiles: 10,
  },
  
  // Video processing configuration
  processing: {
    // Default LLM provider
    defaultLlmProvider: 'anthropic',
    
    // Default to local mode
    defaultLocalMode: true,
  },
};

export const FILE_SIZE_MB = 1024 * 1024;

// Helper to format file size
export function formatFileSize(sizeInBytes: number): string {
  if (sizeInBytes < 1024) {
    return `${sizeInBytes} B`;
  } else if (sizeInBytes < 1024 * 1024) {
    return `${(sizeInBytes / 1024).toFixed(1)} KB`;
  } else if (sizeInBytes < 1024 * 1024 * 1024) {
    return `${(sizeInBytes / (1024 * 1024)).toFixed(1)} MB`;
  } else {
    return `${(sizeInBytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
}

// Helper to validate file size
export function isValidFileSize(file: File): boolean {
  return file.size <= config.upload.maxFileSize * FILE_SIZE_MB;
}

// Helper to validate file type
export function isValidFileType(file: File): boolean {
  return config.upload.allowedFileTypes.includes(file.type);
} 