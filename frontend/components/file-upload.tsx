"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, File, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { config, isValidFileSize, isValidFileType, formatFileSize } from "@/config"

interface FileUploadProps {
  onUpload: (files: FileList) => void
}

interface FileValidationError {
  message: string;
  type: 'size' | 'type' | 'count';
}

export default function FileUpload({ onUpload }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [validationError, setValidationError] = useState<FileValidationError | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const validateFiles = (files: FileList): boolean => {
    setValidationError(null)
    
    // Check each file
    for (const file of Array.from(files)) {
      // Check file type
      if (!isValidFileType(file)) {
        setValidationError({
          message: `Unsupported file type: ${file.type}. Please upload MP4, MOV, or AVI files.`,
          type: 'type'
        })
        return false
      }
      
      // Check file size
      if (!isValidFileSize(file)) {
        setValidationError({
          message: `File ${file.name} is too large (${formatFileSize(file.size)}). Maximum size is ${config.upload.maxFileSize}MB.`,
          type: 'size'
        })
        return false
      }
    }
    
    return true
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (validateFiles(e.dataTransfer.files)) {
        onUpload(e.dataTransfer.files)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      if (validateFiles(e.target.files)) {
        onUpload(e.target.files)
      }
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const clearValidationError = () => {
    setValidationError(null)
  }

  return (
    <div className="space-y-4">
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          isDragging ? "border-purple-500 bg-purple-500/10" : "border-zinc-800 bg-zinc-900 hover:border-zinc-700"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={clearValidationError}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
          className="hidden" 
          accept={config.upload.allowedFileTypes.join(',')}
          multiple={true}
        />

        <Upload className="h-10 w-10 mx-auto mb-4 text-zinc-500" />

        <h3 className="text-lg font-medium mb-2">Drag and drop your video</h3>
        <p className="text-zinc-500 mb-4">
          Support for MP4, MOV, AVI (max {config.upload.maxFileSize}MB)
        </p>

        <Button
          onClick={handleButtonClick}
          variant="outline"
          className="bg-zinc-800 border-zinc-700 hover:bg-zinc-700 text-white"
        >
          <File className="mr-2 h-4 w-4" />
          Browse Files
        </Button>
      </div>
      
      {validationError && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Upload Error</AlertTitle>
          <AlertDescription>{validationError.message}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}

