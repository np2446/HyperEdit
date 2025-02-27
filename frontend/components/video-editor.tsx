"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Upload, Play } from "lucide-react"
import FileUpload from "@/components/file-upload"
import ClipList from "@/components/clip-list"
import LoadingAnimation from "@/components/loading-animation"
import VideoPreview from "@/components/video-preview"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { config } from "@/config"

export type VideoClip = {
  id: string
  name: string
  type: string
  size: number
  url: string
  file?: File
}

interface TaskStatus {
  status: "processing" | "completed" | "failed"
  message: string
  video_url?: string
  result?: string
}

export default function VideoEditor() {
  const [prompt, setPrompt] = useState("")
  const [clips, setClips] = useState<VideoClip[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedVideo, setProcessedVideo] = useState<string | null>(null)
  const [isLocalMode, setIsLocalMode] = useState(config.processing.defaultLocalMode)
  const [error, setError] = useState<string | null>(null)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [processingProgress, setProcessingProgress] = useState<number>(0)

  // Poll for task status when taskId changes
  useEffect(() => {
    if (!taskId) return
    
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/task-status/${taskId}`)
        
        if (!response.ok) {
          throw new Error("Failed to fetch task status")
        }
        
        const taskStatus: TaskStatus = await response.json()
        setStatusMessage(taskStatus.message)
        
        // Set a fake progress percentage based on messages
        if (taskStatus.status === "processing") {
          // Increment progress to show something is happening
          setProcessingProgress(prev => Math.min(prev + 5, 95))
        }
        
        if (taskStatus.status === "completed") {
          clearInterval(pollInterval)
          setIsProcessing(false)
          setProcessingProgress(100)
          if (taskStatus.video_url) {
            setProcessedVideo(taskStatus.video_url)
          }
        } else if (taskStatus.status === "failed") {
          clearInterval(pollInterval)
          setIsProcessing(false)
          setError(taskStatus.message)
        }
      } catch (err) {
        console.error("Error polling task status:", err)
      }
    }, config.api.pollingInterval) // Poll interval from config
    
    return () => clearInterval(pollInterval)
  }, [taskId])

  const handleFileUpload = (files: FileList) => {
    const newClips = Array.from(files).map((file) => {
      return {
        id: Math.random().toString(36).substring(7),
        name: file.name,
        type: file.type.split("/")[1],
        size: file.size,
        url: URL.createObjectURL(file),
        file: file,
      }
    })

    setClips([...clips, ...newClips])
  }

  const removeClip = (id: string) => {
    setClips(clips.filter((clip) => clip.id !== id))
  }

  const processVideo = async () => {
    if (clips.length === 0) return
    setError(null)
    setIsProcessing(true)
    setProcessedVideo(null)
    setTaskId(null)
    setStatusMessage(null)
    setProcessingProgress(0)

    try {
      // Create FormData to send files to the backend
      const formData = new FormData()
      formData.append("prompt", prompt)
      formData.append("local_mode", isLocalMode.toString())
      
      // Add all video files
      clips.forEach((clip) => {
        if (clip.file) {
          formData.append("videos", clip.file)
        }
      })

      // Set up request timeout
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), config.api.timeout)

      // Call the backend API
      const response = await fetch("/api/process-video", {
        method: "POST",
        body: formData,
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Failed to process video")
      }

      const data = await response.json()
      
      // Set task ID for polling
      if (data.task_id) {
        setTaskId(data.task_id)
        setStatusMessage(data.message || "Processing video...")
        setProcessingProgress(10) // Start progress at 10%
      } else {
        // If there's a direct result
        setIsProcessing(false)
        if (data.video_url) {
          setProcessedVideo(data.video_url)
        }
      }
    } catch (err) {
      console.error("Error processing video:", err)
      if (err instanceof DOMException && err.name === 'AbortError') {
        setError('Request timed out. Please try again.')
      } else {
        setError(err instanceof Error ? err.message : "An unknown error occurred")
      }
      setIsProcessing(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-cyan-500 to-emerald-400 mb-2">
          AI Video Editor
        </h1>
        <p className="text-zinc-400">Upload your clips, describe your edits, and let AI do the magic</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-white">Describe Your Edit</h2>
            <Textarea
              placeholder="Describe how you want your video to be edited..."
              className="h-32 bg-zinc-900 border-zinc-800 focus-visible:ring-purple-500"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-white">Upload Clips</h2>
            <FileUpload onUpload={handleFileUpload} />
          </div>

          {clips.length > 0 && (
            <div className="space-y-2">
              <h2 className="text-xl font-semibold text-white">Your Clips</h2>
              <ClipList clips={clips} onRemove={removeClip} />
            </div>
          )}
          
          <div className="flex items-center space-x-2 py-2">
            <Checkbox 
              id="local-mode" 
              checked={isLocalMode}
              onCheckedChange={(checked) => setIsLocalMode(checked === true)}
            />
            <Label htmlFor="local-mode" className="text-zinc-400">
              Run in local mode (faster, but less powerful)
            </Label>
          </div>

          <Button
            onClick={processVideo}
            disabled={clips.length === 0 || prompt.trim() === "" || isProcessing}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
          >
            <Play className="mr-2 h-4 w-4" />
            Process Video
          </Button>
          
          {error && (
            <div className="text-red-500 text-sm mt-2 p-2 bg-red-500/10 rounded border border-red-500/20">
              {error}
            </div>
          )}
        </div>

        <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-6 h-[500px] flex items-center justify-center">
          {isProcessing ? (
            <LoadingAnimation statusMessage={statusMessage} progress={processingProgress} />
          ) : processedVideo ? (
            <VideoPreview videoUrl={processedVideo} />
          ) : (
            <div className="text-center text-zinc-500">
              <Upload className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p>Upload clips and process to see the result here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

