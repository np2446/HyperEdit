"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Download, Play, Pause } from "lucide-react"

interface VideoPreviewProps {
  videoUrl: string
}

export default function VideoPreview({ videoUrl }: VideoPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false)

  const togglePlay = () => {
    const video = document.getElementById("preview-video") as HTMLVideoElement
    if (video.paused) {
      video.play()
      setIsPlaying(true)
    } else {
      video.pause()
      setIsPlaying(false)
    }
  }

  const handleVideoEnd = () => {
    setIsPlaying(false)
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="relative flex-1 bg-black rounded-lg overflow-hidden">
        <video id="preview-video" src={videoUrl} className="w-full h-full object-contain" onEnded={handleVideoEnd} />

        <div className="absolute inset-0 flex items-center justify-center">
          {!isPlaying && (
            <Button
              onClick={togglePlay}
              className="rounded-full w-16 h-16 bg-purple-600/80 hover:bg-purple-600 text-white"
            >
              <Play className="h-8 w-8 fill-current" />
            </Button>
          )}
        </div>
      </div>

      <div className="flex justify-between items-center mt-4">
        <Button
          variant="outline"
          className="bg-zinc-800 border-zinc-700 hover:bg-zinc-700 text-white"
          onClick={togglePlay}
        >
          {isPlaying ? (
            <>
              <Pause className="mr-2 h-4 w-4" />
              Pause
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Play
            </>
          )}
        </Button>

        <Button
          className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
          onClick={() => window.open(videoUrl, "_blank")}
        >
          <Download className="mr-2 h-4 w-4" />
          Download
        </Button>
      </div>
    </div>
  )
}

