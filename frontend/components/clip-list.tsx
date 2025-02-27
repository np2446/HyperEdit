import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { VideoClip } from "@/components/video-editor"

interface ClipListProps {
  clips: VideoClip[]
  onRemove: (id: string) => void
}

export default function ClipList({ clips, onRemove }: ClipListProps) {
  return (
    <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
      {clips.map((clip) => (
        <div key={clip.id} className="flex items-center gap-3 p-3 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="relative w-16 h-12 bg-zinc-800 rounded overflow-hidden flex-shrink-0">
            {clip.type === "mp4" || clip.type === "mov" || clip.type === "avi" ? (
              <video src={clip.url} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-zinc-800 text-zinc-500">
                {clip.type}
              </div>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{clip.name}</p>
            <p className="text-xs text-zinc-500">
              {clip.type.toUpperCase()} • {(clip.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-zinc-500 hover:text-white hover:bg-zinc-800"
            onClick={() => onRemove(clip.id)}
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Remove</span>
          </Button>
        </div>
      ))}
    </div>
  )
}

