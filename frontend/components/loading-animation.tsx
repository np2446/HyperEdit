import { Progress } from "@/components/ui/progress"

interface LoadingAnimationProps {
  statusMessage?: string | null;
  progress?: number;
}

export default function LoadingAnimation({ statusMessage, progress = 0 }: LoadingAnimationProps) {
  return (
    <div className="text-center w-full">
      <div className="relative w-24 h-24 mx-auto mb-6">
        <div className="absolute top-0 left-0 w-full h-full border-4 border-t-purple-500 border-r-blue-500 border-b-cyan-500 border-l-emerald-500 rounded-full animate-spin"></div>
        <div className="absolute top-2 left-2 right-2 bottom-2 border-4 border-t-purple-400 border-r-blue-400 border-b-cyan-400 border-l-emerald-400 rounded-full animate-spin animation-delay-150"></div>
        <div className="absolute top-4 left-4 right-4 bottom-4 border-4 border-t-purple-300 border-r-blue-300 border-b-cyan-300 border-l-emerald-300 rounded-full animate-spin animation-delay-300"></div>
      </div>

      <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-blue-500 mb-2">
        Processing Your Video
      </h3>

      <div className="mb-4 w-full max-w-md mx-auto">
        <Progress value={progress} className="h-2" />
        <p className="text-xs text-right mt-1 text-zinc-500">{progress}%</p>
      </div>

      <p className="text-zinc-400 max-w-md">
        {statusMessage || "Our AI is working on your edit. This might take a few minutes depending on the complexity of your request."}
      </p>
    </div>
  )
}

