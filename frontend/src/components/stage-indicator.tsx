import { Loader2, Pencil, Search, Sparkles, Tag } from "lucide-react";
import type { StreamStage } from "@/types/api";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<StreamStage, string> = {
  rewriting: "Rewriting query",
  retrieving: "Retrieving knowledge",
  generating: "Generating answer",
  titling: "Naming this chat",
};

const STAGE_ICONS: Record<StreamStage, React.ComponentType<{ className?: string }>> = {
  rewriting: Pencil,
  retrieving: Search,
  generating: Sparkles,
  titling: Tag,
};

interface StageIndicatorProps {
  stage: StreamStage | null;
  message?: string | null;
  className?: string;
}

export function StageIndicator({
  stage,
  message,
  className,
}: StageIndicatorProps) {
  if (!stage) return null;
  const Icon = STAGE_ICONS[stage];
  const label = STAGE_LABELS[stage];

  return (
    <div
      className={cn(
        "glass-soft inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium",
        "animate-fade-in",
        className,
      )}
      role="status"
      aria-live="polite"
    >
      <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-300" />
      <Icon className="h-3.5 w-3.5 text-white/80" />
      <span className="text-white/90">{message ?? label}</span>
    </div>
  );
}
