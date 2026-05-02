import {
  ArrowDownWideNarrow,
  Brain,
  Loader2,
  Pencil,
  Search,
  Sparkles,
  Tag,
  Wrench,
} from "lucide-react";
import type { StreamStage } from "@/types/api";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<StreamStage, string> = {
  rewriting: "Rewriting query",
  retrieving: "Searching knowledge base",
  reranking: "Reranking retrieved snippets",
  generating: "Generating answer",
  titling: "Naming this chat",
  // Agent-mode stages.
  deciding: "Thinking",
  tool_running: "Calling tool",
};

const STAGE_ICONS: Record<StreamStage, React.ComponentType<{ className?: string }>> = {
  rewriting: Pencil,
  retrieving: Search,
  reranking: ArrowDownWideNarrow,
  generating: Sparkles,
  titling: Tag,
  deciding: Brain,
  tool_running: Wrench,
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
      <Loader2 className="h-3.5 w-3.5 animate-spin text-primary/85" />
      <Icon className="h-3.5 w-3.5 text-foreground/70" />
      <span className="text-foreground/90">{message ?? label}</span>
    </div>
  );
}
