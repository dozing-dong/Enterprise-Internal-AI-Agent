import { BookOpen } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import type { SourceItem } from "@/types/api";

interface SourcesPopoverProps {
  sources: SourceItem[];
  retrievalQuestion?: string;
}

function formatMetadata(metadata: Record<string, unknown>): string {
  const entries = Object.entries(metadata).filter(
    ([, value]) => value !== undefined && value !== null && value !== "",
  );
  if (entries.length === 0) return "";
  return entries
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(" · ");
}

export function SourcesPopover({
  sources,
  retrievalQuestion,
}: SourcesPopoverProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="secondary"
          size="sm"
          className="h-7 gap-1.5 px-2.5 text-xs"
        >
          <BookOpen className="h-3.5 w-3.5" />
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[28rem] max-w-[calc(100vw-2rem)]"
        align="start"
      >
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-semibold text-foreground">
              Retrieval sources
            </h4>
            {retrievalQuestion ? (
              <p className="mt-1 text-xs text-muted-foreground">
                Search query:{" "}
                <span className="text-foreground/80">{retrievalQuestion}</span>
              </p>
            ) : null}
          </div>
          <ScrollArea className="max-h-[24rem] pr-3">
            <ol className="space-y-2.5">
              {sources.map((source) => {
                const meta = formatMetadata(source.metadata ?? {});
                return (
                  <li
                    key={source.rank}
                    className="rounded-xl border border-white/10 bg-white/5 p-3"
                  >
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span className="rounded-full bg-white/10 px-2 py-0.5 font-mono text-[10px] text-white/80">
                        #{source.rank}
                      </span>
                      {meta ? <span className="truncate">{meta}</span> : null}
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-foreground/90">
                      {source.content}
                    </p>
                  </li>
                );
              })}
            </ol>
          </ScrollArea>
        </div>
      </PopoverContent>
    </Popover>
  );
}
