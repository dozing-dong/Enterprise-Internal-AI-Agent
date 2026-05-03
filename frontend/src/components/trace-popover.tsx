import { ListTree, AlertCircle, CheckCircle2 } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import type { TraceStep } from "@/types/api";

interface TracePopoverProps {
  trace: TraceStep[];
}

function formatLatency(ms: number | null | undefined): string | null {
  if (ms === null || ms === undefined) return null;
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function TracePopover({ trace }: TracePopoverProps) {
  if (!trace || trace.length === 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="secondary"
          size="sm"
          className="h-7 gap-1.5 px-2.5 text-xs"
        >
          <ListTree className="h-3.5 w-3.5" />
          {trace.length} {trace.length === 1 ? "step" : "steps"}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[28rem] max-w-[calc(100vw-2rem)]"
        align="start"
      >
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-semibold text-foreground">
              Call trace
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              Steps executed by this turn.
            </p>
          </div>
          <ScrollArea className="max-h-[24rem] pr-3">
            <ol className="space-y-2.5">
              {trace.map((step) => {
                const ok = step.ok !== false;
                const Icon = ok ? CheckCircle2 : AlertCircle;
                const latency = formatLatency(step.latency_ms);
                return (
                  <li
                    key={step.step}
                    className="rounded-xl border border-white/10 bg-white/5 p-3"
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <span className="rounded-full bg-white/10 px-2 py-0.5 font-mono text-[10px] text-white/80">
                        #{step.step}
                      </span>
                      <Icon
                        className={
                          "h-3.5 w-3.5 " +
                          (ok ? "text-emerald-400/85" : "text-red-400/85")
                        }
                      />
                      <span className="font-medium text-foreground/90">
                        {step.name}
                      </span>
                      {latency ? (
                        <span className="ml-auto text-[10px] text-muted-foreground">
                          {latency}
                        </span>
                      ) : null}
                    </div>
                    {step.input_summary ? (
                      <p className="mt-2 break-words text-[11px] leading-relaxed text-muted-foreground">
                        <span className="text-foreground/60">input:</span>{" "}
                        {step.input_summary}
                      </p>
                    ) : null}
                    {step.output_summary ? (
                      <p className="mt-1 break-words text-[11px] leading-relaxed text-foreground/85">
                        <span className="text-foreground/60">output:</span>{" "}
                        {step.output_summary}
                      </p>
                    ) : null}
                    {step.error ? (
                      <p className="mt-1 break-words text-[11px] leading-relaxed text-red-300/90">
                        <span className="text-red-200/80">error:</span>{" "}
                        {step.error}
                      </p>
                    ) : null}
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
