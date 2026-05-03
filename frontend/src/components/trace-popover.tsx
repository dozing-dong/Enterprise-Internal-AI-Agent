import {
  ListTree,
  AlertCircle,
  CheckCircle2,
  Database,
  BookOpen,
  Cog,
} from "lucide-react";
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

type StepCategory = "structured" | "knowledge" | "other";

interface CategoryStyle {
  label: string;
  Icon: typeof Database;
  badgeClass: string;
  iconClass: string;
}

const CATEGORY_STYLES: Record<StepCategory, CategoryStyle> = {
  structured: {
    label: "Directory",
    Icon: Database,
    badgeClass: "bg-emerald-200 text-black",
    iconClass: "text-black",
  },
  knowledge: {
    label: "RAG",
    Icon: BookOpen,
    badgeClass: "bg-sky-200 text-black",
    iconClass: "text-black",
  },
  other: {
    label: "Other",
    Icon: Cog,
    badgeClass: "bg-gray-200 text-black",
    iconClass: "text-black",
  },
};

const STRUCTURED_STEP_NAMES = new Set(["employee_retrieve", "employee_lookup"]);
const KNOWLEDGE_STEP_NAMES = new Set([
  "rag_answer",
  "rewrite_query",
  "vector_retrieve",
  "keyword_retrieve",
  "fuse_docs",
  "rerank_docs",
  "generate_answer",
  "quality_gate",
]);

function classify(step: TraceStep): StepCategory {
  if (STRUCTURED_STEP_NAMES.has(step.name)) return "structured";
  if (KNOWLEDGE_STEP_NAMES.has(step.name)) return "knowledge";
  return "other";
}

function formatLatency(ms: number | null | undefined): string | null {
  if (ms === null || ms === undefined) return null;
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function TracePopover({ trace }: TracePopoverProps) {
  if (!trace || trace.length === 0) return null;

  const counts: Record<StepCategory, number> = {
    structured: 0,
    knowledge: 0,
    other: 0,
  };
  const enriched = trace.map((step) => {
    const category = classify(step);
    counts[category] += 1;
    return { step, category };
  });

  const summaryEntries = (Object.keys(counts) as StepCategory[]).filter(
    (key) => counts[key] > 0,
  );

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

          {summaryEntries.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5">
              {summaryEntries.map((category) => {
                const style = CATEGORY_STYLES[category];
                const Icon = style.Icon;
                return (
                  <span
                    key={category}
                    className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${style.badgeClass}`}
                  >
                    <Icon className="h-2.5 w-2.5" />
                    {style.label} · {counts[category]}
                  </span>
                );
              })}
            </div>
          ) : null}

          <ScrollArea className="max-h-[24rem] pr-3">
            <ol className="space-y-2.5">
              {enriched.map(({ step, category }) => {
                const ok = step.ok !== false;
                const StatusIcon = ok ? CheckCircle2 : AlertCircle;
                const latency = formatLatency(step.latency_ms);
                const style = CATEGORY_STYLES[category];
                const CategoryIcon = style.Icon;
                return (
                  <li
                    key={step.step}
                    className="rounded-xl border border-white/10 bg-white/5 p-3"
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <span className="rounded-full bg-gray-200 px-2 py-0.5 font-mono text-[10px] text-black">
                        #{step.step}
                      </span>
                      <CategoryIcon
                        className={`h-3.5 w-3.5 ${style.iconClass}`}
                      />
                      <StatusIcon
                        className={
                          "h-3.5 w-3.5 " +
                          (ok ? "text-emerald-400/85" : "text-red-400/85")
                        }
                      />
                      <span className="font-medium text-foreground/90">
                        {step.name}
                      </span>
                      {step.agent ? (
                        <span className="rounded-full bg-violet-200 px-1.5 py-0.5 text-[10px] font-medium text-black">
                          {step.agent}
                        </span>
                      ) : null}
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
