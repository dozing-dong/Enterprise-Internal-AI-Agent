import { BookOpen, Database } from "lucide-react";
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

const STRUCTURED_ROLE = "employee_structured";

const STRUCTURED_FIELDS = [
  "employee_id",
  "department",
  "job_title",
  "email",
] as const;

function isStructuredEmployee(source: SourceItem): boolean {
  const metadata = source.metadata ?? {};
  return metadata.document_role === STRUCTURED_ROLE;
}

function partitionSources(sources: SourceItem[]) {
  const structured: SourceItem[] = [];
  const knowledge: SourceItem[] = [];
  for (const source of sources) {
    if (isStructuredEmployee(source)) {
      structured.push(source);
    } else {
      knowledge.push(source);
    }
  }
  return { structured, knowledge };
}

function formatKnowledgeMetadata(metadata: Record<string, unknown>): string {
  const entries = Object.entries(metadata).filter(
    ([, value]) => value !== undefined && value !== null && value !== "",
  );
  if (entries.length === 0) return "";
  return entries
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(" · ");
}

function StructuredEmployeeCard({ source }: { source: SourceItem }) {
  const metadata = source.metadata ?? {};
  const employeeName =
    typeof metadata.title === "string"
      ? metadata.title.replace(/^Employee profile - /i, "")
      : "Employee";
  return (
    <li className="rounded-xl border border-emerald-300/20 bg-emerald-400/[0.06] p-3">
      <div className="flex items-center gap-2 text-xs">
        <span className="rounded-full bg-emerald-200 px-2 py-0.5 font-mono text-[10px] text-black">
          #{source.rank}
        </span>
        <span className="font-medium text-foreground/90">{employeeName}</span>
      </div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[11px] leading-relaxed">
        {STRUCTURED_FIELDS.map((field) => {
          const value = metadata[field];
          if (value === undefined || value === null || value === "")
            return null;
          return (
            <div key={field} className="contents">
              <dt className="text-muted-foreground">{field}</dt>
              <dd className="break-all text-foreground/85">{String(value)}</dd>
            </div>
          );
        })}
      </dl>
    </li>
  );
}

function KnowledgeSnippetCard({ source }: { source: SourceItem }) {
  const meta = formatKnowledgeMetadata(source.metadata ?? {});
  return (
    <li className="rounded-xl border border-white/10 bg-white/5 p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="rounded-full bg-gray-200 px-2 py-0.5 font-mono text-[10px] text-black">
          #{source.rank}
        </span>
        {meta ? <span className="truncate">{meta}</span> : null}
      </div>
      <p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-foreground/90">
        {source.content}
      </p>
    </li>
  );
}

export function SourcesPopover({
  sources,
  retrievalQuestion,
}: SourcesPopoverProps) {
  if (!sources || sources.length === 0) return null;

  const { structured, knowledge } = partitionSources(sources);

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
          {structured.length > 0 ? (
            <span className="ml-1 inline-flex items-center gap-1 rounded-full bg-emerald-200 px-1.5 py-0.5 text-[10px] text-black">
              <Database className="h-2.5 w-2.5" />
              {structured.length}
            </span>
          ) : null}
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
            <div className="space-y-4">
              {structured.length > 0 ? (
                <section>
                  <div className="mb-2 flex items-center gap-2">
                    <Database className="h-3.5 w-3.5 text-black" />
                    <h5 className="text-[11px] font-semibold uppercase tracking-wider text-black">
                      Directory matches
                    </h5>
                    <span className="text-[10px] text-black">
                      structured PostgreSQL records
                    </span>
                    <span className="ml-auto text-[10px] text-black">
                      {structured.length}
                    </span>
                  </div>
                  <ol className="space-y-2.5">
                    {structured.map((source) => (
                      <StructuredEmployeeCard
                        key={`structured-${source.rank}`}
                        source={source}
                      />
                    ))}
                  </ol>
                </section>
              ) : null}

              {knowledge.length > 0 ? (
                <section>
                  <div className="mb-2 flex items-center gap-2">
                    <BookOpen className="h-3.5 w-3.5 text-black" />
                    <h5 className="text-[11px] font-semibold uppercase tracking-wider text-black">
                      Knowledge base
                    </h5>
                    <span className="text-[10px] text-black">
                      RAG retrieval snippets
                    </span>
                    <span className="ml-auto text-[10px] text-black">
                      {knowledge.length}
                    </span>
                  </div>
                  <ol className="space-y-2.5">
                    {knowledge.map((source) => (
                      <KnowledgeSnippetCard
                        key={`kb-${source.rank}`}
                        source={source}
                      />
                    ))}
                  </ol>
                </section>
              ) : null}
            </div>
          </ScrollArea>
        </div>
      </PopoverContent>
    </Popover>
  );
}
