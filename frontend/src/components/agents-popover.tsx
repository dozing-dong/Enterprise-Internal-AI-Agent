import { Sparkles, GitBranch, BookOpen, Globe2, PenLine } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";

interface AgentsPopoverProps {
  agentsInvoked: string[];
}

interface AgentMeta {
  label: string;
  description: string;
  Icon: typeof Sparkles;
  badgeClass: string;
}

// Visual metadata for each known sub-agent. Anything we don't know about
// falls back to the "Other" style; we surface it but de-emphasise it.
const AGENT_META: Record<string, AgentMeta> = {
  supervisor: {
    label: "Supervisor",
    description: "Plans which sub-agents to dispatch.",
    Icon: GitBranch,
    badgeClass: "bg-violet-200 text-black",
  },
  policy: {
    label: "PolicyAgent",
    description: "Searches the internal policy knowledge base.",
    Icon: BookOpen,
    badgeClass: "bg-sky-200 text-black",
  },
  external_context: {
    label: "ExternalContextAgent",
    description: "Fetches outside-world info via MCP tools.",
    Icon: Globe2,
    badgeClass: "bg-emerald-200 text-black",
  },
  writer: {
    label: "WriterAgent",
    description: "Assembles the final user-facing answer.",
    Icon: PenLine,
    badgeClass: "bg-amber-200 text-black",
  },
};

const FALLBACK_META: AgentMeta = {
  label: "Agent",
  description: "Sub-agent",
  Icon: Sparkles,
  badgeClass: "bg-gray-200 text-black",
};

function metaFor(name: string): AgentMeta {
  return AGENT_META[name] ?? { ...FALLBACK_META, label: name };
}

export function AgentsPopover({ agentsInvoked }: AgentsPopoverProps) {
  if (!agentsInvoked || agentsInvoked.length === 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="secondary"
          size="sm"
          className="h-7 gap-1.5 px-2.5 text-xs"
        >
          <Sparkles className="h-3.5 w-3.5" />
          {agentsInvoked.length}{" "}
          {agentsInvoked.length === 1 ? "agent" : "agents"}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[22rem] max-w-[calc(100vw-2rem)]"
        align="start"
      >
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-semibold text-foreground">
              Sub-agents invoked
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              Order in which the supervisor dispatched sub-agents this turn.
            </p>
          </div>

          <ol className="space-y-2">
            {agentsInvoked.map((name, idx) => {
              const meta = metaFor(name);
              const Icon = meta.Icon;
              return (
                <li
                  key={`${name}-${idx}`}
                  className="rounded-xl border border-white/10 bg-white/5 p-3"
                >
                  <div className="flex items-center gap-2 text-xs">
                    <span className="rounded-full bg-gray-200 px-2 py-0.5 font-mono text-[10px] text-black">
                      #{idx + 1}
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${meta.badgeClass}`}
                    >
                      <Icon className="h-2.5 w-2.5" />
                      {meta.label}
                    </span>
                  </div>
                  <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground">
                    {meta.description}
                  </p>
                </li>
              );
            })}
          </ol>
        </div>
      </PopoverContent>
    </Popover>
  );
}
