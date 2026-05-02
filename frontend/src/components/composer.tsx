import * as React from "react";
import { ArrowUp, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";
import type { ChatMode } from "@/types/api";

const MODE_OPTIONS: { id: ChatMode; label: string; hint: string }[] = [
  { id: "rag", label: "RAG", hint: "Fixed pipeline" },
  { id: "agent", label: "Agent", hint: "LLM picks tools" },
];

export function Composer() {
  const [value, setValue] = React.useState("");
  const isStreaming = useChatStore((s) => s.isStreaming);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const mode = useChatStore((s) => s.mode);
  const setMode = useChatStore((s) => s.setMode);
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);

  const autosize = React.useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const next = Math.min(el.scrollHeight, 200);
    el.style.height = `${next}px`;
  }, []);

  React.useEffect(() => {
    autosize();
  }, [value, autosize]);

  const handleSubmit = async (event?: React.FormEvent) => {
    event?.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    setValue("");
    await sendMessage(trimmed);
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void handleSubmit();
    }
  };

  const canSend = value.trim().length > 0 && !isStreaming;

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        "glass-panel flex flex-col gap-2 rounded-3xl p-2 pl-4",
        "transition-shadow focus-within:shadow-[0_0_0_2px_rgba(96,165,250,0.4)]",
      )}
    >
      <div className="flex items-end gap-2">
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isStreaming
              ? "Waiting for the assistant to finish..."
              : "Ask anything..."
          }
          rows={1}
          className="max-h-[200px] flex-1 px-0 py-2.5"
          disabled={isStreaming}
          aria-label="Message input"
        />
        <Button
          type="submit"
          size="icon"
          variant="default"
          disabled={!canSend}
          aria-label="Send message"
          className="h-10 w-10 shrink-0 rounded-full"
        >
          {isStreaming ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <ArrowUp className="h-4 w-4" />
          )}
        </Button>
      </div>

      <div
        className="flex items-center gap-1 pl-0 pr-2 pb-1"
        role="radiogroup"
        aria-label="Execution mode"
      >
        {MODE_OPTIONS.map((option) => {
          const active = mode === option.id;
          return (
            <button
              key={option.id}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={isStreaming}
              onClick={() => setMode(option.id)}
              title={option.hint}
              className={cn(
                "rounded-full px-2.5 py-0.5 text-[11px] font-medium",
                "transition-colors disabled:cursor-not-allowed",
                active
                  ? "bg-primary/15 text-primary"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </form>
  );
}
