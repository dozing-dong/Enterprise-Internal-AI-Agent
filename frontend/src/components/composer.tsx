import * as React from "react";
import { ArrowUp, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";

export function Composer() {
  const [value, setValue] = React.useState("");
  const isStreaming = useChatStore((s) => s.isStreaming);
  const sendMessage = useChatStore((s) => s.sendMessage);
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
        "glass-panel flex items-end gap-2 rounded-3xl p-2 pl-4",
        "transition-shadow focus-within:shadow-[0_0_0_2px_rgba(96,165,250,0.4)]",
      )}
    >
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
    </form>
  );
}
