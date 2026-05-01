import * as React from "react";
import { Sparkles } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Composer } from "@/components/composer";
import { MessageBubble } from "@/components/message-bubble";
import { StageIndicator } from "@/components/stage-indicator";
import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";

const SUGGESTIONS = [
  "What does the knowledge base cover?",
  "Summarize the most recent document.",
  "Compare the top-rated answers on retrieval quality.",
];

function EmptyState() {
  const startNewChat = useChatStore((s) => s.startNewChat);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const currentSessionId = useChatStore((s) => s.currentSessionId);

  const handleSuggestion = async (text: string) => {
    if (isStreaming) return;
    if (!currentSessionId) {
      try {
        await startNewChat();
      } catch (err) {
        console.error("Failed to create chat", err);
        return;
      }
    }
    void sendMessage(text);
  };

  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 px-6 text-center animate-fade-in">
      <div className="flex h-16 w-16 items-center justify-center rounded-3xl bg-gradient-to-br from-blue-400/30 to-violet-500/30 shadow-2xl ring-1 ring-white/15 backdrop-blur-xl">
        <Sparkles className="h-7 w-7 text-white" />
      </div>
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">
          How can I help today?
        </h1>
        <p className="max-w-md text-sm text-muted-foreground">
          Ask anything about your enterprise knowledge base. Answers are
          grounded in retrieved sources you can inspect.
        </p>
      </div>
      <div className="flex w-full max-w-xl flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <Button
            key={suggestion}
            variant="secondary"
            size="sm"
            className="rounded-full text-xs"
            onClick={() => void handleSuggestion(suggestion)}
            disabled={isStreaming}
          >
            {suggestion}
          </Button>
        ))}
      </div>
    </div>
  );
}

export function ChatWindow() {
  const messages = useChatStore((s) => s.messages);
  const stage = useChatStore((s) => s.stage);
  const stageMessage = useChatStore((s) => s.stageMessage);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const isLoadingHistory = useChatStore((s) => s.isLoadingHistory);
  const error = useChatStore((s) => s.error);
  const clearError = useChatStore((s) => s.clearError);
  const currentSessionId = useChatStore((s) => s.currentSessionId);
  const sessions = useChatStore((s) => s.sessions);

  const scrollContainerRef = React.useRef<HTMLDivElement | null>(null);
  const bottomRef = React.useRef<HTMLDivElement | null>(null);

  const currentTitle = React.useMemo(
    () =>
      sessions.find((s) => s.session_id === currentSessionId)?.title ??
      "New chat",
    [sessions, currentSessionId],
  );

  React.useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages.length, isStreaming]);

  React.useEffect(() => {
    // Smoothly stick to bottom while tokens stream in.
    if (!isStreaming) return;
    const id = window.setInterval(() => {
      bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
    }, 250);
    return () => window.clearInterval(id);
  }, [isStreaming]);

  const showEmpty =
    !isLoadingHistory && messages.length === 0 && !isStreaming;

  return (
    <section className="flex h-full min-w-0 flex-1 flex-col gap-4">
      <header className="glass-soft flex items-center justify-between rounded-3xl px-5 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold tracking-tight">
            {currentTitle}
          </h2>
          {currentSessionId ? (
            <p className="truncate text-[10px] text-muted-foreground">
              session_id:{" "}
              <code className="font-mono text-foreground/80">
                {currentSessionId}
              </code>
            </p>
          ) : (
            <p className="text-[10px] text-muted-foreground">
              No active session.
            </p>
          )}
        </div>
        <StageIndicator stage={stage} message={stageMessage} />
      </header>

      <div
        ref={scrollContainerRef}
        className={cn(
          "glass-panel relative flex flex-1 flex-col overflow-hidden rounded-3xl",
        )}
      >
        {error ? (
          <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-2 rounded-t-3xl bg-destructive/30 px-4 py-2 text-xs text-destructive-foreground">
            <span>{error}</span>
            <button
              type="button"
              onClick={clearError}
              className="font-semibold underline-offset-2 hover:underline"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {showEmpty ? (
          <EmptyState />
        ) : (
          <ScrollArea className="flex-1 scrollbar-glass">
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-5 py-6">
              {isLoadingHistory && messages.length === 0 ? (
                <div className="space-y-3">
                  {Array.from({ length: 3 }).map((_, idx) => (
                    <div
                      key={idx}
                      className="h-16 rounded-3xl shimmer"
                      aria-hidden
                    />
                  ))}
                </div>
              ) : (
                messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))
              )}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>
        )}
      </div>

      <div className="mx-auto w-full max-w-3xl">
        <Composer />
        <p className="mt-2 text-center text-[10px] text-muted-foreground">
          Enter to send · Shift + Enter for a new line
        </p>
      </div>
    </section>
  );
}
