import { Bot, User } from "lucide-react";
import type { ChatMessage } from "@/types/api";
import { cn } from "@/lib/utils";
import { SourcesPopover } from "@/components/sources-popover";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full gap-3 animate-fade-in",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      {!isUser && (
        <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full glass-soft">
          <Bot className="h-4 w-4 text-blue-300" />
        </div>
      )}

      <div
        className={cn(
          "flex max-w-[78%] flex-col gap-2",
          isUser ? "items-end" : "items-start",
        )}
      >
        <div
          className={cn(
            "rounded-3xl px-4 py-3 text-sm leading-relaxed",
            isUser ? "glass-bubble-user" : "glass-bubble-assistant",
          )}
        >
          {message.content ? (
            <p
              className={cn(
                "whitespace-pre-wrap break-words",
                message.isStreaming && !isUser && "stream-caret",
              )}
            >
              {message.content}
            </p>
          ) : message.isStreaming && !isUser ? (
            <div className="flex items-center gap-1.5 py-0.5">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/60 [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/60 [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/60" />
            </div>
          ) : null}
        </div>

        {!isUser && message.sources && message.sources.length > 0 ? (
          <SourcesPopover
            sources={message.sources}
            retrievalQuestion={message.retrievalQuestion}
          />
        ) : null}
      </div>

      {isUser && (
        <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full glass-soft">
          <User className="h-4 w-4 text-white/80" />
        </div>
      )}
    </div>
  );
}
