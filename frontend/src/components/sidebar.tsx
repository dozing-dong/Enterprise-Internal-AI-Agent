import * as React from "react";
import {
  Check,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useChatStore } from "@/stores/chat-store";
import { cn, formatRelativeTime } from "@/lib/utils";
import type { SessionItem } from "@/types/api";

export function Sidebar() {
  const sessions = useChatStore((s) => s.sessions);
  const currentSessionId = useChatStore((s) => s.currentSessionId);
  const isLoadingSessions = useChatStore((s) => s.isLoadingSessions);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const startNewChat = useChatStore((s) => s.startNewChat);
  const selectSession = useChatStore((s) => s.selectSession);
  const renameSessionById = useChatStore((s) => s.renameSessionById);
  const deleteSessionById = useChatStore((s) => s.deleteSessionById);

  const [pendingDelete, setPendingDelete] = React.useState<SessionItem | null>(
    null,
  );
  const [renamingId, setRenamingId] = React.useState<string | null>(null);
  const [renameValue, setRenameValue] = React.useState("");

  const handleNewChat = async () => {
    if (isStreaming) return;
    try {
      await startNewChat();
    } catch (err) {
      console.error("Failed to create chat", err);
    }
  };

  const startRename = (session: SessionItem) => {
    setRenamingId(session.session_id);
    setRenameValue(session.title);
  };

  const commitRename = async () => {
    if (!renamingId) return;
    const trimmed = renameValue.trim();
    if (trimmed) {
      await renameSessionById(renamingId, trimmed);
    }
    setRenamingId(null);
    setRenameValue("");
  };

  const cancelRename = () => {
    setRenamingId(null);
    setRenameValue("");
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setPendingDelete(null);
    await deleteSessionById(target.session_id);
  };

  return (
    <aside className="glass-panel flex h-full w-72 shrink-0 flex-col rounded-3xl">
      <div className="flex items-center justify-between p-4 pb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-400 to-violet-500 shadow-lg">
            <MessageSquare className="h-4 w-4 text-primary-foreground" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold tracking-tight">
              Company Copilot
            </span>
          </div>
        </div>
      </div>

      <div className="px-3">
        <Button
          onClick={handleNewChat}
          disabled={isStreaming}
          variant="glass"
          className="w-full justify-start gap-2 rounded-2xl"
        >
          <Plus className="h-4 w-4" />
          New chat
        </Button>
      </div>

      <Separator className="mx-4 my-3 w-auto" />

      <div className="px-4 pb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
        Conversations
      </div>

      <ScrollArea className="flex-1 px-2 scrollbar-glass">
        {isLoadingSessions && sessions.length === 0 ? (
          <div className="space-y-2 p-2">
            {Array.from({ length: 4 }).map((_, idx) => (
              <div
                key={idx}
                className="h-12 rounded-2xl shimmer"
                aria-hidden
              />
            ))}
          </div>
        ) : sessions.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-muted-foreground">
            No conversations yet. Click <span className="font-semibold">New chat</span>{" "}
            to start one.
          </div>
        ) : (
          <ul className="space-y-1 p-1">
            {sessions.map((session) => {
              const isActive = session.session_id === currentSessionId;
              const isRenaming = renamingId === session.session_id;
              return (
                <li key={session.session_id}>
                  <div
                    data-active={isActive}
                    className={cn(
                      "group glass-row grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 overflow-hidden rounded-2xl border border-transparent px-3 py-2",
                      "cursor-pointer",
                    )}
                    onClick={() => {
                      if (isRenaming) return;
                      if (!isActive) void selectSession(session.session_id);
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        if (!isRenaming && !isActive) {
                          void selectSession(session.session_id);
                        }
                      }
                    }}
                  >
                    <div className="min-w-0">
                      {isRenaming ? (
                        <div
                          className="flex items-center gap-1"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <Input
                            autoFocus
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                e.preventDefault();
                                void commitRename();
                              } else if (e.key === "Escape") {
                                e.preventDefault();
                                cancelRename();
                              }
                            }}
                            className="h-7 px-2 text-sm"
                            maxLength={80}
                            aria-label="Rename chat"
                          />
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              void commitRename();
                            }}
                            aria-label="Save rename"
                          >
                            <Check className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              cancelRename();
                            }}
                            aria-label="Cancel rename"
                          >
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      ) : (
                        <>
                          <div
                            className="w-full truncate pr-1 text-sm font-medium text-foreground"
                            title={session.title}
                          >
                            {session.title}
                          </div>
                          <div className="text-[10px] text-muted-foreground">
                            {formatRelativeTime(session.updated_at)}
                          </div>
                        </>
                      )}
                    </div>

                    {!isRenaming && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-sm"
                            onClick={(e) => e.stopPropagation()}
                            className={cn(
                              "h-7 w-7 opacity-0 transition-opacity",
                              "group-hover:opacity-100 data-[state=open]:opacity-100",
                              isActive && "opacity-70",
                            )}
                            aria-label="Chat actions"
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <DropdownMenuItem
                            onSelect={(e) => {
                              e.preventDefault();
                              startRename(session);
                            }}
                          >
                            <Pencil className="h-3.5 w-3.5" />
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            onSelect={(e) => {
                              e.preventDefault();
                              setPendingDelete(session);
                            }}
                            className="text-destructive focus:text-destructive"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            Delete chat
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </ScrollArea>

      <div className="border-t border-border/70 p-3 text-[10px] text-muted-foreground">
        Sessions are identified by <code className="text-foreground/80">session_id</code>;
        each new chat gets a fresh ID.
      </div>

      <Dialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this chat?</DialogTitle>
            <DialogDescription>
              This will permanently remove{" "}
              <span className="font-medium text-foreground">
                {pendingDelete?.title}
              </span>{" "}
              and its message history. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="ghost"
              onClick={() => setPendingDelete(null)}
              type="button"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDelete}
              type="button"
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  );
}
