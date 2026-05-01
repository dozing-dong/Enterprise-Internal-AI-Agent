import type {
  DoneEvent,
  ErrorEvent,
  ProgressEvent,
  SourcesEvent,
  TokenEvent,
} from "@/types/api";

// Lightweight SSE client over fetch + ReadableStream.
// We can't use the browser's EventSource because it only supports GET,
// while POST /chat/stream takes a JSON body.

export interface StreamHandlers {
  onProgress?: (event: ProgressEvent) => void;
  onSources?: (event: SourcesEvent) => void;
  onToken?: (event: TokenEvent) => void;
  onDone?: (event: DoneEvent) => void;
  onError?: (event: ErrorEvent) => void;
}

export interface StreamChatOptions extends StreamHandlers {
  sessionId: string;
  question: string;
  signal?: AbortSignal;
}

interface ParsedEvent {
  event: string;
  data: string;
}

function parseEventBlock(raw: string): ParsedEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of raw.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colonIdx = line.indexOf(":");
    const field = colonIdx === -1 ? line : line.slice(0, colonIdx);
    const value =
      colonIdx === -1
        ? ""
        : line.slice(colonIdx + 1).replace(/^ /, "");
    if (field === "event") {
      event = value;
    } else if (field === "data") {
      dataLines.push(value);
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

function dispatch(parsed: ParsedEvent, handlers: StreamHandlers): void {
  let payload: unknown;
  try {
    payload = JSON.parse(parsed.data);
  } catch {
    return;
  }

  switch (parsed.event) {
    case "progress":
      handlers.onProgress?.(payload as ProgressEvent);
      break;
    case "sources":
      handlers.onSources?.(payload as SourcesEvent);
      break;
    case "token":
      handlers.onToken?.(payload as TokenEvent);
      break;
    case "done":
      handlers.onDone?.(payload as DoneEvent);
      break;
    case "error":
      handlers.onError?.(payload as ErrorEvent);
      break;
    default:
      break;
  }
}

export async function streamChat(opts: StreamChatOptions): Promise<void> {
  const { sessionId, question, signal, ...handlers } = opts;

  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ session_id: sessionId, question }),
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = `Stream failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // ignore
    }
    handlers.onError?.({ detail });
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line. Tolerate both \n\n and \r\n\r\n.
      let separatorIdx = -1;
      while (
        (separatorIdx = buffer.search(/\r?\n\r?\n/)) !== -1
      ) {
        const rawBlock = buffer.slice(0, separatorIdx);
        const matchLength = buffer.match(/\r?\n\r?\n/)![0].length;
        buffer = buffer.slice(separatorIdx + matchLength);
        const parsed = parseEventBlock(rawBlock);
        if (parsed) dispatch(parsed, handlers);
      }
    }

    // Flush any trailing event without a final blank line.
    const trailing = buffer.trim();
    if (trailing) {
      const parsed = parseEventBlock(trailing);
      if (parsed) dispatch(parsed, handlers);
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}
