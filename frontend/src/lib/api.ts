import type {
  HistoryResponse,
  SessionItem,
  SessionListResponse,
} from "@/types/api";

// All paths are relative — Vite dev proxies them to FastAPI on :8000.
// In production the same-origin reverse proxy should preserve these prefixes.

async function jsonRequest<T>(
  url: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep default detail
    }
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as unknown as T;
  return (await response.json()) as T;
}

export function createSession(title?: string): Promise<SessionItem> {
  return jsonRequest<SessionItem>("/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });
}

export async function listSessions(): Promise<SessionItem[]> {
  const data = await jsonRequest<SessionListResponse>("/sessions", {
    method: "GET",
  });
  return data.sessions;
}

export function renameSession(
  sessionId: string,
  title: string,
): Promise<SessionItem> {
  return jsonRequest<SessionItem>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({ title }),
    },
  );
}

export function deleteSession(sessionId: string): Promise<void> {
  return jsonRequest<void>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

export function getHistory(sessionId: string): Promise<HistoryResponse> {
  return jsonRequest<HistoryResponse>(
    `/history/${encodeURIComponent(sessionId)}`,
    { method: "GET" },
  );
}
