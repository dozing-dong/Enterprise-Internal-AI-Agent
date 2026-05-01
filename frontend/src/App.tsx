import * as React from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/sidebar";
import { ChatWindow } from "@/components/chat-window";
import { useChatStore } from "@/stores/chat-store";

function AmbientBackground() {
  // Three softly drifting blurred orbs give the glass surface something
  // colorful to refract — this is what sells the iOS 26 / macOS Tahoe vibe.
  return (
    <div
      className="pointer-events-none absolute inset-0 overflow-hidden"
      aria-hidden
    >
      <div
        className="ambient-blob h-[42rem] w-[42rem] animate-blob-drift"
        style={{
          top: "-12rem",
          left: "-10rem",
          background:
            "radial-gradient(circle, rgba(96,165,250,0.30) 0%, rgba(96,165,250,0) 72%)",
          animationDelay: "0s",
        }}
      />
      <div
        className="ambient-blob h-[36rem] w-[36rem] animate-blob-drift"
        style={{
          bottom: "-10rem",
          right: "-8rem",
          background:
            "radial-gradient(circle, rgba(167,139,250,0.28) 0%, rgba(167,139,250,0) 72%)",
          animationDelay: "-8s",
        }}
      />
      <div
        className="ambient-blob h-[30rem] w-[30rem] animate-blob-drift"
        style={{
          top: "30%",
          left: "55%",
          background:
            "radial-gradient(circle, rgba(244,114,182,0.20) 0%, rgba(244,114,182,0) 72%)",
          animationDelay: "-16s",
        }}
      />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(1200px 800px at 50% -10%, rgba(255,255,255,0.55), transparent 62%)",
        }}
      />
    </div>
  );
}

export function App() {
  const initialize = useChatStore((s) => s.initialize);

  React.useEffect(() => {
    void initialize();
  }, [initialize]);

  return (
    <TooltipProvider delayDuration={200}>
      <div className="relative h-screen w-screen overflow-hidden bg-background">
        <AmbientBackground />
        <main className="relative z-10 flex h-full w-full gap-4 p-4">
          <Sidebar />
          <ChatWindow />
        </main>
      </div>
    </TooltipProvider>
  );
}
