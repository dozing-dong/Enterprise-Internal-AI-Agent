import * as React from "react";
import { cn } from "@/lib/utils";

const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => {
  return (
    <textarea
      ref={ref}
      className={cn(
        "flex min-h-[2.5rem] w-full resize-none rounded-2xl bg-transparent px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground",
        "focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
        "scrollbar-glass",
        className,
      )}
      {...props}
    />
  );
});
Textarea.displayName = "Textarea";

export { Textarea };
