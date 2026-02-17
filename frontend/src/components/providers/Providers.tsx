"use client";

import { LogStreamProvider } from "@/lib/websocket";
import { ThemeProvider } from "@/lib/theme";

/**
 * Client-side providers that wrap the entire app.
 * Keeps the WebSocket log stream alive across all route navigations.
 */
export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <LogStreamProvider>{children}</LogStreamProvider>
    </ThemeProvider>
  );
}
