import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/layout/Sidebar";
import Providers from "@/components/providers/Providers";

export const metadata: Metadata = {
  title: "SeaSpan Ferries Knowledge-Base Retrieval System",
  description: "Adaptive retrieval system with PageIndex, Vector, Graph RAG, and CAG",
  icons: {
    icon: "/images/Browser-Tab-Logo.png",
  },
};

const themeScript = `
  (function() {
    var t = localStorage.getItem('app-theme');
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  })();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <Providers>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-auto">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
