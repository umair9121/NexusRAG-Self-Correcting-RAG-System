import Link from "next/link";
import { FileQuestion } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
      <FileQuestion size={48} className="text-[var(--text-secondary)] mb-4" />
      <h1 className="text-xl font-semibold text-[var(--text-primary)] mb-1">Page not found</h1>
      <p className="text-sm text-[var(--text-secondary)] mb-6">The page you’re looking for doesn’t exist or was moved.</p>
      <Link
        href="/"
        className="px-4 py-2 rounded-lg bg-[var(--accent)] text-white text-sm font-medium hover:bg-[var(--accent-hover)]"
      >
        Back to Documents
      </Link>
    </div>
  );
}
