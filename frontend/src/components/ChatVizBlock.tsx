"use client";

import { useEffect, useRef, useState } from "react";
import type { VizPayload } from "@/lib/api";

export default function ChatVizBlock({ spec, data, sql, title }: VizPayload) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSql, setShowSql] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    // No spec — show data as table fallback
    if (!spec) return;

    let cancelled = false;

    (async () => {
      try {
        // Dynamic import to avoid SSR issues and reduce bundle size
        const vegaEmbed = (await import("vega-embed")).default;

        if (cancelled) return;

        const fullSpec = {
          ...spec,
          data: { values: data },
        };

        await vegaEmbed(containerRef.current!, fullSpec as never, {
          actions: { export: true, source: false, compiled: false },
          theme: "dark" as never,
          renderer: "svg",
        });
      } catch (e) {
        if (!cancelled) {
          setError(`Chart render failed: ${(e as Error).message}`);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [spec, data]);

  // Table fallback when no spec
  if (!spec && data.length > 0) {
    const columns = Object.keys(data[0]);
    return (
      <div className="my-3 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] overflow-hidden">
        {title && (
          <div className="px-3 py-2 text-xs font-medium text-[var(--text-primary)] border-b border-[var(--border)]">
            {title}
          </div>
        )}
        <div className="overflow-auto max-h-64">
          <table className="w-full text-xs">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th
                    key={col}
                    className="px-2 py-1.5 text-left font-medium text-[var(--text-secondary)] bg-[var(--bg-panel)] sticky top-0"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.slice(0, 50).map((row, i) => (
                <tr key={i} className="border-t border-[var(--border-subtle)]">
                  {columns.map((col) => (
                    <td
                      key={col}
                      className="px-2 py-1 text-[var(--text-primary)] font-mono"
                    >
                      {row[col] != null ? String(row[col]) : ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.length > 50 && (
          <div className="px-3 py-1 text-[10px] text-[var(--text-secondary)]">
            Showing 50 of {data.length} rows
          </div>
        )}
        {sql && (
          <details className="border-t border-[var(--border)]">
            <summary className="px-3 py-1.5 text-[10px] text-[var(--text-secondary)] cursor-pointer hover:text-[var(--text-primary)]">
              SQL Query
            </summary>
            <pre className="px-3 py-2 text-[10px] text-[var(--text-secondary)] font-mono overflow-auto bg-[var(--bg-primary)]">
              {sql}
            </pre>
          </details>
        )}
      </div>
    );
  }

  return (
    <div className="my-3 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] overflow-hidden">
      {title && (
        <div className="px-3 py-2 text-xs font-medium text-[var(--text-primary)] border-b border-[var(--border)]">
          {title}
        </div>
      )}

      {error ? (
        <div className="px-3 py-4 text-xs text-red-400">{error}</div>
      ) : (
        <div ref={containerRef} className="px-2 py-2 [&_svg]:max-w-full" />
      )}

      {sql && (
        <div className="border-t border-[var(--border)]">
          <button
            onClick={() => setShowSql(!showSql)}
            className="w-full px-3 py-1.5 text-[10px] text-[var(--text-secondary)] text-left cursor-pointer hover:text-[var(--text-primary)] transition-colors"
          >
            {showSql ? "Hide" : "Show"} SQL Query
          </button>
          {showSql && (
            <pre className="px-3 py-2 text-[10px] text-[var(--text-secondary)] font-mono overflow-auto bg-[var(--bg-primary)]">
              {sql}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
