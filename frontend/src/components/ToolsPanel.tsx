"use client";

import { useEffect, useState } from "react";
import { Loader2, Wrench } from "lucide-react";
import { listTools, AgentToolSummary } from "@/lib/api";

export default function ToolsPanel() {
  const [tools, setTools] = useState<AgentToolSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listTools()
      .then(setTools)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border-subtle)]">
        <span className="text-h2 text-[var(--text-primary)]">AGENT TOOLS</span>
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)] border border-[var(--border)]">
          MCP OFFLINE
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin text-[var(--text-secondary)]" />
          </div>
        ) : tools.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <Wrench size={24} className="mx-auto text-[var(--text-secondary)] opacity-30 mb-2" />
            <p className="text-[12px] text-[var(--text-secondary)]">No tools published</p>
            <p className="text-[11px] text-[var(--text-secondary)] mt-1 opacity-60">
              Publish a prompt or workflow, then expose it as a tool from its version history.
            </p>
          </div>
        ) : (
          <div className="py-1">
            {tools.map((tool) => (
              <div
                key={tool.id}
                className="flex items-start gap-2.5 px-3 py-2.5
                  border-b border-[var(--border-subtle)] last:border-0"
              >
                <span className="text-[16px] mt-0.5 shrink-0">{tool.icon || "~"}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-[12px] text-[var(--text-primary)] truncate">
                    {tool.name}
                  </div>
                  {tool.description && (
                    <div className="text-[11px] text-[var(--text-secondary)] truncate mt-0.5">
                      {tool.description}
                    </div>
                  )}
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] text-[var(--text-secondary)] opacity-60">
                      {tool.output_format}
                    </span>
                    <span className="text-[10px] text-[var(--text-secondary)] opacity-60">
                      {tool.prompt_version_id ? "prompt" : "workflow"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
