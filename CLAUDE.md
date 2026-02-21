# Centaur — Project Instructions

These rules are **invariants**. Do not remove, override, or regress them when adding new features.

## UI Invariants

- **Boot splash**: 1000ms "CENTAUR" wordmark splash plays on every page load and refresh (not session-gated)
- **Dark "machined precision" theme**: Subtractive lighting, Z-axis panel shadows, `--bg-desk` as deepest layer
- **Geist Sans + Geist Mono** typography everywhere (never system fonts)
- **Collapsible left sidebar** with CSS width transition
- **Zoom controls**: Floating HUD at bottom-center of DocumentViewer + Ctrl+Scroll on PDF container
- **PDF canvas**: `brightness(0.96)` filter, box-shadow, border (physical desk feel)
- **Segmented control tab bar** with sliding indicator (not plain text tabs)
- **Message arrive animation**: 200ms translateY + opacity on chat messages
- **3-state button presses**: rest → hover → pressed (mechanical feel)

## Frontend Architecture

- **Zustand stores** for all shared state (useDocStore, useViewerStore, useChatStore, useInspectStore, useStudioStore) — never useState waterfalls in page.tsx
- **SSE streaming** for chat responses (ghost bubble stepper, not sync fetch)
- **Doc-scoped queries** via context pill in chat input
- **Structured citations** with fine-grained bbox highlighting (not free-text `[N]` markers)
- **Multi-turn conversation**: Last 6 messages sent to backend

## Backend Architecture

- **Two-phase VLM pipeline**: Layout Analyzer (Glance) → Visual Extractor (Read)
- **Post-extraction validators over prompt rules** (deterministic > probabilistic)
- **Per-series granularity** for accounting_basis (not page-level)
- **Jinja2 SandboxedEnvironment** for template rendering (never raw Template)
- **SQL execution safety**: Semicolon guard + SELECT-only guard + READ ONLY transaction
- **Error sanitization**: Generic messages to clients, full traces to server logs
- **Alembic** manages Studio tables; legacy tables (document_ledger, metric_facts) on create_all()

## Design Philosophy

- No risk classification at ingestion — neutral fact record only
- series_nature/stated_direction = objective (capture at extraction); sentiment_direction = subjective (defer to downstream)
- Principled/mathematical fixes over arbitrary heuristics
