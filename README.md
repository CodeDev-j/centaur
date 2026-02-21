# CENTAUR

**Financial Document Intelligence Platform**

Centaur is a full-stack RAG engine purpose-built for high-finance document analysis. It ingests complex financial documents (CIMs, earnings slides, lender presentations, financial models), extracts structured data from charts and tables with forensic precision, and serves it through a cited conversational interface where every claim links back to its source.

**Python 3.11+** | **Next.js 16** | **Docker** | **Postgres** | **Qdrant** | **LangGraph**

---

## What Makes This Different

Most RAG systems treat documents as flat text. Financial documents aren't flat text — they're dense visual artifacts where a single stacked bar chart encodes more information than three pages of prose. Centaur solves this with three architectural decisions:

1. **Two-Phase Visual Pipeline (Glance + Read).** A layout analyzer detects chart regions and their archetypes (Cartesian, Waterfall, Valuation Field). A forensic vision extractor then reads each region with full context — axis labels, legends, footnotes, annotations — producing typed `MetricSeries` with per-datapoint provenance.

2. **Hybrid Search with Structured Analytics.** Qualitative content (narratives, summaries, insights) lives in Qdrant as dense+sparse vectors. Quantitative content (every extracted datapoint) lives in Postgres as pre-computed `metric_facts` rows. The query router decides which path — or both — to use.

3. **Fact-Scoped Citations.** Every generated answer carries `[N]` citation markers produced by structured output — the LLM never formats markers itself. Each badge resolves to a `Citation` with a fine-grained bounding box that highlights the specific value or sentence on the document page, not the entire table or chart region.

---

## Architecture

```
                                    CENTAUR
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   INGESTION (Write Path)                                        │
    │   ┌───────────┐    ┌──────────────┐    ┌────────────────────┐   │
    │   │  Smart    │───>│  Dual-Helix  │───>│  Unified Document  │   │
    │   │  Router   │    │  Parsers     │    │  (Typed Stream)    │   │
    │   └───────────┘    │              │    └────────┬───────────┘   │
    │                    │  Helix A:    │             │               │
    │                    │  PDF/Visual  │        ┌────┴────┐          │
    │                    │  (Docling +  │        │         │          │
    │                    │   VLM x2)    │        ▼         ▼          │
    │                    │              │   ┌────────┐ ┌────────┐     │
    │                    │  Helix B:    │   │ Qdrant │ │Postgres│     │
    │                    │  Native      │   │ Hybrid │ │ Metric │     │
    │                    │  (Excel)     │   │ Index  │ │ Facts  │     │
    │                    └──────────────┘   └────────┘ └────────┘     │
    │                                           │           │         │
    │   RETRIEVAL (Read Path)                   │           │         │
    │   ┌──────────┐    ┌──────────────┐   ┌────┴───────────┴───┐     │
    │   │  Query   │───>│  Term        │──>│  Hybrid Search     │     │
    │   │  Router  │    │  Injector    │   │  (Dense + BM25)    │     │
    │   │  (LLM)   │    │  (Multilin.) │   │  + Voyage Rerank   │     │
    │   └──────────┘    └──────────────┘   │  + Text-to-SQL     │     │
    │        │                             └──────────┬─────────┘     │
    │        │                                        │               │
    │   GENERATION                                    │               │
    │   ┌───────────────────────────────────────┐     │               │
    │   │  GPT-4.1 Structured Output            │<────┘               │
    │   │  + Citation Sidecar (fact-based dedup)│                     │
    │   │  + Fine-Grained BBox Resolution       │                     │
    │   └──────────────┬────────────────────────┘                     │
    │                  │                                              │
    │   PROMPT STUDIO                                                 │
    │   ┌───────────────────────────────────────┐                     │
    │   │  Saved Prompts (Jinja2 + versioning)  │                     │
    │   │  Workflow Chains (sequential + HITL)   │                     │
    │   │  4 Exec Modes (RAG/Structured/Direct/ │                     │
    │   │  SQL) with retrieval caching           │                     │
    │   └───────────────────────────────────────┘                     │
    │                                                                 │
    └──────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   FastAPI  ──>  SSE Stream  ──>  Next.js 16 + Zustand          │
    │                    ┌─────┬────────┬───────────────────┐         │
    │                    │Docs │ PDF    │ Chat | Inspect |  │         │
    │                    │List │ Viewer │ Explorer | Studio │         │
    │                    │     │ +BBox  │ (segmented tabs)  │         │
    │                    │     │ +Zoom  │                   │         │
    │                    └─────┴────────┴───────────────────┘         │
    └─────────────────────────────────────────────────────────────────┘
```

---

## The Ingestion Pipeline

Documents enter through the **Smart Router** which classifies by file type and routes to the appropriate parser.

### Helix A: Visual Documents (PDF)

The core differentiator. A two-phase VLM pipeline orchestrated by `pdf_parser.py`:

**Phase 1 — Glance (Layout Analysis).** GPT-4.1-mini performs object detection on each page, producing `ChartRegion` objects with bounding boxes (0-1000 integer grid), visual archetypes (Cartesian, Waterfall, Table, ValuationField, MarketMap), axis labels, legends, and annotation texts.

**Phase 2 — Read (Forensic Extraction).** GPT-4.1-mini receives the page image plus a "Scout Dossier" assembled from Phase 1 (OCR text sorted by position, layout hints, Column Manifests for waterfalls). It produces typed `MetricSeries` — each containing labeled `DataPoint` objects with currency, magnitude, measure, periodicity, and anti-hallucination `original_text` fields.

**Parallel: Docling.** IBM Docling v2 runs concurrently for table extraction, producing `FinancialTableItem` objects with HTML representation and row hierarchy (indentation depth for accounting bridges like Net Income to EBITDA).

**Post-Extraction Validators.** Deterministic Python code (not LLM) runs after extraction:
- Stacked Total Guard — prevents nonsensical sums across side-by-side charts
- Summary Consistency Check — verifies summary claims against extracted metrics
- Category Validation — ensures insight categories match the 5-way taxonomy
- Split-Brain Table Guard — skips VLM table extraction when Docling already succeeded

### Helix B: Native Documents (Excel, PPTX)

MarkItDown + OpenPyXL extract structured data. Playwright generates a "Shadow PDF" for visual citation on spreadsheet cells.

### Output: UnifiedDocument

Both helices produce a `UnifiedDocument` — a typed stream of discriminated union items:

| Item Type | Content | Downstream Use |
|-----------|---------|----------------|
| `HeaderItem` | Section titles with hierarchy level | Document structure, retrieval context |
| `NarrativeItem` | Qualitative text with sentiment and category | Semantic search, insight retrieval |
| `FinancialTableItem` | HTML table with row hierarchy and accounting basis | Table search, Docling-sourced structured data |
| `VisualItem` | Chart title, summary, and `MetricSeries[]` | Numeric search, analytics, citations |
| `ChartTableItem` | Hybrid chart-table artifacts (Football Fields) | Dual-mode retrieval |

---

## The Indexing Pipeline

Each `UnifiedDocument` is indexed into two stores simultaneously:

### Qdrant (Semantic Search)

The **Document Chunker** (`chunker.py`) flattens items into `IndexableChunk` objects with type-specific strategies:

- **VisualItem** — one summary chunk (`"{title}. {summary}. Series: {labels}"`) + one chunk per MetricSeries with full data representation
- **FinancialTableItem** — markdown-converted table summary
- **NarrativeItem** — raw text with category and sentiment metadata
- **HeaderItem** — header text with hierarchy level

Each chunk carries spatial metadata: the coarse region bbox and (when available) a `value_bboxes` dictionary mapping individual values and sentences to their fine-grained pixel locations. This metadata flows through to Qdrant as payload fields, enabling per-value citation highlighting at query time without any re-parsing.

Each chunk is embedded via **Cohere embed-v4** (1024 dims, multilingual, 100+ languages) and indexed with both dense vectors and **Qdrant native BM25** sparse vectors. Search uses **Reciprocal Rank Fusion (RRF)** to combine both signals.

### Postgres (Structured Analytics)

The **Analytics Driver** (`analytics_driver.py`) flattens every `DataPoint` from every `MetricSeries` into a `metric_facts` row:

```sql
series_label, numeric_value, currency, magnitude, measure,
resolved_value,  -- PRE-COMPUTED: numeric_value * magnitude_multiplier
period_date,     -- PRE-PARSED from label text
accounting_basis, data_provenance, periodicity, archetype
```

`resolved_value` is computed deterministically in Python at ingestion time (never by the LLM). This means a Text-to-SQL query like "revenue > $1B" translates to `WHERE resolved_value > 1000000000` — no magnitude CASE statements, no LLM math.

---

## The Retrieval Pipeline

### Query Router

An LLM classifier routes each query to the optimal retrieval strategy:

| Route | Strategy | Example |
|-------|----------|---------|
| **Qualitative** | Qdrant hybrid search only | "What are the key investment risks?" |
| **Quantitative** | Text-to-SQL over `metric_facts` + light vector search | "What was EBITDA in Q3 2024?" |
| **Hybrid** | Both paths, merged | "Why did revenue decline in Q3?" |

### Document-Scoped Queries

Users can scope queries to a single document via the doc-scope pill in the chat input. When active, a `doc_filter` (SHA-256 hash) is passed through the entire retrieval pipeline — both Qdrant hybrid search and Postgres Text-to-SQL respect the filter.

### Multilingual Query Expansion

The **Term Injector** solves the cross-lingual BM25 gap. German "Umsatz" won't match English "Revenue" in sparse search. Before search:
1. An LLM expands the query with multilingual synonyms and abbreviations
2. Known series labels from Postgres are fuzzy-matched and appended

The expanded query feeds the BM25 sparse leg. The original query feeds the dense embedding leg.

### Reranking

After RRF fusion, results pass through **Voyage Rerank 2.5** — a cross-encoder that re-scores each chunk against the original query for precision. Optional — the system works without it if no Voyage API key is configured.

---

## The Generation Pipeline

### Why Structured Output (Not Free-Text Markers)

Early iterations used free-text `[N]` markers — the LLM was instructed to write `[1]` inline. This failed in production for three reasons:

1. **Hallucinated markers.** The LLM would emit `[7]` when only 5 sources existed.
2. **Non-consecutive numbering.** `[1]`, `[3]`, `[5]` with gaps confused users.
3. **Non-deterministic formatting.** Sometimes `[1]`, sometimes `(1)`, sometimes superscripts.

The fix: **the LLM never writes `[N]` markers at all.** Instead it returns structured data, and Python controls all formatting.

### Citation Sidecar

Retrieved chunks are mapped to ephemeral integer IDs and formatted as XML source blocks for the generation LLM:

```xml
<source id="1" file="deal_memo.pdf p.12" type="visual">
Revenue grew 15% YoY driven by APAC expansion...
</source>
```

The sidecar maintains a `sidecar_map: Dict[int, RetrievedChunk]` that maps each ID back to its chunk with full metadata (page, bbox, `value_bboxes`, `item_type`, `doc_hash`).

### Structured Answer Generation

GPT-4.1 uses `with_structured_output(StructuredAnswer)` to return a Pydantic-validated response:

```python
class CitedSegment(BaseModel):
    text: str          # "Alphabet's free cash flow for Q4 2024 was $24,837 million."
    source_ids: list   # [3]  — integer ID from the <source> tags

class StructuredAnswer(BaseModel):
    segments: list     # List[CitedSegment]
```

Each segment is a sentence or short passage with explicit source IDs. The LLM's job is only to write the answer text and link it to sources — it never formats citation markers.

### Fact-Based Citation Assembly

`assemble_cited_answer()` transforms structured segments into the final answer with `[N]` badges:

**Step 1: Per-segment dedup.** Within each segment, all `source_ids` support the same factual claim. We keep only the first valid source. This eliminates redundant badges like `[1][2]` when both cite the same number from different pages.

**Step 2: Badge assignment.** Unique source IDs are numbered consecutively by first-appearance order. No global dedup across segments — different claims always get separate badges, even if their sources are on the same page.

**Step 3: Fact-scoped citing texts.** Each badge's "citing texts" are collected only from the segments where that badge appears. This is critical for bbox resolution: if badge `[1]` appears in "$24,837 million" and badge `[2]` appears in "FCF is defined as net cash less capital expenditures", each badge resolves its bbox from its own sentence context — not from a combined pool.

**Why per-fact, not per-page or per-item-type?** Earlier versions deduped by page number (all sources on page 10 → one badge). This collapsed distinct content: a table cell showing `$24,837` and a narrative paragraph defining FCF both lived on page 10 but served completely different citation purposes. The fact-based approach uses the LLM's own segmentation as the dedup boundary — the segment text IS the fact, so dedup within it removes true redundancy while preserving cross-fact diversity.

### Fine-Grained BBox Resolution

The standard approach to citation highlighting — draw a box around the entire table or chart region — is unacceptable in finance. When a user asks "what was Q4 free cash flow?" and clicks `[1]`, they expect to see the specific `$24,837` cell highlighted, not the entire 8-column reconciliation table.

Centaur solves this with a two-phase system:

**Phase 1 (Ingestion Time): Build `value_bboxes`.** During document parsing, each item computes a `Dict[str, List[List[float]]]` mapping values to their pixel locations:

| Item Type | Key Format | Source |
|-----------|-----------|--------|
| **FinancialTableItem** | Cell text (`"24,837"`, `"(13,186)"`) | Docling `table_cells[i].bbox` |
| **VisualItem** | Normalized float (`"8090.0"`) + series label (`"YouTube Ads"`) | OCR word bboxes matched against VLM-extracted MetricSeries |
| **NarrativeItem** | Sentence text (`"We define free cash flow as..."`) | PyMuPDF word bboxes grouped by `(block_no, line_no)` for per-line AABBs |

Each key maps to `List[List[float]]` (not a single bbox) to handle:
- **Duplicate values:** `"0"` appears in 5 table cells → 5 bboxes
- **Multi-line wrapping:** A sentence spanning 2 visual lines → 2 bbox entries

`value_bboxes` is stored as metadata on each chunk in Qdrant.

**Phase 2 (Query Time): Resolve from citing text.** When building a Citation, `_resolve_fine_bbox()` matches values from the answer text against the chunk's `value_bboxes` through a 6-tier cascade:

| Tier | Pattern | Example Match |
|------|---------|---------------|
| 1 | Dollar amounts | `$24,837` → try `"$24,837"`, `"24,837"`, `"24837"`, `"24837.0"` |
| 2 | Percentages | `25%` → try `"25%"`, `"25"` |
| 3 | Plain numbers | `24,837` → try raw, stripped, float form, parenthesized `(24,837)` |
| 4 | Series labels | Chunk metadata `series_label` → try as key |
| 5 | Sentence text | Word-overlap (Jaccard) matching against sentence-keyed entries (>35% threshold) |
| 6 | Coarse fallback | Entire table/chart region bbox from chunk metadata |

Tier 5 handles the narrative case. The LLM might write "FCF is defined as net cash provided by operating activities less capital expenditures" while the document says "We define free cash flow as net cash provided by operating activities less capital expenditures." Exact matching fails, but word overlap (Jaccard similarity) catches it — both sentences share most of their words despite different phrasing.

The tier ordering reflects a principle: **numbers are unique identifiers, text is ambiguous.** `$24,837` almost certainly refers to one specific cell. But a sentence like "Revenue grew in Q1" could match several narrative passages, so we only try text matching after numeric tiers fail.

---

## The Visual Cortex

Centaur's vision pipeline solves problems that standard RAG parsers cannot handle:

### Relative Luminance Resolution
LLMs fail when a chart uses "Brand Red" (`#EA4335`) vs "Dark Red" (`#A61C00`). They see both as "Red." Centaur converts absolute hex codes into **Relative Semantic Names** using HSL math, giving the VLM `Luminance-73 Red (Light)` vs `Luminance-40 Red (Dark)`.

### Hybrid OCR Safety Net
Financial decks contain "Zombie Charts" — Excel screenshots pasted as flat images. The system first probes the PDF's internal vector structure. If a region has fewer than 5 vector text items, it triggers **RapidOCR (ONNX)** automatically.

### Column Manifest Builder
For waterfalls and grouped bar charts, the system builds a spatial manifest of column positions using OCR X-coordinates, enabling precise data-to-label binding without hallucination.

### Anti-Hallucination Guards
- **Forced Chain of Thought:** The `audit_log` field makes the VLM analyze chart structure (units, legends, anomalies, math checks) before extracting any numbers
- **Soft-Fail Validators:** `BeforeValidator` hooks convert messy financial strings ("N/A", "($50.2)", "1,200") to clean floats or None — never crash the pipeline
- **Math Verification:** Waterfall sums, stacked totals, and pie percentages are checked in the audit log

---

## The Frontend

A four-panel Next.js 16 application built on **Zustand** state management with **Geist** typography:

```
┌──────────┬──────────────────────┬──────────────────────┐
│          │                      │ [Chat|Inspect|Explore│
│  CENTAUR │                      │  r|Studio]           │
│  ● Live  │                      │ ────────────────────│
│          │                      │                      │
│  doc1.pdf│    PDF Viewer        │  Chat Panel          │
│  doc2.pdf│    + Zoom HUD        │  or Chunk Inspector  │
│  doc3.pdf│    + BBox Overlay    │  or Metric Explorer  │
│          │    + Citation Hl.    │  or Prompt Studio    │
│          │                      │                      │
│          │                      │                      │
│ [Upload] │  [- 100% + | Fit]   │  [scope] [new topic] │
└──────────┴──────────────────────┴──────────────────────┘
  Sidebar       Center (Desk)         Right Panel Bay
  (Collapsible)                       (Resizable)
```

### Panel Descriptions

| Panel | Component | Function |
|-------|-----------|----------|
| **Left Sidebar** | `DocumentList` | CENTAUR wordmark with status dot, document list with upload + drag-and-drop, SSE ingestion progress, collapsible (`PanelLeftClose`) |
| **Center** | `DocumentViewer` | react-pdf renderer with floating zoom HUD (Ctrl+scroll), active citation bbox overlay, toolbar with page nav + sidebar toggle. Shows `WelcomeDropzone` when no document selected. |
| **Right** | `RightPanel` | Segmented tab control (sliding indicator) hosting 4 sub-panels |

### Right Panel Tabs

| Tab | Component | Function |
|-----|-----------|----------|
| **Chat** | `ChatPanel` | Multi-turn conversational interface with SSE streaming (ghost bubble stepper), clickable `[N]` citation badges, doc-scope pill, "New topic" divider |
| **Inspect** | `ChunkInspectorPanel` | Per-page chunk browser with type badges (visual/narrative/table), metadata inspector, value_bboxes count |
| **Explorer** | `MetricExplorerPanel` | Structured metric ledger — page-grouped or flat table view, sortable columns, series parser with sparkline previews, TSV export |
| **Studio** | `StudioPanel` + `WorkflowBuilder` | Prompt library with Jinja2 editor, version history, test run with retrieval caching, workflow chain builder with HITL approval |

### State Management

Five Zustand stores replace the original `useState` variables, each subscribed by specific slices to prevent waterfall re-renders:

| Store | State | Subscribers |
|-------|-------|-------------|
| `useDocStore` | `documents[]`, `selectedDocHash`, `selectedFilename` | DocumentList, ChatPanel, page.tsx |
| `useViewerStore` | `currentPage`, `numPages`, `zoomScale`, `renderedSize`, `sidebarCollapsed` | DocumentViewer, ChatPanel (citation nav) |
| `useChatStore` | `messages[]`, `isThinking`, `docScope`, `citations`, `activeCitationIndex` | ChatPanel, BboxOverlay |
| `useInspectStore` | `inspectMode`, `inspectChunks`, `activeChunkId`, `docStats`, `activePanel` | ChunkInspectorPanel, MetricExplorerPanel |
| `useStudioStore` | `prompts[]`, `workflows[]`, `activePromptId`, `activeWorkflowId`, editor state | StudioPanel, WorkflowBuilder |

### Streaming & Multi-Turn

Chat uses **Server-Sent Events** (SSE) with a ghost bubble showing terminal-style progress:
```
[✓] Routed via hybrid
[↻] Retrieving relevant chunks
[ ] Generating answer
```

The last 6 messages are sent as conversation history for multi-turn context. A "New topic" button clears the conversation with a visual divider.

### Citation Interaction Flow

1. User asks a question. The backend streams the answer via SSE with structured citation data.
2. `ChatPanel` parses `[N]` markers and renders them as indigo clickable badges.
3. User clicks badge `[1]`. The frontend reads `citation.doc_hash` to auto-select the document (if not already open), navigates `DocumentViewer` to the cited page, and renders a `BboxOverlay` highlight on the specific value or sentence.
4. Only the **active citation** is highlighted — no visual clutter from other citations on the same page.

### Design Language

"Machined Precision" — the DOM treated like milled aluminum:
- **Z-axis depth hierarchy**: Sidebars cast `box-shadow` onto a recessed center desk (`--bg-desk: #050505`)
- **Geist typography**: `GeistSans` for UI text, `GeistMono` for data and labels
- **Subtractive interaction**: Buttons press down on `:active` (not glow up on hover)
- **One-shot boot splash**: CENTAUR wordmark deblurs on first session load (sessionStorage-gated)
- **Error boundary**: React class component wrapping DocumentViewer prevents blank-screen crashes

---

## Prompt Studio

An integrated prompt engineering and workflow orchestration environment, decoupled from the chat router.

### Saved Prompts

- **Jinja2 templates** with `{{ variable }}` detection and a test-run sandbox
- **Version history** with publish/draft lifecycle (only published versions appear in workflow step pickers)
- **Retrieval caching**: First run retrieves from Qdrant and caches context for 5 minutes. Subsequent runs (prompt refinement) skip retrieval and regenerate from cached context. Manual "Re-retrieve" button available.

### Workflow Chains

- **Sequential executor** with Postgres checkpointing after each step
- **HITL (Human-in-the-Loop)**: Step conditions can render to `"pause"` → workflow status becomes `paused_for_approval` → `/approve` endpoint resumes
- **Input mapping**: Steps read from `{{ inputs.* }}` (user-provided) and `{{ steps.<step_name>.text }}` (output of prior steps)
- **Step retry**: Configurable `retry_count` per step with auto-retry on failure
- **Skip conditions**: Step condition renders to `"skip"` or `"false"` → step skipped

### Four Execution Modes

| Mode | Retrieval | Output | Use Case |
|------|-----------|--------|----------|
| `rag` | Qdrant hybrid search | Free text | Standard Q&A |
| `structured` | Qdrant hybrid search | Pydantic JSON schema | Data extraction |
| `direct` | None | Free text | Synthesis, reformatting |
| `sql` | Text-to-SQL | JSON rows | Quantitative queries |

---

## Audit Engine

A first-class data quality layer that runs deterministic, code-only checks against the structured `metric_facts` table after ingestion. Zero LLM calls — pure SQL validators that surface extraction errors, VLM hallucinations, and cross-page inconsistencies.

### The 6 Validators

| Check | What It Does | Severity |
|-------|-------------|----------|
| **StackedTotalCheck** | For stacked bar charts, sums non-total series per label and compares to the stated "Total" series. Flags if mismatch > 2%. | Error |
| **CrossPageConsistencyCheck** | Same metric + same period appearing on multiple pages with different values. Uses `GROUP BY series_label, label HAVING COUNT(DISTINCT resolved_value) > 1`. | Warning |
| **YoYReasonabilityCheck** | Computes YoY growth between consecutive periods. Flags if growth > 300% or decline > 80%. Uses `LAG()` window function. | Warning |
| **CurrencyMagnitudeCheck** | Detects mixed currencies (USD + EUR) or mixed magnitudes (M + B) within a single series — usually a VLM extraction error. | Error |
| **NarrativeMetricCheck** | Extracts dollar amounts from narrative text via regex, cross-references against `metric_facts` `resolved_value`. Flags if narrative claims "$45M" but structured data shows $42.5M. 5% tolerance. | Warning |
| **PercentVerificationCheck** | For percentage series, verifies against base values when identifiable (e.g., Gross Margin = 45% with Revenue and COGS present). | Info |

### Architecture

```
Ingestion Pipeline → metric_facts insertion → AuditEngine.run_all(doc_hash)
  ├─ 6 validators (pure SQL, no LLM)
  ├─ Results → audit_findings table (Postgres, Alembic-managed)
  └─ API: GET /documents/{hash}/audit, POST /documents/{hash}/audit/run
```

### Frontend Integration

- **Document list badge**: Red/amber/green severity dot next to each document
- **Audit section in Inspect panel**: Filter chip alongside Visual/Narrative/Header; finding cards with severity badge, title, expandable detail, click-to-navigate

Performance: <500ms per document (50-200 metric_facts rows → queries complete in <50ms each).

---

## Traceable Excel Export

One-click export of all extracted metrics as a formatted `.xlsx` workbook where every value cell hyperlinks back to the source page in Centaur.

### Workbook Structure

| Sheet | Content |
|-------|---------|
| **Summary** | Document metadata, filename, page count, ingestion date, audit findings count |
| **Metrics** | Pivot-ready fact table: Page, Chart Title, Series, Period, Value, Currency, Magnitude, Resolved Value, Measure, Accounting Basis, Periodicity, Source (hyperlink) |
| **Series Index** | Unique series with metadata: label, archetype, periodicity, data point count |

### Hyperlink Traceability

Every value in the Metrics sheet contains an `=HYPERLINK()` formula:
```
=HYPERLINK("http://localhost:3000?doc={hash}&page={N}", "Page {N}")
```

Clicking a hyperlink opens Centaur, auto-selects the document, and navigates to the exact page. The base URL is configurable via `CENTAUR_BASE_URL` environment variable.

### Formatting

- Frozen header row with auto-filter (pivot-ready)
- `resolved_value` in accounting number format (`#,##0.00`)
- Periods sorted by `period_date` (temporal, not alphabetical)
- Conditional red fill on cells with associated audit findings
- Built with `openpyxl` for fine-grained control over formatting and hyperlinks

### API

```
GET /api/v1/documents/{doc_hash}/export/excel → StreamingResponse (.xlsx)
GET /api/v1/export/excel?doc_hashes=hash1,hash2 → Multi-doc workbook
```

---

## Natural Language Data Visualization

User types "Chart revenue by quarter" in chat → an interactive Vega-Lite chart renders inline in the chat stream, using structured data from the `metric_facts` SQL table.

### Architecture

```
User: "Chart EBITDA margin for 2020-2024"
  ↓
LangGraph Router → route = "visualization"
  ↓
visualize node (terminal — skips generate_answer)
  ├─ Step 1: Text-to-SQL → query metric_facts
  ├─ Step 2: Execute SQL (existing safety guards)
  ├─ Step 3: LLM generates Vega-Lite v5 spec from data + user intent
  └─ Return: {spec, data, sql, title}
  ↓
SSE event: type="viz" → Frontend renders with vega-embed
```

### Why Vega-Lite

- **LLM-friendly**: Single declarative JSON spec, well-documented, LLMs trained on many examples
- **Safe**: No imperative code generation — just a data visualization grammar
- **Feature-rich**: Temporal axes, faceting, aggregation, tooltips, dark theme support
- **Lightweight**: `vega-embed` is ~200KB gzipped, renders to Canvas/SVG

### Data Guards

- **500-row limit**: If SQL returns >500 rows, auto-aggregate before sending to frontend
- **Spec validation**: Vega-Lite JSON parsed server-side before SSE emission
- **Fallback**: If spec generation fails, raw SQL results returned as a table
- **Dark theme**: Background `#111111`, text `#ededed`, grid `#333` — matches Centaur's material design

### Cross-Document Visualization

Since `metric_facts` contains `doc_hash` and `source_file`, cross-document queries work automatically:
```sql
SELECT source_file, label, resolved_value
FROM metric_facts WHERE series_label ILIKE '%revenue%'
ORDER BY period_date
```

The router skips `doc_filter` when the query spans multiple documents.

---

## Security

### Hardening (Applied)

| Layer | Threat | Mitigation |
|-------|--------|------------|
| **Template Injection** | Jinja2 SSTI via user-authored prompt templates | `SandboxedEnvironment` blocks `__class__`, `__mro__`, etc. |
| **SQL Injection** | LLM-generated SQL with embedded DML | 3-layer defense: semicolon guard, SELECT-only guard, `SET TRANSACTION READ ONLY` |
| **Path Traversal** | Crafted filenames like `../../etc/passwd` | `PurePosixPath(name).name` + `Path(name).name` strips all path components |
| **Filename Race** | Concurrent uploads with identical filenames | UUID-prefixed unique filenames (`{uuid4()[:8]}_{safe_name}`) |
| **Upload Size Bypass** | Missing `Content-Length` header skips size check | Post-save `stat()` verification on disk |
| **Input Validation** | Oversized queries, malformed doc hashes | `max_length` on all string fields, regex patterns on `doc_filter` and `role` |
| **Error Leaking** | Stack traces in HTTP responses | Generic error messages to clients; `exc_info=True` to server logs only |
| **Unbounded Cache** | Module-level dict grows without limit | 256-entry max with LRU eviction |
| **Auto-Reload** | `reload=True` in production exposes filesystem watcher | Conditional on `CENTAUR_DEV` environment variable |

### Not Yet Implemented

- **Authentication**: No auth middleware — all endpoints are open. Needs product decision (API keys vs OAuth2).
- **User Isolation**: No tenant partitioning in Qdrant or Postgres. All documents share a single namespace.
- **Concurrency Locks**: Module-level mutable state (`_retrieval_cache`, `_trackers`) lacks `asyncio.Lock` — low risk under CPython GIL but technically unsafe for check-and-set patterns.

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **VLM** | GPT-4.1-mini | Layout analysis + visual extraction (both phases) |
| **Generation** | GPT-4.1 | Answer synthesis with citations |
| **Embedding** | Cohere embed-v4 | Dense vectors (1024 dims, multilingual) |
| **Sparse Search** | Qdrant native BM25 | Exact term matching with IDF weighting |
| **Reranking** | Voyage Rerank 2.5 | Cross-encoder precision layer (optional) |
| **Vector DB** | Qdrant | Hybrid search with RRF fusion |
| **SQL DB** | PostgreSQL 15 | Document ledger + `metric_facts` analytics + Studio tables |
| **Migrations** | Alembic | Schema versioning for Studio tables |
| **Table Extraction** | IBM Docling v2 | HTML table extraction with row hierarchy |
| **OCR** | RapidOCR (ONNX) | Fallback for rasterized chart images |
| **Orchestration** | LangGraph | Stateful query routing and generation workflow |
| **API** | FastAPI | REST + SSE streaming endpoints |
| **Frontend** | Next.js 16 + Tailwind + Zustand | Four-panel document intelligence UI |
| **Typography** | Geist Sans + Geist Mono | Financial-grade crisp rendering |
| **Observability** | LangSmith | Full trace visibility on every pipeline step |

---

## Repository Structure

```
centaur/
├── src/
│   ├── config.py                          # Central config (models, keys, paths)
│   │
│   ├── ingestion/                         # WRITE PATH
│   │   ├── pipeline.py                    # Orchestrator (routing -> parsing -> indexing)
│   │   ├── router.py                      # File type classification
│   │   ├── pdf_parser.py                  # Two-phase VLM pipeline (Glance + Read)
│   │   ├── native_parser.py               # Excel/PPTX extraction
│   │   ├── chunker.py                     # UnifiedDocument -> IndexableChunks + value_bboxes
│   │   ├── filters.py                     # Token firewall (noise stripping)
│   │   └── phantom.py                     # Shadow PDF generation (Playwright)
│   │
│   ├── audit/                             # DATA QUALITY
│   │   └── engine.py                      # 6 deterministic validators + AuditEngine orchestrator
│   │
│   ├── export/                            # EXPORT PIPELINE
│   │   └── excel_builder.py               # openpyxl workbook builder with HYPERLINK traceability
│   │
│   ├── retrieval/                         # READ PATH
│   │   ├── qdrant.py                      # Full pipeline (expand -> search -> rerank)
│   │   ├── term_injector.py               # Multilingual query expansion + label matching
│   │   └── sidecar.py                     # Citation sidecar (build context, assemble, fine-grained bbox)
│   │
│   ├── storage/                           # DATA LAYER
│   │   ├── vector_driver.py               # Qdrant hybrid index (Cohere + BM25 + RRF)
│   │   ├── analytics_driver.py            # Postgres metric_facts (structured analytics)
│   │   ├── db_driver.py                   # Postgres document ledger
│   │   ├── blob_driver.py                 # Local artifact store
│   │   ├── studio_models.py               # ORM models for Prompt Studio (6 tables)
│   │   └── studio_driver.py               # Full CRUD for prompts, workflows, runs
│   │
│   ├── schemas/                           # TYPE CONTRACTS
│   │   ├── deal_stream.py                 # UnifiedDocument, DocItem union
│   │   ├── vision_output.py               # MetricSeries, DataPoint, Insight
│   │   ├── layout_output.py               # ChartRegion, PageLayout, VisualArchetype
│   │   ├── retrieval.py                   # RetrievedChunk, RetrievalResult
│   │   ├── citation.py                    # Citation, BoundingBox (normalized 0-1)
│   │   ├── enums.py                       # Domain taxonomy (Category, Periodicity, etc.)
│   │   ├── state.py                       # LangGraph AgentState
│   │   └── documents.py                   # Ingestion result types
│   │
│   ├── workflows/                         # LANGGRAPH BRAIN
│   │   ├── graph.py                       # Graph assembly (route -> retrieve -> generate)
│   │   ├── router.py                      # Query classifier (qual/quant/hybrid + locale)
│   │   ├── prompt_executor.py             # Decoupled Studio execution (4 modes, Jinja2 sandbox)
│   │   ├── workflow_executor.py           # Sequential workflow runner with HITL + checkpointing
│   │   └── nodes/
│   │       ├── retrieve.py                # Retrieval nodes (qualitative, quantitative, hybrid)
│   │       ├── financial_math.py          # Text-to-SQL over metric_facts
│   │       ├── generate.py                # Structured output generation + citation assembly
│   │       ├── visualize.py               # NL → Text-to-SQL → Vega-Lite spec generation
│   │       └── legal_reasoning.py         # Legal analysis node
│   │
│   ├── api/                               # FASTAPI SERVER
│   │   ├── main.py                        # App setup, CORS, startup initialization
│   │   ├── schemas.py                     # Request/response models (validated)
│   │   └── routes/
│   │       ├── chat.py                    # POST /chat, POST /chat/stream (SSE)
│   │       ├── ingestion.py               # POST /ingest, GET /documents
│   │       ├── documents.py               # PDF serving, page rendering, chunk inspection
│   │       ├── audit.py                   # Audit findings CRUD + re-run trigger
│   │       ├── export.py                  # Excel export (single-doc + multi-doc)
│   │       ├── prompts.py                 # Prompt CRUD + publish + test run (9 endpoints)
│   │       └── workflows.py               # Workflow CRUD + steps + run + approve (14 endpoints)
│   │
│   └── tools/                             # VLM TOOLS
│       ├── layout_analyzer.py             # Phase 1: Object detection (Glance)
│       ├── visual_extractor.py            # Phase 2: Forensic extraction (Read)
│       ├── calculator.py                  # Python math engine
│       ├── database.py                    # SQL lookup tool
│       └── vision.py                      # GPT-4o visual analysis
│
├── frontend/                              # NEXT.JS 16 UI
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                 # Root layout (Geist fonts, BootSplash)
│       │   ├── page.tsx                   # Four-panel app shell + Zustand subscriptions
│       │   └── globals.css                # Material depth theme, animations, typography scale
│       ├── components/
│       │   ├── DocumentViewer.tsx          # react-pdf + zoom HUD + Ctrl+scroll + sidebar toggle
│       │   ├── DocumentList.tsx            # Sidebar: CENTAUR wordmark, upload, SSE status
│       │   ├── ChatPanel.tsx              # SSE streaming, ghost bubble, citations, multi-turn
│       │   ├── ChunkInspectorPanel.tsx    # Per-page chunk browser with type badges
│       │   ├── MetricExplorerPanel.tsx    # Structured metric ledger with sparklines
│       │   ├── StudioPanel.tsx            # Prompt editor, version history, test sandbox
│       │   ├── WorkflowBuilder.tsx        # Workflow chain builder with step stepper
│       │   ├── RightPanel.tsx             # Segmented tab control + resize handle
│       │   ├── WelcomeDropzone.tsx        # Center empty state with drag-and-drop upload
│       │   ├── BootSplash.tsx             # One-shot wordmark animation (sessionStorage-gated)
│       │   ├── ChatVizBlock.tsx            # Vega-Lite chart renderer (vega-embed)
│       │   ├── BboxOverlay.tsx            # Pixel-precise citation highlight
│       │   ├── MiniSparkline.tsx          # Inline SVG sparkline for metric trends
│       │   └── ErrorBoundary.tsx          # React class error boundary for crash recovery
│       ├── stores/
│       │   ├── useDocStore.ts             # Document selection state
│       │   ├── useViewerStore.ts          # PDF viewer state (persisted sidebar)
│       │   ├── useChatStore.ts            # Chat messages, streaming, citations
│       │   ├── useInspectStore.ts         # Chunk inspection + panel routing
│       │   └── useStudioStore.ts          # Prompt Studio state
│       ├── hooks/
│       │   └── useKeyboardShortcuts.ts    # Global keyboard shortcuts (1-4 for tabs)
│       └── lib/
│           ├── api.ts                     # Full API client (REST + SSE streaming)
│           ├── tsvExport.ts               # Metric Explorer → clipboard/file export
│           └── seriesParser.ts            # MetricSeries text → structured data parser
│
├── alembic/                               # DATABASE MIGRATIONS
│   ├── env.py
│   └── versions/
│       └── 6e2e4d9c0fd6_create_studio_tables.py
│
├── data/                                  # DATA LAKE (gitignored)
│   ├── inputs/                            # Raw document drop zone
│   ├── blobs/                             # Extracted artifacts
│   ├── shadow_cache/                      # Rendered shadow PDFs
│   ├── qdrant_storage/                    # Qdrant data volume
│   └── postgres_data/                     # Postgres data volume
│
├── docker-compose.yml                     # Qdrant + Postgres
├── requirements.txt                       # Python dependencies
├── run_ingestion.py                       # Manual ingestion entry point
└── .env                                   # API keys and config
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 24+ (LTS)
- Docker Desktop

### 1. Start Infrastructure

```bash
docker-compose up -d
```

This starts Qdrant (port 6333) and Postgres (port 5432).

### 2. Configure Environment

Add your API keys to `.env`:

```env
OPENAI_API_KEY=sk-...          # Required — GPT-4.1 for VLM + generation
COHERE_API_KEY=...             # Required — embed-v4 for vector search
VOYAGE_API_KEY=...             # Optional — rerank-2.5 for precision
POSTGRES_PASSWORD=...          # Recommended — defaults to "password" with a warning
CENTAUR_DEV=true               # Optional — enables uvicorn auto-reload
```

Where to get keys:
- **OpenAI:** [platform.openai.com](https://platform.openai.com)
- **Cohere:** [dashboard.cohere.com](https://dashboard.cohere.com) — free tier: 1,000 calls/month, no credit card
- **Voyage AI:** [dash.voyageai.com](https://dash.voyageai.com) — free tier: 200M tokens

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

```bash
cd frontend
npm install
```

### 4. Run Database Migrations

```bash
alembic upgrade head
```

### 5. Run

Terminal 1 — Backend:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 — Frontend:
```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 6. Ingest a Document

Upload a PDF through the UI (drag-and-drop on the center dropzone, or click to browse), or run manually:

```bash
python run_ingestion.py
```

---

## Domain Taxonomy

All extracted insights are classified into five categories:

| Category | Definition | Example |
|----------|-----------|---------|
| **Financial** | Ledger facts — P&L, BS, CF line items. Includes bridge/variance drivers. | "EBIT fell due to volume/pricing headwinds" |
| **Operational** | Non-ledger business KPIs | "Headcount grew 12% YoY" |
| **Market** | Industry structure, TAM, competitive dynamics | "European market CAGR of 4.2%" |
| **Strategic** | Corporate decisions by management or shareholders | "Acquired XYZ in 2023 for platform expansion" |
| **Transactional** | Deal terms, financing, covenants, returns | "Senior secured at 5.5x leverage" |

---

## Engineering Standards

1. **No Mental Math.** The LLM never performs arithmetic. All calculations use `resolved_value` (pre-computed in Python) or the calculator tool.

2. **Mandatory Citations.** Every generated factual claim must carry a `[N]` marker that resolves to a verifiable source location.

3. **Token Firewall.** Headers, footers, and decorative artifacts are stripped before embedding to prevent noise in the vector index.

4. **Async Purity.** No blocking operations on the main event loop. CPU-bound tasks (OCR, image processing) are offloaded to thread pools.

5. **Separation of Concerns.** The VLM captures raw visual data. Python handles all transformations, normalization, and calculations. This minimizes hallucination and maximizes auditability.

6. **Sandboxed Execution.** User-authored Jinja2 templates run in `SandboxedEnvironment`. LLM-generated SQL runs in `READ ONLY` transactions with multi-statement guards.

7. **Defense in Depth.** Security-sensitive paths (SQL execution, file upload, template rendering) have multiple independent guards — if one fails, the next catches it.

---

## Observability

Every pipeline step is traced via **LangSmith** under the `Chiron_Adv_RAG` project:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=Chiron_Adv_RAG
```

Traced functions: layout analysis, visual extraction, document chunking, embedding, hybrid search, query expansion, reranking, Text-to-SQL, answer generation, citation verification, prompt execution, workflow runs.

---

## Data Privacy and Azure Deployment

Centaur is designed for sensitive financial data. For production with real deal data:

- **Cohere embed-v4** is available on [Azure AI Foundry](https://docs.cohere.com/docs/cohere-on-azure/cohere-on-azure-ai-foundry) — runs inside your Azure tenant
- **Voyage Rerank 2.5** is available on [Azure Marketplace](https://docs.voyageai.com/docs/azure-marketplace-mongodb-voyage) — data never leaves your VNet
- **OpenAI** is available as Azure OpenAI Service with the same data sovereignty guarantees
- **Qdrant** and **Postgres** run locally — no data leaves your machine

Switching from public APIs to Azure-hosted instances is a config change, not an architectural change.

---

## License

Proprietary. Internal use only.
