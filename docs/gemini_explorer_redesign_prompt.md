# Centaur — Metric Explorer Redesign & Visual Polish

## Your Role
You are a senior product designer and frontend architect reviewing a financial document intelligence platform called **Centaur**. The platform ingests PE/M&A/Capital Markets PDFs (earnings slides, CIMs, market studies), extracts structured metrics via a VLM pipeline, and lets analysts query the data via a RAG chat interface.

The current "Metric Explorer" tab and overall visual design need significant improvement. I want you to brainstorm and propose a redesign that would make this feel like a $10k/seat/year institutional finance product — not a developer tool.

---

## Current Architecture (What Exists Today)

### App Layout
Three-panel layout:
- **Left sidebar** (w-64): Document list with upload, status badges
- **Center** (flex-1): PDF viewer with react-pdf, zoom HUD, bbox overlays for citations
- **Right panel** (w-96): Three tabs — Chat, Inspect, Explorer

### Dark Theme
- Background: `#0a0a0a` (primary), `#141414` (secondary), `#1e1e1e` (tertiary)
- Text: `#ededed` (primary), `#999` (secondary)
- Accent: `#3b82f6` (blue)
- PDF canvas has `brightness(0.96)` filter, `box-shadow`, `border: 1px solid #444`
- All in `globals.css`, using CSS custom properties

### Tech Stack
- **Frontend**: Next.js 15, React, Tailwind CSS, Zustand state management, react-pdf, lucide-react icons
- **Backend**: FastAPI (Python), Qdrant vector DB, Cohere embeddings, LangGraph RAG pipeline
- **No component library** — all components are hand-rolled

---

## The Metric Explorer Problem

### What It Shows
The Explorer tab fetches all series chunks from Qdrant (`item_type=visual&chunk_role=series`) and displays them in a flat HTML `<table>` with columns: Series | Period | Basis | Pg.

### What's Wrong (Screenshot Analysis)
1. **47 rows, massive duplication**: "Operating Income" appears 7 times, "Operating Margin" 6 times. Each VLM extraction pass creates a separate VisualItem, and the side-by-side chart layout (Revenue chart + OpIncome chart) causes the same series to appear in multiple items.

2. **No actual values displayed**: The chunk_text contains full data (`Q1'24=USD 46156.0M, Q1'25=USD 50702.0M`) but the table only shows the series_label. The most valuable information is hidden.

3. **Dead columns**: "Basis" is all `—` (accounting_basis is null for this doc). "Period" is all `Q` (periodicity=Q). Two columns contributing zero information.

4. **No grouping or hierarchy**: Series from different charts on different pages are alphabetically interleaved. "Capital Expenditures" (p.8) sits next to "Cost of Revenues" (p.5).

5. **No chart context**: Which chart does "Operating Income p.4" belong to? There are 4 different VisualItems for "Alphabet Revenues and Operating Income" on page 4 alone (duplicate extraction).

6. **Looks like a debug inspector**: Plain table, no visual richness, no data density, no sparklines or change indicators.

### Actual Data Shape (from Qdrant)
Each series chunk has:
```
chunk_text: "Series: Capital Expenditures. Data: Q1'24=USD 12012.0M, Q2'24=USD 13186.0M, Q3'24=USD 13061.0M, Q4'24=USD 14276.0M, Q1'25=USD 17197.0M"
metadata: {
  series_label: "Capital Expenditures",
  series_nature: "level",          // "level" (absolute) or "delta" (change)
  periodicity: "Q",                // FY, Q, H, M, LTM, etc.
  accounting_basis: null,           // "Adjusted", "GAAP", "Pro Forma", etc.
  data_provenance: null,            // "Third-Party Research; Forecast", etc.
  archetype: "Cartesian",          // Chart type
  source_region_id: 1,             // Which chart region on the page
}
item_id: "ae58600e-..."            // Groups series from the same chart
page_number: 8
```

The `chunk_text` is parseable — data points follow the format `label=CURRENCY value.MAGNITUDE` (e.g., `Q1'24=USD 12012.0M`).

Summary chunks also exist (one per chart) containing the **chart title** and VLM-generated description:
```
chunk_text: "Google Services Revenues and Operating Income. Summary text here..."
chunk_role: "summary"
item_id: "a54fa231-..."    // Same item_id as its series children
```

### Grouping Structure
For the Alphabet Q1 2025 Earnings doc:
- **17 unique item_ids** (charts), **47 series chunks** total
- Many are **duplicates** — the VLM creates multiple VisualItems per page when charts are side-by-side
- Dedup by `(series_label, page_number)` would collapse 47 → ~20 unique series
- Alternative: dedup by `(series_label, page_number, source_region_id)` for finer control
- Grouping by page gives natural slide-by-slide organization:
  - p.4: Alphabet Revenues and Operating Income (Revenue, OpIncome, OpMargin)
  - p.5: Alphabet Cost of Revenues and Operating Expenses (Cost of Rev, R&D, G&A, S&M, OpEx)
  - p.6: Google Services Revenues and Operating Income (Search, YouTube, Network, OpIncome, OpMargin)
  - p.7: Google Cloud Revenues and Operating Income (Revenue, OpIncome)
  - p.8: Alphabet Capital Expenditures (CapEx)

---

## Design Constraints

1. **Right panel width is w-96 (384px)** — data-dense layouts needed. Cannot expand infinitely.
2. **Must work for 5-200 series** — A multi-segment PE CIM might have 200 metric series across 50 pages. The Alphabet doc is a small example.
3. **Click interaction exists**: Clicking a row navigates to the page in the PDF viewer and highlights the chart region. This cross-linking is the core value.
4. **No new backend endpoints** — work with the existing `/chunks` response. Parsing `chunk_text` client-side is fine.
5. **Dark theme only** — no light mode needed.
6. **Target users**: Buy-side analysts, PE associates, credit analysts. They live in Excel and Bloomberg Terminal. They value data density, keyboard navigation, and zero visual noise.

---

## What I Want You To Brainstorm

### A. Metric Explorer Redesign

1. **Information Architecture**: How should 47 (or 200) series chunks be organized? Consider:
   - Grouped by page/chart (collapsible sections with chart titles)
   - Grouped by series_label (deduped, showing all pages where a metric appears)
   - Hierarchical tree (doc → page → chart → series)
   - Searchable flat list with smart dedup

2. **Data Display**: How to show actual values in 384px width?
   - Inline sparklines (SVG, 60px wide)
   - Latest value + YoY/QoQ change pill
   - Mini data table (collapsible per series)
   - Compact "ticker tape" format: `CapEx: 12.0→13.2→13.1→14.3→17.2 (+43% YoY)`

3. **Deduplication**: How to handle the duplicate VisualItems?
   - Client-side dedup by `(series_label, page_number)`, keep chunk with most data points
   - Show a "2 sources" badge for deduped items
   - Or don't dedup — group by chart title and let the user see both extractions

4. **Filtering & Search**:
   - Text search across series labels (instant filter)
   - Smart filters that hide when empty (don't show "Basis: All" if no chunks have basis)
   - Category grouping (Revenue, Expenses, Income, Margins, CapEx) — derived from labels?

5. **Cross-Navigation**: Clicking a series should:
   - Navigate PDF to that page
   - Highlight the specific chart region (not the whole page)
   - Show the series data points in context
   - What about a persistent selection that survives tab switches?

### B. Overall App Visual Polish

The app currently looks like a well-styled developer tool. To justify $10k/seat, it needs to feel like Bloomberg Terminal meets Figma — authoritative, data-dense, premium.

Consider:
1. **Typography hierarchy**: The current app uses one font weight (400/600) and two sizes (text-sm/text-xs). A premium finance tool needs:
   - Monospaced numbers for alignment (tabular nums)
   - Clear weight hierarchy (300/400/500/600)
   - Possibly a serif or distinctive heading font

2. **Color system**: The current palette is generic dark mode. How to make it feel more "institutional"?
   - Goldman-style: warm grays + gold accent?
   - Bloomberg-style: high-contrast + orange/blue?
   - Modern fintech: cool grays + emerald/teal?

3. **Data density**: Finance users want MORE information visible, not less. White space is wasted real estate. Consider:
   - Tighter spacing
   - More visible at once
   - Information scent (subtle badges, counts, indicators)

4. **Micro-interactions**: What subtle animations/transitions signal quality?
   - Smooth page transitions in PDF viewer
   - Skeleton loading states
   - Hover states that reveal additional data
   - Keyboard shortcut hints in tooltips

5. **PDF Viewer**: The center panel is the hero. How to make it feel more premium?
   - Toolbar design (current is minimal but flat)
   - Page transition animations
   - Thumbnail strip (mini page nav)?
   - Split view for comparing pages?

6. **Document sidebar**: Currently a flat file list. For a $10k product:
   - Status indicators (ingestion progress)
   - Document metadata (pages, series count, date)
   - Grouping by upload date or project/deal

7. **Chat panel**: Currently functional but basic. Consider:
   - Markdown rendering in responses
   - Citation badges with better visual treatment
   - Typing indicators
   - Suggested follow-up questions

---

## Competitive Reference Points

Think about what these products look like and feel like:
- **Bloomberg Terminal**: Maximum data density, keyboard-first, orange-on-black
- **FactSet**: Clean professional, blue accent, chart-heavy
- **Bain RADAR / McKinsey Vantage**: Modern SaaS fintech, white-space-balanced
- **Notion / Linear**: Modern product design, but for productivity not finance
- **Datadog / Grafana**: Dense dashboarding with dark themes done well

---

## Deliverables Requested

1. **Metric Explorer redesign**: Propose 2-3 concrete layout options with ASCII mockups showing how the 384px-wide panel would look with real Alphabet data (Capital Expenditures, Google Search Revenue, Operating Margin, etc.). Include the data values.

2. **Visual design direction**: Propose a cohesive visual direction for the whole app — color palette, typography, spacing, and key component treatments. Show before/after for 2-3 components.

3. **Priority ranking**: Which changes would have the highest impact-to-effort ratio? What should be done first?

4. **Anti-patterns to avoid**: What would make this look cheap or amateurish? What screams "developer built this"?

---

## Real Data for Mockups

Use these actual extracted series from the Alphabet Q1 2025 doc:

**Page 4 — Alphabet Revenues and Operating Income:**
- Revenue: Q1'24=$80,539M, Q1'25=$90,234M
- Operating Income: Q1'24=$25,472M, Q1'25=$30,601M
- Operating Margin: Q1'24=31.6%, Q1'25=33.9%

**Page 5 — Alphabet Cost of Revenues and Operating Expenses:**
- Cost of Revenues: Q1'24=$33,712M, Q1'25=$36,361M
- R&D: Q1'24=$11,895M, Q1'25=$13,353M
- General & Administrative: Q1'24=$3,026M, Q1'25=$3,539M
- Sales & Marketing: Q1'24=$6,434M, Q1'25=$6,380M

**Page 6 — Google Services Revenues and Operating Income:**
- Google Search & Other: Q1'24=$46,156M, Q1'25=$50,702M
- YouTube Ads: Q1'24=$8,090M, Q1'25=$8,927M
- Google Network: Q1'24=$7,413M, Q1'25=$7,256M
- Operating Income: Q1'24=$27,897M, Q1'25=$32,682M
- Operating Margin: Q1'24=39.6%, Q1'25=42.3%

**Page 8 — Alphabet Capital Expenditures:**
- CapEx: Q1'24=$12,012M, Q2'24=$13,186M, Q3'24=$13,061M, Q4'24=$14,276M, Q1'25=$17,197M
