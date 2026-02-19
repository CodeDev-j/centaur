# CENTAUR

**Financial Document Intelligence Platform**

Centaur is a full-stack RAG engine purpose-built for high-finance document analysis. It ingests complex financial documents (CIMs, earnings slides, lender presentations, financial models), extracts structured data from charts and tables with forensic precision, and serves it through a cited conversational interface where every claim links back to its source.

**Python 3.11+** | **Next.js 15** | **Docker** | **Postgres** | **Qdrant** | **LangGraph**

---

## What Makes This Different

Most RAG systems treat documents as flat text. Financial documents aren't flat text — they're dense visual artifacts where a single stacked bar chart encodes more information than three pages of prose. Centaur solves this with three architectural decisions:

1. **Two-Phase Visual Pipeline (Glance + Read).** A layout analyzer detects chart regions and their archetypes (Cartesian, Waterfall, Valuation Field). A forensic vision extractor then reads each region with full context — axis labels, legends, footnotes, annotations — producing typed `MetricSeries` with per-datapoint provenance.

2. **Hybrid Search with Structured Analytics.** Qualitative content (narratives, summaries, insights) lives in Qdrant as dense+sparse vectors. Quantitative content (every extracted datapoint) lives in Postgres as pre-computed `metric_facts` rows. The query router decides which path — or both — to use.

3. **Pixel-Precise Citations.** Every generated answer carries `[N]` citation markers. Each resolves to a `Citation` object with source file, page number, and normalized bounding box coordinates. The frontend renders these as clickable highlight overlays on the actual document page.

---

## Architecture

```
                                    CENTAUR
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   INGESTION (Write Path)                                        │
    │   ┌──────────┐    ┌──────────────┐    ┌────────────────────┐   │
    │   │  Smart    │───>│  Dual-Helix  │───>│  Unified Document  │   │
    │   │  Router   │    │  Parsers     │    │  (Typed Stream)    │   │
    │   └──────────┘    │              │    └────────┬───────────┘   │
    │                    │  Helix A:    │             │               │
    │                    │  PDF/Visual  │        ┌────┴────┐         │
    │                    │  (Docling +  │        │         │         │
    │                    │   VLM x2)    │        ▼         ▼         │
    │                    │              │   ┌────────┐ ┌────────┐    │
    │                    │  Helix B:    │   │ Qdrant │ │Postgres│    │
    │                    │  Native      │   │ Hybrid │ │ Metric │    │
    │                    │  (Excel)     │   │ Index  │ │ Facts  │    │
    │                    └──────────────┘   └────────┘ └────────┘    │
    │                                           │           │        │
    │   RETRIEVAL (Read Path)                   │           │        │
    │   ┌──────────┐    ┌──────────────┐   ┌────┴───────────┴───┐   │
    │   │  Query   │───>│  Term        │──>│  Hybrid Search     │   │
    │   │  Router  │    │  Injector    │   │  (Dense + BM25)    │   │
    │   │  (LLM)   │    │  (Multilin.) │   │  + Voyage Rerank   │   │
    │   └──────────┘    └──────────────┘   │  + Text-to-SQL     │   │
    │        │                              └─────────┬─────────┘   │
    │        │                                        │              │
    │   GENERATION                                    │              │
    │   ┌──────────────────────────────────────┐     │              │
    │   │  GPT-4.1 Generation                   │<────┘              │
    │   │  + Citation Sidecar                   │                    │
    │   │  + Citation Verification              │                    │
    │   └──────────────┬───────────────────────┘                    │
    │                  │                                             │
    └──────────────────┼─────────────────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │   FastAPI  ──>  SSE Stream  ──>  Next.js UI                    │
    │                                  ┌─────┬────────┬──────┐       │
    │                                  │Docs │ PDF    │ Chat │       │
    │                                  │List │ Viewer │Panel │       │
    │                                  │     │ +BBox  │ +Cite│       │
    │                                  └─────┴────────┴──────┘       │
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

### Multilingual Query Expansion

The **Term Injector** solves the cross-lingual BM25 gap. German "Umsatz" won't match English "Revenue" in sparse search. Before search:
1. An LLM expands the query with multilingual synonyms and abbreviations
2. Known series labels from Postgres are fuzzy-matched and appended

The expanded query feeds the BM25 sparse leg. The original query feeds the dense embedding leg.

### Reranking

After RRF fusion, results pass through **Voyage Rerank 2.5** — a cross-encoder that re-scores each chunk against the original query for precision. Optional — the system works without it if no Voyage API key is configured.

---

## The Generation Pipeline

### Citation Sidecar

Retrieved chunks are mapped to ephemeral IDs `[1]`, `[2]`, etc. and formatted as XML source blocks for the generation LLM:

```xml
<source id="[1]" file="deal_memo.pdf p.12" type="visual">
Revenue grew 15% YoY driven by APAC expansion...
</source>
```

### Answer Generation

GPT-4.1 generates answers with mandatory `[N]` citation markers. The system prompt enforces: respond in the query's language, cite every factual claim, include currency/magnitude/period for numbers.

### Citation Verification

A lightweight LLM pass checks that each cited source actually supports the claim made. Unsupported citations are dropped before the response reaches the user.

### Citation Resolution

`[N]` markers in the generated text are resolved to `Citation` objects containing:
- `source_file` — original filename
- `page_number` — 1-based page
- `blurb` — the source text snippet
- `bbox` — normalized 0-1 bounding box for frontend highlighting

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

A three-panel Next.js application:

| Panel | Component | Function |
|-------|-----------|----------|
| **Left Sidebar** | `DocumentList` | Upload PDFs, browse ingested documents |
| **Center** | `DocumentViewer` | PDF.js page renderer with bbox highlight overlays |
| **Right** | `ChatPanel` | Conversational interface with clickable `[N]` citation badges |

Clicking a citation badge in the chat highlights the corresponding region on the document page. Coordinates scale from normalized 0-1 values to pixel positions: `left = bbox.x * containerWidth`.

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
| **SQL DB** | PostgreSQL 15 | Document ledger + `metric_facts` analytics |
| **Table Extraction** | IBM Docling v2 | HTML table extraction with row hierarchy |
| **OCR** | RapidOCR (ONNX) | Fallback for rasterized chart images |
| **Orchestration** | LangGraph | Stateful query routing and generation workflow |
| **API** | FastAPI | REST + SSE streaming endpoints |
| **Frontend** | Next.js 15 + Tailwind | Three-panel document intelligence UI |
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
│   │   ├── chunker.py                     # UnifiedDocument -> IndexableChunks
│   │   ├── filters.py                     # Token firewall (noise stripping)
│   │   └── phantom.py                     # Shadow PDF generation (Playwright)
│   │
│   ├── retrieval/                         # READ PATH
│   │   ├── qdrant.py                      # Full pipeline (expand -> search -> rerank)
│   │   ├── term_injector.py               # Multilingual query expansion + label matching
│   │   └── sidecar.py                     # Citation sidecar (build -> resolve -> verify)
│   │
│   ├── storage/                           # DATA LAYER
│   │   ├── vector_driver.py               # Qdrant hybrid index (Cohere + BM25 + RRF)
│   │   ├── analytics_driver.py            # Postgres metric_facts (structured analytics)
│   │   ├── db_driver.py                   # Postgres document ledger
│   │   └── blob_driver.py                 # Local artifact store
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
│   │   └── nodes/
│   │       ├── retrieve.py                # Retrieval nodes (qualitative, quantitative, hybrid)
│   │       ├── financial_math.py          # Text-to-SQL over metric_facts
│   │       └── generate.py                # GPT-4.1 generation + citation verification
│   │
│   ├── api/                               # FASTAPI SERVER
│   │   ├── main.py                        # App setup, CORS, startup initialization
│   │   ├── schemas.py                     # Request/response models
│   │   └── routes/
│   │       ├── chat.py                    # POST /chat, POST /chat/stream (SSE)
│   │       ├── ingestion.py               # POST /ingest, GET /documents
│   │       └── documents.py               # Page image rendering, region overlays
│   │
│   └── tools/                             # VLM TOOLS
│       ├── layout_analyzer.py             # Phase 1: Object detection (Glance)
│       ├── visual_extractor.py            # Phase 2: Forensic extraction (Read)
│       ├── calculator.py                  # Python math engine
│       ├── database.py                    # SQL lookup tool
│       └── vision.py                      # GPT-4o visual analysis
│
├── frontend/                              # NEXT.JS UI
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                 # Root layout
│       │   ├── page.tsx                   # Three-panel app shell
│       │   └── globals.css                # Dark theme, bbox overlay styles
│       ├── components/
│       │   ├── ChatPanel.tsx              # Chat with clickable citation badges
│       │   ├── DocumentViewer.tsx         # PDF.js viewer with bbox overlays
│       │   ├── DocumentList.tsx           # Sidebar with upload
│       │   └── BboxOverlay.tsx            # Pixel-precise citation highlight
│       └── lib/
│           └── api.ts                     # API client
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

### 4. Run

Terminal 1 — Backend:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — Frontend:
```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 5. Ingest a Document

Upload a PDF through the UI, or run manually:

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

---

## Observability

Every pipeline step is traced via **LangSmith** under the `Chiron_Adv_RAG` project:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=Chiron_Adv_RAG
```

Traced functions: layout analysis, visual extraction, document chunking, embedding, hybrid search, query expansion, reranking, Text-to-SQL, answer generation, citation verification.

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
