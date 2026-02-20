# UI Polish & Premium Feel — Design Consultation

## What This Is

Centaur is a **financial document intelligence platform** — think Bloomberg Terminal meets modern RAG. Users upload PDFs (earnings slides, CIMs, credit agreements) and the system extracts structured data, enables natural-language Q&A with cited sources, and provides a "Metric Explorer" for browsing extracted financial series.

The **target users** are investment professionals at PE firms, hedge funds, and investment banks. These people use Bloomberg, PitchBook, Capital IQ, and FactSet daily. They expect **institutional-quality** interfaces — authoritative, information-dense, zero fluff.

The product name "Centaur" references the human-machine hybrid: human financial judgment augmented by AI extraction and retrieval.

## Current Architecture

**Three-panel layout (fixed, desktop-only, min-width 1024px):**

```
┌──────────────┬─────────────────────────────┬──────────────────┐
│  Left Sidebar │      Center: PDF Viewer      │   Right Panel    │
│  (256px)      │      (flex-1)                │   (384px)        │
│               │                              │                  │
│  Documents    │  react-pdf renderer          │  Tab bar:        │
│  - Upload     │  + bbox highlight overlays   │  [Chat|Inspect|  │
│  - List       │  + zoom HUD (floating pill)  │   Explorer]      │
│  - Status     │  + page nav toolbar          │                  │
│               │                              │  Active panel    │
└──────────────┴─────────────────────────────┴──────────────────┘
```

**Tech stack**: Next.js 15 + Tailwind v4 + Zustand + react-pdf + Lucide icons. No component library (all custom). Dark theme only.

## Current CSS Variables (Design Tokens)

```css
:root {
  --bg-primary: #0a0a0a;      /* Deepest background */
  --bg-secondary: #141414;    /* Panel backgrounds */
  --bg-tertiary: #1e1e1e;     /* Hover states, inputs, cards */
  --border: #2a2a2a;          /* All borders */
  --text-primary: #ededed;    /* Main text */
  --text-secondary: #999;     /* Subdued text */
  --accent: #3b82f6;          /* Blue — links, active states, CTAs */
  --accent-hover: #2563eb;    /* Darker blue on hover */
  --highlight: rgba(59, 130, 246, 0.25);  /* Selection highlight */
}
```

## Current Visual Treatment

### What exists:
- **"Physical Desk" PDF styling**: Canvas has `border: 1px solid #444`, layered box-shadows (paper-on-dark-desk feel), `filter: brightness(0.96)` to reduce white glare, `border-radius: 2px`
- **Floating zoom HUD**: Pill at bottom-center with frosted glass (`backdrop-filter: blur(8px)`), rounded-full, shadow
- **Bbox overlays**: Yellow highlight with red border, `mix-blend-mode: multiply`
- **Delta pills** (Metric Explorer): Emerald green / rose red with 10% alpha backgrounds
- **Chat bubbles**: User = solid blue, assistant = dark card (#1e1e1e)
- **Streaming progress**: Terminal-style stepper with check/spinner/circle icons
- **Citation badges**: Blue circles with numbers, clickable

### What's missing / broken:
- **No splash or entry experience** — the app just appears
- **No brand identity** — no logo, no wordmark, no "About" or product name anywhere visible
- **Empty states are barren** — "Select a document to view" in plain text, "No documents ingested yet. Upload a PDF to get started." in 12px gray
- **Upload is browser-default** `<input type="file">` with "Choose file | No file chosen"
- **Zero micro-animation** — tab switches are instant display:none/block, document selection is background color swap, chat messages appear instantly
- **Flat panel separation** — all three panels separated by 1px solid borders, no depth or spatial hierarchy
- **Generic tab bar** — plain text with bottom border accent, no icons, no sliding indicator
- **No typography differentiation** — everything is system fonts at 12-14px, no weight hierarchy beyond "semibold" on section headers

## Current Component Inventory (with problems)

### 1. Left Sidebar — DocumentList.tsx
- Header: "Documents" (14px semibold) + refresh/upload icon buttons
- Upload: Hidden `<input type="file">` triggered by Upload icon — **no drag-and-drop**
- Document items: FileText icon + filename + status label + date
- Active state: background-color change only
- Processing state: spinning Loader2 icon + "Processing..." label
- Empty: "No documents ingested yet. Upload a PDF to get started." in 12px gray, centered

### 2. Center — DocumentViewer.tsx
- Toolbar: sidebar toggle + page nav (prev/next + "Page N of M")
- PDF: react-pdf Page component with zoom support
- Overlay: BboxOverlay component for citation/chunk highlights
- Zoom HUD: floating pill with +/- buttons, percentage display, fit-width button
- Empty: "Select a document to view" in gray, centered
- **Note**: The PDF itself looks great (Physical Desk treatment). The chrome around it is plain.

### 3. Right Panel — RightPanel.tsx
Contains three sub-panels in tabs:

**Chat tab** (ChatPanel.tsx):
- Messages list with user (blue) / assistant (dark card) bubbles
- Streaming: ghost bubble with terminal-style stepper
- Input: text input + send button + doc scope pill + clear button
- Empty: "Ask a question about your documents" in gray

**Inspect tab** (ChunkInspectorPanel.tsx):
- Page header with chunk count + doc stats badges
- Chunk cards with type badges (color-coded), text preview, metadata
- Active state: expanded card with full text + all metadata

**Explorer tab** (MetricExplorerPanel.tsx) — just rebuilt:
- Search input + auto-hiding filter pills + download button
- Collapsible page groups with chart titles
- Ledger rows: label | sparkline | value | delta pill
- Expandable "Ticker Tape" detail for each row

## What We Want

Channel **Jony Ive / Steve Jobs** design philosophy:
- Simple, intuitive, but powerful
- Every pixel intentional
- "It just feels right"
- Premium materials and craftsmanship in every detail
- The interface should feel like it costs money

Specific goals:

### A. Opening / Splash Experience
A very quick (400-700ms) visual moment when the app loads. NOT a loading spinner. More like the feeling of opening a premium product — a reveal, a signature moment. Must be CSS-only or minimal JS, non-blocking (app mounts behind it). Ideas we've considered:
- Wordmark fade-in + shift-up + dissolve
- Horizontal light sweep across dark canvas
- SVG path draw animation

### B. Empty State / Landing Page
When no document is selected, the center panel (70% of viewport) is wasted. This should be:
- A "welcome" experience that communicates product identity
- An intuitive upload zone (drag-and-drop)
- Quick-start orientation (what can this tool do?)
- Not patronizing — these are sophisticated finance professionals

### C. Material & Depth
The three-panel layout should have spatial hierarchy:
- Which surface is "above" or "behind" others?
- Can we use subtle gradients, inner shadows, or translucency to create depth?
- The PDF viewer is the "star" — it should feel like the primary surface

### D. Micro-interactions & Motion
Every state change should have a considered transition:
- Tab switching (sliding indicator? crossfade?)
- Document selection
- Chat message arrival
- Panel collapse/expand
- Page navigation

### E. Typography & Information Hierarchy
- Clear weight/size ladder: H1 → H2 → body → caption → mono
- Section labels need presence (letter-spacing? weight? size?)
- Financial numbers need tabular treatment everywhere
- Consider a refined typeface (Inter? Geist? Or stick with system stack but tune it?)

### F. Upload Experience
- Drag-and-drop zone with visual feedback
- Progress indicator (not just text)
- Success/failure state

### G. Brand Identity
- "Centaur" wordmark — where and how?
- Color palette: stay with blue accent or evolve?
- Icon style: stick with Lucide or augment?

## Constraints

1. **No component library** — everything is hand-built. We like the control this gives us.
2. **CSS/Tailwind only** for styling — no CSS-in-JS runtime.
3. **Performance** — splash must be non-blocking, animations must be 60fps, no layout thrashing. These users have fast machines but zero patience.
4. **Dark theme only** for now (light mode is future).
5. **Desktop only** — min-width 1024px. No mobile considerations.
6. **The PDF viewer "Physical Desk" treatment stays** — it works well. Don't redesign the canvas/shadow/border treatment.
7. **The Metric Explorer (Pitchbook Ledger) was just rebuilt** — don't redesign it, but feel free to suggest refinements.
8. **Keep the 3-panel layout** — it's the right structure. Polish the execution.
9. **No heavy assets** — no large images, videos, or fonts that add download weight. System fonts or a single lightweight webfont max.
10. **Tailwind v4** — uses `@import "tailwindcss"` (no tailwind.config.js), CSS variables, vanilla CSS classes. No `@apply` needed.

## Reference Points

- **Bloomberg Terminal**: Dense, dark, authoritative. Every pixel is information.
- **Linear**: Premium dark UI, beautiful micro-animations, frosted glass, refined typography.
- **Figma**: Three-panel layout done right — clear hierarchy, great empty states.
- **Notion**: Elegant empty states that onboard without being patronizing.
- **Apple.com product pages**: The "reveal" and "materiality" — how surfaces feel.
- **Stripe Dashboard**: Clean dark mode, excellent typography ladder, subtle animations.

## Questions for You

1. **Splash**: What specific animation approach would create the most premium "opening" feeling while staying under 700ms and being pure CSS/minimal JS? Provide specific CSS keyframe code or a detailed specification.

2. **Empty state layout**: Design the "no document selected" center panel and "no documents yet" sidebar empty state. How do we communicate product identity + upload CTA + quick-start without being cluttered? Provide an ASCII mockup or detailed component specification.

3. **Material language**: What specific CSS changes (gradients, shadows, blur, opacity) would create depth hierarchy across the three panels? Be specific — give me the actual CSS properties and values.

4. **Micro-animation catalog**: For each interaction (tab switch, doc select, message arrive, panel collapse, page turn, upload progress), what's the optimal duration, easing, and property to animate? Be specific.

5. **Typography ladder**: What's the exact type scale? Give me sizes, weights, letter-spacing, and line-height for each level. Should we add a webfont or tune the system stack?

6. **Tab bar redesign**: How should the right panel tab bar look? Sliding indicator? Icons? Pill selection? Provide a detailed spec.

7. **Upload zone**: Design the drag-and-drop upload experience. What does the zone look like idle, on dragover, during upload, and on success/failure? How does it integrate with the sidebar layout?

8. **Brand placement**: Where does "Centaur" appear? Just the splash? Sidebar header? Both? What does the wordmark look like (typeface, weight, letter-spacing)?

9. **Color palette evolution**: Should we stay with pure blue (#3b82f6) or evolve? What about the grays — warmer? Cooler? Any secondary accent color?

10. **Priority ordering**: If we can only ship 3 of these improvements, which 3 create the biggest perceived quality jump? Order all items by impact-to-effort ratio.

## Anti-Patterns to Avoid

- **Over-animation**: Linear's animations are 150-200ms. Don't make things feel sluggish with 500ms transitions.
- **Gratuitous blur**: Frosted glass on everything becomes visual noise. Use sparingly (1-2 surfaces max).
- **Gradient abuse**: Subtle = premium. Visible gradient = early 2010s.
- **Dark-on-dark contrast**: Financial data must be immediately readable. Don't sacrifice legibility for aesthetics.
- **"Creative" upload zones**: Animated cloud icons, bouncing arrows, etc. Keep it dignified.
- **Feature onboarding tours**: No tooltips, no "step 1 of 4", no walkthrough modals. The interface should be self-evident.
- **Logo that tries too hard**: No centaur illustration. A wordmark or lettermark is more appropriate for this market.

## Current globals.css (Full)

```css
@import "tailwindcss";

:root {
  --bg-primary: #0a0a0a;
  --bg-secondary: #141414;
  --bg-tertiary: #1e1e1e;
  --border: #2a2a2a;
  --text-primary: #ededed;
  --text-secondary: #999;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --highlight: rgba(59, 130, 246, 0.25);
}

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.app-shell { min-width: 1024px; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.pdf-page-container { position: relative; }
.pdf-page-container canvas {
  border: 1px solid #444;
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.04),
    0 4px 16px rgba(0, 0, 0, 0.6),
    0 16px 48px rgba(0, 0, 0, 0.4);
  border-radius: 2px;
  filter: brightness(0.96);
}

.bbox-overlay {
  position: absolute;
  background: rgba(250, 204, 21, 0.2);
  border: 1.5px solid #ef4444;
  border-radius: 3px;
  pointer-events: none;
  transition: opacity 0.2s;
  mix-blend-mode: multiply;
}
.bbox-overlay.active {
  background: rgba(250, 204, 21, 0.25);
  border-color: #dc2626;
}

.zoom-hud {
  position: absolute;
  bottom: 1.5rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 0.25rem;
  background: rgba(30, 30, 30, 0.92);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 9999px;
  padding: 0.375rem 0.75rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}
.zoom-btn {
  padding: 0.25rem;
  border-radius: 0.25rem;
  color: var(--text-secondary);
  transition: color 0.15s, background 0.15s;
}
.zoom-btn:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.delta-positive { background-color: rgba(16, 185, 129, 0.1); color: #10b981; }
.delta-negative { background-color: rgba(244, 63, 94, 0.1); color: #f43f5e; }
.ledger-row-active {
  background-color: var(--highlight);
  outline: 1px solid var(--accent);
  outline-offset: -1px;
}
.explorer-scroll::-webkit-scrollbar { width: 4px; }
```

## What a Great Response Looks Like

Don't just say "add animations" — give me **specific CSS**, **specific component changes**, **specific values**. I want to be able to hand this to an engineer and have them implement it in one session. ASCII mockups for layout changes, CSS keyframes for animations, exact hex/rgba values for colors, exact px/rem values for spacing.

Prioritize ruthlessly. Not everything needs to change. What are the 5-6 highest-impact changes that will make this feel like a $50k/seat institutional product?
