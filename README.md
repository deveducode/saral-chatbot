# SARAL Chatbot — Audience-Adaptive Script & Bullet Generator

A retrieval-augmented generation (RAG) prototype for SARAL that turns a research
paper into audience-adapted scripts, slide bullets, and short summaries — with
inline provenance citations and iterative, change-tracked editing.

Built for the SARAL recruitment assignment (Part A: Programming Task).

## What it does

Given a research paper (PDF) and a request like *"make a 90-second script for
policymakers"*, the system:

1. Retrieves the most relevant passages from the paper using semantic search
2. Generates a script, slide bullets, and a short summary — parameterized by
   **audience** (policymakers / grad students / press), **length** (30s / 90s /
   5min), and **style** (technical / plain-English / press release)
3. Tags every factual claim with a citation (`[C12]`) linking back to the exact
   source chunk and page number
4. Preserves LaTeX/math notation inline (e.g. `$F1 = \frac{2 \cdot Pr \cdot Re}{Pr + Re}$`)
5. Supports iterative editing — e.g. *"make #2 less technical"* — and shows a
   **diff of what changed and why**, instead of silently replacing the output

## Architecture

```mermaid
flowchart TD
    A[PDF / LaTeX paper] --> B["Ingestion<br/>LaTeX-safe chunking + embed"]
    B --> C["FAISS vector store<br/>chunk text + page number"]
    C --> D["Retriever<br/>top-k similarity search"]
    E["User prompt<br/>audience, length, style, task"] --> D
    D --> F["Generator (Gemini)<br/>prompt template + citations"]
    F --> G["Output<br/>script + bullets + [Cx] citations"]
    G -->|"change instruction<br/>e.g. 'less technical'"| H["Change edit"]
    H -->|"delta + why-changed"| F

    style A fill:#f1efe8,stroke:#2c2c2a,stroke-width:2px,color:#2c2c2a
    style B fill:#9fe1cb,stroke:#04342c,stroke-width:2px,color:#04342c
    style C fill:#9fe1cb,stroke:#04342c,stroke-width:2px,color:#04342c
    style D fill:#9fe1cb,stroke:#04342c,stroke-width:2px,color:#04342c
    style E fill:#f5c4b3,stroke:#4a1b0c,stroke-width:2px,color:#4a1b0c
    style F fill:#cecbf6,stroke:#26215c,stroke-width:2px,color:#26215c
    style G fill:#f5c4b3,stroke:#4a1b0c,stroke-width:2px,color:#4a1b0c
    style H fill:#fac775,stroke:#412402,stroke-width:2px,color:#412402
```
```

**Pipeline stages:**

| Stage | File | What it does |
|---|---|---|
| Ingestion | `ingest.py` | Extracts text per page (PyMuPDF), chunks it while protecting LaTeX math blocks (`$...$`, `\begin{equation}`) from being split mid-expression, embeds chunks with `sentence-transformers` (`all-MiniLM-L6-v2`), and stores them in a FAISS index with page-number metadata |
| Retrieval | `retrieve.py` | Embeds the user's query and does top-k cosine similarity search over the FAISS index, returning chunk text + page number + score |
| Generation | `generate.py` | Builds a parameterized prompt (audience/length/style) from retrieved chunks, calls Gemini, and post-processes citation tags. Also implements `apply_change()` for iterative edits: re-generates the targeted section, computes an old-vs-new diff with `difflib`, and asks the model for a one-line "why changed" explanation |
| Evaluation | `eval.py` | Computes citation coverage (% of lines with a `[Cx]` tag) and a factuality proxy (average cosine similarity between each generated sentence and the source chunk it cites) |
| UI | `app.py` | Minimal Streamlit chat-style interface: set audience/length/style + a source query, generate, then apply follow-up change instructions and see the diff inline |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Usage

**1. Ingest a paper:**
```bash
python ingest.py --pdf data/sample_paper.pdf --out data/index
```

**2. Test retrieval directly (optional sanity check):**
```bash
python retrieve.py --index data/index --query "methodology used" --k 3
```

**3. Generate a script/bullets:**
```bash
python generate.py --index data/index \
  --audience "grad students" --length 90s --style technical \
  --instruction "Summarize the methodology used"
```

**4. Run automatic evaluation:**
```bash
python eval.py --index data/index \
  --audience "grad students" --length 90s --style technical \
  --instruction "Summarize the methodology used"
```

**5. Launch the chat UI:**
```bash
streamlit run app.py
```

**6. See iterative change-tracking end to end:**
```bash
python test_change.py
```

## Example run

Paper used: a landslide susceptibility assessment (LSA) research paper (11 pages,
included at `data/sample_paper.pdf`).

**Input:** audience=grad students, length=90s, style=technical, task="Summarize
the methodology used"

**Output (excerpt):**
> Landslide susceptibility assessment (LSA) is a predictive tool used to estimate
> the likelihood of landslides occurring in a specific area based on local
> conditions [C13]. A key methodology developed for this task is LCFSTE, which
> integrates Landslide Conditioning Factors and a Swin Transformer Ensemble [C1]...

**Change instruction:** "make this less technical for a general audience"

**Result:** the system regenerated the section in plain English (e.g. "Landslide
susceptibility assessment (LSA) is a way to predict where landslides are most
likely to happen..."), produced a unified diff of the exact lines that changed,
and explained: *"The AFTER text replaced complex academic jargon and statistical
modeling terms with simplified, plain-English descriptions to make the technical
concepts of landslide prediction more accessible."*

Full conversation log: [`logs/conversation_example.md`](logs/conversation_example.md)

## Evaluation results

Run on the sample landslide paper (instruction: "Summarize the methodology used",
audience=grad students, length=90s, style=technical):

| Metric | Result |
|---|---|
| Citation coverage | 70.6% of non-empty output lines carry at least one `[Cx]` tag |
| Factuality proxy (avg. sentence-to-source cosine similarity) | 0.574, across 19 cited sentences |

**What's not included, and why:** ROUGE/BERTScore against a human-authored
reference script, and 3-rater human evaluation for audience-appropriateness,
factuality, and helpfulness, are specified in the brief but require a
human-written reference script and human annotators — both out of scope for a
1-day prototype. The natural next step would be to collect 1-2 human-written
reference scripts per test paper and run BERTScore against them, plus a small
(3-rater) blind rating pass on a handful of generated outputs.

## Design notes & trade-offs

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (open, local, free) —
  chosen over an API-based embedding model to keep the pipeline runnable
  offline/cheaply for a small conference-paper-sized corpus, per the brief's
  "use small conference papers to avoid excessive cost" guidance.
- **Generator:** Gemini (`gemini-2.0-flash`) via API — fast and free-tier
  friendly for a 1-day build; the prompt template is model-agnostic and could
  be swapped to an open 7B model or another API with no pipeline changes.
- **Retrieval quality depends on query specificity.** A generic query (e.g.
  "what is the main contribution?") can retrieve boilerplate sections like
  author bios, since those also use generic academic language. Domain-specific
  queries retrieve content-relevant chunks reliably (verified during testing —
  see conversation logs).
- **Safety/style constraints** are enforced via explicit system-prompt
  instructions (avoid offensive language, match requested accessibility level)
  rather than a separate classifier, given the time budget — a dedicated
  toxicity/accessibility classifier is a natural hardening step for production.

## Citations / external resources used

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, NeurIPS 2020 — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
  (RAG design reference)
- `sentence-transformers` / `all-MiniLM-L6-v2` — open embedding model
- Google Gemini API — generation
- FAISS — vector similarity search
- Streamlit — UI
- PyMuPDF — PDF text extraction
