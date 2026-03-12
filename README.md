# Doc Processor with Descriptions

> Most PDFs were designed for human eyes, not machine understanding.
> This project changes that.

This pipeline turns a PDF into **searchable, page-aware Markdown** while preserving the things that usually get lost:

- figures
- tables
- formulas
- page boundaries
- visual context

It does this by combining **Docling** for structured PDF parsing with an **OpenAI vision model** for dense descriptions of non-text content.

---

## Why this exists

A normal PDF-to-text pipeline throws away too much.

Charts become silence. Tables become broken text. Equations become garbage. Images survive as files, but not as meaning.

That is a bad trade if your goal is retrieval, RAG, search, summarization, or downstream question answering.

This project solves that problem with one simple idea:

**do not extract only text — extract understanding.**

The processor reads the document page by page, saves page visuals, crops important assets, describes those assets in plain English, and then stitches everything back into one readable Markdown document.

The result is a document that is not only readable by a human, but far more useful to a model.

---

## How it works

At a high level, the pipeline follows this flow:

1. Read the PDF and determine the page count.
2. Process the PDF **one page at a time**.
3. Use **Docling** to parse the page structure.
4. Save:
   - page images
   - picture crops
   - table crops
   - formula crops
5. Use an **OpenAI vision model** to describe pictures, tables, and formulas.
6. Export page Markdown from Docling.
7. Replace placeholders in the Markdown with:
   - the saved asset image
   - a retrieval-friendly description
8. Save each page as its own Markdown file.
9. Stitch all pages into one final `processed_doc.md` file.

This design is deliberately page-wise.

That matters because page-wise processing is easier to debug, easier on memory, and easier to align with the original PDF than one giant document-wide conversion.

---

## What the pipeline actually produces

When you run the processor, it creates a working folder like this:

```text
output_root/
├── processed_doc.md
├── page_images/
├── image_png_images/
├── table_images/
├── formula_images/
└── pages_md/
```

### `processed_doc.md`
The final stitched Markdown for the whole document.

### `page_images/`
A rendered PNG for each page.

### `image_png_images/`
Cropped figure and image assets extracted from the PDF.

### `table_images/`
Cropped table images.

### `formula_images/`
Cropped equation / formula images.

### `pages_md/`
One Markdown file per page, useful for debugging, chunking, or page-level retrieval.

---

## Core design choices

This version of the processor keeps the flow intentionally simple:

- **CPU-first Docling setup**
- **page-wise conversion only**
- **fixed output structure**
- **per-page Markdown always written**
- **OpenAI vision kept in the loop** for non-text understanding

That means the project favors **clarity and reliability** over endless configuration.

There are fewer knobs. That is intentional.

---

## Libraries used

### Docling
Docling v2 provides the `DocumentConverter` API, supports `PdfPipelineOptions`, exposes document content through a `DoclingDocument`, iterates document items in reading order, and exports Markdown directly from the document object. It is the backbone of the PDF parsing step.

### LangChain OpenAI
The processor uses `ChatOpenAI` through `langchain-openai`. LangChain’s Python docs show that `ChatOpenAI` supports image input and uses the `OPENAI_API_KEY` environment variable for authentication, which is exactly what this project relies on for figure, table, and formula descriptions.

### PyPDF
PyPDF is used only for one thing: counting pages before page-wise processing begins.

---

## Installation

Create and activate a clean environment first.

### Conda

```bash
conda create -n docproc python=3.11
conda activate docproc
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_VISION_MODEL=gpt-5.2
```

`OPENAI_VISION_MODEL` is optional in this codebase. If you do not provide it, the processor defaults to `gpt-5.2`.

---

## Usage

This is the intended usage pattern:

```python
from doc_processor import doc_processor_with_descriptions

pdf_path = "sample.pdf"
output_dir = "doc_assets"

md_path = doc_processor_with_descriptions(
    pdf_path,
    output_root=output_dir,
)

print(f"✅ Finished. Markdown written to: {md_path}")
print("   Folders created: page_images/, image_png_images/, table_images/")
```

### Function signature

```python
doc_processor_with_descriptions(
    pdf_path: str | Path,
    output_root: str | Path,
    page_range: Optional[Tuple[int, int]] = None,
    images_scale: float = 1.6,
) -> Path
```

### Parameters

#### `pdf_path`
Path to the input PDF.

#### `output_root`
Folder where all generated assets and Markdown files will be written.

#### `page_range`
Optional page range tuple like `(5, 12)` if you want to process only part of the PDF.

#### `images_scale`
Controls image rendering scale inside Docling. Higher values usually improve asset clarity, but can also increase processing time and output size.

---

## What happens inside `doc_processor_with_descriptions`

The public API is intentionally small, but the pipeline does real work:

1. Normalize file paths.
2. Create output folders.
3. Build a CPU-only Docling converter.
4. Determine which pages to process.
5. Convert each page separately.
6. Save page images.
7. Collect extracted visual assets from the Docling document.
8. Describe each visual asset with the OpenAI vision model.
9. Export Markdown from Docling with placeholders.
10. Replace placeholders with saved asset paths and generated descriptions.
11. Save each page Markdown file.
12. Stitch all pages into one final Markdown file.

That is the entire system.

Simple idea. Strong outcome.

---

## Why page-wise processing matters

This is one of the most important design choices in the whole project.

Page-wise processing gives you:

- smaller memory footprint
- easier debugging
- cleaner page-level outputs
- simpler recovery when a specific page fails
- easier alignment between source PDF and generated Markdown

It also makes the output far easier to inspect when something looks wrong.

If page 17 breaks, you know where to look.

---

## Why descriptions are added

A raw crop image is useful to a human.

A description is useful to search.

That is why this pipeline does not stop at extraction. It enriches pictures, tables, and formulas with text that a retriever or downstream model can actually use.

### Picture descriptions
Dense factual prose for charts, figures, and diagrams.

### Table descriptions
Retrieval-friendly summaries of what the table contains and how values are organized.

### Formula descriptions
A normalized first line starting with `LaTeX: ...`, followed by a short explanation of what the equation expresses.

---

## Output quality philosophy

This processor is not trying to be flashy.

It is trying to be useful.

That means:

- readable code
- predictable folders
- page-level traceability
- better non-text retention than plain PDF-to-text tools
- Markdown that is useful for both humans and models

This is not just extraction.

It is preparation for understanding.

---

## Limitations

Be honest about what this pipeline does not magically solve.

- It depends on an OpenAI API key for visual descriptions.
- It is only as good as the PDF quality and extractability.
- Table and formula placement still depend on placeholder replacement logic.
- Very complex technical PDFs may still need additional alignment logic or post-processing.
- Higher `images_scale` can improve clarity, but also increases cost and runtime.

---

## Recommended workflow

If you are building RAG or search on top of this project, the clean workflow is:

1. Run the processor.
2. Inspect `pages_md/` for page-level sanity.
3. Use `processed_doc.md` for downstream chunking and embeddings.
4. Keep the generated image folders as evidence and visual grounding assets.

---

## Example use cases

This pipeline is a strong fit for:

- technical PDF retrieval
- multimodal RAG preprocessing
- scientific and engineering document understanding
- PDF knowledge base ingestion
- chart / table / formula-aware indexing

---

## Final thought

A great document pipeline should do more than extract text.

It should preserve intent.
It should preserve structure.
It should preserve meaning.

That is what this project is trying to do.

And that is why it matters.
