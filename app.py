import asyncio
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import io
import base64
import tempfile
import re
from dotenv import load_dotenv
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from PIL import Image
import requests
import markdown
from playwright.sync_api import sync_playwright

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=16000,
    temperature=0.3,
)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def index_docs(files):
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(splits, embeddings)


def format_docs(docs):
    return "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d) for d in docs
    )


def store_uploaded_images(image_files):
    stored = []
    for img in image_files:
        img_bytes = img.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()
        stored.append({
            "base64": img_base64,
            "media_type": img.type or "image/jpeg"
        })
    return stored


def build_vision_content(text, stored_images):
    content = [{"type": "text", "text": text}]
    for img in stored_images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["base64"]
            }
        })
    return content


def generate_image(prompt):
    try:
        import fal_client
        result = fal_client.run(
            "fal-ai/flux/schnell",
            arguments={"prompt": prompt}
        )
        img_url = result["images"][0]["url"]
        response = requests.get(img_url, timeout=30)
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"fal.ai failed: {str(e)}")
        return None


def flux_enhance_prompt(raw_description: str) -> str:
    """
    Wraps a raw UI description in a Flux-optimised prompt that produces
    finished, photorealistic product UI screenshot quality.

    KEY RULES:
    - Explicitly forbids all text/typography — Flux hallucinates garbled text.
    - Steers toward finished product screenshots, not wireframes or diagrams.
    - Adds lighting, depth, and rendering quality boosters for sharpness.
    """
    return (
        "Photorealistic finished product UI screenshot, award-winning app design, "
        "Dribbble portfolio quality, sharp 4K render, "
        + raw_description.strip().rstrip(".,")
        + ", absolutely no text, no words, no labels, no numbers, no typography of any kind, "
        "pure visual UI layout only, pixel-perfect vector-sharp edges, "
        "professional SaaS product design, soft ambient lighting, "
        "subtle depth-of-field background blur, clean consistent spacing, "
        "modern interface components with smooth rounded corners, "
        "realistic screen glow and surface reflections"
    )


def render_mermaid_to_base64(mermaid_code: str) -> str:
    """
    Renders a Mermaid diagram to a base64 PNG using a headless Chromium browser.

    - Uses mermaid.run() Promise API so we KNOW rendering is complete.
    - Polls via page.wait_for_function() until the SVG has a non-zero bounding box.
    - Clips screenshot to the actual SVG bounding box to remove blank space.
    - Wraps in try/except and returns a placeholder on failure so PDF keeps going.
    - Escapes the mermaid code before injecting into HTML.
    """

    safe_code = (
        mermaid_code
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.3/dist/mermaid.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: white; padding: 24px; }}
  #container {{ width: 900px; }}
  .mermaid {{ width: 100%; }}
</style>
</head>
<body>
<div id="container">
  <div class="mermaid" id="mermaid-diagram">
{safe_code}
  </div>
</div>
<script>
  window.mermaidDone = false;
  window.mermaidError = null;

  mermaid.initialize({{
    startOnLoad: false,
    theme: 'default',
    fontSize: 14,
    flowchart: {{ useMaxWidth: true, htmlLabels: true }},
    sequence: {{ useMaxWidth: true }},
    gantt: {{ useMaxWidth: true }},
  }});

  mermaid.run({{
    nodes: [document.getElementById('mermaid-diagram')]
  }}).then(() => {{
    requestAnimationFrame(() => {{
      window.mermaidDone = true;
    }});
  }}).catch((err) => {{
    window.mermaidError = err.message || String(err);
    window.mermaidDone = true;
  }});
</script>
</body>
</html>"""

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            page.set_content(html, wait_until="domcontentloaded")

            page.wait_for_function("window.mermaidDone === true", timeout=15000)

            mermaid_error = page.evaluate("window.mermaidError")
            if mermaid_error:
                st.warning(f"Mermaid render warning: {mermaid_error}")

            page.wait_for_function(
                """() => {
                    const svg = document.querySelector('#mermaid-diagram svg');
                    if (!svg) return false;
                    const rect = svg.getBoundingClientRect();
                    return rect.width > 10 && rect.height > 10;
                }""",
                timeout=10000
            )

            clip_box = page.evaluate("""() => {
                const svg = document.querySelector('#mermaid-diagram svg');
                const rect = svg.getBoundingClientRect();
                return {
                    x: Math.max(0, rect.x - 4),
                    y: Math.max(0, rect.y - 4),
                    width: rect.width + 8,
                    height: rect.height + 8
                };
            }""")

            screenshot = page.screenshot(clip=clip_box, type="png")
            browser.close()

        return base64.b64encode(screenshot).decode()

    except Exception as e:
        st.warning(f"Mermaid diagram could not be rendered: {e}. Skipping diagram.")
        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        return tiny_png


def pil_image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to base64 PNG string."""
    buffered = io.BytesIO()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(buffered, format="PNG", optimize=False)
    return base64.b64encode(buffered.getvalue()).decode()


def build_pdf_html(html_body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @page {{
    size: A4;
    margin: 1.5cm 1.8cm 1.5cm 1.8cm;
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11px;
    line-height: 1.6;
    color: #1a1a1a;
    width: 100%;
    margin: 0;
    padding: 0;
  }}

  h1 {{
    color: #1a73e8;
    font-size: 20px;
    margin-top: 0;
    margin-bottom: 12px;
    page-break-before: auto;
    page-break-after: avoid;
  }}

  h2 {{
    color: #1a73e8;
    font-size: 16px;
    margin-top: 24px;
    margin-bottom: 8px;
    page-break-before: auto;
    page-break-after: avoid;
    break-after: avoid;
  }}

  h3 {{
    color: #333;
    font-size: 13px;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
    break-after: avoid;
  }}

  h4, h5, h6 {{
    color: #444;
    margin-top: 12px;
    margin-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
  }}

  p {{
    margin: 0 0 8px 0;
    orphans: 3;
    widows: 3;
  }}

  ul, ol {{
    margin: 4px 0 8px 0;
    padding-left: 20px;
  }}

  li {{
    margin-bottom: 3px;
    orphans: 2;
    widows: 2;
  }}

  li > ul, li > ol {{
    margin-top: 3px;
    margin-bottom: 3px;
  }}

  img {{
    display: block;
    max-width: 100%;
    width: auto;
    height: auto;
    margin: 16px auto;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 10px;
    page-break-inside: auto;
    word-wrap: break-word;
    table-layout: fixed;
  }}

  thead {{
    display: table-header-group;
  }}

  th {{
    background: #1a73e8;
    color: white;
    padding: 6px 8px;
    text-align: left;
    font-size: 10px;
    word-wrap: break-word;
    border: 1px solid #1558b0;
  }}

  td {{
    padding: 5px 8px;
    border: 1px solid #ddd;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }}

  tr:nth-child(even) td {{
    background: #f8f9fa;
  }}

  tr {{
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  pre {{
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 9px;
    overflow-x: auto;
    page-break-inside: avoid;
    break-inside: avoid;
    white-space: pre-wrap;
    word-wrap: break-word;
  }}

  code {{
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
    font-size: 9px;
    font-family: 'Courier New', monospace;
  }}

  pre code {{
    background: none;
    padding: 0;
  }}

  blockquote {{
    border-left: 4px solid #1a73e8;
    margin: 8px 0;
    padding: 4px 12px;
    color: #555;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 16px 0;
  }}

  .page-section {{
    page-break-before: always;
    break-before: page;
  }}

  figure {{
    page-break-inside: avoid;
    break-inside: avoid;
    margin: 16px 0;
  }}

  strong {{ font-weight: 700; }}
  em {{ font-style: italic; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""


def generate_deck(
    query: str,
    stored_images: list,
    retriever,
    generate_images: bool
) -> bytes:

    context_docs = retriever.invoke(query)
    context = format_docs(context_docs)

    prompt = f"""You are creating a comprehensive professional PDF insights deck.

User request:
"{query}"

MANDATORY RULES — follow ALL of these without exception:
1. Write a COMPLETE document. Never summarize, abbreviate, or stop early.
   Write at minimum 8 substantial sections. Each section MUST have:
   - A descriptive ## heading
   - At least 3 substantial paragraphs OR a detailed table OR a detailed list
   - Real analysis, not filler text
2. Use professional consultant-level writing: specific, evidence-based, detailed.
3. Include at least one markdown table with realistic data/metrics.
4. Include bullet points, numbered lists, pros/cons where appropriate.
5. If the request mentions a timeline or schedule: include a Mermaid diagram
   using this EXACT format (no deviations):
   ```mermaid
   gantt
       title [title here]
       dateFormat YYYY-MM-DD
       section Phase 1
           Task name : 2024-01-01, 30d
6. For UI illustration or mockup improvement requests: add EXACTLY 5 [GENERATE_IMAGE:] markers spread
throughout the document, placed inline near the section they illustrate. Each must depict a SPECIFIC,
FINISHED improved UI screen directly relevant to the uploaded mockups and specs — not abstract concepts.

RULES for every <visual description>:
- Reference the EXACT screen being improved (e.g. "the desktop Kanban board from the mockups",
  "the mobile milestone timeline screen", "the AI suggestion panel")
- Describe the SPECIFIC layout fix being shown (e.g. "collapsed icon sidebar freeing horizontal space",
  "task cards showing only title and avatar in default state", "bottom sheet sliding up over timeline")
- Describe colours, component shapes, spacing, and visual hierarchy precisely
- Do NOT mention text, labels, words, numbers, or typography — Flux cannot render readable text
- Each image should look like a FINISHED product screen, not a wireframe or abstract diagram

Good examples for a project management tool:
[GENERATE_IMAGE: Redesigned desktop Kanban board with a narrow 64px icon-only collapsed left sidebar in dark navy, three columns of minimal white task cards each showing only a circular avatar and a thin left-border priority stripe in red orange or green, generous whitespace between cards, a glowing teal circular AI button floating in the bottom right corner, clean light background]
[GENERATE_IMAGE: Improved mobile timeline view in dark mode, full-width vertical scroll of milestone cards each with a soft rounded rectangle shape in dark charcoal, a thin teal top-border accent, subtle drop shadow, a bottom sheet panel sliding up from below with a frosted glass effect and three action icon buttons arranged in a row, no AI panel blocking the timeline]
[GENERATE_IMAGE: Redesigned AI suggestion component as a compact collapsible right-side panel on a dark dashboard, showing three stacked suggestion cards with a glowing teal left accent border, a small circular confidence meter icon, smooth card shadows, a chevron collapse arrow at the top, not blocking the main kanban content]
7. Output ONLY valid Markdown. No preamble, no "here is your document", no apologies.
8. Do NOT truncate. If approaching your limit, compress earlier sections to ensure all phases of the
timeline and all recommendations are fully written out. Completeness is more important than verbosity.

Document context from uploaded files:
{context}"""

    if stored_images:
        content = build_vision_content(prompt, stored_images)
        md_content = llm.invoke([HumanMessage(content=content)]).content
    else:
        md_content = llm.invoke(prompt).content

    # Mermaid block extraction and rendering
    mermaid_pattern = re.compile(
        r'```mermaid\s*\n(.*?)\n\s*```',
        re.DOTALL | re.IGNORECASE
    )

    mermaid_replacements = []
    for match in mermaid_pattern.finditer(md_content):
        diagram_code = match.group(1).strip()
        full_match = match.group(0)
        b64 = render_mermaid_to_base64(diagram_code)
        replacement = (
            f'\n\n<div style="page-break-inside:avoid; break-inside:avoid; '
            f'text-align:center; margin:20px 0;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%; height:auto; display:block; margin:0 auto;">'
            f'</div>\n\n'
        )
        mermaid_replacements.append((full_match, replacement))

    for original, replacement in mermaid_replacements:
        md_content = md_content.replace(original, replacement)

    # Flux image generation
    image_prompt_pattern = re.compile(
        r'\[GENERATE_IMAGE:\s*(.*?)\]',
        re.DOTALL
    )
    image_prompts = [m.strip() for m in image_prompt_pattern.findall(md_content)]
    md_content = image_prompt_pattern.sub('', md_content)

    if generate_images:
        if not image_prompts and any(
            kw in query.lower() for kw in ["ui", "interface", "mockup", "illustration", "design", "improved", "improvement"]
        ):
            image_prompts = [
                "Redesigned desktop Kanban board with a narrow 64px collapsed icon-only left sidebar "
                "in dark navy, three columns of minimal white task cards each showing only a circular "
                "avatar and a thin left-border priority stripe in red orange or green, generous "
                "whitespace between cards, a glowing teal circular AI button floating bottom right, "
                "clean light background, finished product quality",
                "Improved mobile timeline view in dark mode, full-width vertical milestone cards "
                "with soft rounded rectangle shapes in dark charcoal, thin teal top-border accent "
                "on each card, subtle drop shadows, a frosted glass bottom sheet sliding up with "
                "three icon action buttons, the main timeline fully visible without any overlapping panel",
                "Redesigned AI suggestion panel as a narrow collapsible sidebar on the right of a "
                "dark project dashboard, three stacked suggestion cards with glowing teal left accent "
                "borders, small circular confidence indicator icons, smooth shadows, a collapse chevron "
                "at the top, the main kanban board fully visible and unobstructed to the left",
                "Mobile app dashboard in light mode with a bottom tab navigation bar showing four "
                "icon tabs, full-width content cards stacked vertically showing circular progress "
                "rings in teal and blue, a prominent rounded create button in blue at the top right, "
                "clean white background with subtle card shadows",
                "Desktop task detail view as a right-side slide-in panel over a dimmed kanban board, "
                "the panel in white with a thin teal top accent bar, a large circular avatar at the top, "
                "a horizontal progress bar in teal below it, three coloured priority badge circles "
                "in a row, a section of stacked collaborator avatars with green online indicators",
            ]

        for i, img_prompt in enumerate(image_prompts, start=1):
            enhanced = flux_enhance_prompt(img_prompt)
            with st.spinner(f"Generating AI illustration {i}/{len(image_prompts)}..."):
                img = generate_image(enhanced)

            if img:
                b64 = pil_image_to_base64(img)
                md_content += (
                    f'\n\n<div style="page-break-inside:avoid; break-inside:avoid; '
                    f'text-align:center; margin:24px 0;">\n'
                    f'<h3>AI Illustration {i}</h3>\n'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="max-width:90%; height:auto; display:block; '
                    f'margin:12px auto; border:1px solid #ddd;">\n'
                    f'</div>\n\n'
                )

    # Markdown → HTML → PDF
    html_body = markdown.markdown(
        md_content,
        extensions=[
            'extra',
            'sane_lists',
            'nl2br',
            'toc',
        ],
        extension_configs={
            'toc': {'permalink': False}
        }
    )

    full_html = build_pdf_html(html_body)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.set_content(full_html, wait_until="load")
        page.wait_for_timeout(800)

        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            prefer_css_page_size=False,
            margin={
                "top": "1.5cm",
                "bottom": "1.5cm",
                "left": "1.8cm",
                "right": "1.8cm"
            },
            display_header_footer=True,
            header_template='<div style="font-size:8px; color:#999; width:100%; text-align:right; padding-right:1.8cm;"></div>',
            footer_template='<div style="font-size:8px; color:#999; width:100%; text-align:center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
        )

        browser.close()

    return pdf_bytes


# =============================================================================
# UI
# =============================================================================
st.set_page_config(
    page_title="VizRAG AI",
    page_icon="📊",
    layout="wide"
)

st.title("VizRAG AI — Multi-Modal Document Insights")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("stored_images", [])

with st.sidebar:
    st.header("📁 Upload Documents")
    pdf_files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    image_files = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Session"):
        st.session_state.vectorstore = None
        st.session_state.stored_images = []
        st.session_state.messages = []
        st.success("Cleared")
        st.rerun()

with col2:
    if st.button("📥 Index Docs"):
        if pdf_files:
            with st.spinner("Indexing PDFs..."):
                st.session_state.vectorstore = index_docs(pdf_files)
            st.success(f"✅ {len(pdf_files)} PDFs indexed")
        else:
            st.warning("No PDFs uploaded")

        if image_files:
            st.session_state.stored_images = store_uploaded_images(image_files)
            st.success(f"✅ {len(image_files)} images loaded")

st.divider()
st.header("📄 Generate Deck")

deck_query = st.text_area(
    "Deck Query",
    value="",
    placeholder="Describe the PDF you want to generate...",
    height=120
)

generate_images_flag = st.checkbox(
    "Generate AI illustrations (Flux)",
    value=True
)

if st.button("🧪 Test fal.ai"):
    with st.spinner("Testing fal.ai connection..."):
        test_img = generate_image(
            flux_enhance_prompt(
                "A clean modern product dashboard with a dark sidebar and card grid layout"
            )
        )
        if test_img:
            st.success("✅ fal.ai working")
            st.image(test_img, width=200)

if st.button("🚀 Generate PDF Deck", type="primary"):
    if not st.session_state.vectorstore:
        st.error("Index documents first.")
    elif not deck_query.strip():
        st.error("Enter a deck query.")
    else:
        with st.spinner("Generating deck — this may take a few minutes..."):
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 8}
                )
                pdf_bytes = generate_deck(
                    deck_query.strip(),
                    st.session_state.stored_images,
                    retriever,
                    generate_images_flag
                )
                safe_name = re.sub(
                    r'[^a-zA-Z0-9]+', '_',
                    deck_query.strip()[:40]
                ).strip('_') or "insights"
                filename = f"{safe_name}_deck.pdf"

                st.success("✅ PDF generated!")
                st.download_button(
                    "⬇️ Download PDF",
                    pdf_bytes,
                    filename,
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.exception(e)

# =============================================================================
# Chat Interface
# =============================================================================

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents and images..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            context = format_docs(retriever.invoke(prompt))
            text = (
                f"Context from documents:\n{context}\n\n"
                f"Question: {prompt}\n\nAnswer:"
            )
            if st.session_state.stored_images:
                content = build_vision_content(text, st.session_state.stored_images)
                response = llm.invoke([HumanMessage(content=content)]).content
            else:
                response = llm.invoke(text).content
        else:
            if st.session_state.stored_images:
                content = build_vision_content(prompt, st.session_state.stored_images)
                response = llm.invoke([HumanMessage(content=content)]).content
            else:
                response = llm.invoke(prompt).content

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
