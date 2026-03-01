# VizRAG AI — Multi-Modal Document Insights Engine

**A powerful single-file Streamlit application that turns any PDFs + images into professional, visually rich insights decks using Claude + Flux + Mermaid.**

VizRAG AI lets you upload any PDFs and images, chat with them using full vision capabilities, and instantly generate beautiful, consultant-grade PDF reports that follow your exact instructions. 

Whether you need a polished career summary with timeline, a detailed product evaluation with design critique and improved UI illustrations, a contract risk analysis, a scientific paper review, or any other type of insights report — simply type your prompt and receive a complete, multi-page PDF with embedded Mermaid diagrams and Flux-generated illustrations.

Built as a high-impact portfolio and Upwork showcase project.

## ✨ Key Features

- Full multi-modal RAG with vision (chat naturally with PDFs + images)
- Professional PDF deck generation that follows your prompt 100%
- Hybrid visuals: Clean Mermaid diagrams & graphical timelines + photorealistic Flux illustrations of improved UI/mockups (only when requested)
- No truncation — the deck is as long as needed to fully satisfy your request (5, 6, 7+ pages)
- Completely generic — works for career summaries, product insights, contract analysis, scientific papers, or anything else
- Robust on Windows with reliable Mermaid embedding and minimal white space
- Zero ongoing infrastructure cost (everything runs locally except API calls)

## 🛠 Tech Stack

- **Framework**: Streamlit (single-file Python app)
- **LLM**: Anthropic Claude Sonnet-4-5 (16k context, native vision support)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (lightweight, in-memory)
- **Document Processing**: PyPDFLoader + RecursiveCharacterTextSplitter
- **PDF Generation**: Playwright (headless Chromium) with custom tight CSS
- **Diagrams & Timelines**: Mermaid.js with robust custom renderer
- **AI Illustrations**: fal.ai Flux Schnell with smart prompt enhancement
- **Language**: Python 3.12

## 📋 Full Installation Instructions (Windows)

1. Install **Python 3.12** from the official python.org website. During installation, check the box “Add python.exe to PATH”.

2. Open Command Prompt and navigate to your project folder:
   ```bash
   cd "E:\Upwork2\__DEMO_AND_PORTFOLIO_PROJECTS__\VizRAG_AI\src"
   ```

3. Create a fresh virtual environment:
   ```bash
   py -3.12 -m venv venv
   ```

4. Activate the environment:
   ```bash
   venv\Scripts\activate
   ```

5. Upgrade pip and install all dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install streamlit langchain==0.3.1 langchain-anthropic==0.2.3 langchain-community==0.3.1 langchain-huggingface chromadb pypdf sentence-transformers fal-client pillow python-dotenv playwright markdown
   playwright install
   pip install faiss-cpu
   ```

6. Create a `.env` file in the same folder and add your API keys:
   ```env
   ANTHROPIC_API_KEY=sk-ant-...
   FAL_KEY=your_fal_ai_key_here
   ```

## 🚀 How to Run

Activate the venv and start the app:
```bash
venv\Scripts\activate
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

## 📖 How to Use

1. In the sidebar, upload your PDFs and any supporting images.
2. Click **Index Documents**.
3. In the **Deck Query** box, type exactly what you want the PDF to be. Examples:
   - "Create a professional product insights deck that evaluates the uploaded specs and mockups of TaskFlow AI, identifies design and usability issues, suggests concrete improvements, and includes a graphical timeline for launch with relevant illustrations of the improved UI."
   - "Create a professional career summary with timeline and realistic portraits."
4. Check "Generate AI illustrations" if you want Flux visuals.
5. Click **Generate PDF Deck** and download the result.

The system will produce a complete, dense, multi-page PDF that fully satisfies your request with no truncation, clean Mermaid timelines when requested, and relevant Flux illustrations only where appropriate.

## 💰 Costs

- Claude API: typically $0.05 – $0.30 per full deck
- Flux images: $0.01 – $0.04 per image
- **Typical total cost per deck**: 15–50 cents

Local processing (FAISS, embeddings) is completely free.

## 📌 Limitations

- The vector store is in-memory and resets when you close the app
- No user authentication or multi-user support yet
- Best PDF quality is achieved with decks up to around 10–12 pages (longer decks still work but may render slower)

## 📄 License

MIT License — free to use for personal, portfolio, or commercial demonstration purposes.

---

**Made by Fotios Basagiannis**

A clean demonstration of modern multi-modal AI with vision, structured generation, and professional PDF output.
```
