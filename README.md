# 📄 PageWise — Ask Your PDF

Upload any PDF and ask questions in plain English. PageWise finds the relevant sections and gives you a direct answer — no scrolling, no Ctrl+F.

---

## Why I Built This

I was tired of hunting through 80-page documents manually. PageWise lets you treat any PDF like a conversation.

---

## How It Works

1. **Extract** — PyMuPDF pulls all text from the uploaded PDF
2. **Chunk** — Text is split into overlapping 500-word windows so context isn't lost at boundaries
3. **Embed** — Each chunk is embedded using `sentence-transformers` (runs locally, no API cost)
4. **Search** — Cosine similarity finds the top 4 most relevant chunks for your question
5. **Answer** — GPT-3.5 reads only those chunks and answers your question directly

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pagewise-pdf-chat
cd pagewise-pdf-chat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...

# 4. Run
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Stack

- `Streamlit` — UI
- `PyMuPDF` — PDF parsing
- `sentence-transformers` — local embeddings (all-MiniLM-L6-v2)
- `NumPy` — cosine similarity search
- `OpenAI GPT-3.5` — answer generation

---demo:
<video controls src="20260323-1002-19.1153831.mp4" title="Title"></video>