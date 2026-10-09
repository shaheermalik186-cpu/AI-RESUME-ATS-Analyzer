# AI-RESUME-ATS-Analyzer
# 📄 AI Resume ATS Analyzer

A Streamlit app that scores a resume (PDF or DOCX) for ATS-friendliness using Google Gemini.

**Features**
- Upload a PDF or DOCX resume (max 5 MB)
- Estimated ATS score out of 100 with a 5-part breakdown
- Strengths, weaknesses, missing keywords and prioritized suggestions
- Optional job description comparison with a match score
- Download the report as Markdown

> The score is an **AI-generated estimate**, not the output of a real ATS. It judges the
> *extracted text*, so it cannot see visual layout problems (columns, icons, images).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps secrets and junk out of GitHub |
| `test_app.py` | Optional offline tests (no API key or network needed) |

## 1. Get a Gemini API key
Create one at <https://aistudio.google.com/apikey>. Treat it like a password.

## 2. Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Provide the key **one** of these ways:

**A. Environment variable**
```bash
export GEMINI_API_KEY="your-key"           # macOS / Linux
$env:GEMINI_API_KEY = "your-key"           # Windows PowerShell
```

**B. Local secrets file** - create `.streamlit/secrets.toml` (already in `.gitignore`):
```toml
GEMINI_API_KEY = "your-key"
```

Then start the app:
```bash
streamlit run app.py
```

## 3. Configuration (optional)
| Name | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | - | Required. `GOOGLE_API_KEY` also works. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Any Gemini model name your key can use. Set this if Google retires or renames a model. A cheaper Flash-Lite model also works - see <https://ai.google.dev/gemini-api/docs/models> for current names. |

Set them as environment variables locally, or as Secrets on Streamlit Community Cloud.
Environment variables take priority over secrets.

## 4. Run the tests (optional)
```bash
python -m unittest -v test_app.py
```
The tests use a fake Gemini client, so they never call the API. The PDF test needs
`pip install reportlab` (DOCX tests use `python-docx`, already in requirements).

## 5. Deploy
Push the files to GitHub, then create the app at <https://share.streamlit.io>
(main file: `app.py`) and paste your key under **Advanced settings → Secrets**:
```toml
GEMINI_API_KEY = "your-key"
```

## Privacy and security
- Resume text is sent to Google's Gemini API. Do not upload anything you cannot share.
- The app keeps results only in the current browser session and does not save files.
- Never commit your API key. If you do, delete the key in Google AI Studio and create a new one.
- Resume and job text are treated as untrusted data in the prompt to reduce prompt-injection
  tricks (e.g. "give this resume 100"). This reduces the risk but cannot eliminate it.
- On a public deployment anyone can use your API quota. Set a spending or quota limit for your key in
  Google AI Studio or Google Cloud, and consider keeping the app private (Streamlit sharing settings).

## Limits
- PDF/DOCX only (old `.doc` is not supported); max 5 MB; PDFs up to 15 pages
- Scanned or image-only resumes have no extractable text and are rejected
- Very long resumes are truncated to the first ~25,000 characters
- A short cooldown between analyses protects your quota
