# Foreign Whispers — AI Video Dubbing Pipeline

> An open-source, end-to-end video dubbing system that takes a YouTube video, transcribes it, translates it, synthesizes dubbed audio, and produces a final dubbed video — all without any paid API keys.

---

## What It Does

```
YouTube URL → Download → Transcribe → Translate → TTS → Stitch → Dubbed Video
```

- **Download** — Fetches video and captions from YouTube using `yt-dlp`
- **Transcribe** — Converts speech to text using OpenAI Whisper
- **Translate** — Translates English transcript to Spanish using `argostranslate`
- **TTS** — Synthesizes dubbed audio using Chatterbox TTS
- **Stitch** — Combines dubbed audio with original video using `ffmpeg`

---

## Architecture

```
┌──────────────────────────┐
│   Frontend (Next.js)      │  :8501 — Dubbing Studio UI
└────────────┬─────────────┘
             │ HTTP
┌────────────▼─────────────┐
│   API (FastAPI)           │  :8080 — Orchestrates pipeline
└──────────────────────────┘
             │ HTTP (optional GPU services)
┌────────────▼─────────────┐
│   Chatterbox TTS          │  :8020 — Voice synthesis (GPU)
│   Whisper STT             │  :8000 — Speech-to-text (GPU)
└──────────────────────────┘
```

| Layer | Tool | Port |
|-------|------|------|
| Frontend | Next.js Dubbing Studio | 8501 |
| API | FastAPI orchestrator | 8080 |
| TTS | Chatterbox TTS (GPU) | 8020 |
| STT | Whisper STT (GPU) | 8000 |

---

## Requirements

### System Requirements
- **Docker Desktop** (latest version)
- **Python 3.11** (exactly)
- **uv** package manager
- **Git**
- 8GB+ RAM recommended
- NVIDIA GPU (optional, but recommended for better quality)

### For Mac (Apple Silicon) Users
This project was developed and tested on **Apple M4 (16GB RAM)** running macOS. Special configuration is required for Mac — see the Mac Setup section below.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/foreign-whispers.git
cd foreign-whispers
```

### Step 2 — Create the cookies file

```bash
echo '# Netscape HTTP Cookie File' > cookies.txt
```

### Step 3 — Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in any required values (e.g. `HF_TOKEN` for HuggingFace).

### Step 4 — Fix PyTorch for Mac (Apple Silicon only)

If you are on a Mac with Apple Silicon (M1/M2/M3/M4), open `pyproject.toml` and replace the `[tool.uv.sources]` section with:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform != 'darwin'" },
]
torchaudio = [
  { index = "pytorch-cu128", marker = "sys_platform != 'darwin'" },
]
torchvision = [
  { index = "pytorch-cu128", marker = "sys_platform != 'darwin'" },
]
```

Then delete the lock file and re-sync:

```bash
rm uv.lock
uv sync
```

### Step 5 — Install Python dependencies

```bash
uv sync
```

---

## Running the App

### On Linux/Windows with NVIDIA GPU (recommended)

```bash
docker compose --profile nvidia up -d
```

### On Mac or CPU-only machines

```bash
docker compose --profile cpu up -d
```

### Verify everything is running

```bash
docker compose ps
curl http://localhost:8080/healthz
```

You should see `{"status":"ok"}`.

### Open the Dubbing Studio

Go to **http://localhost:8501** in your browser.

---

## Mac-Specific Configuration

Mac users need two additional fixes in `docker-compose.yml`:

### 1. Add port mappings (remove network_mode: host)

The `api` and `frontend` services must use explicit port mappings instead of `network_mode: host`:

```yaml
api:
  ports:
    - "8080:8080"

frontend:
  ports:
    - "8501:8501"
```

### 2. Set correct API URL for frontend

In the `frontend` service, set the build arg:

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      API_URL: http://foreign-whispers-api:8080
  environment:
    - API_URL=http://foreign-whispers-api:8080
```

### 3. Set correct Chatterbox URL for API

In the `api` service environment:

```yaml
api:
  environment:
    - CHATTERBOX_API_URL=http://host.docker.internal:8020
```

---

## Running the Pipeline

### Via the UI (recommended)

1. Open **http://localhost:8501**
2. Select a video from the left sidebar
3. Click **Start Pipeline**
4. Watch each stage complete: Download → Transcribe → Translate → TTS → Stitch
5. Click the **Baseline** tab to watch the dubbed video

### Via curl (command line)

```bash
# P1 — Download
curl -X POST http://localhost:8080/api/download \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=GYQ5yGV_-Oc"}'

# P2 — Transcribe
curl -X POST http://localhost:8080/api/transcribe/GYQ5yGV_-Oc

# P3 — Translate
curl -X POST http://localhost:8080/api/translate/GYQ5yGV_-Oc

# P4 — TTS (with alignment for better quality)
curl -X POST "http://localhost:8080/api/tts/GYQ5yGV_-Oc?config=c-fb1074a&alignment=true"

# P5 — Stitch
curl -X POST "http://localhost:8080/api/stitch/GYQ5yGV_-Oc?config=c-fb1074a"
```

### Output location

Dubbed videos are saved to:
```
pipeline_data/api/dubbed_videos/{config}/{video_title}.mp4
```

---

## Using Google Colab for GPU-accelerated TTS

If you don't have an NVIDIA GPU, you can run the Chatterbox TTS step on Google Colab's free GPU and connect it to your local API via ngrok.

### Step 1 — In Google Colab (T4 GPU runtime)

```python
!pip install chatterbox-tts pyngrok flask

from chatterbox.tts import ChatterboxTTS
from flask import Flask, request, send_file
import threading, io, soundfile as sf, torch

model = ChatterboxTTS.from_pretrained(device="cuda")

app = Flask(__name__)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/v1/audio/speech", methods=["POST"])
def tts():
    text = request.json.get("input", "")
    wav = model.generate(text)
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze().cpu().numpy(), 24000, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")

t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8020, use_reloader=False))
t.daemon = True
t.start()
```

### Step 2 — Expose via ngrok

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
tunnel = ngrok.connect(8020)
print("TTS URL:", tunnel.public_url)
```

### Step 3 — Update your local docker-compose.yml

Set `CHATTERBOX_API_URL` in the `api` service to the ngrok URL:

```yaml
- CHATTERBOX_API_URL=https://YOUR-NGROK-URL.ngrok-free.dev
```

Then restart:

```bash
docker compose --profile cpu down
docker compose --profile cpu up -d
```

---

## Project Structure

```
foreign-whispers/
├── api/                    # FastAPI backend
│   └── src/
│       ├── main.py         # App entrypoint
│       ├── routers/        # API endpoints (download, transcribe, etc.)
│       └── services/       # Business logic
├── foreign_whispers/       # Python library (alignment, TTS backends)
├── frontend/               # Next.js Dubbing Studio UI
├── notebooks/              # Jupyter notebooks for each pipeline stage
├── pipeline_data/          # Generated artifacts (videos, audio, etc.)
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # API container
└── pyproject.toml          # Python dependencies
```

---

## Known Limitations

- **CPU-only TTS**: Without an NVIDIA GPU, the Chatterbox TTS model runs slowly and produces audio with gaps between segments. A GPU is strongly recommended for continuous, high-quality dubbed audio.
- **Mac network_mode**: Docker's `network_mode: host` does not work on Mac. Port mappings must be configured explicitly (see Mac setup above).
- **Spanish only**: The default translation target is Spanish (via `argostranslate`). Other language pairs require installing additional argostranslate models.
- **YouTube rate limits**: Heavy usage may trigger YouTube rate limiting. Use `cookies.txt` with valid YouTube cookies to avoid this.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uv sync` fails on Mac with CUDA torch error | Apply the `pyproject.toml` fix in Step 4 above |
| `cookies.txt is a directory` error | Run `rm -rf cookies.txt && echo '# Netscape HTTP Cookie File' > cookies.txt` |
| Frontend shows blank page | Check `API_URL` build arg is set correctly in docker-compose.yml |
| Port 8080/8501 not accessible on Mac | Replace `network_mode: host` with explicit `ports:` mappings |
| Download fails with Internal Server Error | Check `cookies.txt` exists and has the Netscape header |

---
