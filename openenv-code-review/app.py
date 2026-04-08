"""
Hugging Face Spaces entrypoint.
HF Spaces expects a file named app.py and sets PORT env var.
This just delegates to the FastAPI server.
"""
import os
import uvicorn
from server import app  # noqa: F401 — imported for HF auto-discovery

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)
