import os
import shutil
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# AEGIS Core Modules
from core.logic import logic_engine
from core.ears import aegis_ears
from core.memory import aegis_memory  # Import the memory manager
from core.vision import aegis_eyes  # Import the new VisionManager instance

app = FastAPI(
    title="AEGIS Sidecar",
    description="Local Inference Engine for AEGIS v2.0",
    version="2.0.0",
)

# CORS configuration for Tauri/Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthStatus(BaseModel):
    status: str
    version: str
    engine: str


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Confirms the Python process and all perception engines are active."""
    return HealthStatus(
        status="active",
        version="2.0.0",
        engine="Gemma-2B + Faster-Whisper + LanceDB + Moondream2",  # Added Moondream2
    )


@app.get("/chat")
async def chat_stream(prompt: str = Query(..., min_length=1)):
    """Streaming Inference Engine (Brain) with RAG."""
    return StreamingResponse(logic_engine.chat_stream(prompt), media_type="text/plain")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    AEGIS Perception Endpoint: Ears.
    Receives audio, transcribes locally, and returns text.
    """
    temp_path = f"stream_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result_text = aegis_ears.transcribe(temp_path)
        return {"text": result_text}
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    AEGIS Memory Endpoint.
    Receives a PDF, extracts text, embeds it, and stores it in LanceDB.
    """
    if not file.filename.endswith(".pdf"):  # type: ignore
        return {"error": "Currently, AEGIS only supports .pdf ingestion."}

    temp_path = f"temp_doc_{file.filename}"

    try:
        # Stream file to disk to protect RAM
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger the ingestion process
        result = aegis_memory.ingest_pdf(temp_path, file.filename)  # type: ignore
        return {"status": "success", "message": result}

    except Exception as e:
        return {"error": f"Ingestion failed: {str(e)}"}

    finally:
        # Strict Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/analyze-image")
async def analyze_image(prompt: str = Query(...), file: UploadFile = File(...)):
    """
    AEGIS Vision Endpoint: Eyes.
    Receives an image and a text prompt, returns the VLM's analysis via Moondream2.
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):  # type: ignore
        return {"error": "Unsupported image format. Please use PNG, JPG, or WEBP."}

    temp_path = f"temp_vision_{file.filename}"

    try:
        # Stream file to disk to protect RAM
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger the Moondream engine
        result = aegis_eyes.analyze(temp_path, prompt)
        return {"status": "success", "text": result}

    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}

    finally:
        # Strict Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    # Launching the Sidecar on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
