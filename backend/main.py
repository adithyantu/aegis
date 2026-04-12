from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# LogicManager handles the LangChain/Ollama connection
from core.logic import logic_engine

app = FastAPI(
    title="AEGIS Sidecar",
    description="Local Inference Engine for AEGIS v2.0",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Authorized for Tauri loopback
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthStatus(BaseModel):
    status: str
    version: str
    engine: str


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Confirms the Python process and LLM bridge are active."""
    return HealthStatus(
        status="active",
        version="2.0.0",
        engine="Gemma-2B-Local",
    )


@app.get("/chat")
async def chat_stream(prompt: str = Query(..., min_length=1)):
    """
    AEGIS Streaming Inference Endpoint.
    Consumes the AsyncGenerator from LogicManager to pipe tokens to the UI.
    """
    # Returns a stream of text chunks to the frontend
    return StreamingResponse(logic_engine.chat_stream(prompt), media_type="text/plain")


if __name__ == "__main__":
    # Standard dev port for sidecar bridge [cite: 12]
    uvicorn.run(app, host="127.0.0.1", port=8000)
