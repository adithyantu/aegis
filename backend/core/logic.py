from langchain_ollama import OllamaLLM
from typing import AsyncGenerator
import asyncio


class LogicManager:
    """
    Handles local LLM orchestration with professional persona steering.
    Targeting Gemini-level clarity on edge hardware[cite: 74, 84].
    """

    def __init__(self):
        # System instructions to define AEGIS's personality [cite: 68]
        self.system_instructions = (
            "You are AEGIS v2.0, a highly intelligent, professional AI assistant. "
            "Your tone is empathetic, clear, and grounded. "
            "Always use Markdown for formatting: use bolding for emphasis, "
            "bullet points for lists, and clear headings. Be concise but insightful."
        )
        self.llm = OllamaLLM(
            model="gemma2:2b",
            temperature=0.4,  # Lower temperature = more stable, "standard" output
            num_ctx=4096,
        )

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        # We wrap the user prompt with the system instructions
        full_prompt = f"{self.system_instructions}\n\nUser: {user_input}\nAssistant:"

        try:
            asyncio.get_event_loop()  # type: ignore
            # Streaming tokens for sub-200ms perceived latency [cite: 88, 89]
            for chunk in self.llm.stream(full_prompt):
                yield chunk
                await asyncio.sleep(0)
        except Exception as e:
            yield f"**System Error:** {str(e)}"


logic_engine = LogicManager()
