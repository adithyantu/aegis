from langchain_ollama import OllamaLLM
from typing import AsyncGenerator
import asyncio
import re

# Import AEGIS Perception Modules
from core.memory import aegis_memory  # Hippocampus (Permanent Memory)
from core.web import aegis_web  # Web Scraping Engine


class LogicManager:
    """
    Handles local LLM orchestration with professional persona steering.
    Now equipped with Retrieval-Augmented Generation (RAG) and Web Intelligence.
    """

    def __init__(self):
        # System instructions optimized for dynamic context injection
        self.system_instructions = (
            "You are AEGIS v2.0, a highly intelligent, professional AI assistant. "
            "Use the provided context (Knowledge Base or Live Web Data) to answer the user's question accurately. "
            "If the context does not contain the answer, rely on your general knowledge. "
            "Be concise, highly accurate, and use Markdown for formatting."
        )
        self.llm = OllamaLLM(
            model="gemma2:2b",
            temperature=0.3,  # Low temperature for factual grounding
            num_ctx=4096,
        )

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        context_block = ""

        # 1. WEB PERCEPTION: Check if the user pasted a URL
        url_pattern = re.compile(r"https?://[^\s]+")
        urls = url_pattern.findall(user_input)

        if urls:
            # If a URL is found, scrape the first one to inject live data
            print(f"Web Engine triggering for URL: {urls[0]}")
            web_data = await aegis_web.scrape(urls[0])
            context_block += (
                f"\n\n### LIVE WEB DATA ###\n[Source: {urls[0]}]\n{web_data}\n"
            )

        else:
            # 2. MEMORY PERCEPTION: Fallback to RAG if no URL is provided
            # This prevents context window overflow (keeping us under the 4096 token limit)
            try:
                memory_data = aegis_memory.search(user_input)
                if memory_data:
                    context_block += f"\n\n### RELEVANT KNOWLEDGE ###\n{memory_data}\n"
            except Exception as e:
                print(f"Warning: Memory search failed - {e}")

        # 3. Build the Master Prompt
        full_prompt = f"{self.system_instructions}{context_block}\n\nUser: {user_input}\nAssistant:"

        try:
            # 4. Stream the Response for sub-200ms perceived latency
            for chunk in self.llm.stream(full_prompt):
                yield chunk
                await asyncio.sleep(0)
        except Exception as e:
            yield f"**System Error:** {str(e)}"


logic_engine = LogicManager()
