import asyncio
from playwright.async_api import async_playwright


class WebManager:
    """
    AEGIS Perception: Web Intelligence.
    Uses headless Chromium to scrape real-time data.
    """

    async def scrape(self, url: str) -> str:
        async with async_playwright() as p:
            # Launching in headless mode is mandatory for 4GB RAM
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                # Set a strict 10s timeout to prevent system hang
                await page.goto(url, timeout=10000, wait_until="domcontentloaded")

                # Extract the visible text only
                content = await page.inner_text("body")

                # Truncate to 2000 characters to protect Gemma's context window
                return content[:2000].strip()
            except Exception as e:
                return f"Web Access Failed: {str(e)}"
            finally:
                await browser.close()


# THIS IS THE LINE PYTHON IS LOOKING FOR
aegis_web = WebManager()
