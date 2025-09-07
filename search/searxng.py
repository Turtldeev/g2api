# Original SearX-ng 
import os
import aiohttp
import asyncio
from typing import List, Dict, Any, AsyncGenerator

class SearXNG:
    url = "https://search.rhscz.eu/"
    label = "SearXNG"
  
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        prompt: str = None,
        proxy: str = None,
        timeout: int = 30,
        language: str = "it",
        max_results: int = 5,
        max_words: int = 2500,
        add_text: bool = True,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        # Get the last user message as prompt if not provided
        if prompt is None:
            for message in reversed(messages):
                if message.get("role") == "user":
                    if isinstance(message.get("content"), str):
                        prompt = message["content"]
                        break
            if prompt is None:
                prompt = ""

        connector = None
        if proxy:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(proxy)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=connector
        ) as session:
            params = {
                "q": prompt,
                "format": "json",
                "language": language,
                "safesearch": 0,
                "categories": "general",
            }
            
            async with session.get(f"{cls.url}/search", params=params, proxy=proxy) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: Failed to fetch search results")
                
                data = await resp.json()
                results = data.get("results", [])
                if not results:
                    return

                if add_text:
                    async def fetch_content(url: str, word_limit: int):
                        try:
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as page_resp:
                                if page_resp.status != 200:
                                    return ""
                                text = await page_resp.text()
                                # Simple text extraction (you might want to use a proper HTML parser)
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(text, 'html.parser')
                                for script in soup(["script", "style"]):
                                    script.decompose()
                                content = soup.get_text()
                                words = content.split()
                                return " ".join(words[:word_limit])
                        except Exception:
                            return ""

                    requests = []
                    word_limit_per_result = int(max_words / max_results)
                    for r in results[:max_results]:
                        requests.append(fetch_content(r["url"], word_limit_per_result))
                    
                    texts = await asyncio.gather(*requests)
                    for i, r in enumerate(results[:max_results]):
                        r["text"] = texts[i]

                formatted = ""
                used_words = 0
                for i, r in enumerate(results[:max_results]):
                    title = r.get("title", "No title")
                    url = r.get("url", "#")
                    content = r.get("text") or r.get("snippet") or ""
                    formatted += f"Title: {title}\n\n{content}\n\nSource: [[{i}]]({url})\n\n"
                    used_words += len(content.split())
                    if max_words and used_words >= max_words:
                        break

                if formatted.strip():
                    yield formatted.strip()
                
                yield {"type": "finish_reason", "reason": "stop"}
