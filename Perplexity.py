from __future__ import annotations

import random
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any

import websockets
from aiohttp import ClientSession

class PerplexityLabs:
    url = "https://labs.perplexity.ai"
    api_url = "https://www.perplexity.ai/socket.io/"
    ws_url = "wss://www.perplexity.ai/socket.io/"

    default_model = "r1-1776"
    models = [
        default_model,
        "sonar-pro",
        "sonar",
        "sonar-reasoning",
        "sonar-reasoning-pro",
    ]

    @classmethod
    def get_model(cls, model: str) -> str:
        return model if model in cls.models else cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        headers = {
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
        }
        
        connector = None
        if proxy:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(proxy)

        async with ClientSession(headers=headers, connector=connector) as session:
            t = format(random.getrandbits(32), "08x")
            async with session.get(
                f"{cls.api_url}?EIO=4&transport=polling&t={t}"
            ) as response:
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status}: Failed to get session ID")
                text = await response.text()
            
            if not text.startswith("0"):
                raise RuntimeError("Unexpected response format")
            sid = json.loads(text[1:])["sid"]
            
            post_data = '40{"jwt":"anonymous-ask-user"}'
            async with session.post(
                f"{cls.api_url}?EIO=4&transport=polling&t={t}&sid={sid}",
                data=post_data
            ) as response:
                if not response.ok or await response.text() != "OK":
                    raise RuntimeError("Failed to authenticate WebSocket")

            async with session.get(
                f"{cls.api_url}?EIO=4&transport=polling&t={t}&sid={sid}",
                data=post_data
            ) as response:
                if not response.ok or not (await response.text()).startswith("40"):
                    raise RuntimeError("WebSocket handshake failed")

            ws_headers = {
                "Origin": cls.url,
                "Cookie": "; ".join([f"{k}={v}" for k, v in session.cookie_jar.filter_cookies(cls.url).items()]),
            }

            ws_url = f"{cls.ws_url}?EIO=4&transport=websocket&sid={sid}"
            async with websockets.connect(ws_url, extra_headers=ws_headers) as ws:
                await ws.send("2probe")
                if await ws.recv() != "3probe":
                    raise RuntimeError("WebSocket probe failed")
                await ws.send("5")
                if await ws.recv() != "6":
                    raise RuntimeError("WebSocket upgrade failed")

                format_messages = []
                last_is_assistant = False
                for message in messages:
                    if message["role"] == "assistant":
                        if last_is_assistant:
                            continue
                        last_is_assistant = True
                    else:
                        last_is_assistant = False
                    if isinstance(message["content"], str):
                        format_messages.append({
                            "role": message["role"],
                            "content": message["content"]
                        })

                message_data = {
                    "version": "2.18",
                    "source": "default",
                    "model": cls.get_model(model),
                    "messages": format_messages
                }
                await ws.send("42" + json.dumps(["perplexity_labs", message_data]))

                last_message = 0
                while True:
                    message = await ws.recv()
                    if message == "2":
                        if last_message == 0:
                            raise RuntimeError("Unknown error")
                        await ws.send("3")
                        continue
                    if not message.startswith("42"):
                        continue

                    try:
                        parsed_data = json.loads(message[2:])
                        message_type = parsed_data[0]
                        data = parsed_data[1]

                        if message_type.endswith("_query_progress") and data.get("status") == "failed":
                            error_message = data.get("text", "Unknown API error")
                            raise RuntimeError(f"API Error: {error_message}")

                        if "output" in data:
                            if last_message == 0 and model == cls.default_model:
                                yield "<think>"
                            new_text = data["output"][last_message:]
                            if new_text:
                                yield new_text
                            last_message = len(data["output"])
                            if data["final"]:
                                if data.get("citations"):
                                    yield {"type": "sources", "sources": data["citations"]}
                                yield {"type": "finish_reason", "reason": "stop"}
                                break

                    except Exception as e:
                        if isinstance(e, RuntimeError) and "API Error" in str(e):
                            raise e
                        raise RuntimeError(f"Error processing message: {message}") from e
