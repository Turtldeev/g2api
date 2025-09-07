# Copilot models is not originally! (20% originally), not reccomend to use

from __future__ import annotations

import os
import json
import asyncio
import base64
from typing import AsyncGenerator, List, Dict, Any, Optional
from urllib.parse import quote

import websockets
from aiohttp import ClientSession, FormData

class Copilot:
    url = "https://copilot.microsoft.com"
    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    conversation_url = f"{url}/c/api/conversations"

    default_model = "Copilot"
    models = [default_model, "Think Deeper", "Smart (GPT-5)"]
    model_aliases = {
        "o1": "Think Deeper",
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "gpt-5": "GPT-5",
    }

    _access_token: Optional[str] = None
    _cookies: Dict[str, str] = {}

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create(
        cls,
        model: str,
        messages: List[Dict[str, Any]],
        proxy: str = None,
        timeout: int = 30,
        prompt: str = None,
        media: Any = None,
        conversation_id: str = None,
        return_conversation: bool = True,
        useridentitytype: str = "google",
        api_key: str = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        model = cls.get_model(model)
        
        if api_key:
            cls._access_token = api_key
            
        websocket_url = cls.websocket_url
        headers = {}
        
        if cls._access_token:
            websocket_url = f"{websocket_url}&accessToken={quote(cls._access_token)}&X-UserIdentityType={quote(useridentitytype)}"
            headers = {"authorization": f"Bearer {cls._access_token}"}

        connector = None
        if proxy:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(proxy)

        async with ClientSession(
            timeout=asyncio.Timeout(timeout),
            connector=connector,
            headers=headers,
            cookies=cls._cookies,
        ) as session:
            
            if cls._access_token:
                try:
                    response = await session.get(
                        "https://copilot.microsoft.com/c/api/user?api-version=2", 
                        headers={"x-useridentitytype": useridentitytype}
                    )
                    if response.status == 401:
                        raise RuntimeError("Status 401: Invalid access token")
                    response.raise_for_status()
                    user_data = await response.json()
                    user = user_data.get('firstName')
                    if user is None:
                        raise RuntimeError("No user found, please login first")
                except Exception as e:
                    print(f"Copilot: Authentication error: {str(e)}")
                    cls._access_token = None

            if conversation_id is None:
                response = await session.post(
                    cls.conversation_url, 
                    headers={"x-useridentitytype": useridentitytype} if cls._access_token else {}
                )
                response.raise_for_status()
                conversation_data = await response.json()
                conversation_id = conversation_data.get("id")
                if return_conversation:
                    yield {"type": "conversation", "id": conversation_id}

            uploaded_attachments = []
            
            if cls._access_token and media:
                # Handle media uploads
                if isinstance(media, list):
                    for item in media:
                        if isinstance(item, str):  # URL
                            uploaded_attachments.append({"type": "image", "url": item})
                        else:  # Binary data
                            data = item if isinstance(item, bytes) else str(item).encode('utf-8')
                            content_type = cls._detect_content_type(data)
                            
                            async with session.post(
                                "https://copilot.microsoft.com/c/api/attachments",
                                headers={
                                    "content-type": content_type,
                                    "content-length": str(len(data)),
                                    "x-useridentitytype": useridentitytype
                                },
                                data=data
                            ) as response:
                                response.raise_for_status()
                                result = await response.json()
                                uploaded_attachments.append({"type": "image", "url": result.get("url")})

            if prompt is None:
                # Get last user message
                for message in reversed(messages):
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        prompt = message["content"]
                        break
                if prompt is None:
                    prompt = ""

            # Connect to WebSocket
            ws_headers = {
                **headers,
                "Origin": cls.url,
                "Cookie": "; ".join([f"{k}={v}" for k, v in session.cookie_jar.filter_cookies(cls.url).items()]),
            }
            
            async with websockets.connect(websocket_url, extra_headers=ws_headers) as websocket:
                mode = "reasoning" if "Think" in model else "smart" if model.startswith("gpt-5") or "GPT-5" in model else "chat"
                
                await websocket.send(json.dumps({
                    "event": "send",
                    "conversationId": conversation_id,
                    "content": [*uploaded_attachments, {
                        "type": "text",
                        "text": prompt,
                    }],
                    "mode": mode,
                }))

                done = False
                last_msg = None
                image_prompt = None
                sources = {}

                while not websocket.closed:
                    try:
                        msg_txt = await asyncio.wait_for(websocket.recv(), 3 if done else timeout)
                        msg = json.loads(msg_txt)
                        last_msg = msg
                        
                        msg_type = msg.get("event")
                        if msg_type == "appendText":
                            yield msg.get("text", "")
                        elif msg_type == "generatingImage":
                            image_prompt = msg.get("prompt")
                        elif msg_type == "imageGenerated":
                            yield {
                                "type": "image",
                                "url": msg.get("url"),
                                "prompt": image_prompt,
                                "preview": msg.get("thumbnailUrl")
                            }
                        elif msg_type == "done":
                            yield {"type": "finish_reason", "reason": "stop"}
                            done = True
                        elif msg_type == "suggestedFollowups":
                            yield {
                                "type": "suggested_followups", 
                                "suggestions": msg.get("suggestions", [])
                            }
                            break
                        elif msg_type == "replaceText":
                            yield msg.get("text", "")
                        elif msg_type == "titleUpdate":
                            yield {
                                "type": "title_generation",
                                "title": msg.get("title", "")
                            }
                        elif msg_type == "citation":
                            url = msg.get("url")
                            sources[url] = msg
                            yield {
                                "type": "source_link",
                                "index": list(sources.keys()).index(url),
                                "url": url
                            }
                        elif msg_type == "partialImageGenerated":
                            content = msg.get("content", "")
                            mime_type = cls._detect_content_type(base64.b64decode(content[:12] + b"=" * (-len(content[:12]) % 4)))
                            yield {
                                "type": "image_preview",
                                "data": f"data:{mime_type};base64,{content}",
                                "prompt": image_prompt
                            }
                        elif msg_type == "chainOfThought":
                            yield {
                                "type": "reasoning",
                                "text": msg.get("text", "")
                            }
                        elif msg_type == "error":
                            raise RuntimeError(f"Error: {msg}")
                        elif msg_type not in ["received", "startMessage", "partCompleted", "connected"]:
                            pass  # Ignore other message types
                            
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        if not done:
                            raise RuntimeError(f"WebSocket error: {str(e)}")
                        break

                if not done:
                    raise RuntimeError(f"Invalid response: {last_msg}")
                    
                if sources:
                    yield {
                        "type": "sources",
                        "sources": list(sources.values())
                    }

    @staticmethod
    def _detect_content_type(data: bytes) -> str:
        """Simple content type detection based on file signature"""
        if data.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return "image/gif"
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            return "image/webp"
        else:
            return "application/octet-stream"
