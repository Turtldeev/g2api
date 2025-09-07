# LambdaChat (95% original models), will be deleted 25 September 2025
# ❤️
from __future__ import annotations

import json
import re
import uuid
import random
from aiohttp import ClientSession, FormData

class LambdaChat:
    url = "https://lambda.chat"
    conversation_url = f"{url}/conversation"

    default_model = "deepseek-llama3.3-70b"
    models = [
        default_model,
        "apriel-5b-instruct",
        "hermes-3-llama-3.1-405b-fp8",
        "llama3.3-70b-instruct-fp8",
        "qwen3-32b-fp8",
    ]
    model_aliases = {
        "llama-3.3-70b": "llama3.3-70b-instruct-fp8",
        "qwen-3-32b": "qwen3-32b-fp8"
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: list,
        api_key: str = None, 
        proxy: str = None,
        cookies: dict = None,
        **kwargs
    ):
        model = cls.get_model(model)
        headers = {
            "Origin": cls.url,
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": cls.url,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Priority": "u=1, i",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }
        
        if cookies is None:
            cookies = {
                "hf-chat": str(uuid.uuid4())
            }
        
        async with ClientSession(headers=headers, cookies=cookies) as session:
            data = {"model": model}
            async with session.post(cls.conversation_url, json=data, proxy=proxy) as response:
                if not response.ok:
                    response_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {response_text}")
                conversation_response = await response.json()
                conversation_id = conversation_response["conversationId"]
                
                for cookie_name, cookie in response.cookies.items():
                    cookies[cookie_name] = cookie.value
            
            async with session.get(
                f"{cls.conversation_url}/{conversation_id}/__data.json?x-sveltekit-invalidated=11", 
                proxy=proxy
            ) as response:
                if not response.ok:
                    response_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {response_text}")
                response_text = await response.text()
                
                for cookie_name, cookie in response.cookies.items():
                    cookies[cookie_name] = cookie.value
                
                message_id = None
                try:
                    for line in response_text.splitlines():
                        if not line.strip():
                            continue
                        try:
                            data_json = json.loads(line)
                            if "type" in data_json and data_json["type"] == "data" and "nodes" in data_json:
                                for node in data_json["nodes"]:
                                    if "type" in node and node["type"] == "data" and "data" in node:
                                        for item in node["data"]:
                                            if isinstance(item, dict) and "id" in item and "from" in item and item.get("from") == "system":
                                                message_id = item["id"]
                                                break
                                        if message_id:
                                            break
                        except json.JSONDecodeError:
                            continue
                    
                    if not message_id:
                        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                        uuids = re.findall(uuid_pattern, response_text)
                        if uuids:
                            message_id = uuids[0]
                    
                    if not message_id:
                        raise ValueError("Could not find message ID in response")
                        
                except (IndexError, KeyError, ValueError) as e:
                    raise RuntimeError(f"Failed to parse conversation data: {str(e)}")
            
            user_message = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
            
            form_data = FormData()
            form_data.add_field(
                "data",
                json.dumps({
                    "inputs": user_message, 
                    "id": message_id, 
                    "is_retry": False, 
                    "is_continue": False, 
                    "web_search": False, 
                    "tools": []
                }),
                content_type="application/json"
            )
            
            async with session.post(
                f"{cls.conversation_url}/{conversation_id}", 
                data=form_data,
                proxy=proxy
            ) as response:
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status}: Failed to send message")
                
                async for chunk in response.content:
                    if not chunk:
                        continue
                    chunk_str = chunk.decode('utf-8', errors='ignore')
                    try:
                        data = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue
                        
                    if data.get("type") == "stream" and "token" in data:
                        token = data["token"].replace("\u0000", "")
                        if token:
                            yield token
                    elif data.get("type") == "title":
                        yield {"type": "title", "title": data.get("title", "")}
                    elif data.get("type") == "reasoning":
                        subtype = data.get("subtype")
                        token = data.get("token", "").replace("\u0000", "")
                        status = data.get("status", "")
                        if subtype == "stream" and token:
                            yield {"type": "reasoning", "token": token}
                        elif subtype == "status" and status:
                            yield {"type": "reasoning_status", "status": status}
                    elif data.get("type") == "finalAnswer":
                        yield {"type": "finish_reason", "reason": "stop"}
                        break
                    elif data.get("type") == "status" and data.get("status") == "keepAlive":
                        continue
