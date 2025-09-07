# not working without auth
# 100% original model

"""
Kimi AI Chat Client
A standalone Python client for interacting with Kimi's chat service (kimi.com)
Mimics the qwenapi style for easy usage.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
import aiohttp
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional, Union


class KimiError(Exception):
    """Base exception for Kimi client errors"""
    pass


class RateLimitError(KimiError):
    """Raised when rate limit is exceeded"""
    pass


class AuthenticationError(KimiError):
    """Raised when authentication fails"""
    pass


class AnonymousLimitError(KimiError):
    """Raised when anonymous chat limit is exceeded"""
    pass


@dataclass
class Message:
    """Represents a chat message"""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from chat completion"""
    content: str
    title: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class StreamChunk:
    """Chunk from streaming response"""
    content: str
    title: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass
class AuthResult:
    """Authentication result"""
    api_key: str
    device_id: str


@dataclass
class Conversation:
    """Conversation state"""
    chat_id: str


class KimiClient:
    """
    Kimi AI Chat Client
    
    A client for interacting with Kimi's chat service that mimics the qwenapi style.
    """
    
    BASE_URL = "https://www.kimi.com"
    DEVICE_REGISTER_URL = "https://www.kimi.com/api/device/register"
    CHAT_CREATE_URL = "https://www.kimi.com/api/chat"
    
    DEFAULT_MODEL = "kimi-k2"
    MODELS = ["kimi-k2"]
    
    def __init__(self, 
                 model: str = None,
                 timeout: int = 120,
                 proxy: str = None,
                 debug: bool = False):
        """
        Initialize Kimi client
        
        Args:
            model: Model to use (default: kimi-k2)
            timeout: Request timeout in seconds
            proxy: Proxy URL
            debug: Enable debug logging
        """
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.proxy = proxy
        self.debug = debug
        
        # Internal state
        self._auth_result: Optional[AuthResult] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
    def _log(self, message: str):
        """Log debug message"""
        if self.debug:
            print(f"[KimiClient] {message}")
    
    def _get_headers(self, include_auth: bool = False, device_id: str = None) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': self.BASE_URL,
            'Referer': f'{self.BASE_URL}/',
            'Content-Type': 'application/json',
        }
        
        if device_id:
            headers.update({
                'x-msh-device-id': device_id,
                'x-msh-platform': 'web',
                'x-traffic-id': device_id
            })
        
        if include_auth and self._auth_result:
            headers['Authorization'] = f'Bearer {self._auth_result.api_key}'
            
        return headers
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def _authenticate(self):
        """Get API key by registering device"""
        if self._auth_result:
            return
            
        self._log("Registering device and getting access token...")
        device_id = str(random.randint(1000000000000000, 9999999999999999))
        session = await self._get_session()
        
        async with session.post(
            self.DEVICE_REGISTER_URL,
            json={},
            headers=self._get_headers(device_id=device_id),
            proxy=self.proxy
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise AuthenticationError(f"Device registration failed with status {response.status}: {error_text}")
            
            data = await response.json()
            access_token = data.get("access_token")
            
            if not access_token:
                raise AuthenticationError("No access token received from device registration")
                
            self._auth_result = AuthResult(
                api_key=access_token,
                device_id=device_id
            )
                
        self._log(f"Device registered successfully. Token: {access_token[:20]}...")
    
    async def get_models(self) -> List[str]:
        """Get available models"""
        return self.MODELS.copy()
    
    async def _create_conversation(self) -> Conversation:
        """Create a new conversation"""
        await self._authenticate()
        
        self._log("Creating new conversation...")
        session = await self._get_session()
        
        payload = {
            "name": "未命名会话",
            "born_from": "home",
            "kimiplus_id": "kimi",
            "is_example": False,
            "source": "web",
            "tags": []
        }
        
        async with session.post(
            self.CHAT_CREATE_URL,
            json=payload,
            headers=self._get_headers(include_auth=True),
            proxy=self.proxy
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                if "匿名聊天使用次数超过" in error_text:
                    raise AnonymousLimitError("Anonymous chat usage limit exceeded")
                raise KimiError(f"Failed to create conversation with status {response.status}: {error_text}")
            
            data = await response.json()
            chat_id = data.get("id")
            
            if not chat_id:
                raise KimiError("No chat ID received from conversation creation")
                
            self._log(f"Conversation created with ID: {chat_id}")
            return Conversation(chat_id=chat_id)
    
    async def _parse_sse_stream(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict[str, Any], None]:
        """Parse SSE stream"""
        buffer = ""
        
        async for line in response.content:
            try:
                line = line.decode('utf-8')
                buffer += line
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str and data_str != "[DONE]":
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                self._log(f"Failed to decode JSON: {data_str}")
                                continue
            except UnicodeDecodeError:
                self._log("Failed to decode line")
                continue
    
    async def chat_completion(self, 
                            messages: Union[List[Message], List[Dict[str, str]]], 
                            model: str = None,
                            stream: bool = False,
                            web_search: bool = False,
                            conversation: Conversation = None,
                            **kwargs) -> Union[ChatResponse, AsyncGenerator[StreamChunk, None]]:
        """
        Create a chat completion
        
        Args:
            messages: List of messages
            model: Model to use (optional, uses default if not specified)
            stream: Whether to stream the response
            web_search: Enable web search
            conversation: Existing conversation to continue
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse if not streaming, AsyncGenerator[StreamChunk] if streaming
        """
        await self._authenticate()
        
        # Convert Message objects to dicts if needed
        if messages and isinstance(messages[0], Message):
            messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Get last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg["content"]
                break
        
        if not user_message:
            raise KimiError("No user message found in messages")
        
        # Create conversation if not provided
        if conversation is None:
            conversation = await self._create_conversation()
        
        data = {
            "kimiplus_id": "kimi",
            "extend": {"sidebar": True},
            "model": "k2",
            "use_search": web_search,
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "refs": [],
            "history": [],
            "scene_labels": [],
            "use_semantic_memory": False,
            "use_deep_research": False
        }
        
        if stream:
            return self._stream_completion(data, conversation)
        else:
            return await self._complete_completion(data, conversation)
    
    async def _stream_completion(self, data: Dict[str, Any], conversation: Conversation) -> AsyncGenerator[StreamChunk, None]:
        """Handle streaming completion"""
        session = await self._get_session()
        
        try:
            completion_url = f"{self.BASE_URL}/api/chat/{conversation.chat_id}/completion/stream"
            
            async with session.post(
                completion_url,
                json=data,
                headers=self._get_headers(include_auth=True),
                proxy=self.proxy
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise KimiError(f"API request failed with status {response.status}: {error_text}")
                
                async for chunk in self._parse_sse_stream(response):
                    try:
                        event = chunk.get("event")
                        
                        if event == "cmpl":
                            text = chunk.get("text", "")
                            if text:
                                yield StreamChunk(content=text)
                        
                        elif event == "rename":
                            title = chunk.get("text", "")
                            if title:
                                yield StreamChunk(content="", title=title)
                        
                        elif event == "all_done":
                            yield StreamChunk(content="", finish_reason="stop")
                            break
                            
                    except Exception as e:
                        self._log(f"Error processing chunk: {e}")
                        continue
                        
        except Exception as e:
            self._log(f"Stream completion error: {e}")
            raise KimiError(f"Failed to complete request: {e}")
    
    async def _complete_completion(self, data: Dict[str, Any], conversation: Conversation) -> ChatResponse:
        """Handle non-streaming completion"""
        content = ""
        title = None
        
        try:
            async for chunk in self._stream_completion(data, conversation):
                if chunk.content:
                    content += chunk.content
                if chunk.title:
                    title = chunk.title
                if chunk.finish_reason == "stop":
                    break
        except Exception as e:
            self._log(f"Error in _complete_completion: {e}")
            raise
        
        return ChatResponse(
            content=content,
            title=title
        )
    
    async def chat(self, 
                  message: str, 
                  model: str = None,
                  web_search: bool = False,
                  **kwargs) -> str:
        """
        Simple chat method - send a message and get response content
        
        Args:
            message: User message
            model: Model to use
            web_search: Enable web search
            **kwargs: Additional parameters
            
        Returns:
            Response content as string
        """
        messages = [{"role": "user", "content": message}]
        response = await self.chat_completion(messages, model=model, web_search=web_search, **kwargs)
        return response.content
    
    async def chat_stream(self, 
                         message: str, 
                         model: str = None,
                         web_search: bool = False,
                         **kwargs) -> AsyncGenerator[str, None]:
        """
        Simple streaming chat method
        
        Args:
            message: User message  
            model: Model to use
            web_search: Enable web search
            **kwargs: Additional parameters
            
        Yields:
            Response content chunks
        """
        messages = [{"role": "user", "content": message}]
        stream_gen = await self.chat_completion(messages, model=model, stream=True, web_search=web_search, **kwargs)
        async for chunk in stream_gen:
            if chunk.content:
                yield chunk.content
    
    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Convenience functions for qwenapi-style usage
async def chat(message: str, model: str = None, web_search: bool = False, debug: bool = False, **kwargs) -> str:
    """
    Simple chat function
    
    Args:
        message: User message
        model: Model to use
        web_search: Enable web search
        debug: Enable debug logging
        **kwargs: Additional parameters
        
    Returns:
        Response content
    """
    async with KimiClient(model=model, debug=debug, **kwargs) as client:
        return await client.chat(message, web_search=web_search)


async def chat_stream(message: str, model: str = None, web_search: bool = False, debug: bool = False, **kwargs) -> AsyncGenerator[str, None]:
    """
    Simple streaming chat function
    
    Args:
        message: User message
        model: Model to use
        web_search: Enable web search
        debug: Enable debug logging  
        **kwargs: Additional parameters
        
    Yields:
        Response content chunks
    """
    async with KimiClient(model=model, debug=debug, **kwargs) as client:
        stream_gen = client.chat_stream(message, web_search=web_search)
        async for chunk in stream_gen:
            yield chunk


# Example usage
if __name__ == "__main__":
    async def main():
        try:
            # Simple usage with debug
            print("Testing simple chat...")
            response = await chat("Hello, how are you?", debug=True)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error in simple chat: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            # Streaming usage
            print("\nTesting streaming response...")
            async for chunk in chat_stream("Tell me a short story", debug=True):
                print(chunk, end="", flush=True)
            print()
            
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Advanced usage with client
            print("\nTesting advanced usage...")
            async with KimiClient(debug=True) as client:
                # Get available models
                models = await client.get_models()
                print(f"\nAvailable models: {models}")
                
                # Chat with web search
                messages = [
                    {"role": "user", "content": "What are the latest developments in AI?"}
                ]
                
                response = await client.chat_completion(
                    messages=messages,
                    model="kimi-k2",
                    web_search=True
                )
                
                print(f"\nResponse: {response.content}")
                if response.title:
                    print(f"Title: {response.title}")
                    
        except Exception as e:
            print(f"Error in advanced usage: {e}")
            import traceback
            traceback.print_exc()
    
    # Run example
    asyncio.run(main())
