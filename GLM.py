# GLM
# 100% original

"""
GLM AI Chat Client
A standalone Python client for interacting with GLM's chat service (chat.z.ai)
Mimics the qwenapi style for easy usage.
"""

from __future__ import annotations

import asyncio
import json
import uuid
import aiohttp
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional, Union


class GLMError(Exception):
    """Base exception for GLM client errors"""
    pass


class RateLimitError(GLMError):
    """Raised when rate limit is exceeded"""
    pass


class AuthenticationError(GLMError):
    """Raised when authentication fails"""
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
    reasoning: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class StreamChunk:
    """Chunk from streaming response"""
    content: str
    is_reasoning: bool = False
    usage: Optional[Dict[str, int]] = None


@dataclass
class Usage:
    """Token usage information"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'Usage':
        """Create Usage from API response data"""
        return cls(
            input_tokens=data.get('prompt_tokens', data.get('input_tokens', 0)),
            output_tokens=data.get('completion_tokens', data.get('output_tokens', 0)),
            total_tokens=data.get('total_tokens', 0)
        )


@dataclass
class Reasoning:
    """Reasoning/thinking content"""
    content: str


class GLMClient:
    """
    GLM AI Chat Client
    
    A client for interacting with GLM's chat service that mimics the qwenapi style.
    """
    
    BASE_URL = "https://chat.z.ai"
    API_ENDPOINT = "https://chat.z.ai/api/chat/completions"
    AUTH_ENDPOINT = "https://chat.z.ai/api/v1/auths/"
    MODELS_ENDPOINT = "https://chat.z.ai/api/models"
    
    DEFAULT_MODEL = "GLM-4.5"
    
    def __init__(self, 
                 model: str = None,
                 timeout: int = 120,
                 proxy: str = None,
                 debug: bool = False):
        """
        Initialize GLM client
        
        Args:
            model: Model to use (default: GLM-4.5)
            timeout: Request timeout in seconds
            proxy: Proxy URL
            debug: Enable debug logging
        """
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.proxy = proxy
        self.debug = debug
        
        # Internal state
        self._api_key: Optional[str] = None
        self._models: List[str] = []
        self._model_aliases: Dict[str, str] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
    def _log(self, message: str):
        """Log debug message"""
        if self.debug:
            print(f"[GLMClient] {message}")
    
    def _get_headers(self, include_auth: bool = False) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': self.BASE_URL,
            'Referer': f'{self.BASE_URL}/',
            'Content-Type': 'application/json',
            'x-fe-version': 'prod-fe-1.0.57'
        }
        
        if include_auth and self._api_key:
            headers['Authorization'] = f'Bearer {self._api_key}'
            
        return headers
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def _authenticate(self):
        """Get API key"""
        if self._api_key:
            return
            
        self._log("Getting API key...")
        session = await self._get_session()
        
        async with session.get(
            self.AUTH_ENDPOINT,
            headers=self._get_headers(),
            proxy=self.proxy
        ) as response:
            response.raise_for_status()
            data = await response.json()
            self._api_key = data.get("token")
            
            if not self._api_key:
                raise AuthenticationError("Failed to get API key")
                
        self._log(f"API key obtained: {self._api_key[:10]}...")
    
    async def get_models(self) -> List[str]:
        """Get available models"""
        if self._models:
            return self._models
            
        await self._authenticate()
        
        self._log("Fetching available models...")
        session = await self._get_session()
        
        async with session.get(
            self.MODELS_ENDPOINT,
            headers=self._get_headers(include_auth=True),
            proxy=self.proxy
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            models_data = data.get("data", [])
            self._model_aliases = {
                model_data.get("name", "").replace("任务专用", "ChatGLM"): model_data.get("id")
                for model_data in models_data
            }
            self._models = list(self._model_aliases.keys())
            
        self._log(f"Found {len(self._models)} models: {self._models}")
        return self._models
    
    def _get_model_id(self, model: str) -> str:
        """Get model ID from name"""
        if model in self._model_aliases:
            return self._model_aliases[model]
        return model
    
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
                            enable_thinking: bool = True,
                            **kwargs) -> Union[ChatResponse, AsyncGenerator[StreamChunk, None]]:
        """
        Create a chat completion
        
        Args:
            messages: List of messages
            model: Model to use (optional, uses default if not specified)
            stream: Whether to stream the response
            enable_thinking: Enable reasoning/thinking mode
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse if not streaming, AsyncGenerator[StreamChunk] if streaming
        """
        await self.get_models()  # Ensure we have models and API key
        
        model = model or self.model
        model_id = self._get_model_id(model)
        
        # Convert Message objects to dicts if needed
        if messages and isinstance(messages[0], Message):
            messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        data = {
            "chat_id": "local",
            "id": str(uuid.uuid4()),
            "stream": stream,
            "model": model_id,
            "messages": messages,
            "params": {},
            "tool_servers": [],
            "features": {
                "enable_thinking": enable_thinking
            }
        }
        
        if stream:
            return self._stream_completion(data)
        else:
            return await self._complete_completion(data)
    
    async def _stream_completion(self, data: Dict[str, Any]) -> AsyncGenerator[StreamChunk, None]:
        """Handle streaming completion"""
        session = await self._get_session()
        
        try:
            async with session.post(
                self.API_ENDPOINT,
                json=data,
                headers=self._get_headers(include_auth=True),
                proxy=self.proxy
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise GLMError(f"API request failed with status {response.status}: {error_text}")
                
                usage_processed = False
                async for chunk in self._parse_sse_stream(response):
                    try:
                        if chunk.get("type") == "chat:completion":
                            # Handle usage info
                            if not usage_processed:
                                usage_data = chunk.get("data", {}).get("usage")
                                if usage_data:
                                    self._log(f"Raw usage data: {usage_data}")
                                    # Convert to standardized format
                                    usage_dict = {
                                        'input_tokens': usage_data.get('prompt_tokens', usage_data.get('input_tokens', 0)),
                                        'output_tokens': usage_data.get('completion_tokens', usage_data.get('output_tokens', 0)),
                                        'total_tokens': usage_data.get('total_tokens', 0)
                                    }
                                    usage_processed = True
                                    yield StreamChunk("", usage=usage_dict)
                            
                            # Handle thinking/reasoning content
                            if chunk.get("data", {}).get("phase") == "thinking":
                                delta_content = chunk.get("data", {}).get("delta_content", "")
                                delta_content = delta_content.split("</summary>\n>")[-1] if delta_content else ""
                                if delta_content:
                                    yield StreamChunk(delta_content, is_reasoning=True)
                            
                            # Handle regular content
                            else:
                                edit_content = chunk.get("data", {}).get("edit_content")
                                if edit_content:
                                    content = edit_content.split("\n</details>\n")[-1]
                                    if content:
                                        yield StreamChunk(content)
                                else:
                                    delta_content = chunk.get("data", {}).get("delta_content")
                                    if delta_content:
                                        yield StreamChunk(delta_content)
                    except Exception as e:
                        self._log(f"Error processing chunk: {e}")
                        continue
        except Exception as e:
            self._log(f"Stream completion error: {e}")
            raise GLMError(f"Failed to complete request: {e}")
    
    async def _complete_completion(self, data: Dict[str, Any]) -> ChatResponse:
        """Handle non-streaming completion"""
        data["stream"] = True  # We still use streaming internally for consistency
        
        content = ""
        reasoning = ""
        usage = None
        
        try:
            async for chunk in self._stream_completion(data):
                if chunk.usage:
                    usage = chunk.usage
                    self._log(f"Received usage: {usage}")
                elif chunk.is_reasoning:
                    reasoning += chunk.content
                    self._log(f"Reasoning chunk: {chunk.content[:50]}...")
                else:
                    content += chunk.content
                    self._log(f"Content chunk: {chunk.content[:50]}...")
        except Exception as e:
            self._log(f"Error in _complete_completion: {e}")
            raise
        
        return ChatResponse(
            content=content,
            reasoning=reasoning if reasoning else None,
            usage=usage
        )
    
    async def chat(self, 
                  message: str, 
                  model: str = None,
                  **kwargs) -> str:
        """
        Simple chat method - send a message and get response content
        
        Args:
            message: User message
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Response content as string
        """
        messages = [{"role": "user", "content": message}]
        response = await self.chat_completion(messages, model=model, **kwargs)
        return response.content
    
    async def chat_stream(self, 
                         message: str, 
                         model: str = None,
                         **kwargs) -> AsyncGenerator[str, None]:
        """
        Simple streaming chat method
        
        Args:
            message: User message  
            model: Model to use
            **kwargs: Additional parameters
            
        Yields:
            Response content chunks
        """
        messages = [{"role": "user", "content": message}]
        stream_gen = await self.chat_completion(messages, model=model, stream=True, **kwargs)
        async for chunk in stream_gen:
            if not chunk.is_reasoning and chunk.content:
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
async def chat(message: str, model: str = None, debug: bool = False, **kwargs) -> str:
    """
    Simple chat function
    
    Args:
        message: User message
        model: Model to use
        debug: Enable debug logging
        **kwargs: Additional parameters
        
    Returns:
        Response content
    """
    async with GLMClient(model=model, debug=debug, **kwargs) as client:
        return await client.chat(message)


async def chat_stream(message: str, model: str = None, debug: bool = False, **kwargs) -> AsyncGenerator[str, None]:
    """
    Simple streaming chat function
    
    Args:
        message: User message
        model: Model to use
        debug: Enable debug logging  
        **kwargs: Additional parameters
        
    Yields:
        Response content chunks
    """
    async with GLMClient(model=model, debug=debug, **kwargs) as client:
        stream_gen = client.chat_stream(message)
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
            async with GLMClient(debug=True) as client:
                # Get available models
                models = await client.get_models()
                print(f"\nAvailable models: {models}")
                
                # Chat with specific model
                messages = [
                    {"role": "user", "content": "What is artificial intelligence?"}
                ]
                
                response = await client.chat_completion(
                    messages=messages,
                    model="GLM-4.5",
                    enable_thinking=True
                )
                
                print(f"\nResponse: {response.content}")
                if response.reasoning:
                    print(f"Reasoning: {response.reasoning}")
                if response.usage:
                    print(f"Usage: {response.usage}")
                    
        except Exception as e:
            print(f"Error in advanced usage: {e}")
            import traceback
            traceback.print_exc()
    
    # Run example
    asyncio.run(main())
