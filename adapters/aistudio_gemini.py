import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from fastapi.responses import JSONResponse

from .base import BaseAIAdapter

logger = logging.getLogger("copilot-openai-adapter")

ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum",
    "properties", "required", "items", "minItems", "maxItems"
}

RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.PoolTimeout,
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def clean_json_schema(schema: Any) -> Any:
    """Sanitize JSON schema for Gemini's OpenAPI 3.0 specification."""
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for k, v in schema.items():
        if k in ALLOWED_SCHEMA_KEYS:
            if k == "properties" and isinstance(v, dict):
                cleaned["properties"] = {pk: clean_json_schema(pv) for pk, pv in v.items()}
            elif k == "items":
                cleaned["items"] = clean_json_schema(v) if isinstance(v, dict) else v
            else:
                cleaned[k] = v
    if "type" in cleaned and isinstance(cleaned["type"], str):
        cleaned["type"] = cleaned["type"].upper()
    return cleaned


class AIStudioGeminiAdapter(BaseAIAdapter):
    """
    Dedicated Adapter for Google AI Studio Gemini API.
    Translates OpenAI requests into Gemini REST API calls, supporting:
    - Dynamic Gemini models listing & caching
    - Visual Studio Copilot Agent Function / Tool Calling with JSON Schema sanitization
    - Multi-turn thoughtSignature tracking and caching for Gemini 2.5/3.x models
    - Real-time SSE streaming and non-streaming responses
    - ThinkingConfig optimization for low-latency coding assistance
    """

    def __init__(self):
        self.api_key_env = os.getenv("GEMINI_API_KEY", "").strip()
        self.base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
        self._default_model = os.getenv("DEFAULT_MODEL", "gemini-3.8-flash")
        self.thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

        # Model aliases mapping common OpenAI / Copilot models to Gemini models
        self.model_aliases: Dict[str, str] = {
            "gpt-4o": "gemini-3.8-flash",
            "gpt-4o-mini": "gemini-3.8-flash",
            "gpt-4": "gemini-3.8-flash",
            "gpt-4-turbo": "gemini-3.8-flash",
            "gpt-3.5-turbo": "gemini-3.8-flash",
            "copilot": "gemini-3.8-flash",
            "default": "gemini-3.8-flash",
            "claude-3-5-sonnet": "gemini-3.8-flash",
            "text-embedding-3-small": "gemini-embedding-001",
            "text-embedding-3-large": "gemini-embedding-001",
            "text-embedding-ada-002": "gemini-embedding-001",
        }

        # Cache for models list
        self._cached_models: List[Dict[str, Any]] = []
        self._cache_timestamp: float = 0
        self.cache_ttl = 300  # 5 minutes

        # Cache for preserving thought_signatures on tool calls for multi-turn Gemini 2.5/3.x validation
        self._call_signatures: Dict[str, str] = {}
        self.fallback_thought_signature = (
            "EmkKZwFpFH0Tgww0EVtvrjwSiIUTzdJuHgz8Z9bQQRvg8NAPCDDu99KGgsoDkkslAa0Lv11QSZQ/"
            "QmXb29K1UVg6HF4WlJwE3U/FBDc0U1Oz577uRrFuwhj7myKra6bE21Ticl3YZ3Dr2Sw="
        )

        # Resilient network and timeout configurations
        self.timeout_seconds = float(os.getenv("GEMINI_TIMEOUT", "180.0"))
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

    def _create_client(self, timeout: Optional[httpx.Timeout] = None) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with HTTP/2 enabled and resilient connection pooling."""
        if timeout is None:
            timeout = httpx.Timeout(
                self.timeout_seconds,
                connect=25.0,
                read=self.timeout_seconds,
                write=60.0,
                pool=30.0,
            )
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=30, keepalive_expiry=60.0)
        try:
            return httpx.AsyncClient(http2=True, timeout=timeout, limits=limits)
        except Exception:
            return httpx.AsyncClient(http2=False, timeout=timeout, limits=limits)

    @property
    def provider_name(self) -> str:
        return "aistudio_gemini"

    @property
    def default_model(self) -> str:
        return self._default_model

    def resolve_api_key(self, authorization: Optional[str] = None) -> str:
        """Extract valid Gemini API key from Authorization header or fallback to environment configuration."""
        if authorization:
            parts = authorization.strip().split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1].strip()
                if token.startswith("AQ.") or token.startswith("AIzaSy"):
                    return token
                if token and not token.startswith("sk-") and token.lower() not in ("dummy", "test", "none", "null"):
                    return token
        return self.api_key_env

    def map_model_name(self, model_name: str) -> str:
        """Map client model name to an available Gemini model ID."""
        clean_name = model_name.strip()
        if clean_name.startswith("models/"):
            clean_name = clean_name[7:]
        return self.model_aliases.get(clean_name, clean_name)

    def convert_tools(self, tools: Optional[List[Union[Dict[str, Any], Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI tools array into Gemini functionDeclarations format."""
        if not tools:
            return None
        normalized_tools = []
        for t in tools:
            if hasattr(t, "model_dump"):
                normalized_tools.append(t.model_dump(exclude_none=True))
            elif hasattr(t, "dict"):
                normalized_tools.append(t.dict(exclude_none=True))
            elif isinstance(t, dict):
                normalized_tools.append(t)
            else:
                try:
                    normalized_tools.append(dict(t))
                except Exception:
                    pass

        gemini_functions = []
        for t in normalized_tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function":
                fn = t.get("function", {})
                fn_name = fn.get("name")
                if not fn_name:
                    continue
                decl = {
                    "name": fn_name,
                    "description": (fn.get("description") or "")[:1024],
                }
                params = fn.get("parameters")
                if params and isinstance(params, dict):
                    cleaned = clean_json_schema(params)
                    if cleaned:
                        decl["parameters"] = cleaned
                gemini_functions.append(decl)
        if not gemini_functions:
            return None
        return [{"functionDeclarations": gemini_functions}]

    def convert_messages(self, messages: List[Union[Dict[str, Any], Any]]):
        """
        Convert OpenAI chat messages to Gemini's systemInstruction and contents format.
        Preserves tool_calls, tool responses, thought_signatures, and alternates turns.
        """
        normalized_messages = []
        for m in (messages or []):
            if hasattr(m, "model_dump"):
                normalized_messages.append(m.model_dump(exclude_none=True))
            elif hasattr(m, "dict"):
                normalized_messages.append(m.dict(exclude_none=True))
            elif isinstance(m, dict):
                normalized_messages.append(m)
            else:
                try:
                    normalized_messages.append(dict(m))
                except Exception:
                    normalized_messages.append({"role": "user", "content": str(m)})
        messages = normalized_messages

        system_parts = []
        contents = []

        # Map tool_call_id to function name across turns
        tool_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    tid = tc.get("id")
                    tname = tc.get("function", {}).get("name")
                    if tid and tname:
                        tool_id_to_name[tid] = tname

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Handle system instructions
            if role in ("system", "developer"):
                if isinstance(content, str) and content.strip():
                    system_parts.append({"text": content.strip()})
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            t = p.get("text", "").strip()
                            if t:
                                system_parts.append({"text": t})
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = []

            # 1. Handle tool_calls in assistant message
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                if isinstance(content, str) and content.strip():
                    parts.append({"text": content.strip()})
                for tc in msg["tool_calls"]:
                    tid = tc.get("id")
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        if not isinstance(args, dict):
                            args = {"args": args}
                    except Exception:
                        args = {}
                    # Attach thought_signature to prevent Gemini 400 validation error on tool history
                    sig = self._call_signatures.get(tid) or self.fallback_thought_signature
                    parts.append({
                        "functionCall": {
                            "name": fn.get("name", "unknown"),
                            "args": args
                        },
                        "thoughtSignature": sig
                    })

            # 2. Handle tool response message
            elif role == "tool":
                tid = msg.get("tool_call_id")
                fn_name = msg.get("name") or tool_id_to_name.get(tid, "unknown_tool")
                tool_content = content or ""
                try:
                    resp_dict = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                    if not isinstance(resp_dict, dict):
                        resp_dict = {"output": resp_dict}
                except Exception:
                    resp_dict = {"output": tool_content}

                parts.append({
                    "functionResponse": {
                        "name": fn_name,
                        "response": resp_dict
                    }
                })
                gemini_role = "user"

            # 3. Handle standard user / assistant text and images
            else:
                if isinstance(content, str):
                    if content:
                        parts.append({"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            ptype = part.get("type", "")
                            if ptype == "text":
                                text_val = part.get("text", "")
                                if text_val:
                                    parts.append({"text": text_val})
                            elif ptype == "image_url":
                                image_info = part.get("image_url", {})
                                url = image_info.get("url", "")
                                match = re.match(r"^data:(image/[a-zA-Z0-9\+\-\.]+);base64,(.+)$", url)
                                if match:
                                    mime_type, base64_data = match.groups()
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": base64_data
                                        }
                                    })

            if not parts:
                parts.append({"text": " "})

            # Merge consecutive turns of the same role
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": gemini_role, "parts": parts})

        if contents and contents[0]["role"] == "model":
            contents.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        system_instruction = {"parts": system_parts} if system_parts else None
        return system_instruction, contents

    async def list_models(self, api_key: str) -> List[Dict[str, Any]]:
        """List models in OpenAI standard format."""
        current_time = time.time()
        if self._cached_models and (current_time - self._cache_timestamp < self.cache_ttl):
            return self._cached_models

        if not api_key:
            return [
                {
                    "id": self.default_model,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "google",
                    "permission": [],
                    "root": self.default_model,
                    "parent": None,
                }
            ] + [
                {
                    "id": alias,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "openai",
                    "permission": [],
                    "root": alias,
                    "parent": None,
                }
                for alias in self.model_aliases.keys()
            ]

        url = f"{self.base_url}/models"
        headers = {"x-goog-api-key": api_key}
        models_data = []

        try:
            async with self._create_client(httpx.Timeout(15.0, connect=10.0)) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("models", []):
                        raw_name = m.get("name", "")
                        model_id = raw_name[7:] if raw_name.startswith("models/") else raw_name
                        methods = m.get("supportedGenerationMethods", [])
                        if "generateContent" in methods or "embedContent" in methods:
                            models_data.append(
                                {
                                    "id": model_id,
                                    "object": "model",
                                    "created": 1700000000,
                                    "owned_by": "google",
                                    "permission": [
                                        {
                                            "id": f"modelperm-{model_id}",
                                            "object": "model_permission",
                                            "created": 1700000000,
                                            "allow_create_engine": False,
                                            "allow_sampling": True,
                                            "allow_logprobs": True,
                                            "allow_search_indices": False,
                                            "allow_view": True,
                                            "allow_fine_tuning": False,
                                            "organization": "*",
                                            "group": None,
                                            "is_blocking": False,
                                        }
                                    ],
                                    "root": model_id,
                                    "parent": None,
                                }
                            )
        except Exception as e:
            logger.warning(f"Failed to fetch dynamic models from Gemini API: {e}")

        existing_ids = {m["id"] for m in models_data}
        if self.default_model not in existing_ids:
            models_data.insert(
                0,
                {
                    "id": self.default_model,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "google",
                    "permission": [],
                    "root": self.default_model,
                    "parent": None,
                },
            )
            existing_ids.add(self.default_model)

        for alias in self.model_aliases.keys():
            if alias not in existing_ids:
                models_data.append(
                    {
                        "id": alias,
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "openai",
                        "permission": [],
                        "root": alias,
                        "parent": None,
                    }
                )

        self._cached_models = models_data
        self._cache_timestamp = current_time
        return models_data

    async def get_model(self, model_id: str, api_key: str) -> Dict[str, Any]:
        """Retrieve specific model details in OpenAI standard format."""
        clean_id = model_id.strip()
        if clean_id.startswith("models/"):
            clean_id = clean_id[7:]

        models = await self.list_models(api_key)
        for m in models:
            if m["id"].lower() == clean_id.lower() or m["id"].lower() == model_id.lower():
                return m

        target_gemini_model = self.map_model_name(clean_id)
        return {
            "id": clean_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "google" if "gemini" in target_gemini_model else "openai",
            "permission": [
                {
                    "id": f"modelperm-{clean_id}",
                    "object": "model_permission",
                    "created": 1700000000,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
            "root": clean_id,
            "parent": None,
        }

    async def chat_completions(self, req: Any, api_key: str) -> Dict[str, Any]:
        """Handle non-streaming chat completions."""
        target_model = self.map_model_name(req.model or self.default_model)
        system_instruction, contents = self.convert_messages(req.messages)
        gemini_tools = self.convert_tools(req.tools)

        generation_config: Dict[str, Any] = {}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature
        if req.top_p is not None:
            generation_config["topP"] = req.top_p

        max_tokens = req.max_completion_tokens or req.max_tokens
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        if req.stop:
            if isinstance(req.stop, str):
                generation_config["stopSequences"] = [req.stop]
            elif isinstance(req.stop, list):
                generation_config["stopSequences"] = req.stop

        if self.thinking_budget >= 0:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if gemini_tools:
            payload["tools"] = gemini_tools
        if generation_config:
            payload["generationConfig"] = generation_config

        generate_url = f"{self.base_url}/models/{target_model}:generateContent"
        headers = {"x-goog-api-key": api_key}

        for attempt in range(self.max_retries):
            try:
                async with self._create_client() as client:
                    resp = await client.post(generate_url, json=payload, headers=headers)
                    if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                        retry_wait = 2.0 if resp.status_code == 429 else (1.0 * (2 ** attempt))
                        logger.warning(
                            f"Gemini API returned retryable status {resp.status_code}, retrying attempt {attempt + 1}/{self.max_retries} in {retry_wait:.1f}s..."
                        )
                        await asyncio.sleep(retry_wait)
                        continue

                    if resp.status_code != 200:
                        logger.error(f"Gemini API returned error: {resp.status_code} {resp.text}")
                        return {
                            "error": {
                                "message": f"Gemini API error: {resp.text}",
                                "type": "invalid_request_error",
                                "code": resp.status_code,
                            }
                        }

                    data = resp.json()
                    candidates = data.get("candidates", [])
                    completion_text = ""
                    tool_calls_out = []
                    finish_reason = "stop"

                    if candidates:
                        cand = candidates[0]
                        parts = cand.get("content", {}).get("parts", [])
                        for p in parts:
                            if "text" in p and p["text"]:
                                completion_text += p["text"]
                            elif "functionCall" in p:
                                fn_call = p["functionCall"]
                                call_id = fn_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                                thought_sig = p.get("thoughtSignature")
                                if thought_sig and call_id:
                                    self._call_signatures[call_id] = thought_sig
                                tool_calls_out.append({
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": fn_call.get("name", ""),
                                        "arguments": json.dumps(fn_call.get("args", {}), ensure_ascii=False)
                                    }
                                })

                        g_finish = cand.get("finishReason")
                        if tool_calls_out or (g_finish == "STOP" and tool_calls_out):
                            finish_reason = "tool_calls"
                        elif g_finish and g_finish == "MAX_TOKENS":
                            finish_reason = "length"

                    usage_meta = data.get("usageMetadata", {})
                    prompt_tokens = usage_meta.get("promptTokenCount", 0)
                    completion_tokens = usage_meta.get("candidatesTokenCount", 0)
                    total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

                    choice_message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": completion_text if (not tool_calls_out or completion_text) else None,
                    }
                    if tool_calls_out:
                        choice_message["tool_calls"] = tool_calls_out

                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": req.model or target_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": choice_message,
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
            except RETRYABLE_EXCEPTIONS as exc:
                if attempt < self.max_retries - 1:
                    retry_wait = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"Gemini API connection error ({type(exc).__name__}: {exc}), retrying attempt {attempt + 1}/{self.max_retries} in {retry_wait:.1f}s..."
                    )
                    await asyncio.sleep(retry_wait)
                    continue
                logger.error(f"Gemini API connection failed after {self.max_retries} attempts: {exc}")
                return {
                    "error": {
                        "message": f"Gemini API connection failed: {str(exc)}",
                        "type": "connection_error",
                        "code": 502,
                    }
                }

    async def chat_completions_stream(self, req: Any, api_key: str) -> AsyncGenerator[str, None]:
        """Handle streaming chat completions yielding OpenAI SSE lines."""
        target_model = self.map_model_name(req.model or self.default_model)
        system_instruction, contents = self.convert_messages(req.messages)
        gemini_tools = self.convert_tools(req.tools)

        generation_config: Dict[str, Any] = {}
        if req.temperature is not None:
            generation_config["temperature"] = req.temperature
        if req.top_p is not None:
            generation_config["topP"] = req.top_p

        max_tokens = req.max_completion_tokens or req.max_tokens
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        if req.stop:
            if isinstance(req.stop, str):
                generation_config["stopSequences"] = [req.stop]
            elif isinstance(req.stop, list):
                generation_config["stopSequences"] = req.stop

        if self.thinking_budget >= 0:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if gemini_tools:
            payload["tools"] = gemini_tools
        if generation_config:
            payload["generationConfig"] = generation_config

        stream_url = f"{self.base_url}/models/{target_model}:streamGenerateContent?alt=sse"
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        # Prepared initial chunk with role
        initial_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": req.model or target_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }

        headers = {"x-goog-api-key": api_key}
        has_tool_call = False
        tool_call_index = 0
        initial_chunk_sent = False

        for attempt in range(self.max_retries):
            try:
                async with self._create_client() as client:
                    async with client.stream("POST", stream_url, json=payload, headers=headers) as response:
                        if response.status_code in RETRYABLE_STATUS_CODES and not initial_chunk_sent and attempt < self.max_retries - 1:
                            retry_wait = 2.0 if response.status_code == 429 else (1.0 * (2 ** attempt))
                            logger.warning(
                                f"Gemini streaming returned retryable status {response.status_code}, retrying attempt {attempt + 1}/{self.max_retries} in {retry_wait:.1f}s..."
                            )
                            await asyncio.sleep(retry_wait)
                            continue

                        if response.status_code != 200:
                            err_text = await response.aread()
                            logger.error(f"Gemini streaming error ({response.status_code}): {err_text.decode('utf-8', errors='ignore')}")
                            if not initial_chunk_sent:
                                yield f"data: {json.dumps(initial_chunk)}\n\n"
                                initial_chunk_sent = True
                            err_chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": req.model or target_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": f"\n[Error from Gemini: {response.status_code}]"},
                                        "finish_reason": "stop",
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(err_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                raw_data = line[5:].strip()
                                if not raw_data:
                                    continue
                                try:
                                    gemini_chunk = json.loads(raw_data)
                                    candidates = gemini_chunk.get("candidates", [])
                                    if candidates:
                                        cand = candidates[0]
                                        parts = cand.get("content", {}).get("parts", [])
                                        finish_reason = cand.get("finishReason")
                                        finish_message = cand.get("finishMessage", "")

                                        for part in parts:
                                            # 1. Text delta
                                            if "text" in part and part["text"]:
                                                if not initial_chunk_sent:
                                                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                                                    initial_chunk_sent = True

                                                chunk_payload = {
                                                    "id": chat_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created_time,
                                                    "model": req.model or target_model,
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {"content": part["text"]},
                                                            "finish_reason": None,
                                                        }
                                                    ],
                                                }
                                                yield f"data: {json.dumps(chunk_payload)}\n\n"

                                            # 2. Tool / Function call delta
                                            elif "functionCall" in part:
                                                if not initial_chunk_sent:
                                                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                                                    initial_chunk_sent = True

                                                has_tool_call = True
                                                fn_call = part["functionCall"]
                                                fn_name = fn_call.get("name", "")
                                                fn_args = fn_call.get("args", {})
                                                call_id = fn_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                                                thought_sig = part.get("thoughtSignature")
                                                if thought_sig and call_id:
                                                    self._call_signatures[call_id] = thought_sig

                                                tool_chunk = {
                                                    "id": chat_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created_time,
                                                    "model": req.model or target_model,
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {
                                                                "tool_calls": [
                                                                    {
                                                                        "index": tool_call_index,
                                                                        "id": call_id,
                                                                        "type": "function",
                                                                        "function": {
                                                                            "name": fn_name,
                                                                            "arguments": json.dumps(fn_args, ensure_ascii=False)
                                                                        }
                                                                    }
                                                                ]
                                                            },
                                                            "finish_reason": None,
                                                        }
                                                    ],
                                                }
                                                tool_call_index += 1
                                                yield f"data: {json.dumps(tool_chunk)}\n\n"

                                        # Fallback rescue for MALFORMED_FUNCTION_CALL
                                        if finish_reason == "MALFORMED_FUNCTION_CALL" and not has_tool_call and finish_message:
                                            m = re.search(r"call:([a-zA-Z0-9_\-]+)\s*(\{.*\})", finish_message, re.DOTALL)
                                            if m:
                                                if not initial_chunk_sent:
                                                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                                                    initial_chunk_sent = True

                                                has_tool_call = True
                                                fn_name, fn_args_str = m.groups()
                                                call_id = f"call_{uuid.uuid4().hex[:12]}"
                                                tool_chunk = {
                                                    "id": chat_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created_time,
                                                    "model": req.model or target_model,
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {
                                                                "tool_calls": [
                                                                    {
                                                                        "index": tool_call_index,
                                                                        "id": call_id,
                                                                        "type": "function",
                                                                        "function": {
                                                                            "name": fn_name,
                                                                            "arguments": fn_args_str
                                                                        }
                                                                    }
                                                                ]
                                                            },
                                                            "finish_reason": None,
                                                        }
                                                    ],
                                                }
                                                yield f"data: {json.dumps(tool_chunk)}\n\n"

                                        if finish_reason and finish_reason != "FINISH_REASON_UNSPECIFIED":
                                            if not initial_chunk_sent:
                                                yield f"data: {json.dumps(initial_chunk)}\n\n"
                                                initial_chunk_sent = True

                                            openai_finish = "tool_calls" if has_tool_call else "stop"
                                            if finish_reason == "MAX_TOKENS":
                                                openai_finish = "length"

                                            stop_chunk = {
                                                "id": chat_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_time,
                                                "model": req.model or target_model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {},
                                                        "finish_reason": openai_finish,
                                                    }
                                                ],
                                            }
                                            yield f"data: {json.dumps(stop_chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue

                # Stream completed normally
                break

            except asyncio.CancelledError:
                logger.info("SSE generator cancelled by client.")
                raise
            except RETRYABLE_EXCEPTIONS as exc:
                if not initial_chunk_sent and attempt < self.max_retries - 1:
                    retry_wait = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"Upstream stream connection dropped ({type(exc).__name__}: {exc}), retrying attempt {attempt + 1}/{self.max_retries} in {retry_wait:.1f}s..."
                    )
                    await asyncio.sleep(retry_wait)
                    continue

                logger.error(f"Exception during SSE stream: {type(exc).__name__}: {exc}")
                if not initial_chunk_sent:
                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                    initial_chunk_sent = True
                err_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model or target_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[Stream Error: {str(exc)}]"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                break
            except Exception as exc:
                logger.error(f"Unexpected exception during SSE stream: {type(exc).__name__}: {exc}")
                if not initial_chunk_sent:
                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                    initial_chunk_sent = True
                err_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model or target_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[Stream Error: {str(exc)}]"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                break

        if not initial_chunk_sent:
            yield f"data: {json.dumps(initial_chunk)}\n\n"
            stop_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": req.model or target_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def completions(self, req: Any, api_key: str) -> Dict[str, Any]:
        """Handle legacy code completions (inline/FIM) with automatic retry on transient errors."""
        target_model = self.map_model_name(req.model or self.default_model)
        prompt_text = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
        if req.suffix:
            instruction = (
                "You are an expert inline code completion engine. Fill in the code between <prefix> and <suffix>. "
                "Output ONLY the exact code that should be inserted in the middle. Do NOT include markdown code blocks, "
                "do NOT repeat prefix or suffix, and do NOT include any explanations.\n\n"
                f"<prefix>{prompt_text}</prefix>\n<suffix>{req.suffix}</suffix>"
            )
        else:
            instruction = (
                "You are an expert inline code completion engine. Continue the following code directly. "
                "Output ONLY the raw continuation code. Do NOT output markdown code blocks or explanations:\n\n"
                f"{prompt_text}"
            )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": instruction}]}],
            "generationConfig": {
                "temperature": req.temperature if req.temperature is not None else 0.2,
                "maxOutputTokens": req.max_tokens or 150,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if req.stop:
            stops = [req.stop] if isinstance(req.stop, str) else req.stop
            payload["generationConfig"]["stopSequences"] = stops

        generate_url = f"{self.base_url}/models/{target_model}:generateContent"
        headers = {"x-goog-api-key": api_key}

        for attempt in range(self.max_retries):
            try:
                async with self._create_client() as client:
                    resp = await client.post(generate_url, json=payload, headers=headers)
                    if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                        retry_wait = 2.0 if resp.status_code == 429 else (1.0 * (2 ** attempt))
                        logger.warning(
                            f"Gemini completions returned retryable status {resp.status_code}, retrying attempt {attempt + 1}/{self.max_retries} in {retry_wait:.1f}s..."
                        )
                        await asyncio.sleep(retry_wait)
                        continue

                    if resp.status_code != 200:
                        return {
                            "error": {
                                "message": f"Gemini error: {resp.text}",
                                "type": "invalid_request_error",
                            }
                        }

                    data = resp.json()
                    candidates = data.get("candidates", [])
                    text = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for p in parts:
                            if "text" in p:
                                text += p["text"]

                    return {
                        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model or target_model,
                        "choices": [
                            {
                                "text": text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt_text) // 4,
                            "completion_tokens": len(text) // 4,
                            "total_tokens": (len(prompt_text) + len(text)) // 4,
                        },
                    }
            except RETRYABLE_EXCEPTIONS as exc:
                if attempt < self.max_retries - 1:
                    retry_wait = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"Gemini completions connection error ({type(exc).__name__}: {exc}), retrying attempt {attempt + 1}/{self.max_retries} in {retry_wait:.1f}s..."
                    )
                    await asyncio.sleep(retry_wait)
                    continue
                return {"error": {"message": f"Gemini completion connection failed: {str(exc)}"}}

    async def embeddings(self, req: Any, api_key: str) -> Dict[str, Any]:
        """Handle embeddings requests with automatic retry on transient errors."""
        target_model = "gemini-embedding-001"
        texts = [req.input] if isinstance(req.input, str) else req.input
        embed_url = f"{self.base_url}/models/{target_model}:embedContent"
        headers = {"x-goog-api-key": api_key}

        results = []
        for idx, text in enumerate(texts):
            payload = {"content": {"parts": [{"text": text}]}}
            success = False
            for attempt in range(self.max_retries):
                try:
                    async with self._create_client() as client:
                        resp = await client.post(embed_url, json=payload, headers=headers)
                        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                            await asyncio.sleep(1.0 * (2 ** attempt))
                            continue
                        if resp.status_code == 200:
                            values = resp.json().get("embedding", {}).get("values", [])
                            results.append(
                                {
                                    "object": "embedding",
                                    "index": idx,
                                    "embedding": values,
                                }
                            )
                            success = True
                            break
                        else:
                            return {"error": {"message": f"Embedding error: {resp.text}"}}
                except RETRYABLE_EXCEPTIONS as exc:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    return {"error": {"message": f"Embedding connection error: {str(exc)}"}}

        return {
            "object": "list",
            "data": results,
            "model": req.model or target_model,
            "usage": {
                "prompt_tokens": sum(len(t) // 4 for t in texts),
                "total_tokens": sum(len(t) // 4 for t in texts),
            },
        }
