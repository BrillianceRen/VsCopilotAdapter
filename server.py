import argparse
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Union

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback basic .env loader if python-dotenv is not installed
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

import uvicorn
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from adapters import BaseAIAdapter, get_adapter

# Setup logging: console + rotating log file (10MB per file, 3 backups)
LOG_FILE = os.getenv("LOG_FILE", "adapter.log")
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper().strip()

log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("copilot-openai-adapter")

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# File handler with UTF-8 encoding
handlers = [console_handler]
try:
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(log_formatter)
    handlers.append(file_handler)
except Exception as e:
    print(f"Warning: Failed to setup log file '{LOG_FILE}': {e}")


def set_log_level(level_name: str):
    """Update logger and handler log levels."""
    numeric_level = getattr(logging, (level_name or "INFO").upper().strip(), logging.INFO)
    logger.setLevel(numeric_level)
    for h in logger.handlers:
        h.setLevel(numeric_level)


if not logger.handlers:
    for h in handlers:
        logger.addHandler(h)

set_log_level(LOG_LEVEL_NAME)
logger.propagate = False

# Active provider adapter instance
AI_PROVIDER = os.getenv("AI_PROVIDER", "aistudio_gemini").strip()
adapter: BaseAIAdapter = get_adapter(AI_PROVIDER)

app = FastAPI(
    title="Copilot OpenAI Adapter",
    description=f"Multi-provider OpenAI-compatible adapter for Visual Studio Copilot (Active: {adapter.provider_name})",
    version="2.0.0",
)

# Enable CORS for Visual Studio, webviews, and external tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def mask_secret(secret: str) -> str:
    """Mask secret strings in logs to prevent accidental leakage."""
    if not secret:
        return ""
    clean = secret.strip()
    if clean.lower().startswith("bearer "):
        clean = clean[7:].strip()
    if len(clean) <= 8:
        return "******"
    return f"{clean[:6]}...{clean[-4:]}"


@app.middleware("http")
async def log_requests_and_responses(request: Request, call_next):
    """
    Middleware that logs requests and responses.
    - INFO level: logs only method, URL, client IP, target model, stream/tool counts, duration, and status (NO body content).
    - DEBUG level: logs full request headers, request JSON body, full response body or aggregated SSE text.
    - WARNING/ERROR level: logs only warnings and errors.
    """
    req_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url_path = request.url.path
    query_params = str(request.query_params)
    full_url = f"{url_path}?{query_params}" if query_params else url_path

    # Read and buffer request body
    req_body_bytes = await request.body()
    parsed_json = None
    model_name = None
    stream_flag = False
    tools_count = 0
    messages_count = 0

    if req_body_bytes:
        try:
            parsed_json = json.loads(req_body_bytes.decode("utf-8"))
            if isinstance(parsed_json, dict):
                model_name = parsed_json.get("model")
                stream_flag = bool(parsed_json.get("stream", False))
                messages = parsed_json.get("messages")
                if isinstance(messages, list):
                    messages_count = len(messages)
                tools = parsed_json.get("tools")
                if isinstance(tools, list):
                    tools_count = len(tools)
        except Exception:
            pass

    if not model_name:
        for prefix in ("/v1/models/", "/models/"):
            if url_path.startswith(prefix) and len(url_path) > len(prefix):
                model_name = url_path[len(prefix):]
                break

    # Format concise metadata tags for INFO level
    req_tags = []
    if model_name:
        req_tags.append(f"Model: {model_name}")
    if stream_flag:
        req_tags.append("Stream: true")
    if messages_count:
        req_tags.append(f"Messages: {messages_count}")
    if tools_count:
        req_tags.append(f"Tools: {tools_count}")
    req_tags_str = f" | {' | '.join(req_tags)}" if req_tags else ""

    logger.info(f"[REQUEST #{req_id}] {method} {full_url}{req_tags_str} | Client: {client_host}")

    # DEBUG level: output detailed headers and request body
    if logger.isEnabledFor(logging.DEBUG):
        display_headers = {}
        for k, v in request.headers.items():
            if k.lower() in ("authorization", "x-goog-api-key"):
                display_headers[k] = mask_secret(v)
            elif k.lower() in ("user-agent", "content-type", "accept", "x-request-id", "host"):
                display_headers[k] = v

        body_display = "(empty)"
        if req_body_bytes:
            if parsed_json is not None:
                body_display = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            else:
                body_display = req_body_bytes.decode("utf-8", errors="replace")

        logger.debug(
            f"\n{'='*25} [REQUEST DETAILS #{req_id}] {'='*25}\n"
            f"Endpoint   : {method} {full_url}\n"
            f"Headers    :\n{json.dumps(display_headers, indent=2, ensure_ascii=False)}\n"
            f"Body       :\n{body_display}\n"
            f"{'-'*70}"
        )

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[ERROR #{req_id}] {method} {full_url} | Duration: {duration_ms:.2f} ms | Exception: {type(exc).__name__}: {exc}"
        )
        raise exc

    content_type = response.headers.get("content-type", "")

    # For streaming SSE responses
    if "text/event-stream" in content_type:
        orig_iterator = response.body_iterator
        is_debug = logger.isEnabledFor(logging.DEBUG)
        accumulated_chunks = [] if is_debug else None
        accumulated_text = [] if is_debug else None
        chunk_count = 0
        tool_calls_preview = []

        async def stream_wrapper():
            nonlocal chunk_count
            try:
                async for chunk in orig_iterator:
                    chunk_count += 1
                    if is_debug and accumulated_chunks is not None:
                        accumulated_chunks.append(chunk)

                    # Extract content/tool calls for summary & debug
                    try:
                        chunk_str = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
                        for line in chunk_str.splitlines():
                            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                                d = json.loads(line[5:].strip())
                                choice = d.get("choices", [{}])[0]
                                delta = choice.get("delta", {})
                                if is_debug and accumulated_text is not None:
                                    delta_text = delta.get("content", "")
                                    if delta_text:
                                        accumulated_text.append(delta_text)
                                for tc in delta.get("tool_calls", []):
                                    fname = tc.get("function", {}).get("name")
                                    if fname and fname not in tool_calls_preview:
                                        tool_calls_preview.append(fname)
                    except Exception:
                        pass

                    yield chunk
            except asyncio.CancelledError:
                logger.info(f"[{req_id}] Client disconnected from SSE stream.")
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                stream_tags = []
                if model_name:
                    stream_tags.append(f"Model: {model_name}")
                if chunk_count:
                    stream_tags.append(f"Chunks: {chunk_count}")
                if tool_calls_preview:
                    stream_tags.append(f"Tool Calls: {', '.join(tool_calls_preview)}")
                stream_tags_str = f" | {' | '.join(stream_tags)}" if stream_tags else ""

                logger.info(
                    f"[STREAM RESPONSE #{req_id}] {method} {full_url} | Status: {response.status_code} | Duration: {duration_ms:.2f} ms{stream_tags_str}"
                )

                if is_debug:
                    full_text = "".join(accumulated_text or [])
                    logger.debug(
                        f"\n{'='*25} [STREAM DETAILS #{req_id}] {'='*25}\n"
                        f"Status     : {response.status_code}\n"
                        f"Duration   : {duration_ms:.2f} ms\n"
                        f"Chunks     : {chunk_count} SSE events\n"
                        f"Output Text:\n{full_text}\n"
                        f"{'='*70}"
                    )

        return StreamingResponse(
            stream_wrapper(),
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    else:
        # For non-streaming responses
        resp_parts = [part async for part in response.body_iterator]
        duration_ms = (time.time() - start_time) * 1000
        full_resp_bytes = b"".join(resp_parts)

        resp_tags = []
        if model_name:
            resp_tags.append(f"Model: {model_name}")

        parsed_resp = None
        if full_resp_bytes:
            try:
                parsed_resp = json.loads(full_resp_bytes.decode("utf-8"))
                if isinstance(parsed_resp, dict):
                    usage = parsed_resp.get("usage")
                    if isinstance(usage, dict) and "total_tokens" in usage:
                        resp_tags.append(f"Tokens: {usage['total_tokens']}")
                    if "error" in parsed_resp:
                        err_msg = str(parsed_resp["error"].get("message", "Error"))
                        resp_tags.append(f"Error: {err_msg[:60]}")
            except Exception:
                pass

        resp_tags_str = f" | {' | '.join(resp_tags)}" if resp_tags else ""

        logger.info(
            f"[RESPONSE #{req_id}] {method} {full_url} | Status: {response.status_code} | Duration: {duration_ms:.2f} ms{resp_tags_str}"
        )

        if logger.isEnabledFor(logging.DEBUG):
            resp_display = "(empty)"
            if full_resp_bytes:
                if parsed_resp is not None:
                    resp_display = json.dumps(parsed_resp, indent=2, ensure_ascii=False)
                else:
                    resp_display = full_resp_bytes.decode("utf-8", errors="replace")

            logger.debug(
                f"\n{'='*25} [RESPONSE BODY #{req_id}] {'='*25}\n"
                f"{resp_display}\n"
                f"{'='*70}"
            )

        headers = dict(response.headers)
        headers.pop("content-length", None)
        return Response(
            content=full_resp_bytes,
            status_code=response.status_code,
            headers=headers,
        )


# -------------------------------------------------------------
# Request Schemas (OpenAI Compatible)
# -------------------------------------------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[Union[ChatMessage, Dict[str, Any]]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    input: Union[str, List[str]]


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "copilot-openai-adapter",
        "provider": adapter.provider_name,
        "default_model": adapter.default_model,
        "version": "2.0.0",
        "documentation": "/docs",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "embeddings": "/v1/embeddings",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "provider": adapter.provider_name}


@app.get("/models")
@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models in OpenAI standard format."""
    api_key = adapter.resolve_api_key(authorization)
    data = await adapter.list_models(api_key)
    return {"object": "list", "data": data}


@app.get("/models/{model_id:path}")
@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    """Retrieve specific model details in OpenAI standard format."""
    api_key = adapter.resolve_api_key(authorization)
    model_info = await adapter.get_model(model_id, api_key)
    return model_info


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """Handle OpenAI chat completions with streaming, function calling, and provider routing."""
    api_key = adapter.resolve_api_key(authorization)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": f"Missing API Key for provider '{adapter.provider_name}'. Please configure it in .env or pass Authorization header.",
                    "type": "invalid_request_error",
                    "code": 401,
                }
            },
        )

    if req.stream:
        return StreamingResponse(
            adapter.chat_completions_stream(req, api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await adapter.chat_completions(req, api_key)
    if isinstance(result, dict) and "error" in result:
        status_code = result.get("error", {}).get("code", 400)
        if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
            status_code = 400
        return JSONResponse(status_code=status_code, content=result)
    return result


@app.post("/completions")
@app.post("/v1/completions")
async def completions(
    req: CompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """Handle OpenAI legacy code completions (used by Copilot for inline autocompletion)."""
    api_key = adapter.resolve_api_key(authorization)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": f"Missing API Key for provider '{adapter.provider_name}'. Please configure it in .env or pass Authorization header.",
                    "type": "invalid_request_error",
                    "code": 401,
                }
            },
        )

    result = await adapter.completions(req, api_key)
    if isinstance(result, dict) and "error" in result:
        status_code = result.get("error", {}).get("code", 400)
        if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
            status_code = 400
        return JSONResponse(status_code=status_code, content=result)
    return result


@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(
    req: EmbeddingRequest,
    authorization: Optional[str] = Header(None),
):
    """Handle OpenAI embeddings requests."""
    api_key = adapter.resolve_api_key(authorization)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": f"Missing API Key for provider '{adapter.provider_name}'. Please configure it in .env or pass Authorization header.",
                    "type": "invalid_request_error",
                    "code": 401,
                }
            },
        )

    result = await adapter.embeddings(req, api_key)
    if isinstance(result, dict) and "error" in result:
        status_code = result.get("error", {}).get("code", 400)
        if not isinstance(status_code, int) or status_code < 100 or status_code > 599:
            status_code = 400
        return JSONResponse(status_code=status_code, content=result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Copilot OpenAI Adapter Server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host address to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to listen on")
    parser.add_argument("--provider", default=None, help="AI Provider name (default: aistudio_gemini)")
    parser.add_argument("--key", default=None, help="API Key override")
    parser.add_argument("--model", default=None, help="Default Model override")
    parser.add_argument("--log-file", default=None, help="Path to log file")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "debug", "info", "warning", "error"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)",
    )

    args = parser.parse_args()

    global adapter, LOG_FILE
    if args.provider:
        adapter = get_adapter(args.provider)
    if args.key:
        if hasattr(adapter, "api_key_env"):
            adapter.api_key_env = args.key
    if args.model:
        if hasattr(adapter, "_default_model"):
            adapter._default_model = args.model
    if args.log_file:
        LOG_FILE = args.log_file
    if args.log_level:
        set_log_level(args.log_level)

    resolved_key = adapter.resolve_api_key()
    masked_key = mask_secret(resolved_key) if resolved_key else "(not set - configure in .env)"
    current_level = logging.getLevelName(logger.level)

    print("=" * 60)
    print(" Copilot OpenAI Adapter Server (Multi-Provider)")
    print("=" * 60)
    print(f" AI Provider        : {adapter.provider_name}")
    print(f" API Key            : {masked_key}")
    print(f" Default Model      : {adapter.default_model}")
    print(f" Log Level          : {current_level}")
    print(f" Listening Address  : http://{args.host}:{args.port}")
    print(f" OpenAI Base URL    : http://127.0.0.1:{args.port}/v1")
    print(f" Models Endpoint    : http://127.0.0.1:{args.port}/v1/models")
    print(f" Log File           : {os.path.abspath(LOG_FILE)}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
