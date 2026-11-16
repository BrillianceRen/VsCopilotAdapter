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
log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("copilot-openai-adapter")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
if not logger.handlers:
    logger.addHandler(console_handler)

# File handler with UTF-8 encoding
try:
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Warning: Failed to setup log file '{LOG_FILE}': {e}")

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
    Middleware that logs every incoming request and outgoing response in full detail
    (including headers, JSON body, latency, and streaming chunks) to console and adapter.log.
    Secrets in headers are automatically masked.
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

    # Filter and mask headers for display
    display_headers = {}
    for k, v in request.headers.items():
        if k.lower() in ("authorization", "x-goog-api-key"):
            display_headers[k] = mask_secret(v)
        elif k.lower() in ("user-agent", "content-type", "accept", "x-request-id", "host"):
            display_headers[k] = v

    # Format request body
    body_display = "(empty)"
    if req_body_bytes:
        try:
            parsed_json = json.loads(req_body_bytes.decode("utf-8"))
            body_display = json.dumps(parsed_json, indent=2, ensure_ascii=False)
        except Exception:
            body_display = req_body_bytes.decode("utf-8", errors="replace")

    logger.info(
        f"\n{'='*25} [REQUEST #{req_id}] {'='*25}\n"
        f"Time       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Endpoint   : {method} {full_url}\n"
        f"Client IP  : {client_host}\n"
        f"Headers    :\n{json.dumps(display_headers, indent=2, ensure_ascii=False)}\n"
        f"Body       :\n{body_display}\n"
        f"{'-'*70}"
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"\n{'='*25} [ERROR #{req_id}] {'='*25}\n"
            f"Endpoint   : {method} {full_url}\n"
            f"Duration   : {duration_ms:.2f} ms\n"
            f"Exception  : {type(exc).__name__}: {exc}\n"
            f"{'='*70}\n"
        )
        raise exc

    content_type = response.headers.get("content-type", "")

    # For streaming SSE responses: stream chunks in real-time and aggregate for logging
    if "text/event-stream" in content_type:
        orig_iterator = response.body_iterator
        accumulated_chunks = []
        accumulated_text = []

        async def stream_wrapper():
            try:
                async for chunk in orig_iterator:
                    accumulated_chunks.append(chunk)
                    yield chunk
            except asyncio.CancelledError:
                logger.info(f"[{req_id}] Client disconnected from SSE stream.")
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                full_stream_bytes = b"".join(accumulated_chunks)
                full_stream_str = full_stream_bytes.decode("utf-8", errors="replace")

                # Extract content text and tool calls from SSE chunks for clean preview
                tool_calls_preview = []
                for line in full_stream_str.splitlines():
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        try:
                            d = json.loads(line[5:].strip())
                            choice = d.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            delta_text = delta.get("content", "")
                            if delta_text:
                                accumulated_text.append(delta_text)
                            tcs = delta.get("tool_calls", [])
                            for tc in tcs:
                                fn = tc.get("function", {})
                                tool_calls_preview.append(f"{fn.get('name')}({fn.get('arguments', '')})")
                        except Exception:
                            pass

                aggregated_output = "".join(accumulated_text)
                if tool_calls_preview:
                    aggregated_output += f"\n[Tool Calls: {', '.join(tool_calls_preview)}]"

                logger.info(
                    f"\n{'='*25} [STREAM RESPONSE #{req_id}] {'='*25}\n"
                    f"Status     : {response.status_code}\n"
                    f"Duration   : {duration_ms:.2f} ms\n"
                    f"Endpoint   : {method} {full_url}\n"
                    f"Chunks     : {len(accumulated_chunks)} SSE events\n"
                    f"Output Text:\n{aggregated_output}\n"
                    f"{'='*70}\n"
                )

        return StreamingResponse(
            stream_wrapper(),
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    else:
        # For non-streaming responses: read body parts, log and return
        resp_parts = [part async for part in response.body_iterator]
        duration_ms = (time.time() - start_time) * 1000
        full_resp_bytes = b"".join(resp_parts)

        resp_display = "(empty)"
        if full_resp_bytes:
            try:
                parsed_json = json.loads(full_resp_bytes.decode("utf-8"))
                resp_display = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            except Exception:
                resp_display = full_resp_bytes.decode("utf-8", errors="replace")

        logger.info(
            f"\n{'='*25} [RESPONSE #{req_id}] {'='*25}\n"
            f"Status     : {response.status_code}\n"
            f"Duration   : {duration_ms:.2f} ms\n"
            f"Endpoint   : {method} {full_url}\n"
            f"Body       :\n{resp_display}\n"
            f"{'='*70}\n"
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

    resolved_key = adapter.resolve_api_key()
    masked_key = mask_secret(resolved_key) if resolved_key else "(not set - configure in .env)"

    print("=" * 60)
    print(" Copilot OpenAI Adapter Server (Multi-Provider)")
    print("=" * 60)
    print(f" AI Provider        : {adapter.provider_name}")
    print(f" API Key            : {masked_key}")
    print(f" Default Model      : {adapter.default_model}")
    print(f" Listening Address  : http://{args.host}:{args.port}")
    print(f" OpenAI Base URL    : http://127.0.0.1:{args.port}/v1")
    print(f" Models Endpoint    : http://127.0.0.1:{args.port}/v1/models")
    print(f" Log File           : {os.path.abspath(LOG_FILE)}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
