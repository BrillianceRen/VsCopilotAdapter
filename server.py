import argparse
import asyncio
from datetime import datetime
import json
import logging
from logging.handlers import BaseRotatingHandler
import os
import re
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
from fallback_manager import fallback_manager


class DailySizeRotatingHandler(BaseRotatingHandler):
    """
    Rotating file handler that:
    1. Stores log files in a dedicated directory (log_dir).
    2. Rotates daily based on current local date (yyyy-MM-dd).
    3. Names files in format: yyyy-MM-dd.##.log (e.g. 2026-11-23.01.log).
    4. Enforces a maximum file size (default 30MB).
    5. When file size reaches limit, increments the sequence number (e.g. .01.log -> .02.log).
    """

    def __init__(
        self,
        log_dir: str,
        max_bytes: int = 30 * 1024 * 1024,
        encoding: str = "utf-8",
        delay: bool = False,
    ):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.max_bytes = max_bytes
        self.current_date = self._get_current_date()
        self.current_seq = self._find_or_start_seq(self.current_date)
        initial_file = self._build_filepath(self.current_date, self.current_seq)

        super().__init__(initial_file, mode="a", encoding=encoding, delay=delay)

    def _get_current_date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _build_filepath(self, date_str: str, seq: int) -> str:
        return os.path.join(self.log_dir, f"{date_str}.{seq:02d}.log")

    def _find_or_start_seq(self, date_str: str) -> int:
        """
        Scan log_dir for existing files matching date_str.##.log.
        If any exist, check if the latest one has space (< max_bytes).
        If it has space, reuse it. Otherwise, increment seq.
        If none exist, start at 1.
        """
        pattern = re.compile(rf"^{re.escape(date_str)}\.(\d+)\.log$")
        existing_seqs = []
        try:
            for fname in os.listdir(self.log_dir):
                m = pattern.match(fname)
                if m:
                    existing_seqs.append(int(m.group(1)))
        except Exception:
            pass

        if not existing_seqs:
            return 1

        existing_seqs.sort()
        latest_seq = existing_seqs[-1]
        latest_path = self._build_filepath(date_str, latest_seq)
        try:
            if os.path.exists(latest_path) and os.path.getsize(latest_path) >= self.max_bytes:
                return latest_seq + 1
        except Exception:
            pass
        return latest_seq

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """
        Determine if rollover should occur.
        Rollover triggers when:
        1. Date changed (new day -> reset to new date seq).
        2. File size plus new formatted record exceeds max_bytes (-> next seq).
        """
        now_date = self._get_current_date()
        if now_date != self.current_date:
            return True

        if self.max_bytes > 0:
            if self.stream is None:
                self.stream = self._open()
            try:
                pos = self.stream.tell()
            except Exception:
                try:
                    pos = os.fstat(self.stream.fileno()).st_size
                except Exception:
                    pos = os.path.getsize(self.baseFilename) if os.path.exists(self.baseFilename) else 0

            msg = "%s\n" % self.format(record)
            msg_len = len(msg.encode(self.encoding or "utf-8", errors="replace"))
            if pos > 0 and (pos + msg_len >= self.max_bytes):
                return True

        return False

    def doRollover(self):
        """
        Execute rollover:
        - Flush and close current stream.
        - Advance date / seq.
        - Open new file stream.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        now_date = self._get_current_date()
        if now_date != self.current_date:
            self.current_date = now_date
            self.current_seq = self._find_or_start_seq(now_date)
        else:
            self.current_seq += 1

        self.baseFilename = self._build_filepath(self.current_date, self.current_seq)
        if not self.delay:
            self.stream = self._open()


# Setup logging: console + daily rotating log files in specified directory (yyyy-MM-dd.##.log, max 30MB)
LOG_DIR = os.getenv("LOG_DIR", os.getenv("LOG_FILE", "logs")).strip()
if LOG_DIR.lower().endswith(".log"):
    LOG_DIR = os.path.dirname(LOG_DIR) or "logs"

MAX_LOG_SIZE_MB = int(os.getenv("MAX_LOG_SIZE_MB", "30"))
MAX_LOG_BYTES = MAX_LOG_SIZE_MB * 1024 * 1024
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper().strip()

log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("copilot-openai-adapter")

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Daily size-rotating file handler with UTF-8 encoding
file_handler: Optional[DailySizeRotatingHandler] = None
try:
    file_handler = DailySizeRotatingHandler(
        log_dir=LOG_DIR,
        max_bytes=MAX_LOG_BYTES,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_formatter)
except Exception as e:
    print(f"Warning: Failed to setup rotating log directory '{LOG_DIR}': {e}")

handlers = [console_handler]
if file_handler:
    handlers.append(file_handler)


def set_log_level(level_name: str):
    """Update logger and handler log levels."""
    numeric_level = getattr(logging, (level_name or "INFO").upper().strip(), logging.INFO)
    logger.setLevel(numeric_level)
    for h in logger.handlers:
        h.setLevel(numeric_level)


def configure_log_handler(target_dir: str, max_mb: Optional[int] = None):
    """Reconfigure or update rotating file handler with a new directory or size limit."""
    global file_handler, LOG_DIR, MAX_LOG_SIZE_MB, MAX_LOG_BYTES
    LOG_DIR = target_dir
    if max_mb is not None:
        MAX_LOG_SIZE_MB = max_mb
        MAX_LOG_BYTES = MAX_LOG_SIZE_MB * 1024 * 1024

    if file_handler and file_handler in logger.handlers:
        logger.removeHandler(file_handler)
        file_handler.close()

    try:
        file_handler = DailySizeRotatingHandler(
            log_dir=LOG_DIR,
            max_bytes=MAX_LOG_BYTES,
            encoding="utf-8",
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to configure log handler for directory '{LOG_DIR}': {e}")


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


def analyze_stream_chunk(chunk: Union[str, bytes]) -> Tuple[bool, str, bool]:
    """
    Inspect a stream chunk.
    Returns: (is_error, error_message, has_real_content)
    """
    try:
        chunk_str = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
        for line in chunk_str.splitlines():
            line = line.strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                d = json.loads(line[5:].strip())
                if "error" in d:
                    return True, str(d["error"].get("message", "Upstream Error")), False
                choices = d.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content.startswith("\n[Stream Error:") or content.startswith("\n[Error from Gemini:"):
                        return True, content.strip(), False
                    if bool(content) or bool(delta.get("tool_calls")):
                        return False, "", True
    except Exception:
        pass
    return False, "", False


async def handle_fallback_chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    api_key: str,
):
    """
    Execute chat completions with multi-turn session degradation fallback strategy.
    """
    session_id, is_new = fallback_manager.resolve_session(req, dict(request.headers))
    fallback_manager.get_or_create_session(session_id, is_new=is_new)

    if not req.stream:
        while True:
            target_model = fallback_manager.get_active_model(session_id)
            cloned_req = req.model_copy(update={"model": target_model})
            logger.info(
                f"[FALLBACK] Session '{session_id}' -> executing non-streaming with model '{target_model}'"
            )
            try:
                result = await adapter.chat_completions(cloned_req, api_key)
                if isinstance(result, dict) and "error" in result:
                    status_code = result.get("error", {}).get("code", 400)
                    if not isinstance(status_code, int) or status_code >= 400:
                        err_msg = str(result["error"].get("message", "Error"))
                        next_model = fallback_manager.record_failure(session_id, error_desc=err_msg)
                        if next_model:
                            continue
                        return JSONResponse(
                            status_code=status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else 502,
                            content=result,
                        )

                fallback_manager.record_success(session_id)
                return result
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                next_model = fallback_manager.record_failure(session_id, error_desc=err_msg)
                if next_model:
                    continue
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"All fallback models failed. Last error: {err_msg}",
                            "type": "fallback_error",
                            "code": 502,
                        }
                    },
                )

    # Streaming SSE response with automatic pre-stream failure interception & fallback
    async def fallback_stream_generator():
        nonlocal session_id
        while True:
            target_model = fallback_manager.get_active_model(session_id)
            cloned_req = req.model_copy(update={"model": target_model})
            logger.info(
                f"[FALLBACK] Session '{session_id}' -> executing stream with model '{target_model}'"
            )

            stream_started = False
            model_failed = False
            fail_reason = ""
            buffered_initial_chunks = []

            try:
                stream_iter = adapter.chat_completions_stream(cloned_req, api_key)
                async for chunk in stream_iter:
                    if not stream_started:
                        is_err, err_msg, has_real_content = analyze_stream_chunk(chunk)
                        if is_err:
                            model_failed = True
                            fail_reason = err_msg
                            break
                        if has_real_content:
                            stream_started = True
                            for b_chunk in buffered_initial_chunks:
                                yield b_chunk
                            buffered_initial_chunks.clear()
                            yield chunk
                        else:
                            buffered_initial_chunks.append(chunk)
                    else:
                        yield chunk

            except Exception as exc:
                if not stream_started:
                    model_failed = True
                    fail_reason = f"{type(exc).__name__}: {exc}"
                else:
                    err_payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": target_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"\n[Stream Error: {type(exc).__name__}: {exc}]"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            if model_failed:
                buffered_initial_chunks.clear()
                next_model = fallback_manager.record_failure(session_id, error_desc=fail_reason)
                if next_model:
                    logger.warning(
                        f"[FALLBACK] Upstream error detected before stream output. Switching session '{session_id}' to '{next_model}'..."
                    )
                    continue
                else:
                    err_payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": fallback_manager.virtual_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"\n[All fallback models failed. Last error: {fail_reason}]"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(err_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            else:
                fallback_manager.record_success(session_id)
                return

    return StreamingResponse(
        fallback_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "provider": adapter.provider_name}


@app.get("/models")
@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models in OpenAI standard format, including virtual fallback model."""
    api_key = adapter.resolve_api_key(authorization)
    data = await adapter.list_models(api_key)
    # Prepend virtual model entry if not already in list
    v_id = fallback_manager.virtual_model_name
    if not any(m.get("id") == v_id for m in data):
        data.insert(
            0,
            {
                "id": v_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "adapter",
                "permission": [],
                "root": v_id,
                "parent": None,
            },
        )
    return {"object": "list", "data": data}


@app.get("/models/{model_id:path}")
@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    """Retrieve specific model details in OpenAI standard format."""
    if fallback_manager.is_virtual_model(model_id):
        return {
            "id": fallback_manager.virtual_model_name,
            "object": "model",
            "created": 1700000000,
            "owned_by": "adapter",
            "permission": [],
            "root": fallback_manager.virtual_model_name,
            "parent": None,
        }
    api_key = adapter.resolve_api_key(authorization)
    model_info = await adapter.get_model(model_id, api_key)
    return model_info


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Handle OpenAI chat completions with streaming, function calling, virtual fallback, and provider routing."""
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

    effective_model = req.model or adapter.default_model
    if fallback_manager.is_virtual_model(effective_model):
        return await handle_fallback_chat_completions(req, request, api_key)

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

    effective_model = req.model or adapter.default_model
    if fallback_manager.is_virtual_model(effective_model):
        for model_name in fallback_manager.fallback_models:
            cloned_req = req.model_copy(update={"model": model_name})
            result = await adapter.completions(cloned_req, api_key)
            if isinstance(result, dict) and "error" in result:
                continue
            return result
        return JSONResponse(status_code=400, content=result)

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
    parser.add_argument(
        "--virtual-model",
        default=None,
        help="Virtual fallback model name (default: auto)",
    )
    parser.add_argument(
        "--fallback-models",
        default=None,
        help="Comma-separated model list for fallback (default: gemini-4-hydrogen,gemini-3.8-flash)",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=int,
        default=None,
        help="Consecutive failure threshold before degrading to next model (default: 1)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to store daily rotating log files (default: logs)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Deprecated: please use --log-dir instead",
    )
    parser.add_argument(
        "--max-log-size-mb",
        type=int,
        default=None,
        help="Maximum size in MB per log file before sequence increment (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "debug", "info", "warning", "error"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)",
    )

    args = parser.parse_args()

    global adapter, LOG_DIR, MAX_LOG_SIZE_MB
    if args.provider:
        adapter = get_adapter(args.provider)
    if args.key:
        if hasattr(adapter, "api_key_env"):
            adapter.api_key_env = args.key
    if args.model:
        if hasattr(adapter, "_default_model"):
            adapter._default_model = args.model

    target_log_dir = args.log_dir
    if not target_log_dir and args.log_file:
        target_log_dir = (
            os.path.dirname(args.log_file)
            if args.log_file.lower().endswith(".log")
            else args.log_file
        )

    if target_log_dir or args.max_log_size_mb:
        configure_log_handler(
            target_dir=target_log_dir or LOG_DIR,
            max_mb=args.max_log_size_mb,
        )

    fallback_models_list = (
        [m.strip() for m in args.fallback_models.split(",") if m.strip()]
        if args.fallback_models
        else None
    )
    fallback_manager.reconfigure(
        virtual_model_name=args.virtual_model,
        fallback_models=fallback_models_list,
        failure_threshold=args.fallback_threshold,
    )

    if args.log_level:
        set_log_level(args.log_level)

    resolved_key = adapter.resolve_api_key()
    masked_key = mask_secret(resolved_key) if resolved_key else "(not set - configure in .env)"
    current_level = logging.getLevelName(logger.level)
    active_log_path = os.path.abspath(file_handler.baseFilename) if file_handler else "(disabled)"

    print("=" * 60)
    print(" Copilot OpenAI Adapter Server (Multi-Provider)")
    print("=" * 60)
    print(f" AI Provider        : {adapter.provider_name}")
    print(f" API Key            : {masked_key}")
    print(f" Default Model      : {adapter.default_model}")
    print(f" Virtual Model      : {fallback_manager.virtual_model_name}")
    print(f" Fallback Models    : {' -> '.join(fallback_manager.fallback_models)}")
    print(f" Fallback Threshold : {fallback_manager.failure_threshold} failure(s)")
    print(f" Log Level          : {current_level}")
    print(f" Listening Address  : http://{args.host}:{args.port}")
    print(f" OpenAI Base URL    : http://127.0.0.1:{args.port}/v1")
    print(f" Models Endpoint    : http://127.0.0.1:{args.port}/v1/models")
    print(f" Log Directory      : {os.path.abspath(LOG_DIR)}")
    print(f" Active Log File    : {active_log_path}")
    print(f" Max File Size      : {MAX_LOG_SIZE_MB} MB (Rollover: yyyy-MM-dd.##.log)")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
