import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("copilot-openai-adapter")


class SessionState:
    """Tracks model degradation state for a single conversation session."""

    def __init__(self, session_id: str, initial_model: str):
        self.session_id = session_id
        self.model_index: int = 0
        self.consecutive_failures: int = 0
        self.last_active: float = time.time()
        self.history: List[str] = [initial_model]


class FallbackManager:
    """
    Manages virtual model degradation strategy across multi-turn sessions.
    - Exposes a virtual model (default: 'auto').
    - Sequentially tries configured fallback models (e.g. gemini-4-hydrogen, gemini-3.8-flash).
    - Degrades to the next model if consecutive failures reach threshold.
    - Automatically resets to the first model when a new session begins.
    """

    def __init__(
        self,
        virtual_model_name: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        failure_threshold: Optional[int] = None,
        session_ttl: int = 1800,
    ):
        self.virtual_model_name = (
            virtual_model_name
            or os.getenv("VIRTUAL_MODEL_NAME", "auto").strip().lower()
        )
        raw_models = fallback_models or [
            m.strip()
            for m in os.getenv(
                "FALLBACK_MODELS", "gemini-4-hydrogen,gemini-3.8-flash"
            ).split(",")
            if m.strip()
        ]
        self.fallback_models: List[str] = (
            raw_models if raw_models else ["gemini-4-hydrogen", "gemini-3.8-flash"]
        )
        self.failure_threshold = int(
            failure_threshold
            if failure_threshold is not None
            else os.getenv("FALLBACK_THRESHOLD", "1")
        )
        self.session_ttl = session_ttl
        self.sessions: Dict[str, SessionState] = {}

    def reconfigure(
        self,
        virtual_model_name: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        failure_threshold: Optional[int] = None,
    ):
        """Update fallback configuration dynamically from CLI or runtime."""
        if virtual_model_name:
            self.virtual_model_name = virtual_model_name.strip().lower()
        if fallback_models:
            self.fallback_models = fallback_models
        if failure_threshold is not None:
            self.failure_threshold = failure_threshold
        logger.info(
            f"[FALLBACK] Configured virtual model '{self.virtual_model_name}' -> models: {self.fallback_models}, threshold: {self.failure_threshold}"
        )

    def is_virtual_model(self, model_name: Optional[str]) -> bool:
        """Check if requested model matches virtual model or its standard aliases."""
        if not model_name:
            return False
        name = model_name.strip().lower()
        if name.startswith("models/"):
            name = name[7:]
        return name in (self.virtual_model_name, "auto", "fallback-auto", "virtual-model")

    def resolve_session(
        self, req: Any, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[str, bool]:
        """
        Identify session ID and determine whether this is a new conversation.
        Returns: (session_id, is_new_session)
        """
        self._cleanup_expired_sessions()
        headers = headers or {}

        # 1. Check explicit headers from clients
        session_id = None
        for hdr in (
            "x-session-id",
            "x-conversation-id",
            "x-copilot-conversation-id",
            "session-id",
            "conversation-id",
        ):
            val = headers.get(hdr)
            if val and val.strip():
                session_id = val.strip()
                break

        # 2. Check body user / session_id
        if not session_id and hasattr(req, "user") and req.user:
            session_id = str(req.user).strip()

        messages = getattr(req, "messages", [])
        has_assistant = False
        first_user_content = ""

        for m in messages:
            role = (
                getattr(m, "role", None)
                if hasattr(m, "role")
                else (m.get("role") if isinstance(m, dict) else None)
            )
            content = (
                getattr(m, "content", "")
                if hasattr(m, "content")
                else (m.get("content", "") if isinstance(m, dict) else "")
            )
            if role in ("assistant", "tool"):
                has_assistant = True
            elif role == "user" and not first_user_content:
                if isinstance(content, str):
                    first_user_content = content
                elif isinstance(content, list):
                    first_user_content = json.dumps(content, ensure_ascii=False)

        if not session_id:
            if first_user_content:
                h = hashlib.md5(
                    first_user_content.strip().encode("utf-8", errors="ignore")
                ).hexdigest()[:12]
                session_id = f"sess_{h}"
            else:
                session_id = "sess_default"

        # If message history has NO assistant / tool responses, it is a brand new conversation!
        is_new_session = not has_assistant
        return session_id, is_new_session

    def get_or_create_session(
        self, session_id: str, is_new: bool = False
    ) -> SessionState:
        now = time.time()
        if is_new or session_id not in self.sessions:
            state = SessionState(session_id, self.fallback_models[0])
            self.sessions[session_id] = state
            if is_new:
                logger.info(
                    f"[FALLBACK] New session '{session_id}' started -> reset to initial model '{self.fallback_models[0]}'"
                )
            return state

        state = self.sessions[session_id]
        state.last_active = now
        return state

    def get_active_model(self, session_id: str) -> str:
        state = self.get_or_create_session(session_id)
        idx = min(state.model_index, len(self.fallback_models) - 1)
        return self.fallback_models[idx]

    def record_success(self, session_id: str):
        """Reset consecutive failures upon successful response."""
        if session_id in self.sessions:
            state = self.sessions[session_id]
            state.consecutive_failures = 0
            state.last_active = time.time()

    def record_failure(
        self, session_id: str, error_desc: str = ""
    ) -> Optional[str]:
        """
        Record a failure for the session's active model.
        If consecutive failure count reaches threshold, degrade to next model.
        Returns the new active model name if degraded, or None if no more models.
        """
        state = self.get_or_create_session(session_id)
        state.consecutive_failures += 1
        state.last_active = time.time()
        current_model = self.fallback_models[state.model_index]

        logger.warning(
            f"[FALLBACK] Session '{session_id}' model '{current_model}' failure {state.consecutive_failures}/{self.failure_threshold} ({error_desc})"
        )

        if state.consecutive_failures >= self.failure_threshold:
            if state.model_index + 1 < len(self.fallback_models):
                state.model_index += 1
                state.consecutive_failures = 0
                next_model = self.fallback_models[state.model_index]
                state.history.append(next_model)
                logger.warning(
                    f"[FALLBACK] Threshold reached. Session '{session_id}' DEGRADED from '{current_model}' to '{next_model}' (tier {state.model_index + 1}/{len(self.fallback_models)})"
                )
                return next_model
            else:
                logger.error(
                    f"[FALLBACK] Session '{session_id}' exhausted all {len(self.fallback_models)} fallback models."
                )
                return None
        return None

    def _cleanup_expired_sessions(self):
        now = time.time()
        expired = [
            sid
            for sid, s in self.sessions.items()
            if now - s.last_active > self.session_ttl
        ]
        for sid in expired:
            del self.sessions[sid]


# Global instance
fallback_manager = FallbackManager()
