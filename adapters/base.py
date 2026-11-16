from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional


class BaseAIAdapter(ABC):
    """
    Abstract base class for AI Model Adapters.
    New provider adapters (e.g., Claude, DeepSeek, Ollama) should inherit from this.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'aistudio_gemini')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name for this provider."""
        pass

    @abstractmethod
    def resolve_api_key(self, authorization: Optional[str] = None) -> str:
        """Extract or fallback to the provider's API key."""
        pass

    @abstractmethod
    async def list_models(self, api_key: str) -> List[Dict[str, Any]]:
        """List models in OpenAI standard format."""
        pass

    @abstractmethod
    async def get_model(self, model_id: str, api_key: str) -> Dict[str, Any]:
        """Retrieve specific model details in OpenAI standard format."""
        pass

    @abstractmethod
    async def chat_completions(
        self,
        req: Any,
        api_key: str,
    ) -> Dict[str, Any]:
        """Handle non-streaming chat completions."""
        pass

    @abstractmethod
    async def chat_completions_stream(
        self,
        req: Any,
        api_key: str,
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat completions yielding OpenAI SSE lines."""
        pass

    @abstractmethod
    async def completions(
        self,
        req: Any,
        api_key: str,
    ) -> Dict[str, Any]:
        """Handle legacy code completions (inline/FIM)."""
        pass

    @abstractmethod
    async def embeddings(
        self,
        req: Any,
        api_key: str,
    ) -> Dict[str, Any]:
        """Handle embeddings requests."""
        pass
