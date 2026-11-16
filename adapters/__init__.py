from typing import Dict, Type
from .base import BaseAIAdapter
from .aistudio_gemini import AIStudioGeminiAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseAIAdapter]] = {
    "aistudio_gemini": AIStudioGeminiAdapter,
    "gemini": AIStudioGeminiAdapter,
}


def get_adapter(provider_name: str) -> BaseAIAdapter:
    """
    Factory function to get an initialized AI adapter instance by provider name.
    """
    normalized_name = (provider_name or "").strip().lower()
    adapter_cls = ADAPTER_REGISTRY.get(normalized_name)
    if not adapter_cls:
        available = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported AI provider: '{provider_name}'. Available providers: {available}"
        )
    return adapter_cls()


__all__ = ["BaseAIAdapter", "AIStudioGeminiAdapter", "ADAPTER_REGISTRY", "get_adapter"]
