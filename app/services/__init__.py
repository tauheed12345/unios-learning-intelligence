from app.services.llm_provider import (
    BaseLLMProvider,
    MockLLMProvider,
    GroqLLMProvider,
    get_llm_provider,
    LLMProviderError,
    LLMParseError,
    extract_json_payload,
    normalize_intelligence_dict,
)
from app.services.prompt_builder import (
    get_stage_guidelines,
    build_pedagogy_prompt,
    build_lesson_prompt,
)
from app.services.onboarding_prompt_builder import (
    build_onboarding_intelligence_prompt,
)

from app.services.memory_repository import (
    BaseMemoryRepository,
    InMemoryMemoryRepository,
)
from app.services.memory_engine import MemoryEngine

_global_memory_repository = InMemoryMemoryRepository()
_global_memory_engine = MemoryEngine(repository=_global_memory_repository)


def get_memory_repository() -> BaseMemoryRepository:
    """Provides the active memory repository instance."""
    return _global_memory_repository


def get_memory_engine(repository: BaseMemoryRepository = None) -> MemoryEngine:
    """Provides the active MemoryEngine instance."""
    if repository:
        return MemoryEngine(repository=repository)
    return _global_memory_engine


__all__ = [
    "BaseLLMProvider",
    "MockLLMProvider",
    "GroqLLMProvider",
    "get_llm_provider",
    "LLMProviderError",
    "LLMParseError",
    "extract_json_payload",
    "normalize_intelligence_dict",
    "get_stage_guidelines",
    "build_pedagogy_prompt",
    "build_lesson_prompt",
    "build_onboarding_intelligence_prompt",
    "BaseMemoryRepository",
    "InMemoryMemoryRepository",
    "MemoryEngine",
    "get_memory_repository",
    "get_memory_engine",
]
