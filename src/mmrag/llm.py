"""Chat model clients.

OpenRouter exposes an OpenAI-compatible API, so the standard ChatOpenAI client
works by pointing base_url at OpenRouter.
"""

from langchain_openai import ChatOpenAI

from mmrag.config import get_settings


def get_llm(model: str) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=2,
        # OpenRouter-specific option, passed through in the request body.
        extra_body={"reasoning": {"effort": settings.llm_reasoning_effort}},
    )


def text_llm() -> ChatOpenAI:
    return get_llm(get_settings().text_model)


def vision_llm() -> ChatOpenAI:
    return get_llm(get_settings().vision_model)
