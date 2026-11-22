"""Chat model clients.

OpenRouter exposes an OpenAI-compatible API, so the standard ChatOpenAI client
works by pointing base_url at OpenRouter.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from mmrag.config import get_settings


def get_llm(model: str, timeout: int | None = None, max_retries: int = 2) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
        timeout=timeout or settings.llm_timeout,
        max_retries=max_retries,
        # OpenRouter-specific option, passed through in the request body.
        extra_body={"reasoning": {"effort": settings.llm_reasoning_effort}},
    )


def invoke_hedged(runnable: Runnable, prompt, hedge_after: float, timeout: float):
    """Invoke a model call, sending a backup request if the first one is slow.

    Hosted models often have a long latency tail (most replies in ~2s, some stuck
    in a provider queue for 30s+). If no reply arrives within `hedge_after`
    seconds, an identical second request is sent and whichever finishes first
    is returned ("hedged requests"). Raises TimeoutError after `timeout` seconds.
    This is a wall-clock limit: an HTTP timeout alone can be reset by keep-alive
    bytes that some providers send while a request is queued.
    """
    pool = ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(runnable.invoke, prompt)]
    done, _ = wait(futures, timeout=hedge_after)
    if not done:
        futures.append(pool.submit(runnable.invoke, prompt))
        done, _ = wait(futures, timeout=timeout - hedge_after, return_when=FIRST_COMPLETED)
    pool.shutdown(wait=False)  # losing or abandoned calls finish in the background
    if not done:
        raise TimeoutError(f"no reply within {timeout}s")
    return done.pop().result()


def text_llm() -> ChatOpenAI:
    return get_llm(get_settings().text_model)


def vision_llm() -> ChatOpenAI:
    return get_llm(get_settings().vision_model)
