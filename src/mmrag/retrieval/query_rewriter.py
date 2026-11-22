"""Turn the user's message into search queries before retrieval.

One LLM call fixes two retrieval problems:
  * follow-ups   "What is its battery life?" cannot be searched on its own, so it is
                 rewritten into a standalone question using the chat history.
  * multi-part   a question combining facts from different places ("How much larger
                 was A than B?") tends to retrieve chunks for only one of them, so it
                 is split into one search query per fact. Each query is retrieved
                 separately and the results are merged.
"""

from pydantic import BaseModel, Field

from mmrag.config import get_settings
from mmrag.llm import get_llm, invoke_hedged

REWRITE_PROMPT = """You prepare search queries for a document search system. Do not answer the question.

Conversation so far:
{history}

Latest question: {question}

1. Rewrite the latest question so it is understandable without the conversation:
   replace references such as "it", "that product" or "and in 2024?" with what they refer to.
   If it is already standalone, keep it exactly as it is.
2. Write the search queries:
   - If the answer needs one fact, return one query: the rewritten question.
   - If it combines facts that may be in different documents (a comparison, sum,
     difference or ratio of separate figures), return one short query per fact,
     at most {max_queries}."""


class QueryPlan(BaseModel):
    question: str = Field(description="The latest question, rewritten to be standalone.")
    search_queries: list[str] = Field(description="One search query per fact needed to answer.")


def format_history(history: list[dict], turns: int) -> str:
    """The last `turns` question/answer pairs as "User: ..." / "Assistant: ..." lines."""
    recent = history[-2 * turns:] if turns else []
    lines = [f"{m['role'].capitalize()}: {m['content'][:500]}" for m in recent]
    return "\n".join(lines) or "(no previous messages)"


def plan_queries(question: str, history: list[dict] | None = None) -> tuple[QueryPlan, dict]:
    """Return the search plan and the token usage of the rewrite call.

    Falls back to searching with the original question when rewriting is
    disabled, times out or returns nothing usable, so a slow model never
    blocks an answer.
    """
    settings = get_settings()
    fallback = QueryPlan(question=question, search_queries=[question])
    if not settings.query_rewrite:
        return fallback, {}

    prompt = REWRITE_PROMPT.format(
        history=format_history(history or [], settings.history_turns),
        question=question,
        max_queries=settings.max_search_queries,
    )
    # include_raw=True keeps the raw reply so its token usage can be reported.
    chain = get_llm(settings.text_model, max_retries=0).with_structured_output(
        QueryPlan, method="json_schema", include_raw=True
    )
    try:
        result = invoke_hedged(chain, prompt, settings.llm_hedge_after, settings.rewrite_timeout)
    except Exception:  # timeout or API error
        return fallback, {}

    plan, usage = result["parsed"], result["raw"].usage_metadata or {}
    if plan is None or not plan.search_queries:
        return fallback, usage
    if len(plan.search_queries) == 1:
        # Single fact: search with the full question. Rerankers are trained on natural
        # questions and score keyword-style queries worse.
        plan.search_queries = [plan.question]
    plan.search_queries = plan.search_queries[: settings.max_search_queries]
    return plan, usage


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "How much does the Atlas-M robot weigh?"},
        {"role": "assistant", "content": "The Atlas-M weighs 86 kg [1]."},
    ]
    for q in ["What is its reference battery endurance?", "How much larger was A's revenue than B's CapEx?"]:
        plan, _ = plan_queries(q, history)
        print(f"{q}\n  -> {plan.question}\n  -> {plan.search_queries}\n")
