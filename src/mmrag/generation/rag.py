"""Retrieval-augmented question answering with modality routing.

Route selection:
  * "text":   no charts/figures were retrieved -> the text LLM answers from the context.
  * "visual": at least one chart/figure was retrieved -> the VLM receives the context
              AND the original images, so it can read values the summary may have missed.
"""

import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from mmrag.config import PROJECT_ROOT, get_settings
from mmrag.generation.prompts import SYSTEM_PROMPT, USER_PROMPT
from mmrag.ingestion.image_utils import image_to_data_uri
from mmrag.llm import text_llm, vision_llm
from mmrag.retrieval.query_rewriter import plan_queries
from mmrag.retrieval.retriever import format_context, retrieve_many
from mmrag.schemas import RAGResponse, Source


def choose_route(docs: list[Document]) -> str:
    return "visual" if any(d.metadata["modality"] == "visual" for d in docs) else "text"


def build_message(question: str, docs: list[Document]) -> HumanMessage:
    """User message with the numbered context, plus the original image of each visual source."""
    content = [{"type": "text", "text": USER_PROMPT.format(context=format_context(docs), question=question)}]

    for number, doc in enumerate(docs, start=1):
        if doc.metadata["modality"] == "visual":
            content.append({"type": "text", "text": f"Image for source [{number}]:"})
            data_uri = image_to_data_uri(PROJECT_ROOT / doc.metadata["image_path"])
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

    return HumanMessage(content=content)


def to_source(number: int, doc: Document) -> Source:
    meta = doc.metadata
    return Source(
        number=number,
        source=meta["source"],
        page=int(meta["page"]),
        modality=meta["modality"],
        score=meta["score"],
        chunk_id=meta["chunk_id"],
        content=doc.page_content,
        image_path=meta.get("image_path"),
    )


def add_usage(*usages: dict) -> dict:
    """Sum token counts from several LLM calls."""
    keys = ("input_tokens", "output_tokens", "total_tokens")
    return {key: sum(u.get(key, 0) for u in usages) for key in keys}


def ask_rag(question: str, history: list[dict] | None = None, k: int | None = None) -> RAGResponse:
    """Answer a question, using earlier chat messages to resolve follow-ups.

    Steps: rewrite into search queries -> retrieve for each -> pick a route -> generate.
    history: earlier messages as {"role": "user" | "assistant", "content": str}.
    """
    settings = get_settings()
    times = [time.perf_counter()]

    plan, rewrite_usage = plan_queries(question, history)
    times.append(time.perf_counter())

    docs = retrieve_many(plan.search_queries, k)
    times.append(time.perf_counter())

    route = choose_route(docs)
    llm, model = (vision_llm(), settings.vision_model) if route == "visual" else (text_llm(), settings.text_model)
    # The standalone question is answered, so the chat history is not needed here.
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), build_message(plan.question, docs)])
    times.append(time.perf_counter())

    return RAGResponse(
        question=question,
        rewritten_question=plan.question,
        search_queries=plan.search_queries,
        answer=reply.content.strip(),
        route=route,
        model=model,
        sources=[to_source(i, d) for i, d in enumerate(docs, start=1)],
        timings={
            "rewrite": round(times[1] - times[0], 2),
            "retrieval": round(times[2] - times[1], 2),
            "generation": round(times[3] - times[2], 2),
            "total": round(times[3] - times[0], 2),
        },
        usage=add_usage(rewrite_usage, reply.usage_metadata or {}),
    )


if __name__ == "__main__":
    import sys

    response = ask_rag(" ".join(sys.argv[1:]) or "Which campus region has the highest PUE?")
    print(response.answer)
    print(f"\nsearch queries={response.search_queries}")
    print(f"route={response.route}  model={response.model}  timings={response.timings}")
    for s in response.sources:
        print(f"  [{s.number}] {s.score:.3f}  {s.modality:6}  p{s.page}  {s.source}")
