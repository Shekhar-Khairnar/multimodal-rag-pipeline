"""Run the evaluation set through the RAG pipeline and report quality and latency.

Usage:
    python evals/run_evals.py --label baseline
    python evals/run_evals.py --label k5 --k 5
    python evals/run_evals.py --label r20 --k 20 --retrieval-only   # no LLM calls, seconds to run

Metrics
  recall    share of a question's evidence strings that appear in the retrieved chunks.
            Evidence strings are the exact text the answer depends on (e.g. "1.14"),
            so labels stay valid when chunking changes.
  mrr       mean reciprocal rank: 1 / rank of the first retrieved chunk containing evidence
            (1.0 = it was ranked first, 0.5 = second, 0 = not retrieved).
  accuracy  unanswerable questions must abstain ("Not found..."). Answerable questions
            fail on abstention; otherwise an LLM judge compares with the reference answer.

Per-question results and a summary (with the configuration used) are saved to
evals/results/ so runs can be compared.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime

from langchain_core.messages import HumanMessage

from mmrag.config import PROJECT_ROOT, get_settings
from mmrag.generation.rag import ask_rag
from mmrag.llm import text_llm
from mmrag.retrieval.retriever import retrieve

QA_PATH = PROJECT_ROOT / "data" / "ground_truth" / "qa.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
NOT_FOUND = "not found in the documents"

JUDGE_PROMPT = """You are grading an answer to a question about business documents.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

The answer is correct if it states the same key facts as the reference (same numbers, names or conclusion).
Different wording, extra detail and small rounding differences are fine.
A missing, different or contradictory key fact makes it incorrect.

Reply with exactly one word: CORRECT or INCORRECT."""


def retrieval_scores(evidence: list[str], chunks: list[str]) -> tuple[float | None, float | None]:
    """Return (recall, reciprocal rank) for one question. None for unanswerable questions."""
    if not evidence:
        return None, None
    recall = sum(any(e in c for c in chunks) for e in evidence) / len(evidence)
    first_hit = next((rank for rank, c in enumerate(chunks, start=1) if any(e in c for e in evidence)), None)
    return recall, (1 / first_hit if first_hit else 0.0)


def is_correct(item: dict, answer: str) -> bool:
    abstained = NOT_FOUND in answer.lower()
    if item["category"] == "unanswerable":
        return abstained
    if abstained:
        return False
    prompt = JUDGE_PROMPT.format(question=item["question"], reference=item["answer"], answer=answer)
    verdict = text_llm().invoke([HumanMessage(prompt)]).content.upper()
    return "CORRECT" in verdict and "INCORRECT" not in verdict


def run_question(item: dict, k: int, retrieval_only: bool) -> dict:
    row = {"qid": item["qid"], "category": item["category"], "question": item["question"]}

    if retrieval_only:
        start = time.perf_counter()
        docs = retrieve(item["question"], k)
        row["latency"] = round(time.perf_counter() - start, 3)
        chunks = [d.page_content for d in docs]
        row["retrieved"] = [d.metadata["chunk_id"] for d in docs]
    else:
        response = ask_rag(item["question"], k=k)
        chunks = [s.content for s in response.sources]
        row.update({
            "reference": item["answer"],
            "answer": response.answer,
            "correct": is_correct(item, response.answer),
            "route": response.route,
            "latency": response.timings["total"],
            "tokens": response.usage.get("total_tokens", 0),
            "retrieved": [s.chunk_id for s in response.sources],
        })

    row["recall"], row["rr"] = retrieval_scores(item["evidence"], chunks)
    return row


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["recall"] is not None]
    latencies = sorted(r["latency"] for r in rows)
    summary = {
        "n": len(rows),
        "recall": round(statistics.mean(r["recall"] for r in answerable), 3) if answerable else None,
        "mrr": round(statistics.mean(r["rr"] for r in answerable), 3) if answerable else None,
        "latency_p50": round(statistics.median(latencies), 3),
    }
    if "correct" in rows[0]:
        summary["accuracy"] = round(sum(r["correct"] for r in rows) / len(rows), 3)
        summary["tokens"] = sum(r["tokens"] for r in rows)
    return summary


def print_table(summary: dict) -> None:
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"\n{'category':16} {'n':>3} {'accuracy':>9} {'recall':>7} {'mrr':>6} {'p50 s':>6}")
    for name, s in [*summary["by_category"].items(), ("OVERALL", summary["overall"])]:
        print(f"{name:16} {s['n']:>3} {fmt(s.get('accuracy')):>9} {fmt(s['recall']):>7} "
              f"{fmt(s['mrr']):>6} {s['latency_p50']:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="name for this run, e.g. baseline")
    parser.add_argument("--k", type=int, default=None, help="chunks to retrieve (default: settings.top_k)")
    parser.add_argument("--retrieval-only", action="store_true", help="score retrieval only, no LLM calls")
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.top_k
    items = [json.loads(line) for line in QA_PATH.open(encoding="utf-8")]

    rows = []
    for item in items:
        row = run_question(item, k, args.retrieval_only)
        rows.append(row)
        if not args.retrieval_only:
            print(f"{'PASS' if row['correct'] else 'FAIL'}  {row['qid']}  {row['category']:15} "
                  f"{row['route']:6} {row['latency']:5.1f}s")

    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    summary = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "k": k,
            "retrieval_only": args.retrieval_only,
            "hybrid_search": settings.hybrid_search,
            "rerank": settings.rerank,
            "retrieval_candidates": settings.retrieval_candidates,
            "text_model": settings.text_model,
            "vision_model": settings.vision_model,
            "embedding_model": settings.embedding_model,
        },
        "overall": summarize(rows),
        "by_category": {cat: summarize(r) for cat, r in sorted(by_category.items())},
    }
    print_table(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{args.label}"
    (RESULTS_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (RESULTS_DIR / f"{stem}_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved to evals/results/{stem}_*")


if __name__ == "__main__":
    main()
