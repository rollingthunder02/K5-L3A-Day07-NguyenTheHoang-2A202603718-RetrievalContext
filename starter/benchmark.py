"""Run Day 7 retrieval and the real agent on a selected benchmark split."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from agentic import run_agent
from baseline import hit
from controller import ChatController
from retrievers import DenseRetriever, HybridRetriever, LexicalRetriever


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "dense-base": ("dense", 800, 1),
    "tfidf-base": ("tfidf", 800, 1),
    "hybrid-base": ("hybrid", 800, 1),
    "hybrid-budget": ("hybrid", 2000, 1),
    "hybrid-multisearch": ("hybrid", 2000, 3),
}


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8-sig"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_case(case, retriever, controller, budget, max_searches):
    started = perf_counter()
    usage_start = len(controller.usage)
    question = case["question"]
    trace = {
        "query_id": case["id"], "split": case["split"], "question": question,
        "model": controller.model, "top_k": 3, "budget_chars": budget,
        "max_searches": max_searches, "events": [], "citations": [],
    }
    try:
        ranking = retriever.search(question, top_k=3)
        trace["query_top3"] = ranking
        trace["ranking_ms"] = (perf_counter() - started) * 1000
        agent_start = perf_counter()
        trace.update(run_agent(question, controller, retriever, top_k=3,
                               budget=budget, max_searches=max_searches))
        trace["agent_ms"] = (perf_counter() - agent_start) * 1000
        if case.get("gold"):
            gold = case["gold"]
            final_search = next((e for e in reversed(trace["events"]) if e["event"] == "search"), {})
            packed = set(final_search.get("context_ids", []))
            trace.update(
                hit1=bool(ranking and hit(ranking[0], gold)),
                hit3=any(hit(c, gold) for c in ranking),
                context_hit=any(c["id"] in packed and hit(c, gold)
                                for e in trace["events"] if e["event"] == "search"
                                for c in e["results"]),
            )
    except (RuntimeError, ValueError, OSError, KeyError, IndexError) as error:
        trace.update(status="error", error_type=type(error).__name__, error=str(error))
    trace["chat_usage"] = controller.usage[usage_start:]
    trace["elapsed_ms"] = (perf_counter() - started) * 1000
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day6-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", nargs="+", help="Chọn ID để chạy smoke test")
    args = parser.parse_args()

    import psycopg

    load_dotenv(ROOT.parent / ".env")
    load_dotenv(args.day6_root / "starter" / ".env")
    method, budget, max_searches = CONFIGS[args.config]
    base_url = os.getenv("AGENT_BASE_URL", "https://api.openai.com/v1")
    key_env = os.getenv("AGENT_API_KEY_ENV", "OPENAI_API_KEY")
    if not os.getenv(key_env):
        parser.error(f"Thiếu biến môi trường {key_env}.")
    model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
    selected = [row for name in ("questions_answerable.json", "questions_unanswerable.json")
                for row in load(name) if row["split"] == args.split
                and (not args.ids or row["id"] in args.ids)]
    if not selected or (args.ids and set(args.ids) != {q["id"] for q in selected}):
        parser.error("Không tìm thấy tất cả ID trong split đã chọn.")

    try:
        lexical = LexicalRetriever(load("chunks.json")) if method != "dense" else None
        dense = DenseRetriever(args.day6_root) if method != "tfidf" else None
    except (ValueError, OSError, KeyError, psycopg.Error) as error:
        parser.exit(1, f"Không khởi tạo được retrieval: {error}\n")
    if method == "hybrid":
        retriever = HybridRetriever(dense, lexical)
    else:
        retriever = dense if method == "dense" else lexical
    controller = ChatController(base_url, os.environ[key_env], model, max_searches)
    traces = []
    for case in selected:
        try:
            trace = run_case(case, retriever, controller, budget, max_searches)
        except psycopg.Error as error:
            trace = {"query_id": case["id"], "split": args.split, "status": "error",
                     "error_type": type(error).__name__, "error": "PostgreSQL retrieval failed.",
                     "events": [], "citations": []}
        trace["config"] = args.config
        traces.append(trace)
        print(f"{case['id']}: {trace['status']} — {trace.get('answer', trace.get('error', ''))}")
    save(args.output / "traces.json", traces)
    if any(t["status"] == "error" for t in traces):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
