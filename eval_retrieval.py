import argparse
import contextlib
import json
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HopRAG retrieval on a JSONL QA dataset.")
    parser.add_argument("--label", required=True, help="Neo4j node label to evaluate.")
    parser.add_argument(
        "--relationship",
        default=None,
        help="Neo4j relationship type. Defaults to pen2ans_<label>.",
    )
    parser.add_argument("--data", default="quickstart_dataset/hotpot_example.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--traversal", default="bfs_sim_node")
    parser.add_argument("--entry-type", default="node")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-hop", type=int, default=2)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show retriever internal prints.",
    )
    return parser.parse_args()


def load_questions(path, limit):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
            if limit and len(questions) >= limit:
                break
    return questions


def supporting_sentences(example):
    title_to_sentences = {title: sentences for title, sentences in example.get("context", [])}
    sentences = []
    for title, sentence_idx in example.get("supporting_facts", []):
        doc_sentences = title_to_sentences.get(title)
        if doc_sentences is None or sentence_idx >= len(doc_sentences):
            continue
        sentences.append(
            {
                "title": title,
                "sentence_idx": sentence_idx,
                "sentence": doc_sentences[sentence_idx].strip(),
            }
        )
    return sentences


def text_hit(needle, haystack):
    needle = " ".join(needle.lower().split())
    haystack = " ".join(haystack.lower().split())
    return needle in haystack


def evaluate_hit(example, retrieved_contexts):
    supports = supporting_sentences(example)
    support_hits = []
    for support in supports:
        hit = any(text_hit(support["sentence"], context) for context in retrieved_contexts)
        support_hits.append({**support, "hit": hit})
    return support_hits


def main():
    args = parse_args()
    os.environ["HOPRAG_NODE_NAME"] = args.label
    if args.relationship:
        # config.py derives edge_name from HOPRAG_NODE_NAME, so this script currently assumes
        # the standard pen2ans_<label> naming convention used by graph_augment.py.
        expected = f"pen2ans_{args.label}"
        if args.relationship != expected:
            raise ValueError(
                f"Custom relationship {args.relationship!r} is not supported by config.py yet. "
                f"Use the standard relationship name {expected!r}."
            )

    from HopRetriever import HopRetriever
    from config import edge_name, local_model_name, node_name

    questions = load_questions(args.data, args.limit)
    retriever = HopRetriever(
        llm=local_model_name,
        max_hop=args.max_hop,
        entry_type=args.entry_type,
        if_hybrid=False,
        topk=args.topk,
        traversal=args.traversal,
    )

    results = []
    start = time.time()
    for idx, example in enumerate(questions, start=1):
        query = example["question"]
        item_start = time.time()
        if args.verbose:
            contexts, scores = retriever.search_docs(query)
        else:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull):
                    contexts, scores = retriever.search_docs(query)
        elapsed = time.time() - item_start

        support_hits = evaluate_hit(example, contexts)
        hit_count = sum(1 for hit in support_hits if hit["hit"])
        required_count = len(support_hits)
        results.append(
            {
                "id": example.get("_id"),
                "question": query,
                "answer": example.get("answer"),
                "retrieved_count": len(contexts),
                "support_hit_count": hit_count,
                "support_required_count": required_count,
                "all_supports_hit": hit_count == required_count and required_count > 0,
                "any_support_hit": hit_count > 0,
                "elapsed_seconds": elapsed,
                "support_hits": support_hits,
                "contexts": contexts,
                "scores": [str(score) for score in scores],
            }
        )
        print(
            f"[{idx}/{len(questions)}] supports {hit_count}/{required_count} "
            f"elapsed {elapsed:.2f}s :: {query}"
        )

    total_elapsed = time.time() - start
    evaluated = len(results)
    total_supports = sum(row["support_required_count"] for row in results)
    total_support_hits = sum(row["support_hit_count"] for row in results)
    summary = {
        "label": node_name,
        "relationship": edge_name,
        "data": args.data,
        "limit": args.limit,
        "traversal": args.traversal,
        "entry_type": args.entry_type,
        "topk": args.topk,
        "max_hop": args.max_hop,
        "evaluated_questions": evaluated,
        "questions_with_any_support_hit": sum(1 for row in results if row["any_support_hit"]),
        "questions_with_all_supports_hit": sum(1 for row in results if row["all_supports_hit"]),
        "support_fact_hit_rate": total_support_hits / total_supports if total_supports else 0.0,
        "avg_elapsed_seconds": total_elapsed / evaluated if evaluated else 0.0,
        "total_elapsed_seconds": total_elapsed,
    }

    payload = {"summary": summary, "results": results}
    print("=" * 80)
    print("Retrieval Evaluation Summary")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation to: {args.output}")


if __name__ == "__main__":
    main()
