import json
import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from rag import build_chain, format_docs, load_retriever

load_dotenv()

TEST_SET_PATH = "eval/test_set.json"
RESULTS_PATH = "eval/eval_results.json"
REPORT_PATH = "eval/eval_report.md"
JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are grading a RAG system's answer. You are given the question, the context it retrieved, "
        "and the answer it generated. Score strictly from the context only — do not use outside knowledge "
        "to decide if the answer is correct, only to decide if it is SUPPORTED by the given context.\n\n"
        "Return ONLY valid JSON, no markdown fences, no commentary, with exactly these fields:\n"
        '{{"faithfulness": <1-5 int, is every claim in the answer backed by the context>, '
        '"relevance": <1-5 int, does the answer actually address the question>, '
        '"correctly_abstained": <true, false, or null — null if abstaining was not expected>, '
        '"reasoning": "<one or two sentences>"}}',
    ),
    (
        "user",
        "Question: {question}\n\nExpected to abstain (say it doesn't know): {expect_abstain}\n\n"
        "Retrieved context:\n{context}\n\nGenerated answer:\n{answer}",
    ),
])


def keyword_check(answer, expected_facts):
    if not expected_facts:
        return None
    lower = answer.lower()
    hits = [f for f in expected_facts if f.lower() in lower]
    return {"hits": hits, "total": len(expected_facts), "ratio": len(hits) / len(expected_facts)}


def parse_judge_json(raw):
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise SystemExit("GROQ_API_KEY not set. Add it to a .env file before running eval.py.")

    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    retriever = load_retriever()
    answer_chain = build_chain(retriever, groq_api_key)

    judge_llm = ChatGroq(model=JUDGE_MODEL, api_key=groq_api_key, temperature=0)
    judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()

    results = []
    for item in test_set:
        question = item["question"]
        docs = retriever.invoke(question)
        context = format_docs(docs)
        answer = answer_chain.invoke(question)

        kw = keyword_check(answer, item.get("expected_facts", []))

        judge_raw = judge_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
            "expect_abstain": item.get("expect_abstain", False),
        })
        try:
            judge = parse_judge_json(judge_raw)
            judge_error = None
        except Exception as e:
            judge = None
            judge_error = f"{e}: {judge_raw[:200]}"

        results.append({
            "id": item["id"],
            "question": question,
            "answer": answer,
            "sources": [d.metadata.get("page") for d in docs],
            "expect_abstain": item.get("expect_abstain", False),
            "keyword_check": kw,
            "judge": judge,
            "judge_error": judge_error,
        })
        print(f"[{item['id']}] done")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    write_report(results)
    print(f"\nWrote {RESULTS_PATH} and {REPORT_PATH}")


def write_report(results):
    scored = [r for r in results if r["judge"]]
    avg_faith = sum(r["judge"]["faithfulness"] for r in scored) / len(scored) if scored else 0
    avg_rel = sum(r["judge"]["relevance"] for r in scored) / len(scored) if scored else 0

    lines = [
        "# Eval report\n",
        f"- Questions: {len(results)}",
        f"- Judge-scored: {len(scored)} (parse failures: {len(results) - len(scored)})",
        f"- Average faithfulness: {avg_faith:.2f} / 5",
        f"- Average relevance: {avg_rel:.2f} / 5\n",
        "| id | faithfulness | relevance | abstain expected | abstained correctly | keyword match | judge reasoning |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        j = r["judge"] or {}
        kw = r["keyword_check"]
        kw_str = f"{len(kw['hits'])}/{kw['total']}" if kw else "n/a"
        lines.append(
            f"| {r['id']} | {j.get('faithfulness', '-')} | {j.get('relevance', '-')} | "
            f"{r['expect_abstain']} | {j.get('correctly_abstained', '-')} | {kw_str} | "
            f"{(j.get('reasoning') or r['judge_error'] or '').replace(chr(10), ' ')[:140]} |"
        )
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
