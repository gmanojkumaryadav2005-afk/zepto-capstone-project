"""
Runs two example calls against the graph directly (equivalent to what the
/ask endpoint returns) and prints their raw JSON — one that should trigger
retrieval (policy question) and one that should not (general question).
Paste this output into support_assistant/README.md per the assignment.

Run:
    python ingest.py     # build the ChromaDB index first
    python example_calls.py
"""
import json
from graph import ask

examples = [
    "How long do I have to return a damaged grocery item?",
    "What's your favorite movie?",
]

for q in examples:
    result = ask(q)
    print(f"Query: {q}")
    print(json.dumps(result.model_dump(), indent=2))
    print()
