"""
Module 3 - Support Assistant: LangGraph pipeline.

Nodes:
    classify_intent      -> policy_question | general_question
    retrieve_and_answer   -> retrieval always runs for real; generation branches on MOCK_LLM
    direct_answer         -> generation branches on MOCK_LLM, no retrieval

MOCK_LLM unset or "1" (default, graded baseline): fully deterministic, rule-based,
no LLM call, no network call, no API key.
MOCK_LLM="0" (optional, ungraded extension): calls a real LLM (Groq free tier by
default) for classification and generation.
"""
import os
import re
from typing import TypedDict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

PROMPT_TEMPLATE = """\
# Role
You are Zepto's customer support assistant.

# Context
Use ONLY the context below to answer. Do not answer using information not
present in the provided context. If the context does not contain the
answer, say you don't have that information.

Context:
{context}

# Task
Answer the customer's question grounded strictly in the context above.

# Format
Respond with a concise, direct answer in 1-3 sentences.

# Length
Keep the answer under 60 words.

# Example
Question: "How long does delivery take?"
Context: "Zepto delivers ... within 10 to 30 minutes ..."
Answer: "Delivery typically takes 10 to 30 minutes after order confirmation,
depending on your zone and current order volume."

# Customer question
{query}
"""


class AssistantResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved: Optional[List[dict]]
    response: Optional[dict]


_embedder = None
_collection = None


def _get_resources():
    global _embedder, _collection
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _embedder, _collection


def _call_llm(prompt: str) -> str:
    """Optional real-LLM path (MOCK_LLM=0). Uses Groq's free-tier API."""
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    if MOCK_LLM:
        lower = query.lower()
        intent = "policy_question" if any(kw in lower for kw in KEYWORDS) else "general_question"
    else:
        prompt = (
            "Classify this customer query as exactly one word, either "
            "'policy_question' (about Zepto delivery/returns/membership/tracking/"
            "cancellation/gift cards/support) or 'general_question' (anything else).\n"
            f"Query: {query}\nAnswer with one word only."
        )
        raw = _call_llm(prompt).strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"
    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    embedder, collection = _get_resources()
    query = state["query"]
    query_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)

    retrieved = [
        {"id": results["ids"][0][i], "text": results["documents"][0][i], "doc_id": results["metadatas"][0][i]["doc_id"]}
        for i in range(len(results["ids"][0]))
    ]
    top_chunk = retrieved[0]["text"] if retrieved else ""

    if MOCK_LLM:
        snippet = top_chunk[:200]
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        context = "\n\n".join(r["text"] for r in retrieved)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        answer = _validated_llm_answer(prompt)
        confidence = 0.9

    response = {
        "answer": answer,
        "sources": [r["id"] for r in retrieved],
        "confidence": confidence,
    }
    return {**state, "retrieved": retrieved, "response": response}


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        prompt = f"Answer this general question concisely: {query}"
        answer = _call_llm(prompt)

    response = {"answer": answer, "sources": [], "confidence": 1.0 if MOCK_LLM else 0.7}
    return {**state, "response": response}


def _validated_llm_answer(prompt: str, retries: int = 2) -> str:
    """Optional MOCK_LLM=0 path: enforce the Pydantic schema on the LLM's raw
    output, retrying up to `retries` additional times with a corrective
    instruction before giving up."""
    attempt_prompt = prompt
    for attempt in range(retries + 1):
        raw = _call_llm(attempt_prompt)
        try:
            AssistantResponse(answer=raw, sources=[], confidence=0.9)
            return raw
        except ValidationError:
            attempt_prompt = prompt + "\n\nYour previous answer was invalid. Return plain text only."
    return "Error: could not produce a schema-valid answer after retries."


# ---------------------------------------------------------------------------
# Wire the graph
# ---------------------------------------------------------------------------
def route(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent", route,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


def ask(query: str) -> AssistantResponse:
    app = build_graph()
    result = app.invoke({"query": query, "intent": None, "retrieved": None, "response": None})
    return AssistantResponse(**result["response"])


if __name__ == "__main__":
    for q in ["What is your delivery policy?", "What's the capital of France?"]:
        print(q, "->", ask(q).model_dump())
