"""
Module 3 - Support Assistant: FastAPI wrapper.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

Example:
    curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \\
         -d '{"query": "What is your delivery policy?"}'
"""
from fastapi import FastAPI
from pydantic import BaseModel

from graph import ask, AssistantResponse

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AssistantResponse)
def ask_endpoint(req: AskRequest) -> AssistantResponse:
    return ask(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}
