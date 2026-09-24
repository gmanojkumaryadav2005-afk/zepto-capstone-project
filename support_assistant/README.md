# Module 3 — Support Assistant (RAG)

## Run (graded baseline — fully offline, no signup, no API key)
```bash
pip install -r requirements.txt
python ingest.py            # builds the ChromaDB index from docs/*.txt
python example_calls.py     # two example calls, printed as JSON (paste output below)
uvicorn main:app --host 0.0.0.0 --port 7860   # MOCK_LLM left unset -> mock mode
```
Test the API:
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is your delivery policy?"}'
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

## Optional, ungraded extension — real LLM
```bash
export MOCK_LLM=0
export GROQ_API_KEY=your_key_here   # free signup at console.groq.com, no card required
uvicorn main:app --host 0.0.0.0 --port 7860
```
This does not affect the required submission: with `MOCK_LLM` at its
default, `answer`/`sources`/`confidence` are fully correct using only the
mock baseline described below.

## Optional, ungraded extension — Docker / Hugging Face Spaces
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```
The `Dockerfile` builds and runs locally with no external service — this is
the required, graded baseline for containerization. Pushing to Hugging Face
Spaces (free CPU tier, `GROQ_API_KEY` stored as a Space secret, never
hardcoded or committed) is an optional stretch on top of it.

## Architecture (ingestion -> embedding -> retrieval -> generation)

1. **Ingestion** (`ingest.py`): loads the 8 policy files from `docs/`,
   chunks each with `chunk_text()` (docs here are short, so most become a
   single chunk).
2. **Embedding** (`ingest.py`): each chunk is embedded locally with
   `sentence-transformers`' `all-MiniLM-L6-v2` model — no API key, no
   network call beyond the one-time model download — and the vectors are
   stored in a persistent ChromaDB collection (`chroma_store/`,
   collection `zepto_policies`).
3. **Retrieval** (`graph.py::retrieve_and_answer`): embeds the incoming
   query with the same model and retrieves the top-3 most similar chunks
   from ChromaDB via cosine similarity. This step always runs for real, in
   both mock and real-LLM modes, since it needs no API key or network call.
4. **Generation**: branches on the `MOCK_LLM` env var, checked inside each
   node's generation step (routing itself does not depend on `MOCK_LLM`):
   - `classify_intent`: mock mode uses a keyword heuristic over
     `delivery/return/refund/membership/tracking/cancel/gift card/support
     hours` to route to `policy_question` or `general_question`; the
     optional real-LLM extension asks the LLM to classify instead.
   - `retrieve_and_answer` (policy_question branch): mock mode returns a
     canned template built from the top retrieved chunk
     (`f"Based on the retrieved context: {snippet}"`); the optional
     extension prompts the LLM (via the structured template in
     `PROMPT_TEMPLATE`) to answer grounded only in the retrieved chunks.
   - `direct_answer` (general_question branch): mock mode returns a fixed
     canned string with no retrieval and no LLM call; the optional
     extension prompts the LLM directly.
5. **Output schema**: every response is validated against the
   `AssistantResponse` Pydantic model (`answer: str`, `sources: List[str]`,
   `confidence: float`). In mock mode this is populated deterministically
   from code (no LLM output to validate). In the optional real-LLM
   extension, a failed validation is retried up to 2 additional times with
   a corrective instruction before returning a clearly marked error.
6. **API**: `main.py` wraps the compiled LangGraph app in a FastAPI
   `POST /ask` endpoint, run locally with `uvicorn`.

## Example call transcripts (MOCK_LLM at default)
Run `python example_calls.py` after `python ingest.py` and paste its
printed JSON output here, e.g.:
```
Query: How long do I have to return a damaged grocery item?
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect...",
  "sources": ["doc_02_chunk0"],
  "confidence": 1.0
}

Query: What's your favorite movie?
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
