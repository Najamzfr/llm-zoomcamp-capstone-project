# Tracing with LangSmith (optional, free)

LangSmith records each question as a trace tree, so you can see every search, tool call and LLM
call with its inputs, outputs, tokens and latency. It uses the `langsmith` SDK directly
(`@traceable` and `wrap_openai`); no LangChain code is involved. Tracing is **off by default**.

## Setup
1. Create a free account at https://smith.langchain.com (Developer plan, ~5k traces/month).
2. Settings → **API Keys** → create a key.
3. In `.env`:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=paper-scout
   ```
   EU-region accounts also set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.
4. Restart the API (`uv run uvicorn api:app`) and ask a question.
5. Open the **paper-scout** project in LangSmith.

## What a trace looks like
```
research_agent (chain)                 answer_stage, tokens, cost in metadata
├─ hybrid_search_rrf (retriever)
│  ├─ keyword_search_fts5 (retriever)
│  └─ vector_search_qdrant (retriever)
├─ route_abstracts_or_full_text        → OpenAI call (JSON decision)
├─ answer_from_abstracts               → OpenAI call
├─ check_abstract_answer               → OpenAI call (judge)
├─ retrieve_full_text (tool)           PDF download, chunking, chunk ranking
├─ answer_from_full_text               → OpenAI call
└─ check_full_text_answer              → OpenAI call (judge)
```
Steps appear only when the agent takes them (most questions stop after the abstract answer).

Thumbs up / down in the web UI is attached to the run as `user_rating` feedback (1 or 0).

## Privacy
With tracing on, questions, retrieved titles/abstracts, PDF excerpts and answers are sent to
LangSmith. Set `LANGSMITH_TRACING=false` to stop.
