# SHL Assessment Recommendation Agent Approach

## Architecture

The solution is a lightweight FastAPI application that exposes a health endpoint and a chat endpoint backed by a stateless retrieval pipeline. The application keeps its state in the incoming request payload and never relies on server-side memory or a second database. The core components are the FastAPI entrypoint, the route layer, the conversation agent, and the existing FAISS-backed retriever.

## Retrieval Pipeline

The retrieval pipeline uses the existing catalog and FAISS index already provided in the repository. The retriever loads the preprocessed SHL catalog and the vector index at startup, then searches the index with the user’s latest request. Recommendations are built directly from the retrieved catalog records, which prevents hallucinated names and URLs.

## FAISS Indexing

The FAISS index is treated as a prebuilt asset. The project does not recreate or rebuild the index during runtime. This keeps the submission deterministic and avoids unnecessary computational cost while still enabling fast similarity search over the 377 SHL assessments.

## Prompt Design

The agent uses a compact system prompt that limits answers to SHL assessment topics. It prioritizes concise replies and relies on retrieved catalog context rather than free-form generation. When Gemini is unavailable, the system falls back to deterministic logic that still returns a valid schema-safe response.

## Clarification Strategy

Vague prompts such as “I need an assessment” trigger a clarification question instead of returning arbitrary recommendations. The agent asks for the role, seniority, and whether the request is technical or behavioral when the user has not provided enough context.

## Recommendation Strategy

Once the request contains enough hiring context, the agent uses the retrieved SHL assessments to produce a shortlist of 1–10 recommendations. Recommendations are returned in the required schema and contain only the catalog name, URL, and test type. The system uses the full message history so later turns can refine or expand the earlier request.

## Comparison Strategy

Catalog-based comparison requests are answered using only the available SHL catalog entries. The agent does not invent attribute values or make unsupported claims; if the catalog does not contain enough detail, it explains that limitation directly.

## Refusal Strategy

Requests outside SHL assessment recommendations are politely refused. This includes general hiring advice, legal advice, salary questions, and prompt injection attempts. The response still follows the required schema with empty recommendations and an open conversation flag.

## Evaluation Methodology

The solution was validated through direct API calls against the live FastAPI server as well as pytest coverage for health checks, vague queries, detailed recommendations, comparison requests, refinement flow, and off-topic refusal. All responses were verified to conform to the expected schema.

## Limitations

The experience depends on the quality of the existing catalog and the FAISS index. Retrieval quality can vary for very short or ambiguous prompts, and comparison quality depends on the catalog descriptions available for the requested assessments.

## Future Improvements

Future work could include more nuanced intent detection, richer conversation state with explicit slots (role, seniority, domain, and assessment type), and an optional fallback to a curated rule-based recommendation map for highly specific hiring scenarios.
