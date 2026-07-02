---
title: SHL Assessment Recommender
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---


# SHL Assessment Recommendation Agent

This project provides a FastAPI service that helps users discover SHL assessments from a prebuilt catalog and FAISS index. It is designed for the SHL AI Hiring Challenge and keeps the retrieval pipeline stateless by using the existing catalog and index assets.

## Overview

- Exposes a health endpoint for service checks.
- Accepts chat-style requests and returns structured recommendations.
- Uses the existing FAISS-backed retriever to surface relevant SHL assessments.
- Supports clarification questions for vague prompts, catalog-based comparisons, and polite refusal for off-topic requests.

## Architecture

- app/main.py: FastAPI application and global exception handling.
- app/routes.py: API routes for /health and /chat.
- app/agent.py: Conversation handling, clarification logic, and retrieval orchestration.
- app/retriever.py: FAISS-based retrieval against the SHL catalog.
- app/schemas.py: Pydantic models for request and response payloads.
- data/: Preprocessed SHL catalog and FAISS index assets.

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

## API Endpoints

### GET /health

Returns:

```json
{"status":"ok"}
```

### POST /chat

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need a senior Java developer assessment."}]}'
```

Example response:

```json
{
  "reply": "Here are a few SHL assessments that match your request.",
  "recommendations": [
    {
      "name": "Java Developer Test",
      "url": "https://www.shl.com/products/product-catalog/view/java-developer-test/",
      "test_type": "Assessment"
    }
  ],
  "end_of_conversation": true
}
```

## Deployment

The service can be deployed as a standard FastAPI application. For container-based deployment, ensure the data directory and model assets are included in the runtime image.
