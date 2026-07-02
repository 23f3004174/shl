from fastapi import APIRouter

from app.agent import chat
from app.schemas import ChatRequest, ChatResponse, Recommendation

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Handle chat requests using the full message history for each turn."""

    reply, recommendations, end_of_conversation = chat(request.messages)

    payload_recommendations = [
        Recommendation(**item) for item in recommendations
    ]

    return ChatResponse(
        reply=reply,
        recommendations=payload_recommendations,
        end_of_conversation=end_of_conversation,
    )