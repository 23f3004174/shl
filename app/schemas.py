from typing import List

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool