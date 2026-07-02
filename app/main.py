from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routes import router

app = FastAPI(
    title="SHL Assessment Recommendation Agent"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a schema-safe chat payload for malformed requests instead of crashing."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "reply": "I could not process that request. Please send a valid message list.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Prevent unexpected errors from bubbling up to the client."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "reply": "The service hit an unexpected issue. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


app.include_router(router)