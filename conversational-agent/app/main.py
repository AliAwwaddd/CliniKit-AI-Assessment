"""FastAPI entry point exposing the pipeline over HTTP."""

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import process_message

app = FastAPI(title="CliniKit Conversational Agent")


class MessageIn(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/message")
def handle_message(payload: MessageIn) -> dict:
    result = process_message(payload.message)
    return result.model_dump()
