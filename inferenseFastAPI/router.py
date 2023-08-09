from fastapi import APIRouter
import functools
from fastapi.responses import PlainTextResponse, RedirectResponse
import asyncio
from src import TextGenerator
from concurrent.futures import ProcessPoolExecutor

router = APIRouter()
generator = TextGenerator()


@router.post("/generate_text")
async def generate_text(start_prompt: str) -> str:
    """
    Generate text
    """
    result = generator.generate(start_prompt)
    return result


@router.get("/echo", response_class=PlainTextResponse)
async def echo():
    """
    Echo
    """
    return "ok"


@router.get("/")
async def redirect():
    return RedirectResponse("/docs")
