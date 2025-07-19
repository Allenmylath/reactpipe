#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Simplified Modal Example - Gemini + Cartesia Only.

This module shows a simple example of how to deploy a bot using Modal and FastAPI.
Simplified to only use Gemini LLM + Cartesia TTS, synced with React frontend.

It includes:
- FastAPI endpoints for starting agents and checking bot statuses.
- Gemini LLM with Cartesia TTS integration.
- Use of a Daily transport for bot communication.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import aiohttp
import modal
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Container specifications for the FastAPI web server
web_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily]")
    .add_local_dir("src", remote_path="/root/src")
)

# Container specifications for the Pipecat pipeline
bot_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pipecat-ai[daily,google,silero]")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("pipecat-modal", secrets=[modal.Secret.from_dotenv()])

# Global variables for bot management
bot_jobs = {}
daily_helpers = {}


# Define models at module level
class ConnectData(BaseModel):
    """Data provided by client to specify the services.

    This matches the format expected by the React frontend.
    """
    services: Dict[str, str] = {"llm": "gemini", "tts": "cartesia"}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_jobs.values():
        func = modal.FunctionCall.from_id(entry[0])
        if func:
            func.cancel()


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_jobs.values():
        func = modal.FunctionCall.from_id(entry[0])
        if func:
            func.cancel()


async def create_room_and_token() -> tuple[str, str]:
    """Create a Daily room and generate an authentication token.

    This function checks for existing room URL and token in the environment variables.
    If not found, it creates a new room using the Daily API and generates a token for it.

    Returns:
        tuple[str, str]: A tuple containing the room URL and the authentication token.

    Raises:
        HTTPException: If room creation or token generation fails.
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRoomParams

    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    if not room_url:
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")
        room_url = room.url

        token = await daily_helpers["rest"].get_token(room_url)
        if not token:
            raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    return room_url, token


async def start():
    """Internal method to start the Gemini + Cartesia bot agent.

    Returns:
        tuple[str, str]: A tuple containing the room URL and token.
    """
    room_url, token = await create_room_and_token()
    launch_bot_func = modal.Function.from_name("pipecat-modal", "bot_runner")
    function_id = launch_bot_func.spawn(room_url, token)
    bot_jobs[function_id] = (function_id, room_url)

    return room_url, token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper
    
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


@app.function(image=bot_image, min_containers=1)
async def bot_runner(room_url: str, token: str):
    """Launch the Gemini + Cartesia bot process.

    Args:
        room_url (str): The URL of the Daily room where the bot and client will communicate.
        token (str): The authentication token for the room.

    Raises:
        HTTPException: If the bot pipeline fails to start.
    """
    try:
        # Import the bot runner
        from src.bot import run_bot

        print(f"Starting Gemini + Cartesia bot process: -u {room_url} -t {token}")
        await run_bot(room_url, token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bot pipeline: {e}")


# Create router
router = APIRouter()


@router.get("/")
async def start_agent():
    """A user endpoint for launching a bot agent and redirecting to the created room URL.

    This function starts the Gemini + Cartesia bot agent and redirects the user 
    to the room URL to interact with the bot through a Daily Prebuilt Interface.

    Returns:
        RedirectResponse: A response that redirects to the room URL.
    """
    print("Starting Gemini + Cartesia bot")
    room_url, token = await start()

    return RedirectResponse(room_url)


@router.post("/connect")
async def rtvi_connect(data: ConnectData = None) -> Dict[Any, Any]:
    """A user endpoint for launching a bot agent and retrieving the room/token credentials.

    This function starts the Gemini + Cartesia bot agent and returns the room URL 
    and token for the bot. This allows the client to then connect to the bot using 
    their own RTVI interface.

    Args:
        data (ConnectData): Optional. The data containing the services to use.

    Returns:
        Dict[Any, Any]: A dictionary containing the room URL and token.
    """
    print("Starting Gemini + Cartesia bot from /connect endpoint")
    if data and data.services:
        print(f"Services requested: {data.services}")
        
        # Validate that the frontend is requesting the correct services
        if data.services.get("llm") != "gemini" or data.services.get("tts") != "cartesia":
            raise HTTPException(
                status_code=400, 
                detail="This backend only supports Gemini LLM and Cartesia TTS"
            )
    
    room_url, token = await start()

    return {"room_url": room_url, "token": token}


@router.get("/status/{fid}")
def get_status(fid: str):
    """Retrieve the status of a bot process by its function ID.

    Args:
        fid (str): The function ID of the bot process.

    Returns:
        JSONResponse: A JSON response containing the bot's status and result code.

    Raises:
        HTTPException: If the bot process with the given ID is not found.
    """
    func = modal.FunctionCall.from_id(fid)
    if not func:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {fid} not found")

    try:
        result = func.get(timeout=0)
        return JSONResponse({"bot_id": fid, "status": "finished", "code": result})
    except modal.exception.OutputExpiredError:
        return JSONResponse({"bot_id": fid, "status": "finished", "code": 404})
    except TimeoutError:
        return JSONResponse({"bot_id": fid, "status": "running", "code": 202})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


@app.function(image=web_image, min_containers=1)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """Create and configure the FastAPI application.

    This function initializes the FastAPI app with middleware, routes, and lifespan management.
    It is decorated to be used as a Modal ASGI app.
    """
    # Initialize FastAPI app
    web_app = FastAPI(lifespan=lifespan)

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the endpoints from this file
    web_app.include_router(router)

    return web_app
