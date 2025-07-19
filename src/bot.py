#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Multimodal Live Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model
with built-in text-to-speech. It includes:
- Real-time audio/video interaction through Daily
- Speech-to-speech model via Gemini Live
- Built-in TTS via Gemini (no external TTS service needed)
- No sprite animations (clean video output)

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass


async def run_bot(room_url: str, token: str):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily audio transport (no video needed)
    - Gemini Live multimodal model with built-in TTS
    - Voice activity detection
    - RTVI event handling
    """
    # Set up Daily transport with audio-only parameters for Gemini
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            audio_out_enabled=True,
            camera_out_enabled=False,  # No video output needed for audio-only bot
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            transcription_enabled=True,
        ),
    )

    # Initialize the Gemini Multimodal Live model with built-in TTS
    # Available voices: Aoede, Charon, Fenrir, Kore, Puck
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",  # Choose from available Gemini voices
        #transcribe_user_audio=True,
    )

    messages = [
        {
            "role": "user",
            "content": "You are Chatbot, a friendly, helpful AI assistant. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
        },
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Pipeline: Input -> RTVI -> Context -> Gemini LLM (with built-in TTS) -> Output
    # Audio-only pipeline - no video processing needed
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            context_aggregator.user(),
            llm,  # Gemini handles both LLM and TTS
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner()

    await runner.run(task)
