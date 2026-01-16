from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import whisper
import asyncio
import re
import json
from rich.console import Console

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM

from tts_koko import TextToSpeechService

STATIC_PROMPT = """
You are the character described below, participating in a cognitive interview about an aviation incident... Limit your answer to short sentences
"""

console = Console()

app = FastAPI(title="Cognitive Interview - Realtime Voice (Linh)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

console.print("[yellow]Loading Whisper...[/yellow]")
whisper_model = whisper.load_model("base.en")

console.print("[yellow]Initializing TTS...[/yellow]")
tts = TextToSpeechService()

console.print("[yellow]Connecting to Ollama...[/yellow]")
llm = OllamaLLM(model="gemma3", base_url="http://localhost:11434")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{static_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt_template | llm
chat_sessions = {}

def get_session_history(session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
PROCESS_EVERY_SECONDS = 3.5

processing_lock = asyncio.Lock()
playback_events = {}

def get_playback_event(session_id: str) -> asyncio.Event:
    if session_id not in playback_events:
        playback_events[session_id] = asyncio.Event()
    return playback_events[session_id]

def clean_llm_output(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()

async def generate_response(text: str, session_id: str) -> str:
    response = await asyncio.to_thread(
        chain_with_history.invoke,
        {"input": text, "static_prompt": STATIC_PROMPT},
        config={"configurable": {"session_id": session_id}}
    )
    return clean_llm_output(response)

# --- REMOVE playback_events, locks, and waits entirely ---

@app.websocket("/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    console.print("[green]Client connected[/green]")

    buffer = bytearray()
    session_id = f"session_{id(websocket)}"
    is_speaking = False

    try:
        while True:
            msg = await websocket.receive()

            # ---------- ACK HANDLING ----------
            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "audio_playback_complete":
                        console.print("[green]Playback ACK received[/green]")
                        is_speaking = False
                        await websocket.send_json({
                            "type": "status",
                            "state": "listening"
                        })
                        continue
                except:
                    pass

            # ---------- AUDIO INPUT ----------
            data = msg.get("bytes")
            if not data or is_speaking:
                continue

            buffer.extend(data)

            if len(buffer) < BYTES_PER_SECOND * PROCESS_EVERY_SECONDS:
                continue

            audio = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
            buffer = bytearray()

            result = await asyncio.to_thread(
                whisper_model.transcribe,
                audio,
                fp16=False,
                language="en"
            )

            user_text = result["text"].strip()
            if len(user_text) < 3:
                continue

            console.print(f"[yellow]USER: {user_text}[/yellow]")

            response_text = await generate_response(user_text, session_id)
            console.print(f"[cyan]AI: {response_text}[/cyan]")

            sr, audio_array = await asyncio.to_thread(
                tts.long_form_synthesize,
                response_text,
                audio_prompt_path=None,
                exaggeration=0.6,
                cfg_weight=0.6
            )

            pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()

            is_speaking = True

            await websocket.send_bytes(pcm_bytes)
            await websocket.send_json({
                "type": "status",
                "state": "speaking"
            })

    except WebSocketDisconnect:
        console.print("[red]Client disconnected[/red]")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
