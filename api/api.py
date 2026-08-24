import os
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core.agent import AetherAgent
import config

app = FastAPI(title="AetherCode API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    # One agent per connection, rebuilt only when the working directory
    # changes. This also fixes `undo`, which previously reset every message
    # because a brand new SafeFileManager (and its history stack) was created
    # on every single prompt.
    agent: Optional[AetherAgent] = None
    current_working_dir: Optional[str] = None

    pending_approvals: Dict[str, asyncio.Future] = {}
    approval_counter = 0

    def ws_log_callback(role: str, text: str):
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"role": role, "text": text}),
            loop
        )

    async def _wait_for_approval(request_id: str, cmd: str) -> bool:
        fut = loop.create_future()
        pending_approvals[request_id] = fut
        await websocket.send_json({"role": "approval_request", "id": request_id, "command": cmd})
        try:
            return await asyncio.wait_for(fut, timeout=115)
        except asyncio.TimeoutError:
            return False
        finally:
            pending_approvals.pop(request_id, None)

    def request_approval(cmd: str) -> bool:
        """
        Runs on the worker thread (agent.run is inside asyncio.to_thread).
        Blocks until the user responds via an 'approval_response' message,
        or 115s pass, whichever comes first.
        """
        nonlocal approval_counter
        approval_counter += 1
        request_id = f"approval_{approval_counter}"

        concurrent_future = asyncio.run_coroutine_threadsafe(
            _wait_for_approval(request_id, cmd), loop
        )
        try:
            return concurrent_future.result(timeout=120)
        except Exception:
            return False

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "approval_response":
                request_id = data.get("id")
                approved = bool(data.get("approved"))
                fut = pending_approvals.get(request_id)
                if fut and not fut.done():
                    fut.set_result(approved)
                continue

            if data.get("type") == "clear_history":
                if agent is not None:
                    agent.reset_history()
                await websocket.send_json({"role": "system", "text": "🧹 New chat started — previous context cleared."})
                continue

            user_prompt = data.get("prompt")
            working_dir = data.get("working_dir")
            execution_mode = data.get("execution_mode", "auto")
            is_general_chat = not bool(working_dir)

            target_dir = working_dir or os.getcwd()

            if agent is None or current_working_dir != target_dir:
                agent = AetherAgent(
                    root_dir=target_dir,
                    gemini_key=config.GEMINI_API_KEY,
                    groq_key=config.GROQ_API_KEY
                )
                current_working_dir = target_dir

            try:
                await asyncio.to_thread(
                    agent.run,
                    user_prompt=user_prompt,
                    log_callback=ws_log_callback,
                    command_approval_callback=request_approval,
                    is_general_chat=is_general_chat,
                    execution_mode=execution_mode,
                )
            except Exception as e:
                # Scoped to this turn only — the connection (and session state)
                # stays alive for the next message instead of dying here.
                await websocket.send_json({"role": "system", "text": f"❌ Agent Error: {str(e)}"})

            await asyncio.sleep(0.1)
            await websocket.send_json({"role": "system", "text": "--- END OF TASK ---"})

    except WebSocketDisconnect:
        print("Client disconnected")