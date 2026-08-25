import os
import asyncio
import threading
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

    agent: Optional[AetherAgent] = None
    current_working_dir: Optional[str] = None

    pending_approvals: Dict[str, asyncio.Future] = {}
    approval_counter = 0
    client_gone = asyncio.Event()

    # Incoming prompts go on this queue instead of being processed inline
    # by the receive loop. This is the actual fix for approvals hanging
    # forever: previously the SAME loop that reads incoming websocket
    # messages was also the one blocked awaiting agent.run() for the
    # whole turn — so an 'approval_response' the user sent mid-task could
    # never be read until the task finished, and the task could never
    # finish until the approval was read. Deadlock, resolved only by the
    # 115s approval timeout (as an automatic DENY, not an approve).
    # Splitting "receive" and "process" into a producer/consumer pair
    # means the receive loop is always free to read the next message —
    # including an approval_response arriving mid-turn.
    prompt_queue: asyncio.Queue = asyncio.Queue()

    async def _safe_send(payload: dict):
        if client_gone.is_set():
            return
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    def ws_log_callback(role: str, text: str):
        asyncio.run_coroutine_threadsafe(
            _safe_send({"role": role, "text": text}),
            loop
        )

    async def _wait_for_approval(request_id: str, cmd: str, cancel_event: threading.Event) -> bool:
        # Checked once more right here, not just by the caller — closes
        # the small race window between request_approval() checking
        # cancel_event and this coroutine actually being scheduled on the
        # event loop via run_coroutine_threadsafe.
        if cancel_event.is_set() or client_gone.is_set():
            return False

        fut = loop.create_future()
        pending_approvals[request_id] = fut
        await _safe_send({"role": "approval_request", "id": request_id, "command": cmd})
        try:
            return await asyncio.wait_for(fut, timeout=115)
        except asyncio.TimeoutError:
            return False
        finally:
            pending_approvals.pop(request_id, None)

    def request_approval(cmd: str, cancel_event: threading.Event) -> bool:
        """
        Runs on the worker thread (agent.run is inside asyncio.to_thread).
        Blocks until the user responds via an 'approval_response' message,
        or 115s pass, whichever comes first.

        cancel_event belongs to the SPECIFIC turn this call is part of
        (created fresh per _process_prompt call, below). If that turn's
        180s ceiling has already fired by the time the agent gets here —
        which can happen, since the worker thread this runs on can't
        actually be killed when the timeout hits, only waited-on-no-longer
        — this returns False immediately instead of sending a fresh
        approval_request card for a task the frontend was already told
        was over.
        """
        if cancel_event.is_set():
            return False

        nonlocal approval_counter
        approval_counter += 1
        request_id = f"approval_{approval_counter}"

        concurrent_future = asyncio.run_coroutine_threadsafe(
            _wait_for_approval(request_id, cmd, cancel_event), loop
        )
        try:
            return concurrent_future.result(timeout=120)
        except Exception:
            return False

    async def _process_prompt(data: dict):
        nonlocal agent, current_working_dir

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

        # Cancellation signal scoped to THIS turn only — fresh Event every
        # call, never reused. Set the moment this turn's timeout fires,
        # below. The agent's tool loop checks it before every Groq call,
        # every tool call, and every approval request; request_approval
        # above checks it before ever touching the websocket. That's what
        # actually stops a timed-out turn from continuing to run commands,
        # request approvals, or mutate agent.chat_history / _recent_files /
        # file_manager.history in the background after the frontend has
        # already moved on — since the worker thread itself genuinely
        # cannot be force-killed once started, only cooperative checks
        # like this can make cancellation take effect.
        cancel_event = threading.Event()

        def turn_log_callback(role: str, text: str):
            # Belt-and-suspenders alongside the agent-side checks: even if
            # something slips through and calls this after cancellation,
            # it's dropped here rather than reaching the frontend.
            if cancel_event.is_set():
                return
            ws_log_callback(role, text)

        def turn_approval_callback(cmd: str) -> bool:
            return request_approval(cmd, cancel_event)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    agent.run,
                    user_prompt=user_prompt,
                    log_callback=turn_log_callback,
                    command_approval_callback=turn_approval_callback,
                    is_general_chat=is_general_chat,
                    execution_mode=execution_mode,
                    cancel_event=cancel_event,
                ),
                timeout=config.AGENT_TURN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            cancel_event.set()
            await _safe_send({
                "role": "system",
                "text": f"❌ Turn timed out after {config.AGENT_TURN_TIMEOUT_SECONDS}s. "
                        "Try again, or check whether Groq/Gemini are slow to respond right now."
            })
        except Exception as e:
            await _safe_send({"role": "system", "text": f"❌ Agent Error: {str(e)}"})

        await asyncio.sleep(0.1)
        await _safe_send({"role": "system", "text": "--- END OF TASK ---"})

    async def _prompt_worker():
        """Processes one turn at a time, pulled off prompt_queue. Kept as
        its own task so a long-running turn never blocks the receive loop
        below from reading new messages in the meantime."""
        while True:
            payload = await prompt_queue.get()
            if payload is None:
                return
            try:
                await _process_prompt(payload)
            finally:
                prompt_queue.task_done()

    worker_task = asyncio.create_task(_prompt_worker())

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                # Starlette can raise this instead of WebSocketDisconnect
                # if receive() is called again after the socket has
                # already fully torn down — treat it the same as a
                # disconnect instead of letting it crash the ASGI app.
                break

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
                await _safe_send({"role": "system", "text": "🧹 New chat started — previous context cleared."})
                continue

            await prompt_queue.put(data)

    finally:
        client_gone.set()
        # Resolve any approval still waiting on this connection immediately
        # (as a deny) instead of leaving it to burn its full 115s timeout
        # now that we know no response is ever coming.
        for fut in pending_approvals.values():
            if not fut.done():
                fut.set_result(False)
        # Lets the worker exit after finishing whatever it's currently on
        # (if anything) — an in-flight turn isn't force-cancelled; its
        # sends just get silently dropped by _safe_send from here on. See
        # earlier note: this can't force-kill the underlying worker
        # thread either, same caveat as before.
        await prompt_queue.put(None)
        print("Client disconnected")