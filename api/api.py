import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core.agent import AetherAgent
import config

app = FastAPI(title="AetherCode API")

# Allow React to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Grab the running async event loop
    loop = asyncio.get_running_loop()
    
    try:
        while True:
            data = await websocket.receive_json()
            user_prompt = data.get("prompt")
            working_dir = data.get("working_dir")
            is_general_chat = not bool(working_dir)

            target_dir = working_dir or os.getcwd()

            # Safely push messages from the background thread to the async WebSocket
            def ws_log_callback(role: str, text: str):
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"role": role, "text": text}),
                    loop
                )

            agent = AetherAgent(
                root_dir=target_dir,
                gemini_key=config.GEMINI_API_KEY,
                groq_key=config.GROQ_API_KEY
            )
            
            def auto_approve(cmd: str):
                ws_log_callback("system", f"⚠️ Auto-running command: {cmd}")
                return True

            # RUN IN BACKGROUND THREAD: This prevents the server from freezing!
            await asyncio.to_thread(
                agent.run,
                user_prompt=user_prompt,
                log_callback=ws_log_callback,
                command_approval_callback=auto_approve,
                is_general_chat=is_general_chat
            )
            
            # Brief pause to ensure all messages flush, then signal completion
            await asyncio.sleep(0.1)
            await websocket.send_json({"role": "system", "text": "--- END OF TASK ---"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"role": "system", "text": f"❌ Server Error: {str(e)}"})