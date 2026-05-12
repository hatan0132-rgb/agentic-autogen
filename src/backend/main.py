from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import autogen
import os
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Debate Club", version="1.0.0")

# Serve React static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str, request: Request):
    static_file_path = os.path.join("static", full_path)
    if os.path.isfile(static_file_path):
        return FileResponse(static_file_path)
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Not Found"}

class DebateRequest(BaseModel):
    topic: str

class DebateResponse(BaseModel):
    messages: List[Dict[str, Any]]
    topic: str

# CẤU HÌNH DÀNH RIÊNG CHO GEMINI
def get_llm_config():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in Secrets")
    return {
        "config_list": [
            {
                "model": "gemini-1.5-flash",
                "api_key": api_key,
                "api_type": "google"
            }
        ]
    }

def create_debate_agents():
    llm_config = get_llm_config()
    pro_agent = autogen.ConversableAgent(
        name="ProAgent",
        system_message="You are a skilled debater arguing in FAVOR of the topic.",
        llm_config=llm_config,
    )
    con_agent = autogen.ConversableAgent(
        name="ConAgent",
        system_message="You are a skilled debater arguing AGAINST the topic.",
        llm_config=llm_config,
    )
    return pro_agent, con_agent

def init_autogen_chat(manager, agent, debate_prompt):
    manager.initiate_chat(agent, message=debate_prompt)

@app.post("/debate", response_model=DebateResponse)
async def start_debate(request: DebateRequest):
    try:
        pro_agent, con_agent = create_debate_agents()
        groupchat = autogen.GroupChat(
            agents=[pro_agent, con_agent], 
            messages=[], 
            max_round=4,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False
        )
        manager = autogen.GroupChatManager(
            groupchat=groupchat, 
            llm_config=get_llm_config()
        )
        
        debate_prompt = f"Topic: {request.topic}. ProAgent argues FOR, ConAgent argues AGAINST."
        await asyncio.to_thread(init_autogen_chat, manager, pro_agent, debate_prompt)

        messages = []
        for msg in groupchat.messages:
            if msg.get("content") and msg.get("name"):
                messages.append({"role": msg["name"], "content": msg["content"]})
        return DebateResponse(topic=request.topic, messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
