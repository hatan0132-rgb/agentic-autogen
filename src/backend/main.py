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

# Serve React static files from "./static"
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str, request: Request):
    # Serve static files and frontend src files for SPA routing and development
    # Try to serve files from /static first
    static_file_path = os.path.join("static", full_path)
    frontend_src_path = os.path.join("frontend", "src", full_path)

    if os.path.isfile(static_file_path):
        return FileResponse(static_file_path)
    elif os.path.isfile(frontend_src_path):
        return FileResponse(frontend_src_path)
    else:
        # Fallback to React index.html for SPA routing
        index_path = os.path.join("static", "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Not Found"}

# Debate request and response models
class DebateRequest(BaseModel):
    topic: str

class DebateResponse(BaseModel):
    messages: List[Dict[str, Any]]
    topic: str

def get_llm_config():
    # Đổi từ MISTRAL_API_KEY sang GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY") 
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY chưa được cấu hình")
    
    return {
        "config_list": [
            {
                "model": "gemini-1.5-flash", # Đổi tên mô hình
                "api_key": api_key,
                "api_type": "google"        # Đổi loại API sang google
            }
        ]
    }
def create_debate_agents():
    """Create the debate agents with specific roles"""
    llm_config = get_llm_config()
    
    # Pro agent (argues in favor)
    pro_agent = autogen.ConversableAgent(
        name="ProAgent",
        system_message="You are a skilled debater arguing in FAVOR of the given topic. Present compelling arguments, evidence, and reasoning to support the topic. Be persuasive, logical, and engaging. Keep responses concise but impactful. Always stay in character as the 'pro' side of the debate.",
        llm_config=llm_config,
    )
    
    # Con agent (argues against)
    con_agent = autogen.ConversableAgent(
        name="ConAgent",
        system_message="You are a skilled debater arguing AGAINST the given topic. Present compelling arguments, evidence, and reasoning to oppose the topic. Be persuasive, logical, and engaging. Keep responses concise but impactful. Always stay in character as the 'con' side of the debate.",
        llm_config=llm_config,
    )
    
    # return user_proxy, pro_agent, con_agent
    return pro_agent, con_agent

def init_autogen_chat(manager, agent, debate_prompt):
    manager.initiate_chat(agent, message=debate_prompt)

@app.post("/debate", response_model=DebateResponse)
async def start_debate(request: DebateRequest):
    """Start a debate between Pro and Con agents on the given topic"""
    try:
        # 1. Khởi tạo Agents
        pro_agent, con_agent = create_debate_agents()

        # 2. Cấu hình GroupChat (Sửa lỗi ngoặc ở đây)
        groupchat = autogen.GroupChat(
            agents=[pro_agent, con_agent], 
            messages=[], 
            max_round=4,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False
        )

        # 3. Khởi tạo Manager
        manager = autogen.GroupChatManager(
            groupchat=groupchat, 
            llm_config=get_llm_config()
        )

        # 4. Bắt đầu tranh luận
        debate_prompt = f"Debate topic: {request.topic}. ProAgent argues FOR, ConAgent argues AGAINST. Keep it respectful and engaging."
        
        try:
            await asyncio.to_thread(init_autogen_chat, manager, pro_agent, debate_prompt)
        except Exception as chat_error:
            print(f"AutoGen chat failed: {chat_error}")
            mock_messages = [
                {"role": "ProAgent", "content": f"I will argue in favor of: {request.topic}."},
                {"role": "ConAgent", "content": f"I will argue against: {request.topic}."}
            ]
            return DebateResponse(topic=request.topic, messages=mock_messages)

        # 5. Trả về kết quả
        messages = []
        for msg in groupchat.messages:
            if msg.get("content") and msg.get("name"):
                messages.append({"role": msg["name"], "content": msg["content"]})
        
        return DebateResponse(topic=request.topic, messages=messages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate failed: {str(e)}")
        
        return DebateResponse(
            topic=request.topic,
            messages=messages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Debate Club is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
