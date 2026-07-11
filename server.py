from __future__ import annotations

import asyncio
import os
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

AGENT_ROLE = os.getenv("AGENT_ROLE", "harper")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "anthropic/claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
INTER_SERVICE_SECRET_IN = os.getenv("INTER_SERVICE_SECRET_A", "")
INTER_SERVICE_SECRET_OUT = os.getenv("INTER_SERVICE_SECRET_B", "")
BUILD_SHA = os.getenv("BUILD_SHA", "dev")

# Load agent persona from CLAUDE.md
def _load_persona() -> str:
    try:
        with open("/root/CLAUDE.md", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return f"你是 {AGENT_ROLE}，DaDaAssis AI 核心團隊成員。請用繁體中文回覆。"

SYSTEM_PROMPT = _load_persona()

app = FastAPI(title=f"DaDaAssis cc-{AGENT_ROLE}", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "role": AGENT_ROLE, "sha": BUILD_SHA, "model": AGENT_MODEL}


class JobRequest(BaseModel):
    job_id: str
    team_id: str
    prompt: str
    context: dict | None = None
    callback_url: str
    attempt: int = 1


@app.post("/job")
async def receive_job(
    req: JobRequest,
    x_auth: str | None = Header(default=None, alias="X-DaDaAssis-Auth"),
) -> dict:
    if INTER_SERVICE_SECRET_IN and x_auth != INTER_SERVICE_SECRET_IN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    trace_id = str(uuid.uuid4())
    print(f"[{AGENT_ROLE.upper()}] job={req.job_id} team={req.team_id} trace={trace_id}", flush=True)
    asyncio.create_task(_process(req, trace_id))
    return {"accepted": True, "job_id": req.job_id, "role": AGENT_ROLE}


async def _process(req: JobRequest, trace_id: str) -> None:
    try:
        result = await _call_openrouter(req.prompt, trace_id)
        await _callback(req.callback_url, req.job_id, req.team_id, "done", result, trace_id)
    except Exception as exc:
        print(f"[{AGENT_ROLE.upper()}] error job={req.job_id} err={exc}", flush=True)
        await _callback(req.callback_url, req.job_id, req.team_id, "failed", str(exc), trace_id)


async def _call_openrouter(prompt: str, trace_id: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://dadaassis.zeabur.app",
        "X-Title": f"DaDaAssis {AGENT_ROLE.capitalize()}",
        "X-Trace-Id": trace_id,
    }
    payload = {
        "model": AGENT_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"[{AGENT_ROLE.upper()}] done in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)} trace={trace_id}", flush=True)
    return text


async def _callback(
    url: str, job_id: str, team_id: str, status: str, result: str, trace_id: str,
    max_retries: int = 3,
) -> None:
    payload = {
        "job_id": job_id,
        "team_id": team_id,
        "role": AGENT_ROLE,
        "status": status,
        "result": result,
    }
    headers = {
        "Content-Type": "application/json",
        "X-DaDaAssis-Auth": INTER_SERVICE_SECRET_OUT,
        "X-Trace-Id": trace_id,
    }
    delays = [3, 10, 20]
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (200, 202):
                    print(f"[{AGENT_ROLE.upper()}] callback ok job={job_id} attempt={attempt+1}", flush=True)
                    return
                print(f"[{AGENT_ROLE.upper()}] callback status={resp.status_code} job={job_id}", flush=True)
            except Exception as exc:
                print(f"[{AGENT_ROLE.upper()}] callback attempt {attempt+1} err={exc}", flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(delays[attempt])
    print(f"[{AGENT_ROLE.upper()}] callback exhausted job={job_id}", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
