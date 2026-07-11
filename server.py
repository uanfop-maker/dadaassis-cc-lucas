from __future__ import annotations

import asyncio
import os
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

AGENT_ROLE = os.getenv("AGENT_ROLE", "harper")
INTER_SERVICE_SECRET_IN = os.getenv("INTER_SERVICE_SECRET_A", "")
INTER_SERVICE_SECRET_OUT = os.getenv("INTER_SERVICE_SECRET_B", "")
BUILD_SHA = os.getenv("BUILD_SHA", "dev")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "90"))

app = FastAPI(title=f"DaDaAssis cc-{AGENT_ROLE}", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "role": AGENT_ROLE, "sha": BUILD_SHA}


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
        result = await _run_claude(req.prompt, trace_id)
        await _callback(req.callback_url, req.job_id, req.team_id, "done", result, trace_id)
    except Exception as exc:
        print(f"[{AGENT_ROLE.upper()}] error job={req.job_id} err={exc}", flush=True)
        await _callback(req.callback_url, req.job_id, req.team_id, "failed", str(exc), trace_id)


async def _run_claude(prompt: str, trace_id: str) -> str:
    print(f"[{AGENT_ROLE.upper()}] running claude -p trace={trace_id}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        "--output-format", "text",
        "--no-color",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
        cwd="/root",
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=CLAUDE_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"claude timeout after {CLAUDE_TIMEOUT}s")
    if proc.returncode != 0:
        err = stderr.decode().strip()
        raise RuntimeError(f"claude exit {proc.returncode}: {err[:200]}")
    output = stdout.decode().strip()
    print(f"[{AGENT_ROLE.upper()}] done len={len(output)} trace={trace_id}", flush=True)
    return output


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
            except Exception as exc:
                print(f"[{AGENT_ROLE.upper()}] callback attempt {attempt+1} failed: {exc}", flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(delays[attempt])
    print(f"[{AGENT_ROLE.upper()}] callback exhausted job={job_id}", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
