from __future__ import annotations

import asyncio
import json
import os
import random
import time
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

# LLM 後端切換："openrouter"（按量計費，預設，行為不變）或 "claude_cli"（Max 訂閱 OAuth）
LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter")
CLI_TIMEOUT_SEC = int(os.getenv("CLI_TIMEOUT_SEC", "180"))
CLI_MAX_CONCURRENT = int(os.getenv("CLI_MAX_CONCURRENT", "2"))
CLI_JITTER_MAX_SEC = float(os.getenv("CLI_JITTER_MAX_SEC", "3"))
_cli_semaphore = asyncio.Semaphore(CLI_MAX_CONCURRENT)

# 判定為 OAuth/額度類錯誤的關鍵字（用來觸發熔斷，跟一般 CLI crash 分開處理）
_OAUTH_ERROR_MARKERS = (
    "usage limit", "rate limit", "429", "unauthorized", "401", "403",
    "authentication", "session", "not logged in", "please run",
)

# Load agent persona from CLAUDE.md
def _load_persona() -> str:
    try:
        with open("/root/CLAUDE.md", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return f"你是 {AGENT_ROLE}，DaDaAssis AI 核心團隊成員。請用繁體中文回覆。"

SYSTEM_PROMPT = _load_persona()

app = FastAPI(title=f"DaDaAssis cc-{AGENT_ROLE}", version="1.0.0")

_cli_health: dict = {"checked_at": None, "ok": None, "detail": None}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "role": AGENT_ROLE,
        "sha": BUILD_SHA,
        "model": AGENT_MODEL,
        "backend": LLM_BACKEND,
        "cli_health": _cli_health if LLM_BACKEND == "claude_cli" else None,
    }


@app.on_event("startup")
async def _startup_cli_healthcheck() -> None:
    if LLM_BACKEND != "claude_cli":
        return
    ok, detail = await _claude_whoami()
    _cli_health.update({"checked_at": int(time.time()), "ok": ok, "detail": detail})
    print(f"[{AGENT_ROLE.upper()}/cli] startup healthcheck ok={ok} detail={detail}", flush=True)


async def _claude_whoami() -> tuple[bool, str]:
    """跑一次 `claude whoami` 確認 OAuth session 有效，不佔用 job semaphore。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "whoami",
            cwd="/root",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            return True, out.decode(errors="replace").strip()[:200]
        return False, err.decode(errors="replace").strip()[:200]
    except FileNotFoundError:
        return False, "claude CLI not found in image"
    except Exception as exc:
        return False, str(exc)[:200]


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
    print(f"[{AGENT_ROLE.upper()}] job={req.job_id} team={req.team_id} trace={trace_id} backend={LLM_BACKEND}", flush=True)
    asyncio.create_task(_process(req, trace_id))
    return {"accepted": True, "job_id": req.job_id, "role": AGENT_ROLE}


async def _process(req: JobRequest, trace_id: str) -> None:
    try:
        if LLM_BACKEND == "claude_cli":
            result = await _call_claude_cli(req.prompt, trace_id)
        else:
            result = await _call_openrouter(req.prompt, trace_id)
        await _callback(req.callback_url, req.job_id, req.team_id, "done", result, trace_id)
    except _OAuthError as exc:
        print(f"[{AGENT_ROLE.upper()}] OAuth error job={req.job_id} err={exc}", flush=True)
        _cli_health.update({"checked_at": int(time.time()), "ok": False, "detail": str(exc)[:200]})
        await _callback(req.callback_url, req.job_id, req.team_id, "failed", f"OAUTH_LIMIT: {exc}", trace_id)
    except Exception as exc:
        print(f"[{AGENT_ROLE.upper()}] error job={req.job_id} err={exc}", flush=True)
        await _callback(req.callback_url, req.job_id, req.team_id, "failed", str(exc), trace_id)


class _OAuthError(RuntimeError):
    """CLI 回傳的錯誤判定為額度/認證類，跟一般 crash 分開，方便上游做熔斷。"""


async def _call_claude_cli(prompt: str, trace_id: str) -> str:
    async with _cli_semaphore:
        if CLI_JITTER_MAX_SEC > 0:
            await asyncio.sleep(random.uniform(0, CLI_JITTER_MAX_SEC))
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            "--output-format", "json",
            cwd="/root",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=CLI_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"claude CLI timeout after {CLI_TIMEOUT_SEC}s")

        stderr_text = err.decode(errors="replace").strip()
        if proc.returncode != 0:
            if any(marker in stderr_text.lower() for marker in _OAUTH_ERROR_MARKERS):
                raise _OAuthError(stderr_text[-500:] or f"exit={proc.returncode}")
            raise RuntimeError(f"claude CLI exit={proc.returncode}: {stderr_text[-500:]}")

        try:
            data = json.loads(out.decode(errors="replace"))
        except json.JSONDecodeError:
            # 非 JSON 輸出，直接把 stdout 當結果（保底）
            text = out.decode(errors="replace").strip()
            print(f"[{AGENT_ROLE.upper()}/cli] done (raw) trace={trace_id}", flush=True)
            return text

        if data.get("is_error"):
            detail = str(data.get("result") or data)
            if any(marker in detail.lower() for marker in _OAUTH_ERROR_MARKERS):
                raise _OAuthError(detail[:500])
            raise RuntimeError(detail[:500])

        text = data.get("result", "")
        usage = data.get("usage", {})
        print(
            f"[{AGENT_ROLE.upper()}/cli] done in={usage.get('input_tokens', 0)} "
            f"out={usage.get('output_tokens', 0)} cost=$0(subscription) trace={trace_id}",
            flush=True,
        )
        return text


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
