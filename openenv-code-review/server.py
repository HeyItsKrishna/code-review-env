"""
FastAPI server exposing the CodeReviewEnv via HTTP.
Compliant with the OpenEnv REST spec.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment.env import CodeReviewEnv
from environment.models import Action, Observation, Reward, EnvironmentInfo

app = FastAPI(
    title="CodeReview OpenEnv",
    description="Production-grade pull request code review environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-process session store (single-process deployment)
_sessions: Dict[str, CodeReviewEnv] = {}


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_difficulty: str = "easy"
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "CodeReviewEnv", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    import uuid
    session_id = req.session_id or str(uuid.uuid4())
    try:
        env = CodeReviewEnv(task_difficulty=req.task_difficulty)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    obs = env.reset()
    _sessions[session_id] = env
    return ResetResponse(
        session_id=session_id,
        observation=obs.model_dump(),
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found. Call /reset first.")
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info.model_dump(),
    )


@app.get("/state/{session_id}", response_model=StateResponse)
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return StateResponse(session_id=session_id, state=env.state())


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    removed = _sessions.pop(session_id, None)
    return {"deleted": removed is not None, "session_id": session_id}


@app.get("/tasks")
def list_tasks():
    from tasks.definitions import TASKS
    return {
        difficulty: {
            "pr_id": cfg["pr"].pr_id,
            "pr_title": cfg["pr"].title,
            "max_steps": cfg["max_steps"],
            "num_files": len(cfg["pr"].changes),
            "num_ground_truth_issues": len(cfg["ground_truth"]),
        }
        for difficulty, cfg in TASKS.items()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
