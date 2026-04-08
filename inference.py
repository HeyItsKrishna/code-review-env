"""
inference.py - Baseline inference script for CodeReview OpenEnv.

Logging format (exact):
  [START] task=<difficulty> session=<id>
  [STEP]  step=<n> action=<type> reward=<r> done=<bool>
  [END]   task=<difficulty> session=<id> score=<final_score>
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from openai import OpenAI

API_BASE_URL     = os.getenv("API_BASE_URL", "https://KrishnaAIX-KrishnaAIX.hf.space")
MODEL_NAME       = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN         = os.getenv("HF_TOKEN")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", HF_TOKEN)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL     = API_BASE_URL.rstrip("/")
DIFFICULTIES     = ["easy", "medium", "hard"]

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "dummy",
)

def env_reset(difficulty: str) -> Dict[str, Any]:
    r = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_difficulty": difficulty},
        timeout=30,
        verify=False,
    )
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        f"{API_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
        verify=False,
    )
    r.raise_for_status()
    return r.json()


SYSTEM_PROMPT = """\
You are an expert software engineer performing a pull request code review.
You will receive the PR diff and must identify ONLY real, critical bugs and security issues.

STRICT RULES:
1. Add AT MOST 2 comments total - only for DEFINITE bugs or security issues with exact line numbers.
2. If you are not 100% sure about an issue, do NOT add a comment.
3. After adding comments (or if there are no issues), call set_review_decision ONCE.
4. Then immediately call finish_review to end.
5. Never add duplicate or speculative comments - each wrong comment costs -0.05.

Response format - single JSON object, no markdown:
{
  "action_type": "add_comment" | "set_review_decision" | "finish_review",
  "comment": {
    "filename": "<file>",
    "line_number": <int>,
    "severity": "critical",
    "category": "security" | "correctness",
    "message": "<specific bug description>",
    "suggestion": "<exact fix>"
  },
  "decision": "request_changes" | "approve",
  "reasoning": "<one sentence>"
}

STRATEGY:
- Step 1: Add 1-2 comments ONLY if you see a clear bug/security issue
- Step 2: set_review_decision (request_changes if issues found, approve if clean)
- Step 3: finish_review
"""


def build_user_message(obs: Dict[str, Any]) -> str:
    pr = obs["pull_request"]
    diff_text = ""
    for change in pr["changes"]:
        diff_text += f"\n\n### File: {change['filename']} ({change['language']})\n"
        diff_text += (
            f"Additions: {change['additions']}, Deletions: {change['deletions']}, "
            f"Has tests: {change['has_tests']}\n"
        )
        diff_text += f"```diff\n{change['diff']}\n```"

    comments_so_far = obs.get("review_comments_so_far", [])
    comment_summary = ""
    if comments_so_far:
        comment_summary = f"\n\nComments already added ({len(comments_so_far)}):\n"
        for i, c in enumerate(comments_so_far):
            comment_summary += (
                f"  [{i}] {c['filename']}:{c['line_number']} "
                f"[{c['severity']}] {c['message'][:80]}\n"
            )

    checklist = "\n".join(f"  - {item}" for item in obs.get("checklist_remaining", []))
    decision = obs.get("review_decision")
    decision_text = f"\nCurrent decision: {decision}" if decision else "\nNo decision set yet."

    num_comments = len(comments_so_far)
    urgency = ""
    if num_comments >= 2:
        urgency = "\nYou have added enough comments. NOW call set_review_decision, then finish_review."
    elif decision:
        urgency = "\nDecision is set. NOW call finish_review immediately."

    return (
        f"## Pull Request: {pr['title']}\n"
        f"**PR #{pr['pr_id']}** by {pr['author']} -> {pr['target_branch']}\n"
        f"**Description:** {pr['description']}\n"
        f"**Labels:** {', '.join(pr['labels'])}\n\n"
        f"## Diffs\n{diff_text}\n\n"
        f"## Review Checklist (remaining)\n{checklist}\n"
        f"{comment_summary}\n"
        f"{decision_text}\n"
        f"{urgency}\n\n"
        f"Step {obs['step_number']}/{obs['max_steps']}. "
        f"Respond with a single JSON object."
    )


def parse_action(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action_type": "finish_review", "reasoning": "parse error"}


def run_episode(difficulty: str) -> float:
    # Always print START - even if everything fails
    session_id = "error"
    try:
        reset_data = env_reset(difficulty)
        session_id = reset_data["session_id"]
        obs        = reset_data["observation"]
        print(f"[START] task={difficulty} session={session_id}", flush=True)
    except Exception as e:
        print(f"[START] task={difficulty} session={session_id}", flush=True)
        print(f"[STEP] step=1 action=finish_review reward=0.0 done=True", flush=True)
        print(f"[END] task={difficulty} session={session_id} score=0.0000", flush=True)
        return 0.0

    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    done        = False
    final_score = 0.0
    step_num    = 0

    while not done:
        step_num += 1

        if step_num > 6:
            action = {"action_type": "finish_review", "reasoning": "max steps reached"}
        else:
            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            comments_so_far = obs.get("review_comments_so_far", [])
            current_decision = obs.get("review_decision")

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"[STEP] step={step_num} action=error reward=0.0 done=False", flush=True)
                response_text = '{"action_type": "finish_review", "reasoning": "llm unavailable"}'

            messages.append({"role": "assistant", "content": response_text})
            action = parse_action(response_text)

            if current_decision and action.get("action_type") == "add_comment":
                action = {"action_type": "finish_review", "reasoning": "decision already set"}

            if len(comments_so_far) >= 3 and action.get("action_type") == "add_comment":
                action = {"action_type": "finish_review", "reasoning": "enough comments"}

        try:
            step_data = env_step(session_id, action)
        except Exception as e:
            print(
                f"[STEP] step={step_num} action={action.get('action_type','?')} "
                f"reward=0.0 done=True",
                flush=True,
            )
            break

        reward      = step_data["reward"]["total"]
        done        = step_data["done"]
        obs         = step_data["observation"]
        action_type = action.get("action_type", "unknown")

        print(
            f"[STEP] step={step_num} action={action_type} "
            f"reward={reward:.4f} done={done}",
            flush=True,
        )

        if done:
            final_score = step_data["info"].get("final_score", reward)

        time.sleep(0.1)

    print(f"[END] task={difficulty} session={session_id} score={final_score:.4f}", flush=True)
    return final_score


def main() -> int:
    scores: Dict[str, float] = {}

    for difficulty in DIFFICULTIES:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {difficulty.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            score = run_episode(difficulty)
        except Exception as e:
            print(f"[START] task={difficulty} session=error", flush=True)
            print(f"[STEP] step=1 action=finish_review reward=0.0 done=True", flush=True)
            print(f"[END] task={difficulty} session=error score=0.0000", flush=True)
            score = 0.0
        scores[difficulty] = score

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for diff, sc in scores.items():
        print(f"  {diff:8s}: {sc:.4f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':8s}: {avg:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())