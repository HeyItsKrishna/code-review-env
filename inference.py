"""
inference.py — Baseline inference script for CodeReview OpenEnv.

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
MODEL_NAME       = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = API_BASE_URL.rstrip("/")
DIFFICULTIES = ["easy", "medium", "hard"]

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1",
    api_key=HF_TOKEN,
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
You will receive the PR diff and must identify real bugs, security issues,
and code quality problems.

For each step, respond with a single JSON object (no markdown fences):
{
  "action_type": "add_comment" | "set_review_decision" | "finish_review",
  "comment": {
    "filename": "<file>",
    "line_number": <int>,
    "severity": "info" | "warning" | "critical",
    "category": "security" | "performance" | "correctness" | "style" | "maintainability" | "test_coverage",
    "message": "<description of the issue>",
    "suggestion": "<optional fix>"
  },
  "decision": "approve" | "request_changes" | "comment",
  "reasoning": "<brief chain-of-thought>"
}

Rules:
- Only comment on REAL issues visible in the diff — not hypothetical ones.
- Set the review decision once, then call finish_review.
- Be precise about line numbers from the diff (+lines shown).
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

    return (
        f"## Pull Request: {pr['title']}\n"
        f"**PR #{pr['pr_id']}** by {pr['author']} -> {pr['target_branch']}\n"
        f"**Description:** {pr['description']}\n"
        f"**Labels:** {', '.join(pr['labels'])}\n\n"
        f"## Diffs\n{diff_text}\n\n"
        f"## Review Checklist (remaining)\n{checklist}\n"
        f"{comment_summary}\n"
        f"{decision_text}\n\n"
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
    reset_data  = env_reset(difficulty)
    session_id  = reset_data["session_id"]
    obs         = reset_data["observation"]

    print(f"[START] task={difficulty} session={session_id}", flush=True)

    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    done        = False
    final_score = 0.0
    step_num    = 0

    while not done:
        step_num += 1
        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

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
            response_text = '{"action_type": "finish_review"}'

        messages.append({"role": "assistant", "content": response_text})
        action = parse_action(response_text)

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
            print(f"[END] task={difficulty} session=error score=0.0", flush=True)
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
