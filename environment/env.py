"""
CodeReviewEnv — OpenEnv-compliant environment for AI code review agents.
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Action, EnvironmentInfo, Observation, Reward, ReviewComment
)
from graders.scorer import grade, step_reward
from tasks.definitions import TASKS


class CodeReviewEnv:
    """
    An OpenEnv environment that simulates pull-request code review.

    The agent reads a pull request (diffs, metadata) and must:
      1. Identify real issues by leaving ReviewComments
      2. Classify each by severity and category
      3. Make a final review decision (approve / request_changes / comment)

    Three difficulty tiers — easy, medium, hard — each with a different PR
    and deterministic ground-truth grader.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, task_difficulty: str = "easy"):
        assert task_difficulty in TASKS, f"Unknown difficulty: {task_difficulty}"
        self._difficulty = task_difficulty
        self._task_cfg = TASKS[task_difficulty]
        self._reset_state()

    # -------------------------------------------------------------- OpenEnv API

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._reset_state()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, EnvironmentInfo]:
        """
        Apply one action and return (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() first.")

        self._step_number += 1
        immediate_reward = Reward(total=0.0)
        done = False
        message = ""

        # ---- Dispatch action ----------------------------------------
        if action.action_type == "add_comment":
            immediate_reward, message = self._handle_add_comment(action)

        elif action.action_type == "remove_comment":
            message = self._handle_remove_comment(action)

        elif action.action_type == "set_review_decision":
            message = self._handle_set_decision(action)

        elif action.action_type == "request_more_context":
            # Small time cost — agent asked for clarification
            immediate_reward = Reward(total=-0.01)
            message = "Context request logged."

        elif action.action_type == "finish_review":
            done = True
            message = "Review finished by agent."

        # ---- Check termination conditions ---------------------------
        if self._step_number >= self._task_cfg["max_steps"]:
            done = True
            message = "Maximum steps reached."

        if done:
            self._done = True
            final_reward = self._compute_final_reward()
            info = EnvironmentInfo(
                done=True,
                truncated=(self._step_number >= self._task_cfg["max_steps"]),
                task_id=self._task_id,
                step_number=self._step_number,
                elapsed_steps=self._step_number,
                ground_truth_issues=self._task_cfg["ground_truth"],
                final_score=final_reward.total,
                message=message,
            )
            return self._build_observation(), final_reward, True, info

        info = EnvironmentInfo(
            done=False,
            truncated=False,
            task_id=self._task_id,
            step_number=self._step_number,
            elapsed_steps=self._step_number,
            message=message,
        )
        return self._build_observation(), immediate_reward, False, info

    def state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the full environment state."""
        return {
            "task_id": self._task_id,
            "difficulty": self._difficulty,
            "step_number": self._step_number,
            "max_steps": self._task_cfg["max_steps"],
            "done": self._done,
            "review_comments": [c.model_dump() for c in self._comments],
            "review_decision": self._decision,
            "matched_issues": list(self._matched_issue_indices),
            "pr_id": self._task_cfg["pr"].pr_id,
        }

    # ----------------------------------------------------------------- private

    def _reset_state(self) -> None:
        self._task_id = f"{self._difficulty}-{uuid.uuid4().hex[:8]}"
        self._step_number = 0
        self._done = False
        self._comments: List[ReviewComment] = []
        self._decision: Optional[str] = None
        self._matched_issue_indices: set = set()
        self._checklist = list(self._task_cfg["checklist"])

    def _build_observation(self) -> Observation:
        cfg = self._task_cfg
        return Observation(
            task_id=self._task_id,
            task_difficulty=self._difficulty,
            pull_request=copy.deepcopy(cfg["pr"]),
            step_number=self._step_number,
            max_steps=cfg["max_steps"],
            review_comments_so_far=list(self._comments),
            review_decision=self._decision,
            checklist_remaining=list(self._checklist),
            token_budget_remaining=max(0, 4096 - self._step_number * 128),
        )

    def _handle_add_comment(self, action: Action) -> Tuple[Reward, str]:
        if action.comment is None:
            return Reward(total=-0.02), "No comment provided."

        comment = action.comment
        self._comments.append(comment)

        # Tick off checklist item if comment aligns
        cat_map = {
            "security": 0,
            "test_coverage": len(self._checklist) - 1,
        }

        # Compute immediate step reward
        r = step_reward(
            comment,
            self._task_cfg["ground_truth"],
            self._matched_issue_indices,
            self._difficulty,
        )

        # Update matched set (so same issue not rewarded twice)
        for idx, issue in enumerate(self._task_cfg["ground_truth"]):
            if idx not in self._matched_issue_indices:
                from graders.scorer import _comment_matches_issue
                if _comment_matches_issue(comment, issue):
                    self._matched_issue_indices.add(idx)
                    break

        msg = "Comment added."
        if r > 0:
            msg += " Matched a known issue."
        elif r < 0:
            msg += " No matching issue found (possible false positive)."

        return Reward(total=round(r, 4)), msg

    def _handle_remove_comment(self, action: Action) -> str:
        idx = action.comment_index
        if idx is None or idx < 0 or idx >= len(self._comments):
            return f"Invalid comment index {idx}."
        removed = self._comments.pop(idx)
        # Revoke matched issue if it was linked
        # (simplified: re-run matching on remaining comments)
        self._recompute_matched()
        return f"Removed comment: {removed.message[:60]}"

    def _handle_set_decision(self, action: Action) -> str:
        if action.decision not in ("approve", "request_changes", "comment"):
            return "Invalid decision value."
        self._decision = action.decision
        return f"Review decision set to: {action.decision}"

    def _recompute_matched(self) -> None:
        from graders.scorer import _comment_matches_issue
        self._matched_issue_indices = set()
        for comment in self._comments:
            for idx, issue in enumerate(self._task_cfg["ground_truth"]):
                if idx not in self._matched_issue_indices:
                    if _comment_matches_issue(comment, issue):
                        self._matched_issue_indices.add(idx)
                        break

    def _compute_final_reward(self) -> Reward:
        return grade(
            comments=self._comments,
            decision=self._decision,
            ground_truth=self._task_cfg["ground_truth"],
            correct_decision=self._task_cfg["correct_decision"],
            steps_used=self._step_number,
            max_steps=self._task_cfg["max_steps"],
            task_difficulty=self._difficulty,
        )
