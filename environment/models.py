"""
Pydantic models for the Code Review Triage OpenEnv environment.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core domain types
# ---------------------------------------------------------------------------

class CodeChange(BaseModel):
    """A single file change in a pull request."""
    filename: str
    language: str
    diff: str                          # unified-diff string
    additions: int
    deletions: int
    has_tests: bool = False
    complexity_score: float = 0.0      # cyclomatic complexity (0–10)


class ReviewComment(BaseModel):
    """A comment left by the agent on a specific line."""
    filename: str
    line_number: int
    severity: Literal["info", "warning", "critical"]
    category: Literal[
        "security", "performance", "correctness",
        "style", "maintainability", "test_coverage"
    ]
    message: str
    suggestion: Optional[str] = None   # optional code fix


class PullRequest(BaseModel):
    """Full representation of a pull request to review."""
    pr_id: str
    title: str
    description: str
    author: str
    target_branch: str = "main"
    changes: List[CodeChange]
    existing_comments: List[ReviewComment] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenEnv Observation / Action / Reward
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    task_difficulty: Literal["easy", "medium", "hard"]
    pull_request: PullRequest
    step_number: int
    max_steps: int
    review_comments_so_far: List[ReviewComment] = Field(default_factory=list)
    review_decision: Optional[Literal["approve", "request_changes", "comment"]] = None
    # Hints / scaffolding visible to agent
    checklist_remaining: List[str] = Field(default_factory=list)
    token_budget_remaining: int = 4096


class Action(BaseModel):
    """An action the agent can take."""
    action_type: Literal[
        "add_comment",
        "remove_comment",
        "set_review_decision",
        "request_more_context",
        "finish_review",
    ]
    # For add_comment
    comment: Optional[ReviewComment] = None
    # For remove_comment
    comment_index: Optional[int] = None
    # For set_review_decision
    decision: Optional[Literal["approve", "request_changes", "comment"]] = None
    # For request_more_context
    question: Optional[str] = None
    # Reasoning trace (not graded but logged)
    reasoning: Optional[str] = None


class Reward(BaseModel):
    """Dense, structured reward signal."""
    total: float = Field(ge=-1.0, le=1.0)
    # Sub-components
    issue_detection: float = 0.0       # caught real bugs / security issues
    false_positives: float = 0.0       # penalises spurious comments
    severity_calibration: float = 0.0  # severity matches ground truth
    decision_accuracy: float = 0.0     # approve vs request_changes
    efficiency: float = 0.0            # reward for not wasting steps
    coverage: float = 0.0              # % of files actually reviewed
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentInfo(BaseModel):
    """Extra info returned alongside step()."""
    done: bool
    truncated: bool
    task_id: str
    step_number: int
    elapsed_steps: int
    ground_truth_issues: Optional[List[Dict[str, Any]]] = None  # revealed on done
    final_score: Optional[float] = None
    message: str = ""
