"""
Deterministic graders for each task.
Returns a score in [0.0, 1.0] with a detailed breakdown.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from environment.models import ReviewComment, Reward


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _comment_matches_issue(
    comment: ReviewComment,
    issue: Dict[str, Any],
    line_tolerance: int = 3,
) -> bool:
    """
    Fuzzy match: a comment matches a ground-truth issue if:
      - same filename (exact)
      - line within ±line_tolerance (or issue line == 0 meaning file-level)
      - same category
      - severity >= issue severity (being more severe is acceptable)
    """
    SEV_RANK = {"info": 0, "warning": 1, "critical": 2}

    if comment.filename != issue["filename"]:
        return False
    if comment.category != issue["category"]:
        return False

    issue_line = issue["line_number"]
    if issue_line == 0:
        # file-level issue — any line matches
        pass
    elif abs(comment.line_number - issue_line) > line_tolerance:
        return False

    # Severity: agent must be at least as severe
    if SEV_RANK[comment.severity] < SEV_RANK[issue["severity"]]:
        return False

    return True


def _severity_exact_match(comment: ReviewComment, issue: Dict[str, Any]) -> bool:
    return comment.severity == issue["severity"]


# ---------------------------------------------------------------------------
# Core grader
# ---------------------------------------------------------------------------

def grade(
    comments: List[ReviewComment],
    decision: Optional[str],
    ground_truth: List[Dict[str, Any]],
    correct_decision: str,
    steps_used: int,
    max_steps: int,
    task_difficulty: str,
) -> Reward:
    """
    Compute a dense reward signal.

    Sub-components (weights sum to 1.0):
      issue_detection        40% — precision/recall F1 on ground truth issues
      false_positives        15% — penalty for spurious comments
      severity_calibration   15% — rewards exact severity match on true positives
      decision_accuracy      20% — correct approve/request_changes/comment
      efficiency             10% — bonus for finishing in fewer steps
    """
    if not ground_truth:
        return Reward(total=0.0)

    # ---- Issue detection (F1) ----------------------------------------
    matched_issues = set()
    true_positives = 0
    false_positive_count = 0

    for comment in comments:
        matched = False
        for idx, issue in enumerate(ground_truth):
            if idx in matched_issues:
                continue
            if _comment_matches_issue(comment, issue):
                matched_issues.add(idx)
                true_positives += 1
                matched = True
                break
        if not matched:
            false_positive_count += 1

    n_gt = len(ground_truth)
    precision = true_positives / len(comments) if comments else 0.0
    recall = true_positives / n_gt

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    issue_detection_score = f1

    # ---- False positive penalty -------------------------------------
    # Penalty scales with ratio of false positives to total comments
    fp_ratio = false_positive_count / max(len(comments), 1)
    false_positive_score = max(0.0, 1.0 - fp_ratio * 1.5)

    # ---- Severity calibration ----------------------------------------
    sev_matches = 0
    for comment in comments:
        for idx, issue in enumerate(ground_truth):
            if _comment_matches_issue(comment, issue) and _severity_exact_match(comment, issue):
                sev_matches += 1
                break
    severity_score = sev_matches / max(true_positives, 1) if true_positives > 0 else 0.0

    # ---- Decision accuracy -------------------------------------------
    if decision is None:
        decision_score = 0.0
    elif decision == correct_decision:
        decision_score = 1.0
    elif decision == "comment" and correct_decision == "request_changes":
        decision_score = 0.4  # partial credit — identified issues but didn't block
    else:
        decision_score = 0.0

    # ---- Efficiency --------------------------------------------------
    step_fraction = steps_used / max_steps
    efficiency_score = max(0.0, 1.0 - step_fraction) * 0.5 + 0.5  # [0.5, 1.0]

    # ---- Difficulty multiplier (hard tasks worth more per correct issue)
    difficulty_bonus = {"easy": 1.0, "medium": 1.0, "hard": 1.05}[task_difficulty]

    # ---- Weighted total ----------------------------------------------
    weights = {
        "issue_detection": 0.40,
        "false_positives": 0.15,
        "severity_calibration": 0.15,
        "decision_accuracy": 0.20,
        "efficiency": 0.10,
    }

    raw_total = (
        weights["issue_detection"] * issue_detection_score
        + weights["false_positives"] * false_positive_score
        + weights["severity_calibration"] * severity_score
        + weights["decision_accuracy"] * decision_score
        + weights["efficiency"] * efficiency_score
    )

    total = min(1.0, max(-1.0, raw_total * difficulty_bonus))

    return Reward(
        total=round(total, 4),
        issue_detection=round(issue_detection_score, 4),
        false_positives=round(false_positive_score, 4),
        severity_calibration=round(severity_score, 4),
        decision_accuracy=round(decision_score, 4),
        efficiency=round(efficiency_score, 4),
        breakdown={
            "true_positives": true_positives,
            "false_positives": false_positive_count,
            "ground_truth_count": n_gt,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "steps_used": steps_used,
            "max_steps": max_steps,
            "decision_given": decision,
            "correct_decision": correct_decision,
        },
    )


# ---------------------------------------------------------------------------
# Step-level dense reward (called at each step, not just end)
# ---------------------------------------------------------------------------

def step_reward(
    new_comment: Optional[ReviewComment],
    ground_truth: List[Dict[str, Any]],
    existing_matched: set,
    task_difficulty: str,
) -> float:
    """
    Immediate per-step reward signal to guide exploration.
    Returns a float in [-0.1, 0.3].
    """
    if new_comment is None:
        return 0.0

    for idx, issue in enumerate(ground_truth):
        if idx in existing_matched:
            continue
        if _comment_matches_issue(new_comment, issue):
            severity_bonus = {"info": 0.05, "warning": 0.10, "critical": 0.20}
            base = severity_bonus.get(issue["severity"], 0.05)
            sev_match = 0.05 if _severity_exact_match(new_comment, issue) else 0.0
            return base + sev_match

    # False positive — small penalty
    return -0.05
