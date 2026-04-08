"""
Unit tests for CodeReviewEnv.
Run with: pytest tests/ -v
"""
import pytest
from environment.env import CodeReviewEnv
from environment.models import Action, ReviewComment


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def easy_env():
    env = CodeReviewEnv(task_difficulty="easy")
    env.reset()
    return env


@pytest.fixture
def hard_env():
    env = CodeReviewEnv(task_difficulty="hard")
    env.reset()
    return env


# ------------------------------------------------------------------ reset

def test_reset_returns_observation():
    env = CodeReviewEnv(task_difficulty="easy")
    obs = env.reset()
    assert obs.task_difficulty == "easy"
    assert obs.step_number == 0
    assert obs.pull_request is not None
    assert len(obs.pull_request.changes) >= 1


def test_reset_clears_state():
    env = CodeReviewEnv(task_difficulty="easy")
    env.reset()
    env.step(Action(
        action_type="add_comment",
        comment=ReviewComment(
            filename="auth/utils.py",
            line_number=5,
            severity="critical",
            category="security",
            message="test",
        )
    ))
    obs = env.reset()
    assert obs.step_number == 0
    assert obs.review_comments_so_far == []


# ------------------------------------------------------------------ state

def test_state_returns_dict(easy_env):
    s = easy_env.state()
    assert isinstance(s, dict)
    assert "step_number" in s
    assert "difficulty" in s
    assert "done" in s


# ------------------------------------------------------------------ step

def test_step_increments_step_number(easy_env):
    obs, _, _, _ = easy_env.step(Action(action_type="request_more_context", question="Why?"))
    assert obs.step_number == 1


def test_add_comment_is_stored(easy_env):
    comment = ReviewComment(
        filename="auth/utils.py",
        line_number=5,
        severity="critical",
        category="security",
        message="Hardcoded secret",
    )
    obs, reward, done, info = easy_env.step(
        Action(action_type="add_comment", comment=comment)
    )
    assert len(obs.review_comments_so_far) == 1
    assert not done


def test_true_positive_gives_positive_reward(easy_env):
    """Matching a ground-truth issue should yield positive immediate reward."""
    comment = ReviewComment(
        filename="auth/utils.py",
        line_number=5,
        severity="critical",
        category="security",
        message="Hardcoded SECRET_KEY is a security risk",
    )
    _, reward, _, _ = easy_env.step(
        Action(action_type="add_comment", comment=comment)
    )
    assert reward.total > 0


def test_false_positive_gives_negative_reward(easy_env):
    """Commenting on a non-existent issue should penalise."""
    comment = ReviewComment(
        filename="auth/utils.py",
        line_number=100,
        severity="warning",
        category="style",
        message="This line is too long",
    )
    _, reward, _, _ = easy_env.step(
        Action(action_type="add_comment", comment=comment)
    )
    assert reward.total < 0


def test_set_decision_works(easy_env):
    obs, _, _, _ = easy_env.step(
        Action(action_type="set_review_decision", decision="request_changes")
    )
    assert obs.review_decision == "request_changes"


def test_finish_review_ends_episode(easy_env):
    easy_env.step(Action(action_type="set_review_decision", decision="request_changes"))
    _, _, done, info = easy_env.step(Action(action_type="finish_review"))
    assert done
    assert info.final_score is not None
    assert 0.0 <= info.final_score <= 1.0


def test_step_after_done_raises(easy_env):
    easy_env.step(Action(action_type="finish_review"))
    with pytest.raises(RuntimeError):
        easy_env.step(Action(action_type="finish_review"))


def test_max_steps_terminates(easy_env):
    done = False
    for _ in range(20):  # easy max_steps=15
        _, _, done, _ = easy_env.step(Action(action_type="request_more_context", question="?"))
        if done:
            break
    assert done


# ------------------------------------------------------------------ grader

def test_perfect_easy_score():
    """An agent that finds all easy issues gets a high score."""
    from graders.scorer import grade
    from tasks.definitions import EASY_GROUND_TRUTH, EASY_CORRECT_DECISION

    comments = [
        ReviewComment(
            filename=issue["filename"],
            line_number=issue["line_number"] if issue["line_number"] > 0 else 1,
            severity=issue["severity"],
            category=issue["category"],
            message=issue["description"],
        )
        for issue in EASY_GROUND_TRUTH
    ]
    reward = grade(
        comments=comments,
        decision=EASY_CORRECT_DECISION,
        ground_truth=EASY_GROUND_TRUTH,
        correct_decision=EASY_CORRECT_DECISION,
        steps_used=8,
        max_steps=15,
        task_difficulty="easy",
    )
    assert reward.total >= 0.8


def test_empty_review_low_score():
    from graders.scorer import grade
    from tasks.definitions import EASY_GROUND_TRUTH, EASY_CORRECT_DECISION

    reward = grade(
        comments=[],
        decision=None,
        ground_truth=EASY_GROUND_TRUTH,
        correct_decision=EASY_CORRECT_DECISION,
        steps_used=15,
        max_steps=15,
        task_difficulty="easy",
    )
    assert reward.total < 0.3


def test_wrong_decision_penalised():
    from graders.scorer import grade
    from tasks.definitions import EASY_GROUND_TRUTH

    reward_wrong = grade(
        comments=[],
        decision="approve",
        ground_truth=EASY_GROUND_TRUTH,
        correct_decision="request_changes",
        steps_used=5,
        max_steps=15,
        task_difficulty="easy",
    )
    reward_correct = grade(
        comments=[],
        decision="request_changes",
        ground_truth=EASY_GROUND_TRUTH,
        correct_decision="request_changes",
        steps_used=5,
        max_steps=15,
        task_difficulty="easy",
    )
    assert reward_correct.decision_accuracy > reward_wrong.decision_accuracy


def test_score_in_valid_range():
    from graders.scorer import grade
    from tasks.definitions import HARD_GROUND_TRUTH, HARD_CORRECT_DECISION

    reward = grade(
        comments=[
            ReviewComment(
                filename="cache/distributed.py",
                line_number=14,
                severity="critical",
                category="security",
                message="Tenant injection",
            )
        ],
        decision="request_changes",
        ground_truth=HARD_GROUND_TRUTH,
        correct_decision=HARD_CORRECT_DECISION,
        steps_used=10,
        max_steps=25,
        task_difficulty="hard",
    )
    assert -1.0 <= reward.total <= 1.0


# ------------------------------------------------------------------ server

def test_server_health(tmp_path):
    """Smoke test: server imports without error."""
    import importlib
    import server  # noqa: F401
    assert True


def test_all_difficulties():
    for difficulty in ["easy", "medium", "hard"]:
        env = CodeReviewEnv(task_difficulty=difficulty)
        obs = env.reset()
        assert obs.task_difficulty == difficulty
        _, _, done, info = env.step(Action(action_type="finish_review"))
        assert done
        assert info.final_score is not None
