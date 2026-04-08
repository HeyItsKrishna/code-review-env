"""
Task definitions: Easy, Medium, Hard pull requests with ground-truth graders.
Each task is a realistic PR scenario with known bugs / issues.
"""
from __future__ import annotations
from typing import Any, Dict, List
from environment.models import (
    CodeChange, PullRequest, ReviewComment
)


# ---------------------------------------------------------------------------
# TASK 1 – EASY: Python utility function with obvious bugs
# ---------------------------------------------------------------------------

EASY_PR = PullRequest(
    pr_id="pr-001",
    title="Add user authentication helper",
    description="Adds a simple helper to validate JWT tokens and check user roles.",
    author="dev-junior",
    target_branch="main",
    labels=["feature", "auth"],
    changes=[
        CodeChange(
            filename="auth/utils.py",
            language="python",
            additions=42,
            deletions=0,
            has_tests=False,
            complexity_score=4.2,
            diff='''\
+import jwt
+import hashlib
+import os
+
+SECRET_KEY = "mysecretkey123"  # hardcoded secret
+
+def validate_token(token: str) -> dict:
+    """Validate a JWT token and return the payload."""
+    try:
+        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
+        return payload
+    except jwt.ExpiredSignatureError:
+        return {}
+    except Exception:
+        return {}  # silences ALL exceptions including invalid tokens
+
+def hash_password(password: str) -> str:
+    """Hash a password for storage."""
+    return hashlib.md5(password.encode()).hexdigest()  # MD5 is insecure
+
+def check_role(user: dict, required_role: str) -> bool:
+    """Check if user has required role."""
+    return user["role"] == required_role  # KeyError if 'role' missing
+
+def get_admin_users() -> list:
+    """Return all admin users from DB."""
+    import sqlite3
+    conn = sqlite3.connect("users.db")
+    cursor = conn.cursor()
+    query = f"SELECT * FROM users WHERE role = 'admin'"
+    cursor.execute(query)
+    return cursor.fetchall()  # connection never closed
''',
        ),
    ],
)

EASY_GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "filename": "auth/utils.py",
        "line_number": 5,
        "severity": "critical",
        "category": "security",
        "key": "hardcoded_secret",
        "description": "Hardcoded SECRET_KEY in source code",
    },
    {
        "filename": "auth/utils.py",
        "line_number": 19,
        "severity": "critical",
        "category": "security",
        "key": "md5_password",
        "description": "MD5 used for password hashing — cryptographically broken",
    },
    {
        "filename": "auth/utils.py",
        "line_number": 11,
        "severity": "warning",
        "category": "correctness",
        "key": "silent_exception",
        "description": "Bare except silences all exceptions including invalid tokens",
    },
    {
        "filename": "auth/utils.py",
        "line_number": 23,
        "severity": "warning",
        "category": "correctness",
        "key": "keyerror_role",
        "description": "KeyError if 'role' key missing from user dict",
    },
    {
        "filename": "auth/utils.py",
        "line_number": 31,
        "severity": "warning",
        "category": "performance",
        "key": "unclosed_connection",
        "description": "SQLite connection never closed — resource leak",
    },
    {
        "filename": "auth/utils.py",
        "line_number": 0,
        "severity": "warning",
        "category": "test_coverage",
        "key": "no_tests",
        "description": "No tests added for authentication utilities",
    },
]

EASY_CORRECT_DECISION = "request_changes"


# ---------------------------------------------------------------------------
# TASK 2 – MEDIUM: Async API client with race conditions & error handling issues
# ---------------------------------------------------------------------------

MEDIUM_PR = PullRequest(
    pr_id="pr-042",
    title="Async batch data ingestion pipeline",
    description=(
        "Refactors the data ingestion service to use asyncio for better "
        "throughput. Adds retry logic and batch processing."
    ),
    author="dev-mid",
    target_branch="main",
    labels=["refactor", "performance", "data"],
    changes=[
        CodeChange(
            filename="ingestion/pipeline.py",
            language="python",
            additions=95,
            deletions=30,
            has_tests=True,
            complexity_score=7.1,
            diff='''\
+import asyncio
+import aiohttp
+import logging
+from typing import List, Dict, Any
+
+MAX_RETRIES = 3
+BATCH_SIZE = 100
+
+# Global session — not thread/coroutine-safe initialisation
+session: aiohttp.ClientSession = None
+
+async def init_session():
+    global session
+    session = aiohttp.ClientSession()
+
+async def fetch_record(url: str, record_id: str) -> Dict[str, Any]:
+    for attempt in range(MAX_RETRIES):
+        try:
+            async with session.get(f"{url}/{record_id}") as resp:
+                if resp.status == 200:
+                    return await resp.json()
+                elif resp.status == 429:
+                    await asyncio.sleep(2 ** attempt)
+        except aiohttp.ClientError as e:
+            logging.warning(f"Attempt {attempt} failed: {e}")
+    return {}  # silently returns empty dict on failure
+
+async def process_batch(records: List[str], url: str) -> List[Dict]:
+    tasks = [fetch_record(url, rid) for rid in records]
+    results = await asyncio.gather(*tasks)  # no return_exceptions=True
+    return list(results)
+
+async def ingest_all(record_ids: List[str], url: str) -> Dict[str, int]:
+    await init_session()
+    total = len(record_ids)
+    processed = 0
+    failed = 0
+
+    for i in range(0, total, BATCH_SIZE):
+        batch = record_ids[i:i + BATCH_SIZE]
+        results = await process_batch(batch, url)
+        processed += len([r for r in results if r])
+        failed += len([r for r in results if not r])
+
+    # session never closed
+    return {"processed": processed, "failed": failed, "total": total}
+
+def run_ingestion(record_ids: List[str], url: str) -> Dict[str, int]:
+    """Synchronous entry point."""
+    loop = asyncio.get_event_loop()  # deprecated pattern; may reuse closed loop
+    return loop.run_until_complete(ingest_all(record_ids, url))
''',
        ),
        CodeChange(
            filename="ingestion/test_pipeline.py",
            language="python",
            additions=25,
            deletions=0,
            has_tests=True,
            complexity_score=2.0,
            diff='''\
+import pytest
+from ingestion.pipeline import run_ingestion
+
+def test_empty_input():
+    result = run_ingestion([], "http://example.com")
+    assert result["total"] == 0
+
+# NOTE: No mocking of network calls — tests will hit real network
+# NOTE: No test for partial failure scenarios
+# NOTE: No test for retry logic
+def test_basic_ingestion():
+    result = run_ingestion(["id1", "id2"], "http://fake-api.test")
+    assert "processed" in result
''',
        ),
    ],
)

MEDIUM_GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "filename": "ingestion/pipeline.py",
        "line_number": 10,
        "severity": "critical",
        "category": "correctness",
        "key": "global_session_race",
        "description": "Global session initialised without lock — race condition under concurrent calls",
    },
    {
        "filename": "ingestion/pipeline.py",
        "line_number": 27,
        "severity": "warning",
        "category": "correctness",
        "key": "gather_no_exceptions",
        "description": "asyncio.gather without return_exceptions=True will propagate first exception and cancel remaining tasks",
    },
    {
        "filename": "ingestion/pipeline.py",
        "line_number": 22,
        "severity": "warning",
        "category": "correctness",
        "key": "silent_failure",
        "description": "Returns empty dict on all retries exhausted — caller cannot distinguish failure from empty record",
    },
    {
        "filename": "ingestion/pipeline.py",
        "line_number": 43,
        "severity": "warning",
        "category": "correctness",
        "key": "session_not_closed",
        "description": "aiohttp.ClientSession never closed — resource/connection leak",
    },
    {
        "filename": "ingestion/pipeline.py",
        "line_number": 47,
        "severity": "warning",
        "category": "correctness",
        "key": "deprecated_event_loop",
        "description": "asyncio.get_event_loop() is deprecated in Python 3.10+; use asyncio.run()",
    },
    {
        "filename": "ingestion/test_pipeline.py",
        "line_number": 8,
        "severity": "warning",
        "category": "test_coverage",
        "key": "no_mock_network",
        "description": "Tests make real network calls — should mock aiohttp session",
    },
]

MEDIUM_CORRECT_DECISION = "request_changes"


# ---------------------------------------------------------------------------
# TASK 3 – HARD: Distributed cache with subtle TTL/invalidation & security bugs
# ---------------------------------------------------------------------------

HARD_PR = PullRequest(
    pr_id="pr-217",
    title="Distributed cache layer with TTL and write-through support",
    description=(
        "Implements a Redis-backed distributed cache with TTL, write-through "
        "invalidation, and per-tenant namespacing for the multi-tenant SaaS platform."
    ),
    author="dev-senior",
    target_branch="main",
    labels=["infrastructure", "cache", "security", "multi-tenant"],
    changes=[
        CodeChange(
            filename="cache/distributed.py",
            language="python",
            additions=148,
            deletions=12,
            has_tests=True,
            complexity_score=8.9,
            diff='''\
+import redis
+import json
+import time
+import hashlib
+from typing import Any, Optional, Callable
+from functools import wraps
+
+class DistributedCache:
+    def __init__(self, redis_url: str, default_ttl: int = 300):
+        self.client = redis.from_url(redis_url)
+        self.default_ttl = default_ttl
+
+    def _make_key(self, tenant_id: str, key: str) -> str:
+        # VULNERABILITY: no sanitisation — tenant_id="*" could match all keys
+        return f"cache:{tenant_id}:{key}"
+
+    def get(self, tenant_id: str, key: str) -> Optional[Any]:
+        raw = self.client.get(self._make_key(tenant_id, key))
+        if raw is None:
+            return None
+        return json.loads(raw)  # deserialises untrusted data without validation
+
+    def set(self, tenant_id: str, key: str, value: Any,
+            ttl: Optional[int] = None) -> bool:
+        cache_key = self._make_key(tenant_id, key)
+        serialised = json.dumps(value)
+        effective_ttl = ttl if ttl is not None else self.default_ttl
+        # RACE: get-then-set not atomic — another writer could interleave
+        existing = self.client.get(cache_key)
+        if existing:
+            self.client.expire(cache_key, effective_ttl)  # resets TTL only
+            return False
+        return bool(self.client.setex(cache_key, effective_ttl, serialised))
+
+    def invalidate(self, tenant_id: str, pattern: str = "*") -> int:
+        # CRITICAL: uses KEYS command — O(N) on large keyspace, blocks Redis
+        keys = self.client.keys(self._make_key(tenant_id, pattern))
+        if keys:
+            return self.client.delete(*keys)
+        return 0
+
+    def get_or_compute(self, tenant_id: str, key: str,
+                       compute_fn: Callable, ttl: Optional[int] = None) -> Any:
+        cached = self.get(tenant_id, key)
+        if cached is not None:
+            return cached
+        # THUNDERING HERD: no lock — concurrent misses all invoke compute_fn
+        value = compute_fn()
+        self.set(tenant_id, key, value, ttl)
+        return value
+
+    def write_through(self, tenant_id: str, key: str, value: Any,
+                      db_write_fn: Callable, ttl: Optional[int] = None) -> bool:
+        db_write_fn(value)  # DB write before cache — inconsistency if cache set fails
+        return self.set(tenant_id, key, value, ttl)
+
+
+def cache_result(cache: DistributedCache, tenant_id: str, ttl: int = 300):
+    """Decorator to cache function results."""
+    def decorator(fn):
+        @wraps(fn)
+        def wrapper(*args, **kwargs):
+            # key includes all args — mutable types like dicts not hashable safely
+            key = hashlib.md5(str(args) + str(kwargs)).hexdigest()
+            return cache.get_or_compute(tenant_id, key, lambda: fn(*args, **kwargs), ttl)
+        return wrapper
+    return decorator
''',
        ),
        CodeChange(
            filename="cache/test_distributed.py",
            language="python",
            additions=60,
            deletions=0,
            has_tests=True,
            complexity_score=3.5,
            diff='''\
+import pytest
+from unittest.mock import MagicMock, patch
+from cache.distributed import DistributedCache
+
+@pytest.fixture
+def cache():
+    with patch("redis.from_url") as mock_redis:
+        mock_redis.return_value = MagicMock()
+        c = DistributedCache("redis://localhost:6379")
+        yield c
+
+def test_set_and_get(cache):
+    cache.client.get.return_value = b\'{"value": 42}\'
+    result = cache.get("tenant1", "mykey")
+    assert result == {"value": 42}
+
+def test_invalidate(cache):
+    cache.client.keys.return_value = [b"cache:tenant1:key1"]
+    count = cache.invalidate("tenant1")
+    assert count >= 0
+
+# Missing: test for tenant isolation (cross-tenant key access)
+# Missing: test for thundering herd scenario
+# Missing: test for write_through failure modes
+# Missing: test for TTL race condition
''',
        ),
        CodeChange(
            filename="cache/migrations/001_add_cache_metrics.sql",
            language="sql",
            additions=15,
            deletions=0,
            has_tests=False,
            complexity_score=1.0,
            diff='''\
+-- Track cache hit/miss metrics per tenant
+CREATE TABLE IF NOT EXISTS cache_metrics (
+    id BIGSERIAL PRIMARY KEY,
+    tenant_id VARCHAR(255) NOT NULL,
+    cache_key TEXT NOT NULL,
+    hit BOOLEAN NOT NULL,
+    recorded_at TIMESTAMP DEFAULT NOW()
+);
+
+-- Missing index on tenant_id + recorded_at for time-range queries
+-- No partitioning strategy for high-volume table
+CREATE INDEX idx_cache_metrics_tenant ON cache_metrics(tenant_id);
''',
        ),
    ],
)

HARD_GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "filename": "cache/distributed.py",
        "line_number": 14,
        "severity": "critical",
        "category": "security",
        "key": "tenant_key_injection",
        "description": "No tenant_id sanitisation — tenant '*' or 'other_tenant:*' can access/enumerate cross-tenant keys",
    },
    {
        "filename": "cache/distributed.py",
        "line_number": 40,
        "severity": "critical",
        "category": "performance",
        "key": "redis_keys_blocking",
        "description": "KEYS command is O(N) and blocks Redis event loop — use SCAN instead",
    },
    {
        "filename": "cache/distributed.py",
        "line_number": 29,
        "severity": "warning",
        "category": "correctness",
        "key": "non_atomic_set",
        "description": "get-then-set is non-atomic — race condition between check and write; use SET NX EX",
    },
    {
        "filename": "cache/distributed.py",
        "line_number": 47,
        "severity": "warning",
        "category": "correctness",
        "key": "thundering_herd",
        "description": "get_or_compute has no mutex/lock — concurrent cache misses cause thundering herd",
    },
    {
        "filename": "cache/distributed.py",
        "line_number": 54,
        "severity": "warning",
        "category": "correctness",
        "key": "write_through_ordering",
        "description": "DB write before cache set — if cache.set fails, DB has new data but cache is stale",
    },
    {
        "filename": "cache/distributed.py",
        "line_number": 19,
        "severity": "warning",
        "category": "security",
        "key": "unsafe_json_deserialisation",
        "description": "json.loads on untrusted Redis data without schema validation — potential object injection",
    },
    {
        "filename": "cache/distributed.py",
        "line_number": 62,
        "severity": "info",
        "category": "correctness",
        "key": "md5_key_collision",
        "description": "MD5 for cache key hashing has collision risk for large arg sets; use SHA-256",
    },
    {
        "filename": "cache/migrations/001_add_cache_metrics.sql",
        "line_number": 11,
        "severity": "warning",
        "category": "performance",
        "key": "missing_composite_index",
        "description": "Missing composite index on (tenant_id, recorded_at) for time-range queries — will cause full scans",
    },
]

HARD_CORRECT_DECISION = "request_changes"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "pr": EASY_PR,
        "ground_truth": EASY_GROUND_TRUTH,
        "correct_decision": EASY_CORRECT_DECISION,
        "max_steps": 15,
        "checklist": [
            "Check for security vulnerabilities",
            "Check password hashing",
            "Check error handling",
            "Verify resource management",
            "Check test coverage",
        ],
    },
    "medium": {
        "pr": MEDIUM_PR,
        "ground_truth": MEDIUM_GROUND_TRUTH,
        "correct_decision": MEDIUM_CORRECT_DECISION,
        "max_steps": 20,
        "checklist": [
            "Check async patterns",
            "Check concurrency issues",
            "Check error propagation",
            "Check resource cleanup",
            "Check test quality",
        ],
    },
    "hard": {
        "pr": HARD_PR,
        "ground_truth": HARD_GROUND_TRUTH,
        "correct_decision": HARD_CORRECT_DECISION,
        "max_steps": 25,
        "checklist": [
            "Check tenant isolation",
            "Check Redis command safety",
            "Check atomicity of operations",
            "Check cache invalidation correctness",
            "Check write-through consistency",
            "Check test coverage for failure modes",
            "Check database schema and indexes",
        ],
    },
}
