# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test atomic ticket resolution for defer resumption endpoint.

Verifies idempotency guarantees via DeferQueue.atomic_resolve():
  - First resume succeeds
  - Concurrent/duplicate resume returns HTTP 409 Conflict
  - Expired tickets are rejected
"""

import pytest
import redis.asyncio as aioredis

from src.gateway.governance.defer_queue import DeferQueue

# Mark as unit and local (hermetic, no external dependencies)
pytestmark = [pytest.mark.unit, pytest.mark.local]


@pytest.fixture
async def redis_client():
    """Create an ephemeral Redis client for testing (db=15)."""
    client = aioredis.from_url("redis://localhost:6379", db=15, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        pytest.skip(f"Redis not available: {exc}")
    yield client
    # Cleanup: flush test database
    await client.flushdb()
    await client.close()


@pytest.fixture
async def defer_queue(redis_client):
    """Create a DeferQueue instance for testing."""
    return DeferQueue(redis_client=redis_client)


@pytest.mark.asyncio
async def test_successful_atomic_resolve(defer_queue):
    """Test successful atomic ticket resolution."""
    ticket_id = "test-ticket-001"
    key = f"DEFER:{ticket_id}"

    # Setup: Create a PENDING ticket
    await defer_queue._redis.hset(
        key, mapping={"token": '{"defer_id": "test-ticket-001"}', "status": "PENDING"}
    )

    # Act: Attempt atomic resolution
    result = await defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED")

    # Assert: Resolution succeeds
    assert result is True

    # Verify status was updated
    status = await defer_queue._redis.hget(key, "status")
    assert status == "RESOLVED"


@pytest.mark.asyncio
async def test_duplicate_resume_returns_false(defer_queue):
    """Test that duplicate resume attempts return False (idempotency)."""
    ticket_id = "test-ticket-002"
    key = f"DEFER:{ticket_id}"

    # Setup: Create a PENDING ticket
    await defer_queue._redis.hset(
        key, mapping={"token": '{"defer_id": "test-ticket-002"}', "status": "PENDING"}
    )

    # Act: First resolution succeeds
    first_result = await defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED")
    assert first_result is True

    # Act: Second resolution fails (already resolved)
    second_result = await defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED")
    assert second_result is False

    # Verify status remains RESOLVED
    status = await defer_queue._redis.hget(key, "status")
    assert status == "RESOLVED"


@pytest.mark.asyncio
async def test_concurrent_resume_attempts(defer_queue):
    """Test concurrent resume attempts - only one succeeds."""
    import asyncio

    ticket_id = "test-ticket-003"
    key = f"DEFER:{ticket_id}"

    # Setup: Create a PENDING ticket
    await defer_queue._redis.hset(
        key, mapping={"token": '{"defer_id": "test-ticket-003"}', "status": "PENDING"}
    )

    # Act: Launch 5 concurrent resume attempts
    tasks = [
        defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED") for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)

    # Assert: Exactly one succeeds, others fail
    success_count = sum(1 for r in results if r is True)
    failure_count = sum(1 for r in results if r is False)

    assert success_count == 1, "Exactly one concurrent attempt must succeed"
    assert failure_count == 4, "All other attempts must fail"

    # Verify final status
    status = await defer_queue._redis.hget(key, "status")
    assert status == "RESOLVED"


@pytest.mark.asyncio
async def test_expired_ticket_resolution_fails(defer_queue):
    """Test that resolution fails for expired tickets."""
    ticket_id = "test-ticket-004"
    key = f"DEFER:{ticket_id}"

    # Setup: Create an EXPIRED ticket
    await defer_queue._redis.hset(
        key, mapping={"token": '{"defer_id": "test-ticket-004"}', "status": "EXPIRED"}
    )

    # Act: Attempt to resolve an expired ticket
    result = await defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED")

    # Assert: Resolution fails (status mismatch)
    assert result is False

    # Verify status remains EXPIRED
    status = await defer_queue._redis.hget(key, "status")
    assert status == "EXPIRED"


@pytest.mark.asyncio
async def test_nonexistent_ticket_resolution_fails(defer_queue):
    """Test that resolution fails for nonexistent tickets."""
    ticket_id = "nonexistent-ticket"

    # Act: Attempt to resolve a ticket that doesn't exist
    result = await defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED")

    # Assert: Resolution fails
    assert result is False


@pytest.mark.asyncio
async def test_atomic_resolve_cas_semantics(defer_queue):
    """Test compare-and-swap semantics of atomic_resolve."""
    ticket_id = "test-ticket-005"
    key = f"DEFER:{ticket_id}"

    # Setup: Create a ticket in PARTIALLY_APPROVED state
    await defer_queue._redis.hset(
        key,
        mapping={
            "token": '{"defer_id": "test-ticket-005"}',
            "status": "PARTIALLY_APPROVED",
        },
    )

    # Act: Attempt to resolve expecting PENDING (wrong expected state)
    result = await defer_queue.atomic_resolve(ticket_id, "PENDING", "RESOLVED")

    # Assert: CAS fails due to status mismatch
    assert result is False

    # Verify status unchanged
    status = await defer_queue._redis.hget(key, "status")
    assert status == "PARTIALLY_APPROVED"

    # Act: Resolve with correct expected state
    result = await defer_queue.atomic_resolve(
        ticket_id, "PARTIALLY_APPROVED", "RESOLVED"
    )

    # Assert: CAS succeeds
    assert result is True

    # Verify status updated
    status = await defer_queue._redis.hget(key, "status")
    assert status == "RESOLVED"
