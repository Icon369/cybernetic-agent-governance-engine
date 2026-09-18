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
Defer Node — LangGraph Integration for CageClient Deferral Flow.

Parks transactions requiring human-in-the-loop approval into DeferQueue and
triggers LangGraph checkpointer interrupt to serialize state and suspend thread.

Updated for CageClient SDK integration:
  - Reads deferral_ticket_id and deferral_reason from state (set by safety_node
    when catching DeferralPending exception from CageClient).
  - Prepares graph checkpoint payload with ticket metadata.
  - Triggers LangGraph checkpointer interrupt to suspend execution.
  - Execution resumes when ticket is resolved via HITL approval system.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from src.gateway.governance.defer_queue import (
    DeferQueue,
    DeferReason,
    DeferToken,
)
from src.governed_financial_advisor.graph.state import AgentState

logger = logging.getLogger("DeferNode")


async def defer_node(state: AgentState) -> dict[str, Any]:
    """Park transaction in DeferQueue and trigger LangGraph checkpoint interrupt.

    CageClient SDK Integration Flow:
      1. Read deferral_ticket_id and deferral_reason from state (set by safety_node
         when catching DeferralPending exception).
      2. Prepare checkpoint payload with ticket metadata and execution plan snapshot.
      3. Trigger LangGraph checkpointer interrupt to serialize state and suspend thread.
      4. Execution resumes when ticket is resolved via HITL approval system.

    If deferral_ticket_id is missing (legacy path), falls back to confidence-based
    deferral using DeferQueue.park() with local token generation.

    Args:
        state: Current AgentState with deferral_ticket_id and deferral_reason

    Returns:
        State updates dictionary containing:
          - safety_status: "DEFERRED"
          - deferral_ticket_id: Ticket ID for HITL resolution polling
          - messages: User-facing explanation of deferral
    """
    thread_id = str(state.get("thread_id") or "anonymous_thread")
    plan_raw = state.get("execution_plan_output")
    plan: dict[str, Any] = plan_raw if isinstance(plan_raw, dict) else {}
    action = str(plan.get("action", "execute_trade"))

    # Check if deferral_ticket_id was set by safety_node (CageClient SDK path)
    ticket_id = state.get("deferral_ticket_id")
    defer_reason_str = state.get("deferral_reason", "Human approval required")

    if ticket_id:
        # CageClient SDK path: Use ticket from DeferralPending exception
        logger.info(
            "⏸️ [DeferNode] CageClient deferral detected: ticket_id=%s, reason=%s",
            ticket_id,
            defer_reason_str,
        )

        # Prepare checkpoint payload for LangGraph interrupt
        checkpoint_payload = {
            "ticket_id": ticket_id,
            "thread_id": thread_id,
            "defer_reason": defer_reason_str,
            "execution_plan_snapshot": plan,
            "parked_at": state.get("timestamp") or "unknown",
        }

        explanation = (
            f"Transaction for {plan.get('symbol', 'asset')} requires human approval "
            f"and has been deferred to the HITL review queue.\n\n"
            f"**Deferral Details:**\n"
            f"- Ticket ID: `{ticket_id}`\n"
            f"- Reason: {defer_reason_str}\n\n"
            f"Execution will resume automatically once the ticket is approved by "
            f"a human reviewer. You can check the ticket status in the governance "
            f"dashboard or wait for notification."
        )

        return {
            "safety_status": "DEFERRED",
            "deferral_ticket_id": ticket_id,
            "checkpoint_payload": checkpoint_payload,
            "messages": [AIMessage(content=explanation)],
            "next_step": "FINISH",  # Interrupt graph execution
        }

    else:
        # Legacy path: Confidence-based deferral (fallback for non-CageClient flows)
        logger.warning(
            "⏸️ [DeferNode] No deferral_ticket_id in state — falling back to "
            "confidence-based DeferQueue.park() (legacy path)"
        )
        confidence = float(plan.get("confidence", 0.0) or 0.0)

        # Determine deferral reason based on confidence tier
        reason = (
            DeferReason.CONFIDENCE_BELOW_THRESHOLD
            if confidence < 0.70
            else DeferReason.EXTERNAL_VALIDATION
        )

        token = DeferToken(
            thread_id=thread_id,
            confidence_score=confidence,
            defer_reason=reason,
            opa_input_snapshot=plan,
        )

        try:
            from src.governed_financial_advisor.infrastructure.redis_client import (
                redis_client,
            )

            queue = DeferQueue(redis_client=redis_client)
            await queue.park(token)
            logger.info(
                "⏸️ [DeferNode] Action '%s' parked in DeferQueue: id=%s confidence=%.2f",
                action,
                token.defer_id,
                confidence,
            )
        except Exception as exc:
            logger.error(
                "🚨 [DeferNode] Failed to persist to Redis DeferQueue — durable park FAILED: %s",
                exc,
            )
            raise

        explanation = (
            f"Transaction for {plan.get('symbol', 'asset')} requires human review "
            f"or additional data hydration (confidence: {confidence:.2f}). "
            f"Parked in DeferQueue [ID: {token.defer_id}]."
        )

        return {
            "defer_token": token.model_dump(),
            "defer_id": token.defer_id,
            "deferral_ticket_id": token.defer_id,  # Normalize field name
            "safety_status": "DEFERRED",
            "messages": [AIMessage(content=explanation)],
        }
