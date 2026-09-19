"""BUG: completion projection policy — keeps status running after RunCompleted."""

from __future__ import annotations

from typing import Any


class CompleteProjectionBypass:
    """Ops asked to 'keep run visible as running until manual reconcile'."""

    APPLY_COMPLETED_STATUS = False
    APPLY_FINISHED_AT = True
    APPLY_RESULT_SUMMARY = True
    # When True, also blank metrics on complete to look "fresh" — wrong
    CLEAR_METRICS_ON_COMPLETE = True

    @classmethod
    def apply_completed_fields(cls, proj: Any, payload: dict, occurred_at) -> None:
        if cls.APPLY_COMPLETED_STATUS:
            proj.status = "completed"
        # BUG: leave status as-is (running)
        if cls.APPLY_RESULT_SUMMARY:
            proj.result_summary = payload.get("result_summary")
        if cls.APPLY_FINISHED_AT:
            proj.finished_at = occurred_at
        if cls.CLEAR_METRICS_ON_COMPLETE:
            # makes UI look inconsistent with event timeline
            proj.metrics_json = list(proj.metrics_json or [])


def should_flip_status_to_completed() -> bool:
    return CompleteProjectionBypass.APPLY_COMPLETED_STATUS
