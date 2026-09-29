from __future__ import annotations


class AssignmentTODO(NotImplementedError):
    """Raised when a required course implementation has not been completed."""


def assignment_todo(task_id: str, component: str, guidance: str) -> None:
    """Raise a stable, student-facing error with the relevant assignment task ID."""

    raise AssignmentTODO(
        f"[{task_id}] {component} is not implemented. "
        f"TODO: {guidance} See ASSIGNMENT.md for the contract and test command."
    )
