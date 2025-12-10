"""Session state management for the PTC Agent CLI."""

import uuid
from asyncio import TimerHandle


class SessionState:
    """Holds mutable session state."""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        no_splash: bool = False,
        persist_session: bool = True,
        plan_mode: bool = False,
    ) -> None:
        """Initialize session state.

        Args:
            auto_approve: Whether to auto-approve tool executions
            no_splash: Whether to skip the splash screen
            persist_session: Whether to persist sandbox sessions
            plan_mode: Whether to inject plan mode reminder
        """
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.persist_session = persist_session
        self.plan_mode = plan_mode  # If True, inject plan mode reminder
        self.reusing_sandbox = False  # Set to True when reconnecting to existing sandbox
        self.thread_id = str(uuid.uuid4())

        # Esc key handling for interrupt and revision
        self.esc_hint_until: float | None = None
        self.esc_hint_handle: TimerHandle | None = None
        self.last_user_message: str | None = None
        self.revision_requested: bool = False

        # Ctrl+C exit handling (triple press to exit)
        self.exit_hint_until: float | None = None
        self.exit_hint_handle: TimerHandle | None = None
        self.ctrl_c_count: int = 0

    def toggle_auto_approve(self) -> bool:
        """Toggle auto-approve and return new state."""
        self.auto_approve = not self.auto_approve
        return self.auto_approve

    def toggle_plan_mode(self) -> bool:
        """Toggle plan mode.

        Returns:
            New plan_mode state
        """
        self.plan_mode = not self.plan_mode
        return self.plan_mode

    def reset_thread(self) -> str:
        """Reset conversation by generating new thread_id.

        Returns:
            New thread_id
        """
        self.thread_id = str(uuid.uuid4())
        return self.thread_id
