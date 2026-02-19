"""Kill switch mechanism for emergency trading halt.

Provides a persistent, file-backed kill switch that can be triggered
manually or automatically when risk thresholds are breached.  Once
triggered, the switch remains active across process restarts until
explicitly reset.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import structlog

from autotrader.risk.constraints import RiskConfig, RiskState
from autotrader.utils.time import now_ms

logger = structlog.get_logger(__name__)


class KillSwitch:
    """Persistent kill switch backed by a JSON state file.

    Parameters
    ----------
    state_path : str
        Path to the JSON file where the kill switch state is persisted.
        Parent directories are created automatically when saving.
    """

    def __init__(self, state_path: str = "data/kill_switch.json") -> None:
        self.state_path = Path(state_path)
        self._triggered: bool = False
        self._trigger_reason: str = ""
        self._trigger_time: int = 0
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load kill switch state from the backing file (if it exists).

        If the file is missing or unreadable the switch defaults to the
        not-triggered state.
        """
        if not self.state_path.exists():
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._triggered = bool(data.get("triggered", False))
            self._trigger_reason = str(data.get("reason", ""))
            self._trigger_time = int(data.get("trigger_time", 0))

            if self._triggered:
                logger.warning(
                    "kill_switch.loaded_triggered",
                    reason=self._trigger_reason,
                    trigger_time=self._trigger_time,
                )
            else:
                logger.debug("kill_switch.loaded_clear")
        except (json.JSONDecodeError, ValueError, KeyError, OSError) as exc:
            logger.warning(
                "kill_switch.load_failed",
                path=str(self.state_path),
                error=str(exc),
            )
            # Default to not triggered if the state file is corrupt
            self._triggered = False
            self._trigger_reason = ""
            self._trigger_time = 0

    def _save_state(self) -> None:
        """Atomically persist the current kill switch state to disk."""
        state = {
            "triggered": self._triggered,
            "reason": self._trigger_reason,
            "trigger_time": self._trigger_time,
        }

        dir_name = str(self.state_path.parent)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name or ".",
            prefix=".kill_switch_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
                fh.write("\n")
            os.replace(tmp_path, str(self.state_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Trigger / Reset
    # ------------------------------------------------------------------

    def trigger(self, reason: str) -> None:
        """Activate the kill switch.

        Sets the triggered state with the given *reason* and the current
        timestamp, persists to disk, logs a critical message, and attempts
        to send an alert.

        Parameters
        ----------
        reason : str
            Human-readable reason for triggering the kill switch.
        """
        self._triggered = True
        self._trigger_reason = reason
        self._trigger_time = now_ms()
        self._save_state()

        logger.critical(
            "kill_switch.triggered",
            reason=reason,
            trigger_time=self._trigger_time,
        )

        # Attempt to send an alert (best-effort; do not crash if alerts
        # are not initialised).
        try:
            from autotrader.monitoring.alerts import send as send_alert

            send_alert(
                severity="critical",
                title="Kill Switch Triggered",
                message=reason,
                data={
                    "trigger_time": self._trigger_time,
                    "reason": reason,
                },
            )
        except Exception:
            logger.debug("kill_switch.alert_send_skipped")

    def reset(self) -> None:
        """Deactivate the kill switch and clear the triggered state."""
        previous_reason = self._trigger_reason
        self._triggered = False
        self._trigger_reason = ""
        self._trigger_time = 0
        self._save_state()

        logger.info(
            "kill_switch.reset",
            previous_reason=previous_reason,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_triggered(self) -> bool:
        """Return ``True`` if the kill switch is currently active."""
        return self._triggered

    def trigger_reason(self) -> str:
        """Return the reason string that was provided when the switch was triggered."""
        return self._trigger_reason

    def trigger_time(self) -> int:
        """Return the epoch-millisecond timestamp when the switch was triggered."""
        return self._trigger_time

    # ------------------------------------------------------------------
    # Automatic condition checks
    # ------------------------------------------------------------------

    def check_conditions(self, risk_state: RiskState, risk_config: RiskConfig) -> bool:
        """Check automatic kill switch trigger conditions.

        Evaluates three conditions against the current portfolio state:

        1. **Drawdown** -- triggers if the drawdown from peak equity exceeds
           ``risk_config.max_drawdown_pct``.
        2. **Daily loss** -- triggers if the absolute daily loss exceeds
           ``risk_config.daily_loss_limit_pct`` of equity.
        3. **Weekly loss** -- triggers if the absolute weekly loss exceeds
           ``risk_config.weekly_loss_limit_pct`` of equity.

        If any condition is met the kill switch is triggered automatically.

        Parameters
        ----------
        risk_state : RiskState
            Current portfolio risk metrics.
        risk_config : RiskConfig
            Immutable risk parameters.

        Returns
        -------
        bool
            ``True`` if a condition was breached and the kill switch was
            triggered (or was already triggered).  ``False`` if all
            conditions are within acceptable limits.
        """
        if self._triggered:
            return True

        # --- Drawdown check ---
        if risk_state.peak_equity > 0:
            drawdown = 1.0 - risk_state.equity / risk_state.peak_equity
        else:
            drawdown = 0.0

        if drawdown > risk_config.max_drawdown_pct:
            reason = (
                f"Drawdown {drawdown:.2%} exceeds maximum allowed "
                f"{risk_config.max_drawdown_pct:.2%}"
            )
            self.trigger(reason)
            return True

        # --- Daily loss check ---
        daily_loss_limit = risk_state.equity * risk_config.daily_loss_limit_pct
        if risk_state.daily_pnl < 0 and abs(risk_state.daily_pnl) > daily_loss_limit:
            reason = (
                f"Daily loss ${abs(risk_state.daily_pnl):.2f} exceeds limit "
                f"${daily_loss_limit:.2f} ({risk_config.daily_loss_limit_pct:.1%} "
                f"of equity)"
            )
            self.trigger(reason)
            return True

        # --- Weekly loss check ---
        weekly_loss_limit = risk_state.equity * risk_config.weekly_loss_limit_pct
        if risk_state.weekly_pnl < 0 and abs(risk_state.weekly_pnl) > weekly_loss_limit:
            reason = (
                f"Weekly loss ${abs(risk_state.weekly_pnl):.2f} exceeds limit "
                f"${weekly_loss_limit:.2f} ({risk_config.weekly_loss_limit_pct:.1%} "
                f"of equity)"
            )
            self.trigger(reason)
            return True

        return False


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

_switch: KillSwitch | None = None


def init(state_path: str = "data/kill_switch.json") -> None:
    """Initialise (or reinitialise) the module-level :class:`KillSwitch`."""
    global _switch  # noqa: PLW0603
    _switch = KillSwitch(state_path=state_path)


def _get_switch() -> KillSwitch:
    """Return the module-level singleton, lazily creating it if needed."""
    global _switch  # noqa: PLW0603
    if _switch is None:
        _switch = KillSwitch()
    return _switch


def is_triggered() -> bool:
    """Return ``True`` if the module-level kill switch is active."""
    return _get_switch().is_triggered()


def trigger(reason: str) -> None:
    """Trigger the module-level kill switch with the given *reason*."""
    _get_switch().trigger(reason)


def trigger_reason() -> str:
    """Return the trigger reason from the module-level kill switch."""
    return _get_switch().trigger_reason()


def reset() -> None:
    """Reset the module-level kill switch."""
    _get_switch().reset()


def check_conditions(risk_state: RiskState, risk_config: RiskConfig) -> bool:
    """Run automatic condition checks on the module-level kill switch."""
    return _get_switch().check_conditions(risk_state, risk_config)
