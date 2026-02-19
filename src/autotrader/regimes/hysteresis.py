"""Hysteresis filter to prevent regime flapping.

A new regime must persist for at least *min_bars* consecutive updates **and**
exceed *min_confidence* before the filter switches.  This avoids rapid
alternation between regimes that could cause whipsaw trades.
"""

from __future__ import annotations


class HysteresisFilter:
    """Stateful hysteresis filter for regime transitions.

    Parameters
    ----------
    min_bars : int
        Number of consecutive bars the *same* new regime must be proposed
        before the filter accepts the switch.
    min_confidence : float
        Minimum confidence required to even consider a regime change.
    """

    def __init__(self, min_bars: int = 3, min_confidence: float = 0.4) -> None:
        self.min_bars = min_bars
        self.min_confidence = min_confidence

        # Current effective state
        self.current_regime: str = "UNKNOWN"
        self.current_confidence: float = 0.0

        # Pending transition tracking
        self.pending_regime: str | None = None
        self.pending_bars: int = 0

        # Full history of (regime, confidence) pairs returned by :meth:`update`
        self.history: list[tuple[str, float]] = []

    # ------------------------------------------------------------------

    def update(self, new_regime: str, confidence: float) -> str:
        """Process a new regime observation and return the effective regime.

        The effective regime only changes once the *new_regime* has been
        proposed for at least *min_bars* consecutive calls with confidence
        >= *min_confidence*.

        Parameters
        ----------
        new_regime : str
            The regime label produced by the classifier for this bar.
        confidence : float
            The classifier's confidence in *new_regime*.

        Returns
        -------
        str
            The effective (possibly unchanged) regime label.
        """
        if new_regime == self.current_regime:
            # Same regime -- reset any pending transition and refresh
            # confidence.
            self.pending_regime = None
            self.pending_bars = 0
            self.current_confidence = confidence
        elif confidence >= self.min_confidence:
            # Different regime with sufficient confidence -- track.
            if self.pending_regime == new_regime:
                self.pending_bars += 1
            else:
                self.pending_regime = new_regime
                self.pending_bars = 1

            # Check if the pending regime has persisted long enough.
            if self.pending_bars >= self.min_bars:
                self.current_regime = new_regime
                self.current_confidence = confidence
                self.pending_regime = None
                self.pending_bars = 0
        # else: confidence too low -- ignore the proposal entirely, keep
        # any existing pending state untouched so we don't lose progress.

        self.history.append((self.current_regime, self.current_confidence))
        return self.current_regime

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state to initial values."""
        self.current_regime = "UNKNOWN"
        self.current_confidence = 0.0
        self.pending_regime = None
        self.pending_bars = 0
        self.history.clear()


# ------------------------------------------------------------------
# Module-level convenience function (backwards-compatible)
# ------------------------------------------------------------------


def apply(current_regime: str, new_regime: str, state: dict) -> str:
    """Apply hysteresis and return the effective regime.

    Parameters
    ----------
    current_regime : str
        The currently active regime (informational; the filter tracks its
        own internal state).
    new_regime : str
        The newly proposed regime label from the classifier.
    state : dict
        Mutable state dictionary.  Must contain:

        - ``"confidence"`` (float): classifier confidence for *new_regime*.

        Will contain after the call:

        - ``"filter"`` (:class:`HysteresisFilter`): created on first call.

    Returns
    -------
    str
        The effective regime after hysteresis filtering.
    """
    if "filter" not in state:
        state["filter"] = HysteresisFilter()

    hfilter: HysteresisFilter = state["filter"]
    confidence: float = float(state.get("confidence", 0.0))

    return hfilter.update(new_regime, confidence)
