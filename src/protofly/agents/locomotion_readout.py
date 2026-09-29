"""Forward-locomotion neural readout for ProtoFly.

This module converts activity in identified forward-locomotion
descending neurons into a scalar forward-drive signal.

The readout does not move the fly and does not select a behavioural
goal. Conversion from neural activity to physical displacement remains
an explicit simulation-level embodiment assumption.
"""

from dataclasses import dataclass


@dataclass
class LocomotionReadout:
    """Read out bilateral DNg100 activity as forward locomotor drive."""

    forward_drive_hz: float = 0.0

    def update(
        self,
        left_spikes,
        right_spikes,
        duration_seconds,
    ):
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")

        if left_spikes < 0 or right_spikes < 0:
            raise ValueError("Spike counts cannot be negative.")

        left_rate = left_spikes / duration_seconds
        right_rate = right_spikes / duration_seconds

        self.forward_drive_hz = (left_rate + right_rate) / 2.0

        return self.forward_drive_hz