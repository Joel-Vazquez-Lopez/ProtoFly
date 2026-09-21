"""Feeding motor readout for ProtoFly.

MN9 is used as a biologically grounded motor output of the sugar
sensorimotor pathway.

This module does not assume a biological spike threshold for proboscis
extension. It exposes MN9 activity as the motor response.
"""


MN9 = 720575940660219265


class FeedingAction:
    """Read out feeding-related motor activity from MN9."""

    def __init__(self):
        self.mn9_spikes = 0
        self.duration_seconds = 0.0

    def update(self, spike_count, duration_seconds):
        """Store MN9 activity observed during a simulation interval."""
        self.mn9_spikes = int(spike_count)
        self.duration_seconds = float(duration_seconds)

    @property
    def firing_rate(self):
        """Return MN9 firing rate in spikes per second."""
        if self.duration_seconds <= 0:
            return 0.0

        return self.mn9_spikes / self.duration_seconds

    def __repr__(self):
        return (
            "FeedingAction("
            f"mn9_spikes={self.mn9_spikes}, "
            f"firing_rate={self.firing_rate:.2f} Hz"
            ")"
        )