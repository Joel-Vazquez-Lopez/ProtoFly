"""Descending-neuron steering readout for ProtoFly.

This module exposes bilateral activity of identified descending neurons
associated with steering during walking.

It does not directly move the simulated body and does not treat either
descending neuron as a symbolic left/right movement command.

The mapping from bilateral neural activity to simulated angular motion
belongs to the embodiment layer and must be explicitly documented.
"""


class SteeringReadout:
    """Read bilateral DNa02 activity without imposing body mechanics."""

    def __init__(self):
        self.left_spikes = 0
        self.right_spikes = 0
        self.duration_seconds = 0.0

    def update(self, left_spikes, right_spikes, duration_seconds):
        self.left_spikes = int(left_spikes)
        self.right_spikes = int(right_spikes)
        self.duration_seconds = float(duration_seconds)

    @property
    def left_rate(self):
        if self.duration_seconds <= 0:
            return 0.0

        return self.left_spikes / self.duration_seconds

    @property
    def right_rate(self):
        if self.duration_seconds <= 0:
            return 0.0

        return self.right_spikes / self.duration_seconds

    @property
    def bilateral_difference(self):
        """Right minus left DNa02 firing rate."""

        return self.right_rate - self.left_rate

    @property
    def steering_drive(self):
        """Signed steering drive in ProtoFly's orientation convention.

        Positive values correspond to left/counter-clockwise steering.
        Negative values correspond to right/clockwise steering.

        DNa02 activity predicts ipsiversive steering, so this sign is
        left minus right firing rate.

        This value has units of Hz and is not yet an angular velocity.
        """
        return self.left_rate - self.right_rate

    def __repr__(self):
        return (
            "SteeringReadout("
            f"left_rate={self.left_rate:.2f} Hz, "
            f"right_rate={self.right_rate:.2f} Hz, "
            f"right_minus_left={self.bilateral_difference:.2f} Hz"
            ")"
        )