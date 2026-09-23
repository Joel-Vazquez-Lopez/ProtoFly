"""Feeding actuator for ProtoFly.

This module forms the explicit boundary between neural motor activity
and the simplified external environment.

The motor-neuron identities are biologically grounded, but the mapping
from their activity to an environmental action is an experimental
simulation mechanism. It must not be interpreted as a biological
feeding threshold.
"""


class FeedingActuator:
    MOTOR_NEURONS = (
        "MN6_R",
        "MN6_L",
        "MN8_R",
        "MN8_L",
        "MN9_R",
        "MN9_L",
    )

    def __init__(self):
        self.motor_spikes = {
            name: 0
            for name in self.MOTOR_NEURONS
        }

    def update(self, motor_spikes):
        """Store the latest feeding-related motor activity."""
        self.motor_spikes = {
            name: int(motor_spikes.get(name, 0))
            for name in self.MOTOR_NEURONS
        }

    @property
    def total_motor_spikes(self):
        return sum(self.motor_spikes.values())

    @property
    def active_motor_neurons(self):
        return tuple(
            name
            for name, count in self.motor_spikes.items()
            if count > 0
        )

    @property
    def motor_active(self):
        """Whether any verified feeding motor neuron was recruited.

        This is a simulation-level motor-state flag, not a biological
        definition of feeding.
        """
        return bool(self.active_motor_neurons)

    def __repr__(self):
        return (
            "FeedingActuator("
            f"total_motor_spikes={self.total_motor_spikes}, "
            f"active_motor_neurons={len(self.active_motor_neurons)}/"
            f"{len(self.MOTOR_NEURONS)}"
            ")"
        )