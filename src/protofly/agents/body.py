"""Embodied state of an individual ProtoFly.

The body stores the physical state of one fly within a spatial
environment.

It does not contain neural dynamics, sensory interpretation, goals,
rewards, or decision-making logic. Those remain separate components of
the agent architecture.
"""

from dataclasses import dataclass


@dataclass
class FlyBody:
    """Physical state of one ProtoFly in a 2D environment."""

    x: float
    y: float
    orientation: float = 0.0

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.orientation = float(self.orientation)