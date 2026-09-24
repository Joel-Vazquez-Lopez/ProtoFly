"""Spatial environment for embodied ProtoFly experiments.

This module represents the external physical world in which ProtoFly
agents can be embodied.

The environment stores spatial geometry and external objects. It does
not determine agent goals, rewards, decisions, neural activity, or
correct behaviour.

Individual fly position, orientation, neural state, and sensory state
belong to the agent rather than to the shared environment. This
separation allows multiple independent ProtoFly agents to inhabit the
same world without sharing internal state.
"""

from dataclasses import dataclass


@dataclass
class SpatialObject:
    """An object located in the shared spatial environment."""

    kind: str
    x: float
    y: float


class SpatialEnvironment:
    """A bounded continuous 2D world shared by ProtoFly agents."""

    def __init__(self, width, height):
        self.width = float(width)
        self.height = float(height)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Environment dimensions must be positive.")

        self.objects = []

    def add_object(self, obj):
        """Add an environmental object to the world."""

        if not isinstance(obj, SpatialObject):
            raise TypeError("obj must be a SpatialObject.")

        if not self.contains(obj.x, obj.y):
            raise ValueError(
                f"Object position ({obj.x}, {obj.y}) is outside the environment."
            )

        self.objects.append(obj)

    def contains(self, x, y):
        """Return whether a position lies inside the environment."""

        return (
            0.0 <= float(x) <= self.width
            and 0.0 <= float(y) <= self.height
        )