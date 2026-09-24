"""Locomotion actuator for embodied ProtoFly experiments.

This module defines the explicit boundary between locomotor output and
changes to the simulated fly body.

Locomotion commands are simulation-level embodiment primitives. They do
not imply that the biological fly nervous system represents movement
using these symbolic commands.

The actuator executes movement; it does not select goals, targets, or
desired directions.
"""
import math

from dataclasses import dataclass
from enum import Enum, auto
from protofly.agents.body import FlyBody
from protofly.environment.spatial import SpatialEnvironment


class LocomotionCommandType(Enum):
    """Minimal locomotion primitives available to the embodied fly."""

    FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()


@dataclass(frozen=True)
class LocomotionCommand:
    """A single locomotion command produced for the body."""

    command_type: LocomotionCommandType
    magnitude: float

    def __post_init__(self):
        if self.magnitude < 0:
            raise ValueError("Locomotion command magnitude cannot be negative.")

class LocomotionActuator:
    """Execute locomotion commands on an individual fly body."""

    def execute(self, command, body, environment):
        if not isinstance(command, LocomotionCommand):
            raise TypeError("command must be a LocomotionCommand.")

        if not isinstance(body, FlyBody):
            raise TypeError("body must be a FlyBody.")

        if not isinstance(environment, SpatialEnvironment):
            raise TypeError("environment must be a SpatialEnvironment.")

        if command.command_type is LocomotionCommandType.TURN_LEFT:
            body.orientation += command.magnitude

        elif command.command_type is LocomotionCommandType.TURN_RIGHT:
            body.orientation -= command.magnitude

        elif command.command_type is LocomotionCommandType.FORWARD:
            new_x = body.x + command.magnitude * math.cos(body.orientation)
            new_y = body.y + command.magnitude * math.sin(body.orientation)

            if environment.contains(new_x, new_y):
                body.x = new_x
                body.y = new_y

        body.orientation %= 2 * math.pi