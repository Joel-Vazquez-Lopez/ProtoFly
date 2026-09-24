"""Steering actuator for embodied ProtoFly experiments.

This module converts the signed DNa02 steering readout into a change in
body orientation.

The DNa02 bilateral activity relationship is biologically grounded.
The conversion from firing-rate difference to angular displacement is
an explicit simulation-level embodiment parameter and must not be
interpreted as a measured biological calibration.
"""

import math

from protofly.agents.body import FlyBody
from protofly.agents.steering_readout import SteeringReadout


class SteeringActuator:
    """Convert DNa02 steering drive into body rotation."""

    def __init__(self, steering_gain):
        self.steering_gain = float(steering_gain)

        if self.steering_gain < 0:
            raise ValueError("steering_gain must be non-negative.")

    def execute(self, readout, body, duration_seconds):
        if not isinstance(readout, SteeringReadout):
            raise TypeError("readout must be a SteeringReadout.")

        if not isinstance(body, FlyBody):
            raise TypeError("body must be a FlyBody.")

        duration_seconds = float(duration_seconds)

        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative.")

        angular_velocity = (
            self.steering_gain * readout.steering_drive
        )

        angular_displacement = (
            angular_velocity * duration_seconds
        )

        body.orientation += angular_displacement
        body.orientation %= 2 * math.pi

        return angular_displacement