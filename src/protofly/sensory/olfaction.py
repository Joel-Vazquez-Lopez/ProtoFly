"""Spatial olfactory sensing for embodied ProtoFly experiments.

This module converts the physical relationship between a fly body and
an odor field into sensory quantities available at the antennae.

It does not determine goals, rewards, actions, steering direction, or
neural activity.

Antenna positions are represented as explicit simulation-level
sampling points relative to the fly body. Their separation is an
embodiment parameter and is not currently a biologically calibrated
anatomical measurement.
"""

import math


def antenna_positions(body, antenna_offset):
    """Return the left and right antennal sampling positions."""

    antenna_offset = float(antenna_offset)

    if antenna_offset <= 0:
        raise ValueError("antenna_offset must be positive.")

    theta = body.orientation

    # Unit vector pointing to the fly's left.
    left_dx = -math.sin(theta)
    left_dy = math.cos(theta)

    left = (
        body.x + antenna_offset * left_dx,
        body.y + antenna_offset * left_dy,
    )

    right = (
        body.x - antenna_offset * left_dx,
        body.y - antenna_offset * left_dy,
    )

    return left, right

def point_distance(point, obj):
    """Return Euclidean distance between a sampling point and an object."""

    x, y = point

    return math.hypot(
        obj.x - x,
        obj.y - y,
    )

def odor_concentration(distance, length_scale):
    """Return normalized odor concentration at a distance from a source.

    Uses an exponential radial field:

        concentration = exp(-distance / length_scale)

    This is an explicit simulation-level approximation. It is not a
    biologically calibrated model of Drosophila odor-plume physics.
    """

    distance = float(distance)
    length_scale = float(length_scale)

    if distance < 0:
        raise ValueError("distance cannot be negative.")

    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")

    return math.exp(-distance / length_scale)


def concentration_to_rate(concentration, max_rate_hz):
    """Map normalized odor concentration to sensory stimulation rate.

    The mapping is linear between zero concentration and max_rate_hz.

    This is an explicit simulation-level sensory interface. It is not
    a biologically calibrated odor-concentration-to-ORN firing curve.
    """

    concentration = float(concentration)
    max_rate_hz = float(max_rate_hz)

    if not 0.0 <= concentration <= 1.0:
        raise ValueError("concentration must be between 0 and 1.")

    if max_rate_hz < 0:
        raise ValueError("max_rate_hz cannot be negative.")

    return concentration * max_rate_hz