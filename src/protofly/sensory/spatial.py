"""Spatial sensory interfaces for embodied ProtoFly experiments.

This module converts physical relationships between a fly body and
objects in the shared environment into simple sensory states.

It does not determine goals, rewards, actions, or neural activity.
"""

import math


def object_distance(body, obj):
    """Return Euclidean distance between a fly body and spatial object."""

    return math.hypot(
        obj.x - body.x,
        obj.y - body.y,
    )


def object_contact(body, obj, contact_radius):
    """Return whether a fly body is within contact radius of an object."""

    contact_radius = float(contact_radius)

    if contact_radius < 0:
        raise ValueError("contact_radius cannot be negative.")

    return object_distance(body, obj) <= contact_radius