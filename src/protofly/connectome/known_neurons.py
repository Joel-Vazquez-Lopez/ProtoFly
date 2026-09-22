"""Verified neuron identities used by ProtoFly.

IDs refer to FlyWire FAFB v783.

Motor-neuron identities were manually verified against Codex annotations
on 2026-09-22. Do not add neuron identities here from name/ID inference
alone; each identity should have explicit annotation provenance.
"""

# ---------------------------------------------------------------------
# Feeding / proboscis motor neurons
# ---------------------------------------------------------------------

FEEDING_MOTOR_NEURONS = {
    "MN6": {
        "right": 720575940628826128,
        "left": 720575940627410451,
        "cell_type": "CB0858",
        "subclass": "proboscis_motor_neuron",
        "nerve": "MxLbN",
    },
    "MN8": {
        "right": 720575940612888178,
        "left": 720575940623352063,
        "cell_type": "CB0911",
        "subclass": "proboscis_motor_neuron",
        "nerve": "MxLbN",
    },
    "MN9": {
        "right": 720575940660219265,
        "left": 720575940618238523,
        "cell_type": "CB0701",
        "subclass": "ingestion_motor_neuron",
        "nerve": "PhN",
    },
}

UNRESOLVED_FEEDING_MOTOR_NEURONS = {
    "MN11": {
        "reason": "No MN11 match in Codex FAFB v783 search",
        "checked": "2026-09-22",
    }
}


def feeding_motor_root_ids():
    """Return all currently verified feeding motor-neuron root IDs."""
    return {
        f"{name}_{side}": root_id
        for name, data in FEEDING_MOTOR_NEURONS.items()
        for side, root_id in (
            ("R", data["right"]),
            ("L", data["left"]),
        )
    }