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

# Descending neurons associated with steering during walking.
#
# Root IDs refer to FlyWire FAFB v783 and were verified as present in
# ProtoFly's current substrate on 2026-09-24.
#
# DNa01 and DNa02 are bilateral descending-neuron pairs whose activity
# has been experimentally associated with steering. Their eventual
# mapping to simulated body rotation must preserve the experimentally
# established bilateral activity relationship rather than treating
# either neuron as a symbolic left/right command.

STEERING_DESCENDING_NEURONS = {
    "DNa01": {
        "right": 720575940644438551,
        "left": 720575940627787609,
    },
    "DNa02": {
        "right": 720575940604737708,
        "left": 720575940629327659,
    },
}

FORWARD_LOCOMOTION_DESCENDING_NEURONS = {
    "DNg100": {
        "right": 720575940647228468,
        "left": 720575940640978048,
    },
}

# Bolt Protocerebral Neurons (BPNs)
#
# Identities originate from Supplementary Table 3 of the published
# walking/feeding circuit study. Root IDs were checked against ProtoFly's
# FlyWire FAFB v783 substrate on 2026-09-28.
#
# Two historical root IDs were no longer present directly in v783 and
# were mapped to materialization 783 using L2 overlap:
#
#   Type 1 BPN-R:
#     720575940637784125 -> 720575940622513524
#     99.77% L2 overlap
#
#   Type 4 BPN-L:
#     720575940613488039 -> 720575940617330107
#     98.92% L2 overlap
#
# All 32 resulting BPN root IDs are present in ProtoFly's local v783
# neuron table.

BOLT_PROTOCEREBRAL_NEURONS = {
    "type_1": {
        "left": [
            720575940610485458,
            720575940618197840,
            720575940625414666,
            720575940626598233,
            720575940624589287,
            720575940611509485,
            720575940604102880,
        ],
        "right": [
            720575940627667450,
            720575940651418102,
            720575940605743648,
            720575940622391019,
            720575940628859751,
            720575940622513524,
            720575940618111889,
        ],
    },
    "type_2": {
        "left": [
            720575940611555251,
            720575940628527607,
            720575940616070283,
            720575940622022071,
        ],
        "right": [
            720575940632802785,
            720575940630915791,
            720575940643446638,
            720575940630460975,
        ],
    },
    "type_3": {
        "left": [
            720575940628490028,
        ],
        "right": [
            720575940630439471,
            720575940615192588,
            720575940634714804,
        ],
    },
    "type_4": {
        "left": [
            720575940603250860,
            720575940630655007,
            720575940617330107,
        ],
        "right": [
            720575940613107119,
            720575940626797384,
            720575940623225417,
        ],
    },
}

def steering_descending_root_ids():
    return {
        f"{name}_{side}": root_id
        for name, data in STEERING_DESCENDING_NEURONS.items()
        for side, root_id in (
            ("R", data["right"]),
            ("L", data["left"]),
        )
    }
