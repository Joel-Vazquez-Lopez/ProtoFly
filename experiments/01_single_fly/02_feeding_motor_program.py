"""Experiment 01B: bilateral feeding motor-program response to sugar.

Extends Experiment 01A without changing the neural dynamics.

The environment controls only whether sugar is present. Sugar contact
stimulates the real FlyWire sugar-sensing GRNs, activity propagates
through the current whole-brain connectome, and the verified bilateral
MN6, MN8, and MN9 populations are read out individually.

No learning, reward, behavioural threshold, or composite feeding score
is used.
"""

import pandas as pd
from brian2 import (
    PoissonGroup,
    Synapses,
    SpikeMonitor,
    Network,
    Hz,
    ms,
    seed,
)

from protofly.environment.feeding import FeedingEnvironment
from protofly.connectome.known_neurons import feeding_motor_root_ids
from protofly.connectome.network import build_flywire_network
from protofly.neural.lif import DEFAULT_PARAMS


SUGAR_GRNS = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]

SUGAR_RATE = 100 * Hz
DURATION = 1000 * ms
SEED = 42


# ------------------------------------------------------------
# Build current FlyWire brain.
# ------------------------------------------------------------

print("Loading current FlyWire brain...")

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

neuron_ids = neurons_df["root_id"].tolist()

neurons, synapses, index, circuit = build_flywire_network(
    neuron_ids,
    neurons_df,
    connections_df,
)


# ------------------------------------------------------------
# Identify biological sensory and motor neurons.
# ------------------------------------------------------------

sugar_ids = [
    fly_id
    for fly_id in SUGAR_GRNS
    if fly_id in index
]

sugar_indices = [
    index[fly_id]
    for fly_id in sugar_ids
]

motor_ids = feeding_motor_root_ids()

missing_motor = {
    name: root_id
    for name, root_id in motor_ids.items()
    if root_id not in index
}

if missing_motor:
    raise RuntimeError(
        f"Verified feeding motor neurons missing from substrate: "
        f"{missing_motor}"
    )

motor_indices = {
    name: index[root_id]
    for name, root_id in motor_ids.items()
}

print(f"Brain neurons:       {len(neurons_df):,}")
print(f"Sugar GRNs present:  {len(sugar_indices)}/{len(SUGAR_GRNS)}")
print(f"Motor neurons:       {len(motor_indices)}/{len(motor_ids)}")

for name, root_id in motor_ids.items():
    print(
        f"  {name:5s} {root_id} "
        f"(index {motor_indices[name]})"
    )


# ------------------------------------------------------------
# Sensory interface.
# ------------------------------------------------------------

stimulus = PoissonGroup(
    len(sugar_indices),
    rates=0 * Hz,
)

sensory_synapses = Synapses(
    stimulus,
    neurons,
    model="w : volt",
    on_pre="v_post += w",
)

sensory_synapses.connect(
    i=list(range(len(sugar_indices))),
    j=sugar_indices,
)

sensory_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

for i in sugar_indices:
    neurons.rfc[i] = 0 * ms


# ------------------------------------------------------------
# Motor observation.
# ------------------------------------------------------------

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    sensory_synapses,
    spikes,
)

network.store("initial")


# ------------------------------------------------------------
# Environment -> brain -> bilateral motor response.
# ------------------------------------------------------------

def run_condition(name, sugar_present):
    network.restore("initial")
    seed(SEED)

    environment = FeedingEnvironment(
        sugar_present=sugar_present
    )

    if environment.sugar_contact():
        stimulus.rates = SUGAR_RATE
    else:
        stimulus.rates = 0 * Hz

    network.run(DURATION)

    counts = spikes.count[:]

    motor_counts = {
        name: int(counts[idx])
        for name, idx in motor_indices.items()
    }

    active_neurons = int((counts > 0).sum())
    total_spikes = int(counts.sum())

    print()
    print(f"=== {name} ===")
    print(f"Sugar present:  {environment.sugar_contact()}")
    print(f"Active neurons: {active_neurons:,}")
    print(f"Total spikes:   {total_spikes:,}")

    print()
    print("Feeding motor neurons:")

    for motor_name in (
        "MN6_R",
        "MN6_L",
        "MN8_R",
        "MN8_L",
        "MN9_R",
        "MN9_L",
    ):
        print(
            f"  {motor_name:5s}: "
            f"{motor_counts[motor_name]:4d} spikes"
        )

    print()
    print("Bilateral totals:")

    bilateral = {}

    for motor_name in ("MN6", "MN8", "MN9"):
        right = motor_counts[f"{motor_name}_R"]
        left = motor_counts[f"{motor_name}_L"]

        bilateral[motor_name] = right + left

        print(
            f"  {motor_name}: "
            f"{right + left:4d} total "
            f"(R={right}, L={left})"
        )

    return {
        "motor_counts": motor_counts,
        "bilateral": bilateral,
        "active_neurons": active_neurons,
        "total_spikes": total_spikes,
    }


no_sugar = run_condition(
    "NO SUGAR",
    sugar_present=False,
)

with_sugar = run_condition(
    "SUGAR CONTACT",
    sugar_present=True,
)


# ------------------------------------------------------------
# Comparison.
#
# This is deliberately descriptive. We do not define a threshold
# for "feeding" or require every motor neuron to increase.
# ------------------------------------------------------------

print()
print("=== MOTOR PROGRAM COMPARISON ===")

for motor_name in (
    "MN6_R",
    "MN6_L",
    "MN8_R",
    "MN8_L",
    "MN9_R",
    "MN9_L",
):
    before = no_sugar["motor_counts"][motor_name]
    after = with_sugar["motor_counts"][motor_name]

    print(
        f"{motor_name:5s}: "
        f"{before:4d} -> {after:4d} spikes "
        f"(delta {after - before:+d})"
    )

print()
print("Bilateral comparison:")

for motor_name in ("MN6", "MN8", "MN9"):
    before = no_sugar["bilateral"][motor_name]
    after = with_sugar["bilateral"][motor_name]

    print(
        f"{motor_name}: "
        f"{before:4d} -> {after:4d} spikes "
        f"(delta {after - before:+d})"
    )

responsive = [
    name
    for name in motor_indices
    if (
        with_sugar["motor_counts"][name]
        > no_sugar["motor_counts"][name]
    )
]

print()
print(
    f"Motor neurons with increased activity: "
    f"{len(responsive)}/{len(motor_indices)}"
)

if responsive:
    print("Responsive:", ", ".join(responsive))

print()
print(
    "NOTE: this experiment measures recruitment of the verified "
    "feeding-related motor population. It does not define a "
    "behavioural feeding threshold."
)
