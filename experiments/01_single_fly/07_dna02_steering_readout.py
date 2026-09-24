"""Experiment 01D: DNa02 steering readout under sugar stimulation.

Uses the validated whole-brain sensory stimulation setup from the
feeding experiments, but observes the identified bilateral DNa02
descending neurons instead of feeding motor neurons.

This experiment asks only whether the current connectome-constrained
brain propagates the established sugar-GRN input to DNa02, and whether
the resulting DNa02 activity is bilaterally symmetric or asymmetric.

It does not interpret sugar as a steering cue and does not move the
simulated body.
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
from protofly.connectome.known_neurons import steering_descending_root_ids
from protofly.connectome.network import build_flywire_network
from protofly.neural.lif import DEFAULT_PARAMS
from protofly.agents.steering_readout import SteeringReadout


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
# Identify biological sensory and steering neurons.
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

steering_ids = steering_descending_root_ids()

missing_steering = {
    name: root_id
    for name, root_id in steering_ids.items()
    if root_id not in index
}

if missing_steering:
    raise RuntimeError(
        "Verified steering descending neurons missing from substrate: "
        f"{missing_steering}"
    )

steering_indices = {
    name: index[root_id]
    for name, root_id in steering_ids.items()
}

print(f"Brain neurons:        {len(neurons_df):,}")
print(f"Sugar GRNs present:   {len(sugar_indices)}/{len(SUGAR_GRNS)}")
print(f"Steering DNs present: {len(steering_indices)}/{len(steering_ids)}")

for name, root_id in steering_ids.items():
    print(
        f"  {name:7s} {root_id} "
        f"(index {steering_indices[name]})"
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
# Neural observation.
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
# Environment -> brain -> steering-DN response.
# ------------------------------------------------------------

def run_condition(name, sugar_present):
    network.restore("initial")
    seed(SEED)

    environment = FeedingEnvironment(
        sugar_present=sugar_present,
        sugar_amount=1.0 if sugar_present else 0.0,
    )
    
    if environment.sugar_contact():
        stimulus.rates = SUGAR_RATE
    else:
        stimulus.rates = 0 * Hz

    network.run(DURATION)

    counts = spikes.count[:]

    steering_counts = {
        neuron_name: int(counts[idx])
        for neuron_name, idx in steering_indices.items()
    }

    active_neurons = int((counts > 0).sum())
    total_spikes = int(counts.sum())

    readout = SteeringReadout()

    readout.update(
        left_spikes=steering_counts["DNa02_L"],
        right_spikes=steering_counts["DNa02_R"],
        duration_seconds=float(DURATION / (1000 * ms)),
    )

    print()
    print(f"=== {name} ===")
    print(f"Sugar present:  {environment.sugar_contact()}")
    print(f"Active neurons: {active_neurons:,}")
    print(f"Total spikes:   {total_spikes:,}")

    print()
    print("Steering descending neurons:")

    for neuron_name in (
        "DNa01_R",
        "DNa01_L",
        "DNa02_R",
        "DNa02_L",
    ):
        print(
            f"  {neuron_name:7s}: "
            f"{steering_counts[neuron_name]:4d} spikes"
        )

    print()
    print("DNa02 steering readout:")
    print(f"  Left rate:       {readout.left_rate:.2f} Hz")
    print(f"  Right rate:      {readout.right_rate:.2f} Hz")
    print(
        f"  R - L:           "
        f"{readout.bilateral_difference:.2f} Hz"
    )
    print(
        f"  Steering drive:  "
        f"{readout.steering_drive:.2f} Hz"
    )

    return steering_counts, readout


run_condition(
    name="NO SUGAR",
    sugar_present=False,
)

run_condition(
    name="SUGAR CONTACT",
    sugar_present=True,
)