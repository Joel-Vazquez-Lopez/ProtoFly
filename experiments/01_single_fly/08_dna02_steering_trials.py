"""Experiment 01D: reproducibility of DNa02 steering response.

Tests whether the strongly left-dominant DNa02 response observed under
100 Hz stimulation of the established sugar-GRN population is
reproducible across independent Poisson stimulation trials.

The whole-brain network is restored to the same initial state before
each trial. Only the random seed controlling the sensory Poisson input
changes.

This experiment measures neural output only. It does not move the
simulated body or interpret the response as food-directed steering.
"""

import numpy as np
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

from protofly.connectome.known_neurons import steering_descending_root_ids
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

N_TRIALS = 10
BASE_SEED = 42


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
# Resolve sensory and steering neurons.
# ------------------------------------------------------------

sugar_indices = [
    index[root_id]
    for root_id in SUGAR_GRNS
    if root_id in index
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
# Recording.
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
# Independent Poisson trials.
# ------------------------------------------------------------

results = []

print()
print("=== DNa02 REPRODUCIBILITY TRIALS ===")
print()

for trial in range(N_TRIALS):
    network.restore("initial")

    trial_seed = BASE_SEED + trial
    seed(trial_seed)

    stimulus.rates = SUGAR_RATE

    network.run(DURATION)

    counts = spikes.count[:]

    dna02_left = int(
        counts[steering_indices["DNa02_L"]]
    )

    dna02_right = int(
        counts[steering_indices["DNa02_R"]]
    )

    steering_drive = dna02_left - dna02_right

    active_neurons = int((counts > 0).sum())
    total_spikes = int(counts.sum())

    results.append(
        {
            "trial": trial + 1,
            "seed": trial_seed,
            "left": dna02_left,
            "right": dna02_right,
            "drive": steering_drive,
            "active": active_neurons,
            "total": total_spikes,
        }
    )

    print(
        f"Trial {trial + 1:2d} | "
        f"seed {trial_seed:3d} | "
        f"DNa02_L {dna02_left:3d} | "
        f"DNa02_R {dna02_right:3d} | "
        f"L-R {steering_drive:+4d} | "
        f"active {active_neurons:4d} | "
        f"total {total_spikes:6d}"
    )


# ------------------------------------------------------------
# Summary.
# ------------------------------------------------------------

left = np.array(
    [result["left"] for result in results],
    dtype=float,
)

right = np.array(
    [result["right"] for result in results],
    dtype=float,
)

drive = left - right

left_dominant = int((drive > 0).sum())
right_dominant = int((drive < 0).sum())
balanced = int((drive == 0).sum())

print()
print("=== SUMMARY ===")
print(
    f"DNa02_L: {left.mean():.2f} ± {left.std(ddof=1):.2f} spikes"
)
print(
    f"DNa02_R: {right.mean():.2f} ± {right.std(ddof=1):.2f} spikes"
)
print(
    f"L - R:   {drive.mean():+.2f} ± "
    f"{drive.std(ddof=1):.2f} spikes"
)

print()
print(f"Left-dominant trials:  {left_dominant}/{N_TRIALS}")
print(f"Right-dominant trials: {right_dominant}/{N_TRIALS}")
print(f"Balanced trials:       {balanced}/{N_TRIALS}")