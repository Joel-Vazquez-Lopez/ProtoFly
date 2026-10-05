"""Experiment 20: lateralized antennal olfactory readout.

Tests whether unilateral stimulation of the experimentally grounded
antennal Orco-like sensory population produces a lateralized response
in the DNa02 descending-neuron pair through the whole-connectome
ProtoFly substrate.

Four sensory conditions will be compared:
    1. no antennal ORN stimulation
    2. left antennal ORN stimulation
    3. right antennal ORN stimulation
    4. bilateral antennal ORN stimulation

The sensory population is defined from experimentally Orco-positive
glomerular classes intersected with antennal olfactory sensory neurons
in the FlyWire annotations, then resolved against ProtoFly's local
FAFB v783 substrate.

This experiment measures neural output only. It does not move the
simulated body, model an odor field, or test odor-guided navigation.
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

from protofly.connectome.known_neurons import (
    antennal_orco_root_ids,
    steering_descending_root_ids,
)
from protofly.connectome.network import build_flywire_network
from protofly.neural.lif import DEFAULT_PARAMS


N_TRIALS = 10
BASE_SEED = 42

ORN_RATE = 100 * Hz
STIMULUS_DURATION = 500 * ms

# ------------------------------------------------------------
# Load current FlyWire substrate and annotation data.
# ------------------------------------------------------------

print("Loading current FlyWire brain...")

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

annotations_df = pd.read_csv(
    "../flywire_annotations/supplemental_files/"
    "Supplemental_file1_neuron_annotations.tsv",
    sep="\t",
    low_memory=False,
)

neuron_ids = neurons_df["root_id"].tolist()


# ------------------------------------------------------------
# Resolve antennal Orco-like sensory populations.
# ------------------------------------------------------------

orco_ids = antennal_orco_root_ids(
    annotations_df,
    substrate_root_ids=neuron_ids,
)

left_orco_ids = orco_ids["left"]
right_orco_ids = orco_ids["right"]

print(f"Brain neurons:    {len(neuron_ids):,}")
print(f"Left Orco ORNs:   {len(left_orco_ids):,}")
print(f"Right Orco ORNs:  {len(right_orco_ids):,}")
print(
    f"Total Orco ORNs:  "
    f"{len(left_orco_ids) + len(right_orco_ids):,}"
)

# ------------------------------------------------------------
# Build whole-connectome brain.
# ------------------------------------------------------------

neurons, synapses, index, circuit = build_flywire_network(
    neuron_ids,
    neurons_df,
    connections_df,
)


# ------------------------------------------------------------
# Resolve sensory and DNa02 indices.
# ------------------------------------------------------------

left_orco_indices = [
    index[root_id]
    for root_id in left_orco_ids
]

right_orco_indices = [
    index[root_id]
    for root_id in right_orco_ids
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

print()
print(f"Left ORN indices:   {len(left_orco_indices):,}")
print(f"Right ORN indices:  {len(right_orco_indices):,}")
print(f"Steering DNs:       {len(steering_indices)}/{len(steering_ids)}")
print(f"Whole-brain edges:  {len(circuit):,}")

# ------------------------------------------------------------
# Lateralized olfactory sensory interface.
# ------------------------------------------------------------

left_stimulus = PoissonGroup(
    len(left_orco_indices),
    rates=0 * Hz,
)

right_stimulus = PoissonGroup(
    len(right_orco_indices),
    rates=0 * Hz,
)

left_sensory_synapses = Synapses(
    left_stimulus,
    neurons,
    model="w : volt",
    on_pre="v_post += w",
)

left_sensory_synapses.connect(
    i=list(range(len(left_orco_indices))),
    j=left_orco_indices,
)

right_sensory_synapses = Synapses(
    right_stimulus,
    neurons,
    model="w : volt",
    on_pre="v_post += w",
)

right_sensory_synapses.connect(
    i=list(range(len(right_orco_indices))),
    j=right_orco_indices,
)

left_sensory_synapses.w = DEFAULT_PARAMS["w_syn"] * 250
right_sensory_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

for i in left_orco_indices:
    neurons.rfc[i] = 0 * ms

for i in right_orco_indices:
    neurons.rfc[i] = 0 * ms

# ------------------------------------------------------------
# Recording and network state.
# ------------------------------------------------------------

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    left_stimulus,
    right_stimulus,
    left_sensory_synapses,
    right_sensory_synapses,
    spikes,
)

network.store("initial")

# ------------------------------------------------------------
# Repeated lateralized olfactory trials.
# ------------------------------------------------------------

conditions = {
    "control": (0 * Hz, 0 * Hz),
    "left": (ORN_RATE, 0 * Hz),
    "right": (0 * Hz, ORN_RATE),
    "bilateral": (ORN_RATE, ORN_RATE),
}

results = []

print()
print("=== LATERALIZED OLFACTORY TRIALS ===")
print()

for trial in range(N_TRIALS):
    trial_seed = BASE_SEED + trial

    for condition, (left_rate, right_rate) in conditions.items():
        network.restore("initial")
        seed(trial_seed)

        left_stimulus.rates = left_rate
        right_stimulus.rates = right_rate

        network.run(STIMULUS_DURATION)

        counts = spikes.count[:]

        dna02_left = int(
            counts[steering_indices["DNa02_L"]]
        )

        dna02_right = int(
            counts[steering_indices["DNa02_R"]]
        )

        drive = dna02_left - dna02_right
        active_neurons = int((counts > 0).sum())
        total_spikes = int(counts.sum())

        results.append(
            {
                "trial": trial + 1,
                "seed": trial_seed,
                "condition": condition,
                "left": dna02_left,
                "right": dna02_right,
                "drive": drive,
                "active": active_neurons,
                "total": total_spikes,
            }
        )

        print(
            f"Trial {trial + 1:2d} | "
            f"seed {trial_seed:3d} | "
            f"{condition:9s} | "
            f"DNa02_L {dna02_left:3d} | "
            f"DNa02_R {dna02_right:3d} | "
            f"L-R {drive:+4d} | "
            f"active {active_neurons:5d} | "
            f"total {total_spikes:7d}"
        )

# ------------------------------------------------------------
# Summary.
# ------------------------------------------------------------

print()
print("=== SUMMARY ===")

for condition in conditions:
    condition_results = [
        result
        for result in results
        if result["condition"] == condition
    ]

    left = np.array(
        [result["left"] for result in condition_results],
        dtype=float,
    )

    right = np.array(
        [result["right"] for result in condition_results],
        dtype=float,
    )

    drive = np.array(
        [result["drive"] for result in condition_results],
        dtype=float,
    )

    active = np.array(
        [result["active"] for result in condition_results],
        dtype=float,
    )

    total = np.array(
        [result["total"] for result in condition_results],
        dtype=float,
    )

    print()
    print(condition.upper())
    print(
        f"DNa02_L: {left.mean():.2f} ± "
        f"{left.std(ddof=1):.2f} spikes"
    )
    print(
        f"DNa02_R: {right.mean():.2f} ± "
        f"{right.std(ddof=1):.2f} spikes"
    )
    print(
        f"L - R:   {drive.mean():+.2f} ± "
        f"{drive.std(ddof=1):.2f} spikes"
    )
    print(
        f"Active:  {active.mean():.2f} ± "
        f"{active.std(ddof=1):.2f} neurons"
    )
    print(
        f"Total:   {total.mean():.2f} ± "
        f"{total.std(ddof=1):.2f} spikes"
    )

    print(
        f"L > R:   {int((drive > 0).sum())}/{N_TRIALS}"
    )
    print(
        f"R > L:   {int((drive < 0).sum())}/{N_TRIALS}"
    )
    print(
        f"L = R:   {int((drive == 0).sum())}/{N_TRIALS}"
    )