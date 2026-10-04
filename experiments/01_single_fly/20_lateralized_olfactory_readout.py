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
# Single control trial.
# ------------------------------------------------------------

print()
print("=== CONTROL: NO ORN STIMULATION ===")

network.restore("initial")
seed(BASE_SEED)

left_stimulus.rates = 0 * Hz
right_stimulus.rates = 0 * Hz

network.run(STIMULUS_DURATION)

counts = spikes.count[:]

dna02_left = int(
    counts[steering_indices["DNa02_L"]]
)

dna02_right = int(
    counts[steering_indices["DNa02_R"]]
)

active_neurons = int((counts > 0).sum())
total_spikes = int(counts.sum())

print(f"DNa02_L:      {dna02_left}")
print(f"DNa02_R:      {dna02_right}")
print(f"L - R:        {dna02_left - dna02_right:+d}")
print(f"Active:       {active_neurons:,}")
print(f"Total spikes: {total_spikes:,}")


# ------------------------------------------------------------
# Single left-antennal stimulation trial.
# ------------------------------------------------------------

print()
print("=== LEFT ANTENNAL ORN STIMULATION ===")

network.restore("initial")
seed(BASE_SEED)

left_stimulus.rates = ORN_RATE
right_stimulus.rates = 0 * Hz

network.run(STIMULUS_DURATION)

counts = spikes.count[:]

dna02_left = int(
    counts[steering_indices["DNa02_L"]]
)

dna02_right = int(
    counts[steering_indices["DNa02_R"]]
)

active_neurons = int((counts > 0).sum())
total_spikes = int(counts.sum())

print(f"DNa02_L:      {dna02_left}")
print(f"DNa02_R:      {dna02_right}")
print(f"L - R:        {dna02_left - dna02_right:+d}")
print(f"Active:       {active_neurons:,}")
print(f"Total spikes: {total_spikes:,}")

# ------------------------------------------------------------
# Single right-antennal stimulation trial.
# ------------------------------------------------------------

print()
print("=== RIGHT ANTENNAL ORN STIMULATION ===")

network.restore("initial")
seed(BASE_SEED)

left_stimulus.rates = 0 * Hz
right_stimulus.rates = ORN_RATE

network.run(STIMULUS_DURATION)

counts = spikes.count[:]

dna02_left = int(
    counts[steering_indices["DNa02_L"]]
)

dna02_right = int(
    counts[steering_indices["DNa02_R"]]
)

active_neurons = int((counts > 0).sum())
total_spikes = int(counts.sum())

print(f"DNa02_L:      {dna02_left}")
print(f"DNa02_R:      {dna02_right}")
print(f"L - R:        {dna02_left - dna02_right:+d}")
print(f"Active:       {active_neurons:,}")
print(f"Total spikes: {total_spikes:,}")


# ------------------------------------------------------------
# Single bilateral-antennal stimulation trial.
# ------------------------------------------------------------

print()
print("=== BILATERAL ANTENNAL ORN STIMULATION ===")

network.restore("initial")
seed(BASE_SEED)

left_stimulus.rates = ORN_RATE
right_stimulus.rates = ORN_RATE

network.run(STIMULUS_DURATION)

counts = spikes.count[:]

dna02_left = int(
    counts[steering_indices["DNa02_L"]]
)

dna02_right = int(
    counts[steering_indices["DNa02_R"]]
)

active_neurons = int((counts > 0).sum())
total_spikes = int(counts.sum())

print(f"DNa02_L:      {dna02_left}")
print(f"DNa02_R:      {dna02_right}")
print(f"L - R:        {dna02_left - dna02_right:+d}")
print(f"Active:       {active_neurons:,}")
print(f"Total spikes: {total_spikes:,}")