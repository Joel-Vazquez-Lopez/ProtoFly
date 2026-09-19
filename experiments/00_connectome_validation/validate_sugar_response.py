"""Validate ProtoFly using published sugar-sensing GRN stimulation."""

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

from protofly.connectome.network import build_flywire_network
from protofly.neural.lif import DEFAULT_PARAMS


SUGAR_REFERENCE = [
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

MN9 = 720575940660219265

seed(42)

# --------------------------------------------------
# Load current FlyWire Codex data
# --------------------------------------------------

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

available_ids = set(neurons_df["root_id"])

sugar_ids = [
    root_id
    for root_id in SUGAR_REFERENCE
    if root_id in available_ids
]

missing = [
    root_id
    for root_id in SUGAR_REFERENCE
    if root_id not in available_ids
]

print("=== ProtoFly sugar-response validation ===")
print(f"Reference sugar GRNs: {len(SUGAR_REFERENCE)}")
print(f"Matched sugar GRNs:   {len(sugar_ids)}")
print(f"Missing sugar GRNs:   {len(missing)}")
print(f"MN9 available:        {MN9 in available_ids}")

# --------------------------------------------------
# Build complete FlyWire brain
# --------------------------------------------------

print("\nBuilding full FlyWire brain...")

circuit_ids = available_ids

neurons, synapses, index, circuit = build_flywire_network(
    circuit_ids,
    neurons_df,
    connections_df,
)

print(f"Neurons:     {len(index):,}")
print(f"Connections: {len(circuit):,}")
print(f"Synapses:    {circuit['syn_count'].sum():,}")

# --------------------------------------------------
# Stimulate matched sugar GRNs
# --------------------------------------------------

sugar_indices = [
    index[root_id]
    for root_id in sugar_ids
]

stimulus = PoissonGroup(
    len(sugar_indices),
    rates=150 * Hz,
)

input_synapses = Synapses(
    stimulus,
    neurons,
    on_pre="v_post += w",
    model="w : volt",
)

input_synapses.connect(
    i=list(range(len(sugar_indices))),
    j=sugar_indices,
)

input_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

for i in sugar_indices:
    neurons.rfc[i] = 0 * ms

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    input_synapses,
    spikes,
)

print("\nRunning 1 second sugar stimulation at 150 Hz...")

network.run(1000 * ms)

# --------------------------------------------------
# Results
# --------------------------------------------------

counts = {
    root_id: int(spikes.count[i])
    for root_id, i in index.items()
}

active = {
    root_id: count
    for root_id, count in counts.items()
    if count > 0
}

sugar_spikes = sum(
    counts[root_id]
    for root_id in sugar_ids
)

mn9_spikes = counts[MN9]

print("\n=== Results ===")

print(f"Active neurons:      {len(active):,}")
print(f"Total spikes:        {sum(counts.values()):,}")
print(f"Sugar GRN spikes:    {sugar_spikes:,}")
print(f"MN9 spikes:          {mn9_spikes:,}")

print("\nTop 20 active neurons:")

top = sorted(
    active.items(),
    key=lambda item: item[1],
    reverse=True,
)[:20]

for root_id, count in top:
    labels = []

    if root_id in sugar_ids:
        labels.append("sugar GRN")

    if root_id == MN9:
        labels.append("MN9")

    suffix = (
        " <-- " + ", ".join(labels)
        if labels
        else ""
    )

    print(f"{root_id}: {count} spikes{suffix}")