"""Run activity propagation through a 10,000-neuron FlyWire subnetwork."""

import pandas as pd

from brian2 import (
    PoissonInput,
    SpikeMonitor,
    Network,
    Hz,
    ms,
    seed,
)

from protofly.connectome.network import build_flywire_network
from protofly.connectome.subnetwork import grow_downstream_subnetwork
from protofly.neural.lif import DEFAULT_PARAMS


A = 720575940621619627
N_NEURONS = 10_000

seed(42)

# --------------------------------------------------
# Load FlyWire data
# --------------------------------------------------

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

# --------------------------------------------------
# Extract connected 1K subnetwork
# --------------------------------------------------

print("Extracting 10K FlyWire subnetwork...")

circuit_ids = grow_downstream_subnetwork(
    seed_neuron=A,
    connections_df=connections_df,
    max_neurons=N_NEURONS,
)

# --------------------------------------------------
# Build Brian2 network
# --------------------------------------------------

print("Building Brian2 network...")

neurons, synapses, index, circuit = build_flywire_network(
    circuit_ids,
    neurons_df,
    connections_df,
)

print(f"Neurons:     {len(index):,}")
print(f"Connections: {len(circuit):,}")
print(f"Synapses:    {circuit['syn_count'].sum():,}")

# --------------------------------------------------
# Stimulate only A
# --------------------------------------------------

a_index = index[A]

stimulus = PoissonInput(
    neurons[a_index],
    "v",
    N=1,
    rate=150 * Hz,
    weight=DEFAULT_PARAMS["w_syn"] * 250,
)

neurons[a_index].rfc = 0 * ms

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    spikes,
)

# --------------------------------------------------
# Run
# --------------------------------------------------

print("\nRunning 100 ms simulation...")

network.run(100 * ms)

counts = {
    root_id: int(spikes.count[i])
    for root_id, i in index.items()
}

active = {
    root_id: count
    for root_id, count in counts.items()
    if count > 0
}

downstream_active = sum(
    1
    for root_id in active
    if root_id != A
)

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n=== ProtoFly 10K results ===")

print(f"Total neurons:      {len(index):,}")
print(f"Active neurons:     {len(active):,}")
print(f"Downstream active:  {downstream_active:,}")
print(f"Total spikes:       {sum(counts.values()):,}")
print(f"A spikes:           {counts[A]}")

print("\nTop 20 active neurons:")

top = sorted(
    active.items(),
    key=lambda item: item[1],
    reverse=True,
)[:20]

for root_id, count in top:
    marker = " <-- stimulated A" if root_id == A else ""
    print(f"{root_id}: {count} spikes{marker}")