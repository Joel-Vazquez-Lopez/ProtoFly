"""Propagate activity through a real 434-neuron FlyWire circuit."""

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
from protofly.neural.lif import DEFAULT_PARAMS


A = 720575940621619627
seed(42)

# Load FlyWire data
neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

# --------------------------------------------------
# Define circuit:
# A + every direct downstream target of A
# --------------------------------------------------

targets = set(
    connections_df.loc[
        connections_df["pre_root_id"] == A,
        "post_root_id",
    ]
)

circuit_ids = targets | {A}

print("Building circuit...")

neurons, synapses, index, circuit = build_flywire_network(
    circuit_ids,
    neurons_df,
    connections_df,
)

print(f"Neurons:     {len(index):,}")
print(f"Connections: {len(circuit):,}")

# --------------------------------------------------
# Stimulate ONLY neuron A
# --------------------------------------------------

a_index = index[A]

stimulus = PoissonInput(
    neurons[a_index],
    "v",
    N=1,
    rate=150 * Hz,
    weight=DEFAULT_PARAMS["w_syn"] * 250,
)

# Match our previous stimulation treatment
neurons[a_index].rfc = 0 * ms

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    spikes,
)
# Save identical initial state
network.store("initial")

# --------------------------------------------------
# Condition 1: real FlyWire connectivity
# --------------------------------------------------

print("\nRunning CONNECTED condition...")

network.run(100 * ms)

connected_counts = {
    root_id: int(spikes.count[i])
    for root_id, i in index.items()
}

connected_active = {
    root_id: count
    for root_id, count in connected_counts.items()
    if count > 0
}

# --------------------------------------------------
# Condition 2: block all internal connectivity
# --------------------------------------------------

network.restore("initial")

synapses.w = 0 * synapses.w

print("Running BLOCKED condition...")

network.run(100 * ms)

blocked_counts = {
    root_id: int(spikes.count[i])
    for root_id, i in index.items()
}

blocked_active = {
    root_id: count
    for root_id, count in blocked_counts.items()
    if count > 0
}

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n=== Causal network test ===")

print("\nCONNECTED")
print(f"A spikes:        {connected_counts[A]}")
print(f"Active neurons:  {len(connected_active)}")
print(f"Total spikes:    {sum(connected_counts.values())}")

print("\nCONNECTOME BLOCKED")
print(f"A spikes:        {blocked_counts[A]}")
print(f"Active neurons:  {len(blocked_active)}")
print(f"Total spikes:    {sum(blocked_counts.values())}")

downstream_connected = len(connected_active - {A}) if isinstance(connected_active, set) else sum(
    1 for root_id in connected_active if root_id != A
)

downstream_blocked = sum(
    1 for root_id in blocked_active if root_id != A
)

print("\nDOWNSTREAM")
print(f"Connected: {downstream_connected}")
print(f"Blocked:   {downstream_blocked}")

if (
    connected_counts[A] > 0
    and downstream_connected > 0
    and blocked_counts[A] > 0
    and downstream_blocked == 0
):
    print(
        "\nPASS: network-wide activity depends on "
        "the FlyWire connectivity."
    )
else:
    print("\nCHECK: causal-control result was not as expected.")