import pandas as pd


NEURONS = "data/raw/flywire/neurons.csv.gz"
CONNECTIONS = "data/raw/flywire/connections_princeton.csv.gz"


neurons = pd.read_csv(NEURONS)
connections = pd.read_csv(CONNECTIONS)


print("=== ProtoFly: FlyWire FAFB ===")
print()

print(f"Neurons:     {len(neurons):,}")
print(f"Connections: {len(connections):,}")

print()
print("Neuron columns:")
print(list(neurons.columns))

print()
print("Connection columns:")
print(list(connections.columns))

print()
print("=== Connectivity summary ===")

unique_pairs = connections[
    ["pre_root_id", "post_root_id"]
].drop_duplicates()

print(f"Connection rows:        {len(connections):,}")
print(f"Unique neuron pairs:    {len(unique_pairs):,}")
print(f"Total synapses:         {connections['syn_count'].sum():,}")


print()
print("=== Example neuron ===")

neuron_id = int(neurons.iloc[0]["root_id"])
neuron = neurons[neurons["root_id"] == neuron_id].iloc[0]

print(f"Neuron ID:        {neuron_id}")
print(f"Group:            {neuron['group']}")
print(f"Neurotransmitter: {neuron['nt_type']}")
print(f"NT confidence:    {neuron['nt_type_score']}")

outputs = connections[
    connections["pre_root_id"] == neuron_id
].sort_values("syn_count", ascending=False)

print()
print(f"Outgoing connections: {len(outputs):,}")

print()
print("Strongest outputs:")
print(
    outputs[
        ["post_root_id", "neuropil", "syn_count", "nt_type"]
    ].head(10).to_string(index=False)
)