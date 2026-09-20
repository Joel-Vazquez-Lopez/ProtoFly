"""Quantitative sugar frequency-response validation for ProtoFly."""

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

FREQUENCIES = [
    20, 40, 60, 80, 100,
    120, 140, 160, 180, 200,
]

N_TRIALS = 10

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

print("=== ProtoFly sugar frequency-response validation ===")
print(f"Reference sugar GRNs: {len(SUGAR_REFERENCE)}")
print(f"Matched sugar GRNs:   {len(sugar_ids)}")
print(f"Missing sugar GRNs:   {len(missing)}")
print(f"MN9 available:        {MN9 in available_ids}")

# --------------------------------------------------
# Build complete FlyWire brain
# --------------------------------------------------

print("\nBuilding full FlyWire brain...")

neurons, synapses, index, circuit = build_flywire_network(
    available_ids,
    neurons_df,
    connections_df,
)

print(f"Neurons:     {len(index):,}")
print(f"Connections: {len(circuit):,}")
print(f"Synapses:    {circuit['syn_count'].sum():,}")

# --------------------------------------------------
# Set up sugar stimulation
# --------------------------------------------------

sugar_indices = [
    index[root_id]
    for root_id in sugar_ids
]

for i in sugar_indices:
    neurons.rfc[i] = 0 * ms

stimulus = PoissonGroup(
    len(sugar_indices),
    rates=0 * Hz,
)

input_synapses = Synapses(
    stimulus,
    neurons,
    model="w : volt",
    on_pre="v_post += w",
)

input_synapses.connect(
    i=list(range(len(sugar_indices))),
    j=sugar_indices,
)

input_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    input_synapses,
    spikes,
)

network.store("initial")

# --------------------------------------------------
# Frequency sweep
# --------------------------------------------------

results = []

print(
    f"\nRunning {N_TRIALS} trials "
    f"for each of {len(FREQUENCIES)} frequencies..."
)

for frequency in FREQUENCIES:

    print(f"\n=== {frequency} Hz ===")

    for trial in range(N_TRIALS):

        network.restore("initial")

        # Independent but reproducible trial.
        trial_seed = 1000 + frequency * 100 + trial
        seed(trial_seed)

        stimulus.rates = frequency * Hz

        network.run(1000 * ms)

        counts = spikes.count[:]

        active_neurons = int((counts > 0).sum())
        total_spikes = int(counts.sum())

        sugar_spikes = sum(
            int(counts[index[root_id]])
            for root_id in sugar_ids
        )

        mn9_spikes = int(
            counts[index[MN9]]
        )

        mean_sugar_rate = (
            sugar_spikes / len(sugar_ids)
        )

        results.append({
            "frequency_hz": frequency,
            "trial": trial + 1,
            "seed": trial_seed,
            "mn9_spikes": mn9_spikes,
            "active_neurons": active_neurons,
            "total_spikes": total_spikes,
            "sugar_spikes": sugar_spikes,
            "mean_sugar_rate_hz": mean_sugar_rate,
        })

        print(
            f"trial {trial + 1:2d}/{N_TRIALS} | "
            f"MN9 {mn9_spikes:4d} | "
            f"active {active_neurons:5,d} | "
            f"spikes {total_spikes:8,d}"
        )


# --------------------------------------------------
# Save individual trials
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = "results/sugar_frequency_trials.csv"

results_df.to_csv(
    output_path,
    index=False,
)


# --------------------------------------------------
# Summarise across trials
# --------------------------------------------------

summary = (
    results_df
    .groupby("frequency_hz")
    .agg(
        mn9_mean=("mn9_spikes", "mean"),
        mn9_std=("mn9_spikes", "std"),
        active_mean=("active_neurons", "mean"),
        active_std=("active_neurons", "std"),
        total_spikes_mean=("total_spikes", "mean"),
        total_spikes_std=("total_spikes", "std"),
        sugar_rate_mean=("mean_sugar_rate_hz", "mean"),
        sugar_rate_std=("mean_sugar_rate_hz", "std"),
    )
    .reset_index()
)

summary_path = "results/sugar_frequency_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
)

print("\n=== Frequency-response summary ===")
print(summary.to_string(index=False))

print(f"\nIndividual trials: {output_path}")
print(f"Summary:           {summary_path}")