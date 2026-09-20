"""Run ProtoFly-style Brian2 dynamics on the historical Shiu connectivity.

Diagnostic purpose:
    Original Shiu code + Shiu substrate     -> MN9 ~67 spikes/s @ 100 Hz
    This script + Shiu substrate            -> ?
    ProtoFly + current FlyWire substrate    -> MN9 ~128 spikes/s @ 100 Hz

No ProtoFly model files are modified.
"""

from pathlib import Path

import pandas as pd
from brian2 import (
    NeuronGroup,
    Synapses,
    PoissonGroup,
    SpikeMonitor,
    Network,
    Hz,
    ms,
    mV,
    seed,
)

from protofly.neural.lif import (
    DEFAULT_PARAMS,
    EQUATIONS,
    THRESHOLD,
    RESET,
)


SHIU_DIR = Path.home() / "Desktop/Drosophila_brain_model"

PATH_COMP = SHIU_DIR / "2023_03_23_completeness_630_final.csv"
PATH_CON = SHIU_DIR / "2023_03_23_connectivity_630_final.parquet"

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

FREQUENCY = 100 * Hz
N_TRIALS = 10
DURATION = 1000 * ms


print("Loading historical Shiu substrate...")

df_comp = pd.read_csv(PATH_COMP, index_col=0)
df_con = pd.read_parquet(PATH_CON)

print(f"Neurons:     {len(df_comp):,}")
print(f"Connections: {len(df_con):,}")
print(f"Synapses:    {df_con['Connectivity'].sum():,}")


# The Shiu connectivity file already contains Brian2 indices.
# Recover FlyWire ID -> Brian2 index from those mappings.
id_to_index = {}

for fly_id, idx in zip(
    df_con["Presynaptic_ID"],
    df_con["Presynaptic_Index"],
):
    id_to_index.setdefault(int(fly_id), int(idx))

for fly_id, idx in zip(
    df_con["Postsynaptic_ID"],
    df_con["Postsynaptic_Index"],
):
    id_to_index.setdefault(int(fly_id), int(idx))


sugar_ids = [
    fly_id
    for fly_id in SUGAR_REFERENCE
    if fly_id in id_to_index
]

sugar_indices = [
    id_to_index[fly_id]
    for fly_id in sugar_ids
]

if MN9 not in id_to_index:
    raise RuntimeError("MN9 is absent from the historical Shiu substrate.")

mn9_index = id_to_index[MN9]

print()
print(f"Sugar GRNs available: {len(sugar_indices)}/{len(SUGAR_REFERENCE)}")
print(f"MN9 index: {mn9_index}")


# ------------------------------------------------------------
# ProtoFly-style LIF network, but using Shiu's connectivity.
# ------------------------------------------------------------

neurons = NeuronGroup(
    len(df_comp),
    model=EQUATIONS,
    threshold=THRESHOLD,
    reset=RESET,
    refractory="rfc",
    method="linear",
    namespace=DEFAULT_PARAMS,
)

neurons.v = DEFAULT_PARAMS["v_0"]
neurons.g = 0 * mV
neurons.rfc = DEFAULT_PARAMS["t_rfc"]


synapses = Synapses(
    neurons,
    neurons,
    model="w : volt",
    on_pre="g += w",
    delay=DEFAULT_PARAMS["t_dly"],
)

synapses.connect(
    i=df_con["Presynaptic_Index"].to_numpy(),
    j=df_con["Postsynaptic_Index"].to_numpy(),
)

# IMPORTANT:
# Use Shiu's own signed connectivity directly.
synapses.w = (
    df_con["Excitatory x Connectivity"].to_numpy()
    * DEFAULT_PARAMS["w_syn"]
)


# ------------------------------------------------------------
# ProtoFly stimulation implementation.
# ------------------------------------------------------------

stimulus = PoissonGroup(
    len(sugar_indices),
    rates=FREQUENCY,
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

network.store("initial")


# ------------------------------------------------------------
# Trials
# ------------------------------------------------------------

mn9_counts = []
total_counts = []
active_counts = []

print()
print(
    f"Running {N_TRIALS} trials at "
    f"{int(FREQUENCY / Hz)} Hz..."
)

for trial in range(N_TRIALS):
    network.restore("initial")

    seed(1000 + trial)

    network.run(DURATION)

    counts = spikes.count[:]

    mn9_spikes = int(counts[mn9_index])
    total_spikes = int(counts.sum())
    active_neurons = int((counts > 0).sum())

    mn9_counts.append(mn9_spikes)
    total_counts.append(total_spikes)
    active_counts.append(active_neurons)

    print(
        f"trial {trial:02d}: "
        f"MN9={mn9_spikes:3d} | "
        f"active={active_neurons:,} | "
        f"total={total_spikes:,}"
    )


mn9_series = pd.Series(mn9_counts)
total_series = pd.Series(total_counts)
active_series = pd.Series(active_counts)


print()
print("=== ProtoFly dynamics + Shiu substrate ===")
print(
    f"MN9:   {mn9_series.mean():.2f} "
    f"± {mn9_series.std():.2f} spikes/s"
)
print(
    f"Active: {active_series.mean():.1f} "
    f"± {active_series.std():.1f}"
)
print(
    f"Total:  {total_series.mean():.1f} "
    f"± {total_series.std():.1f}"
)

print()
print("Reference Shiu result at 100 Hz:")
print("MN9: 67.03 ± 6.71 spikes/s")