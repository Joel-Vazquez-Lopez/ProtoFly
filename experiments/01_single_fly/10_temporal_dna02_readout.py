"""Experiment 01D: temporal DNa02 steering readout.

Runs the validated whole-brain sugar-GRN stimulation experiment and
extracts the actual DNa02_L and DNa02_R spike trains in short temporal
bins.

This experiment does not move the body and does not apply a steering
gain. Its purpose is to expose the temporal neural signal that a future
biologically grounded steering-dynamics model can consume.
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
BIN_SIZE = 50 * ms
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
# Resolve sensory and DNa02 neurons.
# ------------------------------------------------------------

sugar_indices = [
    index[root_id]
    for root_id in SUGAR_GRNS
    if root_id in index
]

steering_ids = steering_descending_root_ids()

dna02_left_index = index[steering_ids["DNa02_L"]]
dna02_right_index = index[steering_ids["DNa02_R"]]

print(f"Brain neurons:       {len(neurons_df):,}")
print(f"Sugar GRNs present:  {len(sugar_indices)}/{len(SUGAR_GRNS)}")
print(f"DNa02_L index:       {dna02_left_index}")
print(f"DNa02_R index:       {dna02_right_index}")


# ------------------------------------------------------------
# Sensory interface.
# ------------------------------------------------------------

stimulus = PoissonGroup(
    len(sugar_indices),
    rates=SUGAR_RATE,
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
# Record actual spike identities and times.
# ------------------------------------------------------------

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    sensory_synapses,
    spikes,
)

seed(SEED)

network.run(DURATION)


# ------------------------------------------------------------
# Extract DNa02 spike trains.
# ------------------------------------------------------------

spike_indices = np.asarray(spikes.i[:])
spike_times_ms = np.asarray(spikes.t[:] / ms)

left_times = spike_times_ms[
    spike_indices == dna02_left_index
]

right_times = spike_times_ms[
    spike_indices == dna02_right_index
]


# ------------------------------------------------------------
# Bin DNa02 activity.
# ------------------------------------------------------------

duration_ms = float(DURATION / ms)
bin_size_ms = float(BIN_SIZE / ms)

bin_edges = np.arange(
    0.0,
    duration_ms + bin_size_ms,
    bin_size_ms,
)

left_counts, _ = np.histogram(
    left_times,
    bins=bin_edges,
)

right_counts, _ = np.histogram(
    right_times,
    bins=bin_edges,
)

bin_seconds = float(BIN_SIZE / (1000 * ms))

left_rates = left_counts / bin_seconds
right_rates = right_counts / bin_seconds

steering_drive = left_rates - right_rates


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

print()
print("=== TEMPORAL DNa02 READOUT ===")
print(f"Duration: {duration_ms:.0f} ms")
print(f"Bin size: {bin_size_ms:.0f} ms")
print()

print(
    f"{'Time (ms)':>12s} | "
    f"{'L spikes':>8s} | "
    f"{'R spikes':>8s} | "
    f"{'L Hz':>8s} | "
    f"{'R Hz':>8s} | "
    f"{'L-R Hz':>8s}"
)

print("-" * 68)

for i in range(len(left_counts)):
    start = bin_edges[i]
    end = bin_edges[i + 1]

    print(
        f"{start:5.0f}-{end:<5.0f} | "
        f"{left_counts[i]:8d} | "
        f"{right_counts[i]:8d} | "
        f"{left_rates[i]:8.1f} | "
        f"{right_rates[i]:8.1f} | "
        f"{steering_drive[i]:+8.1f}"
    )


# ------------------------------------------------------------
# Consistency checks.
# ------------------------------------------------------------

total_left = int(left_counts.sum())
total_right = int(right_counts.sum())

direct_left = int(spikes.count[dna02_left_index])
direct_right = int(spikes.count[dna02_right_index])

print()
print("Totals:")
print(f"  DNa02_L: {total_left} spikes")
print(f"  DNa02_R: {total_right} spikes")

if total_left != direct_left:
    raise RuntimeError(
        f"Left binned total {total_left} != "
        f"SpikeMonitor total {direct_left}"
    )

if total_right != direct_right:
    raise RuntimeError(
        f"Right binned total {total_right} != "
        f"SpikeMonitor total {direct_right}"
    )

print()
print("PASS: temporal DNa02 spike trains extracted consistently.")