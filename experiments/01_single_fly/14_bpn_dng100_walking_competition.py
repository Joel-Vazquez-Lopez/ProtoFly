"""Experiment 14: BPN -> DNg100 walking/feeding competition.

Tests whether stimulation of the verified Bolt Protocerebral Neuron
(BPN) population propagates through the full FlyWire FAFB v783
connectome to bilateral DNg100 descending neurons, and whether
coactivation of the established sugar-GRN pathway changes that output.

Conditions:
    1. Bilateral BPN stimulation at 50 Hz.
    2. Bilateral BPN stimulation at 50 Hz plus sugar GRNs at 150 Hz.

Each condition is run in a freshly constructed brain network so neural
state does not carry between conditions.

This experiment measures DNg100 spikes and membrane potential only.
It does not move the simulated body or impose a locomotion rule.
"""

import numpy as np
import pandas as pd

from brian2 import (
    PoissonGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
    Network,
    Hz,
    ms,
    mV,
    seed,
)

from protofly.connectome.known_neurons import (
    BOLT_PROTOCEREBRAL_NEURONS,
    FORWARD_LOCOMOTION_DESCENDING_NEURONS,
)
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

BPN_RATE = 50 * Hz
SUGAR_RATE = 150 * Hz
DURATION = 1000 * ms
BIN_SIZE = 50 * ms
SEED = 42


BPN_ROOT_IDS = [
    root_id
    for data in BOLT_PROTOCEREBRAL_NEURONS.values()
    for side in ("left", "right")
    for root_id in data[side]
]


# ------------------------------------------------------------
# Load connectome data once.
# ------------------------------------------------------------

print("Loading current FlyWire brain...")

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

neuron_ids = neurons_df["root_id"].tolist()


# ------------------------------------------------------------
# Run one experimental condition in a fresh brain.
# ------------------------------------------------------------

def run_condition(include_sugar, trial_seed):
    neurons, synapses, index, circuit = build_flywire_network(
        neuron_ids,
        neurons_df,
        connections_df,
    )

    bpn_indices = [
        index[root_id]
        for root_id in BPN_ROOT_IDS
        if root_id in index
    ]

    sugar_indices = [
        index[root_id]
        for root_id in SUGAR_GRNS
        if root_id in index
    ]

    dng100_left_root = (
        FORWARD_LOCOMOTION_DESCENDING_NEURONS["DNg100"]["left"]
    )
    dng100_right_root = (
        FORWARD_LOCOMOTION_DESCENDING_NEURONS["DNg100"]["right"]
    )

    dng100_left_index = index[dng100_left_root]
    dng100_right_index = index[dng100_right_root]

    # --------------------------------------------------------
    # BPN stimulation.
    # --------------------------------------------------------

    bpn_stimulus = PoissonGroup(
        len(bpn_indices),
        rates=BPN_RATE,
    )

    bpn_stimulus_synapses = Synapses(
        bpn_stimulus,
        neurons,
        model="w : volt",
        on_pre="v_post += w",
    )

    bpn_stimulus_synapses.connect(
        i=list(range(len(bpn_indices))),
        j=bpn_indices,
    )

    bpn_stimulus_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

    for i in bpn_indices:
        neurons.rfc[i] = 0 * ms

    network_objects = [
        neurons,
        synapses,
        bpn_stimulus,
        bpn_stimulus_synapses,
    ]

    # --------------------------------------------------------
    # Optional sugar-GRN stimulation.
    # --------------------------------------------------------

    if include_sugar:
        sugar_stimulus = PoissonGroup(
            len(sugar_indices),
            rates=SUGAR_RATE,
        )

        sugar_stimulus_synapses = Synapses(
            sugar_stimulus,
            neurons,
            model="w : volt",
            on_pre="v_post += w",
        )

        sugar_stimulus_synapses.connect(
            i=list(range(len(sugar_indices))),
            j=sugar_indices,
        )

        sugar_stimulus_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

        for i in sugar_indices:
            neurons.rfc[i] = 0 * ms

        network_objects.extend([
            sugar_stimulus,
            sugar_stimulus_synapses,
        ])

    # --------------------------------------------------------
    # Record DNg100 activity.
    # --------------------------------------------------------

    spikes = SpikeMonitor(neurons)

    dng100_voltage = StateMonitor(
        neurons,
        "v",
        record=[
            dng100_left_index,
            dng100_right_index,
        ],
    )

    network_objects.extend([
        spikes,
        dng100_voltage,
    ])

    network = Network(*network_objects)

    seed(trial_seed)
    network.run(DURATION)

    # --------------------------------------------------------
    # Extract DNg100 activity.
    # --------------------------------------------------------

    spike_indices = np.asarray(spikes.i[:])
    spike_times_ms = np.asarray(spikes.t[:] / ms)

    left_times = spike_times_ms[
        spike_indices == dng100_left_index
    ]

    right_times = spike_times_ms[
        spike_indices == dng100_right_index
    ]

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

    left_voltage_mv = np.asarray(
        dng100_voltage.v[0] / mV
    )

    right_voltage_mv = np.asarray(
        dng100_voltage.v[1] / mV
    )

    voltage_times_ms = np.asarray(
        dng100_voltage.t[:] / ms
    )

    left_voltage_means = []
    right_voltage_means = []

    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]

        mask = (
            (voltage_times_ms >= start)
            & (voltage_times_ms < end)
        )

        left_voltage_means.append(
            left_voltage_mv[mask].mean()
        )

        right_voltage_means.append(
            right_voltage_mv[mask].mean()
        )

    left_voltage_means = np.asarray(left_voltage_means)
    right_voltage_means = np.asarray(right_voltage_means)

    total_left = int(left_counts.sum())
    total_right = int(right_counts.sum())

    direct_left = int(
        spikes.count[dng100_left_index]
    )
    direct_right = int(
        spikes.count[dng100_right_index]
    )

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

    return {
        "bpn_present": len(bpn_indices),
        "sugar_present": len(sugar_indices),
        "left_counts": left_counts,
        "right_counts": right_counts,
        "left_voltage_means": left_voltage_means,
        "right_voltage_means": right_voltage_means,
        "left_total": total_left,
        "right_total": total_right,
        "left_max_mv": float(left_voltage_mv.max()),
        "right_max_mv": float(right_voltage_mv.max()),
        "bin_edges": bin_edges,
    }


# ------------------------------------------------------------
# Run repeated independent conditions.
# ------------------------------------------------------------

N_TRIALS = 10

condition_1_trials = []
condition_2_trials = []

for trial_seed in range(SEED, SEED + N_TRIALS):
    print(
        f"Running trial {trial_seed - SEED + 1}/{N_TRIALS} "
        f"(seed={trial_seed})..."
    )

    condition_1_trials.append(
        run_condition(
            include_sugar=False,
            trial_seed=trial_seed,
        )
    )

    condition_2_trials.append(
        run_condition(
            include_sugar=True,
            trial_seed=trial_seed,
        )
    )
# ------------------------------------------------------------
# Summarize repeated trials.
# ------------------------------------------------------------

c1_left = np.asarray([
    result["left_total"]
    for result in condition_1_trials
])

c1_right = np.asarray([
    result["right_total"]
    for result in condition_1_trials
])

c2_left = np.asarray([
    result["left_total"]
    for result in condition_2_trials
])

c2_right = np.asarray([
    result["right_total"]
    for result in condition_2_trials
])

left_suppression = 100 * (
    1 - c2_left / c1_left
)

right_suppression = 100 * (
    1 - c2_right / c1_right
)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

print()
print("=== EXPERIMENT 14: BPN -> DNg100 WALKING/FEEDING COMPETITION ===")
print(f"Trials:                {N_TRIALS}")
print(f"Brain neurons:         {len(neurons_df):,}")
print(
    f"BPNs present:          "
    f"{condition_1_trials[0]['bpn_present']}/{len(BPN_ROOT_IDS)}"
)
print(
    f"Sugar GRNs present:    "
    f"{condition_1_trials[0]['sugar_present']}/{len(SUGAR_GRNS)}"
)
print(f"BPN stimulation:       {float(BPN_RATE / Hz):.0f} Hz")
print(f"Sugar stimulation:     {float(SUGAR_RATE / Hz):.0f} Hz")

print()
print("Per-trial DNg100 spike counts:")
print(
    f"{'Trial':>5s} | "
    f"{'BPN L':>6s} | "
    f"{'BPN R':>6s} | "
    f"{'BPN+sugar L':>11s} | "
    f"{'BPN+sugar R':>11s}"
)
print("-" * 55)

for i in range(N_TRIALS):
    print(
        f"{i + 1:5d} | "
        f"{c1_left[i]:6d} | "
        f"{c1_right[i]:6d} | "
        f"{c2_left[i]:11d} | "
        f"{c2_right[i]:11d}"
    )

print()
print("Across-trial summary:")

print(
    f"  BPN only DNg100_L: "
    f"{c1_left.mean():.2f} +/- {c1_left.std(ddof=1):.2f} spikes"
)
print(
    f"  BPN only DNg100_R: "
    f"{c1_right.mean():.2f} +/- {c1_right.std(ddof=1):.2f} spikes"
)

print(
    f"  BPN+sugar DNg100_L: "
    f"{c2_left.mean():.2f} +/- {c2_left.std(ddof=1):.2f} spikes"
)
print(
    f"  BPN+sugar DNg100_R: "
    f"{c2_right.mean():.2f} +/- {c2_right.std(ddof=1):.2f} spikes"
)

print()
print("Suppression relative to BPN-only condition:")

print(
    f"  DNg100_L: "
    f"{left_suppression.mean():.2f}% +/- "
    f"{left_suppression.std(ddof=1):.2f}%"
)

print(
    f"  DNg100_R: "
    f"{right_suppression.mean():.2f}% +/- "
    f"{right_suppression.std(ddof=1):.2f}%"
)

print()
print(
    f"Trials with lower DNg100_L under BPN+sugar: "
    f"{int(np.sum(c2_left < c1_left))}/{N_TRIALS}"
)

print(
    f"Trials with lower DNg100_R under BPN+sugar: "
    f"{int(np.sum(c2_right < c1_right))}/{N_TRIALS}"
)

print()
print(
    "PASS: repeated independent BPN-only and BPN+sugar "
    "conditions completed consistently."
)