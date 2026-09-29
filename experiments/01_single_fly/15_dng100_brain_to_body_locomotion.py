"""Experiment 15: DNg100 brain-to-body forward locomotion.

Tests whether activity reaching the bilateral DNg100 descending neurons
through the full FlyWire FAFB v783 connectome can drive forward movement
of the simulated ProtoFly body.

The neural pathway is:

    BPN stimulation
        -> full FAFB v783 connectome
        -> bilateral DNg100 activity
        -> forward-locomotion readout
        -> simulation-level forward actuator
        -> FlyBody displacement

The DNg100-to-displacement mapping is an explicit simulation-level
embodiment assumption. It does not model the ventral nerve cord,
motor neurons, muscles, or biomechanics, and it is not a biologically
calibrated conversion from DNg100 firing rate to walking speed.

This experiment first tests the BPN-only condition established in
Experiment 14.
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

from protofly.agents.body import FlyBody
from protofly.agents.locomotion_readout import LocomotionReadout
from protofly.agents.locomotion_actuator import (
    LocomotionActuator,
    LocomotionCommand,
    LocomotionCommandType,
)
from protofly.environment.spatial import SpatialEnvironment


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
# Explicit simulation-level embodiment parameter.
# This is NOT a biologically calibrated DNg100 -> walking-speed mapping.
# Units: body-distance units / (s * Hz)
FORWARD_GAIN = 0.01
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
# Run BPN-only neural condition.
# ------------------------------------------------------------

result = run_condition(
    include_sugar=False,
    trial_seed=SEED,
)

# ------------------------------------------------------------
# DNg100 -> body forward-locomotion interface.
# ------------------------------------------------------------

body = FlyBody(
    x=5.0,
    y=4.0,
    orientation=0.0,
)

environment = SpatialEnvironment(
    width=20.0,
    height=10.0,
)

readout = LocomotionReadout()
actuator = LocomotionActuator()

initial_x = body.x
initial_y = body.y

bin_duration_seconds = float(BIN_SIZE / (1000 * ms))

forward_drives = []
forward_displacements = []

for left_spikes, right_spikes in zip(
    result["left_counts"],
    result["right_counts"],
):
    forward_drive = readout.update(
        left_spikes=left_spikes,
        right_spikes=right_spikes,
        duration_seconds=bin_duration_seconds,
    )

    displacement = (
        FORWARD_GAIN
        * forward_drive
        * bin_duration_seconds
    )

    command = LocomotionCommand(
        command_type=LocomotionCommandType.FORWARD,
        magnitude=displacement,
    )

    actuator.execute(
        command=command,
        body=body,
        environment=environment,
    )

    forward_drives.append(forward_drive)
    forward_displacements.append(displacement)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

total_displacement = sum(forward_displacements)

print()
print("=== EXPERIMENT 15: DNg100 BRAIN-TO-BODY LOCOMOTION ===")
print(f"BPN stimulation:       {float(BPN_RATE / Hz):.0f} Hz")
print(f"DNg100_L spikes:       {result['left_total']}")
print(f"DNg100_R spikes:       {result['right_total']}")
print(f"Forward gain:          {FORWARD_GAIN}")
print(f"Initial position:      ({initial_x:.3f}, {initial_y:.3f})")
print(f"Final position:        ({body.x:.3f}, {body.y:.3f})")
print(f"Forward displacement:  {total_displacement:.3f}")

assert body.x > initial_x
assert abs(body.y - initial_y) < 1e-12

print()
print(
    "PASS: activity reaching bilateral DNg100 through the "
    "FAFB v783 substrate produced forward body displacement."
)
