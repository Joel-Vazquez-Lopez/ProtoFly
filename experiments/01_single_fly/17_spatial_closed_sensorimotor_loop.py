"""Experiment 17: spatial closed sensorimotor loop.

Tests whether ProtoFly can close the loop between spatial state,
sensory input, connectome dynamics, descending locomotor output, and
subsequent spatial state.

A single FlyBody and a sugar object inhabit the same bounded 2D
environment. Sugar GRNs are stimulated only when the fly is within an
explicit spatial contact radius of the sugar object.

The interaction loop is:

    current body position
        -> spatial sugar contact
        -> sugar-GRN sensory input
        -> persistent FlyWire FAFB v783 dynamics
        -> bilateral DNg100 activity
        -> forward-locomotion readout
        -> simulation-level locomotion actuator
        -> updated body position
        -> new spatial sugar contact

Bilateral BPN stimulation provides the established walking-promoting
input throughout the experiment. Sugar stimulation is not specified
as an experimental condition in advance; it is determined at each
interaction step from the current physical relationship between the
fly and the sugar object.

There is no explicit "sugar means stop" behavioural rule. Any change
in locomotion after spatial sugar contact must arise through the
connectome-constrained neural dynamics and the same DNg100-to-body
mapping used before contact.

The contact radius and DNg100-to-displacement conversion are explicit
simulation-level embodiment parameters. They are not biologically
calibrated measurements.

No learning, reward, plasticity, goal selection, or distant sugar
sensing is introduced.
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

from protofly.environment.spatial import (
    SpatialEnvironment,
    SpatialObject,
)
from protofly.sensory.spatial import object_contact


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

# One closed-loop interaction step.
STEP_DURATION = 50 * ms

# Total number of environment -> brain -> body cycles.
N_STEPS = 20

# Explicit simulation-level spatial contact parameter.
# This is NOT a biologically calibrated sensory radius.
CONTACT_RADIUS = 0.10
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
# Build one persistent FlyWire brain.
# ------------------------------------------------------------

neurons, synapses, index, circuit = build_flywire_network(
    neuron_ids,
    neurons_df,
    connections_df,
)


# ------------------------------------------------------------
# Resolve experimental neural populations.
# ------------------------------------------------------------

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

print(f"Brain neurons:       {len(neurons_df):,}")
print(f"BPN neurons present: {len(bpn_indices)}/{len(BPN_ROOT_IDS)}")
print(f"Sugar GRNs present:  {len(sugar_indices)}/{len(SUGAR_GRNS)}")

# ------------------------------------------------------------
# Neural input interfaces.
# ------------------------------------------------------------

# BPN stimulation remains active throughout the experiment.
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


# Sugar stimulation is controlled dynamically by spatial contact.
sugar_stimulus = PoissonGroup(
    len(sugar_indices),
    rates=0 * Hz,
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


# ------------------------------------------------------------
# Record descending locomotor activity.
# ------------------------------------------------------------

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    bpn_stimulus,
    bpn_stimulus_synapses,
    sugar_stimulus,
    sugar_stimulus_synapses,
    spikes,
)

seed(SEED)

# ------------------------------------------------------------
# Spatial world and embodied fly.
# ------------------------------------------------------------

environment = SpatialEnvironment(
    width=20.0,
    height=10.0,
)

body = FlyBody(
    x=5.0,
    y=4.0,
    orientation=0.0,
)

sugar = SpatialObject(
    kind="sugar",
    x=5.5,
    y=4.0,
)

environment.add_object(sugar)

locomotion_readout = LocomotionReadout()
locomotion_actuator = LocomotionActuator()

initial_distance = ((sugar.x - body.x) ** 2 + (sugar.y - body.y) ** 2) ** 0.5
initial_contact = object_contact(
    body,
    sugar,
    CONTACT_RADIUS,
)

print()
print("=== EXPERIMENT 17: SPATIAL CLOSED SENSORIMOTOR LOOP ===")
print(f"Initial fly position:  ({body.x:.3f}, {body.y:.3f})")
print(f"Sugar position:        ({sugar.x:.3f}, {sugar.y:.3f})")
print(f"Initial distance:      {initial_distance:.3f}")
print(f"Contact radius:        {CONTACT_RADIUS:.3f}")
print(f"Initial contact:       {initial_contact}")

assert not initial_contact


# ------------------------------------------------------------
# Closed spatial sensorimotor loop.
# ------------------------------------------------------------

previous_counts = spikes.count[:].copy()

step_duration_seconds = float(
    STEP_DURATION / (1000 * ms)
)

history = []

for step in range(1, N_STEPS + 1):

    # --------------------------------------------------------
    # Environment -> sensory state.
    # --------------------------------------------------------

    distance_before = (
        (sugar.x - body.x) ** 2
        + (sugar.y - body.y) ** 2
    ) ** 0.5

    contact_before = object_contact(
        body,
        sugar,
        CONTACT_RADIUS,
    )

    if contact_before:
        sugar_stimulus.rates = SUGAR_RATE
    else:
        sugar_stimulus.rates = 0 * Hz

    # --------------------------------------------------------
    # Advance the SAME persistent brain.
    # --------------------------------------------------------

    network.run(STEP_DURATION)

    current_counts = spikes.count[:].copy()

    step_counts = (
        current_counts - previous_counts
    )

    previous_counts = current_counts

    dng100_left_spikes = int(
        step_counts[dng100_left_index]
    )

    dng100_right_spikes = int(
        step_counts[dng100_right_index]
    )

    # --------------------------------------------------------
    # Descending activity -> embodied locomotion.
    # --------------------------------------------------------

    forward_drive = locomotion_readout.update(
        left_spikes=dng100_left_spikes,
        right_spikes=dng100_right_spikes,
        duration_seconds=step_duration_seconds,
    )

    displacement = (
        FORWARD_GAIN
        * forward_drive
        * step_duration_seconds
    )

    command = LocomotionCommand(
        command_type=LocomotionCommandType.FORWARD,
        magnitude=displacement,
    )

    locomotion_actuator.execute(
        command=command,
        body=body,
        environment=environment,
    )

    # --------------------------------------------------------
    # Record resulting spatial state.
    # --------------------------------------------------------

    distance_after = (
        (sugar.x - body.x) ** 2
        + (sugar.y - body.y) ** 2
    ) ** 0.5

    contact_after = object_contact(
        body,
        sugar,
        CONTACT_RADIUS,
    )

    history.append({
        "step": step,
        "contact_before": contact_before,
        "sugar_rate_hz": (
            float(SUGAR_RATE / Hz)
            if contact_before
            else 0.0
        ),
        "dng100_left_spikes": dng100_left_spikes,
        "dng100_right_spikes": dng100_right_spikes,
        "forward_drive_hz": forward_drive,
        "displacement": displacement,
        "x": body.x,
        "y": body.y,
        "distance_before": distance_before,
        "distance_after": distance_after,
        "contact_after": contact_after,
    })

# ------------------------------------------------------------
# Report closed-loop trajectory.
# ------------------------------------------------------------

print()
print(
    f"{'Step':>4s} | "
    f"{'Contact':>7s} | "
    f"{'Sugar Hz':>8s} | "
    f"{'DNg100 L':>8s} | "
    f"{'DNg100 R':>8s} | "
    f"{'Move':>7s} | "
    f"{'X':>7s} | "
    f"{'Distance':>8s} | "
    f"{'Contact after':>13s}"
)

print("-" * 91)

for record in history:
    print(
        f"{record['step']:4d} | "
        f"{str(record['contact_before']):>7s} | "
        f"{record['sugar_rate_hz']:8.0f} | "
        f"{record['dng100_left_spikes']:8d} | "
        f"{record['dng100_right_spikes']:8d} | "
        f"{record['displacement']:7.3f} | "
        f"{record['x']:7.3f} | "
        f"{record['distance_after']:8.3f} | "
        f"{str(record['contact_after']):>13s}"
    )


# ------------------------------------------------------------
# Closed-loop audit.
# ------------------------------------------------------------

contact_steps = [
    record["step"]
    for record in history
    if record["contact_before"]
]

sensory_steps = [
    record["step"]
    for record in history
    if record["sugar_rate_hz"] > 0
]

first_contact_after = next(
    (
        record["step"]
        for record in history
        if record["contact_after"]
    ),
    None,
)

print()
print("=== FINAL STATE ===")
print(f"Final fly position:    ({body.x:.3f}, {body.y:.3f})")
print(f"Final distance:        {history[-1]['distance_after']:.3f}")
print(f"Final contact:         {history[-1]['contact_after']}")
print(f"First contact reached: {first_contact_after}")
print(f"Contact-input steps:   {contact_steps}")
print(f"Sugar-sensory steps:   {sensory_steps}")

assert contact_steps == sensory_steps

print()
print(
    "PASS: every sugar-GRN sensory-input step was determined by "
    "the fly's spatial contact state."
)