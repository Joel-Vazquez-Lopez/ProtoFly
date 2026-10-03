"""Experiment 19: Spatial closed loop with locomotion and steering.

Integrate the previously validated DNg100 forward-locomotion
and DNa02 steering pathways into one embodied fly.

Both motor outputs must arise from the same persistent
FlyWire FAFB v783 neural simulation and act on the same body.

This experiment tests the integration of translation and
rotation in a spatial sensorimotor loop.

It does not introduce a navigation policy, distant sugar
sensing, learning, or a predefined target-directed behaviour.
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
    STEERING_DESCENDING_NEURONS,
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
from protofly.agents.steering_readout import SteeringReadout
from protofly.agents.steering_actuator import SteeringActuator


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
STEERING_GAIN = 0.01
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

def run_trial(trial_seed):
    """Run one independent spatial closed-loop trial."""

    # --------------------------------------------------------
    # Build one brain that persists throughout this trial.
    # --------------------------------------------------------

    neurons, synapses, index, circuit = build_flywire_network(
        neuron_ids,
        neurons_df,
        connections_df,
    )

    # --------------------------------------------------------
    # Resolve experimental neural populations.
    # --------------------------------------------------------

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

    dna02_left_root = STEERING_DESCENDING_NEURONS["DNa02"]["left"]
    dna02_right_root = STEERING_DESCENDING_NEURONS["DNa02"]["right"]
    dna02_left_index = index[dna02_left_root]
    dna02_right_index = index[dna02_right_root]

    # --------------------------------------------------------
    # Neural input interfaces.
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

    # --------------------------------------------------------
    # Record descending locomotor activity.
    # --------------------------------------------------------

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

    seed(trial_seed)

        # --------------------------------------------------------
    # Spatial world and embodied fly.
    # --------------------------------------------------------

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

    steering_readout = SteeringReadout()
    steering_actuator = SteeringActuator(
        steering_gain=STEERING_GAIN
    )

    initial_contact = object_contact(
        body,
        sugar,
        CONTACT_RADIUS,
    )

    if initial_contact:
        raise RuntimeError(
            "Fly must begin outside the sugar contact radius."
        )

        # --------------------------------------------------------
    # Closed spatial sensorimotor loop.
    # --------------------------------------------------------

    previous_counts = spikes.count[:].copy()

    step_duration_seconds = float(
        STEP_DURATION / (1000 * ms)
    )

    history = []

    for step in range(1, N_STEPS + 1):

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

        # Advance the SAME brain within this trial.
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

        dna02_left_spikes = int(step_counts[dna02_left_index])
        dna02_right_spikes = int(step_counts[dna02_right_index])

        steering_readout.update(
            left_spikes=dna02_left_spikes,
            right_spikes=dna02_right_spikes,
            duration_seconds=step_duration_seconds,
        )

        steering_drive = steering_readout.steering_drive

        angular_displacement = steering_actuator.execute(
            readout=steering_readout,
            body=body,
            duration_seconds=step_duration_seconds,
        )

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
            "dna02_left_spikes": dna02_left_spikes,
            "dna02_right_spikes": dna02_right_spikes,
            "steering_drive_hz": steering_drive,
            "angular_displacement": angular_displacement,
            "orientation": body.orientation,
            "displacement": displacement,
            "x": body.x,
            "y": body.y,
            "distance_before": distance_before,
            "distance_after": distance_after,
            "contact_after": contact_after,
        })
            # --------------------------------------------------------
    # Summarize this trial.
    # --------------------------------------------------------

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

    # Spatial contact must be the sole determinant of sugar input.
    if contact_steps != sensory_steps:
        raise RuntimeError(
            "Sugar sensory input did not match spatial contact state."
        )

    pre_contact_displacements = [
        record["displacement"]
        for record in history
        if not record["contact_before"]
    ]

    post_contact_displacements = [
        record["displacement"]
        for record in history
        if record["contact_before"]
    ]

    return {
        "seed": trial_seed,
        "history": history,
        "first_contact_after": first_contact_after,
        "contact_steps": contact_steps,
        "final_x": body.x,
        "final_y": body.y,
        "final_orientation": body.orientation,
        "final_distance": history[-1]["distance_after"],
        "final_contact": history[-1]["contact_after"],
        "pre_contact_displacements": np.asarray(
            pre_contact_displacements
        ),
        "post_contact_displacements": np.asarray(
            post_contact_displacements
        ),
    }

# ------------------------------------------------------------
# Run repeated independent closed-loop trials.
# ------------------------------------------------------------

N_TRIALS = 10

trials = []

for trial_seed in range(SEED, SEED + N_TRIALS):
    trial_number = trial_seed - SEED + 1

    print(
        f"Running trial {trial_number}/{N_TRIALS} "
        f"(seed={trial_seed})..."
    )

    trials.append(
        run_trial(trial_seed)
    )


# ------------------------------------------------------------
# Report across-trial results.
# ------------------------------------------------------------

print()
print("=== EXPERIMENT 19: SPATIAL LOCOMOTION + STEERING CLOSED LOOP ===")
print(f"Trials:              {N_TRIALS}")
print(f"BPN stimulation:     {float(BPN_RATE / Hz):.0f} Hz")
print(f"Sugar stimulation:   {float(SUGAR_RATE / Hz):.0f} Hz")
print(f"Contact radius:      {CONTACT_RADIUS:.3f}")
print(f"Forward gain:        {FORWARD_GAIN}")
print(
    f"Steering gain:       {STEERING_GAIN} rad/(s*Hz)"
)

print()
print(
    f"{'Seed':>4s} | "
    f"{'Contact':>7s} | "
    f"{'First':>5s} | "
    f"{'Final x':>7s} | "
    f"{'Final y':>7s} | "
    f"{'Orient':>7s} | "
    f"{'Final dist':>10s}"
)

print("-" * 75)

for trial in trials:
    contacted = trial["first_contact_after"] is not None

    print(
        f"{trial['seed']:4d} | "
        f"{str(contacted):>7s} | "
        f"{str(trial['first_contact_after']):>5s} | "
        f"{trial['final_x']:7.3f} | "
        f"{trial['final_y']:7.3f} | "
        f"{trial['final_orientation']:7.3f} | "
        f"{trial['final_distance']:10.3f}"
    )


# ------------------------------------------------------------
# Across-trial summary.
# ------------------------------------------------------------

contacted_trials = [
    trial
    for trial in trials
    if trial["first_contact_after"] is not None
]

trials_with_steering = [
    trial
    for trial in trials
    if any(
        abs(record["angular_displacement"]) > 0
        for record in trial["history"]
    )
]

final_orientations = np.asarray([
    trial["final_orientation"]
    for trial in trials
])

print()
print("Across-trial summary:")
print(
    f"  Trials with non-zero DNa02-derived steering: "
    f"{len(trials_with_steering)}/{N_TRIALS}"
)
print(
    f"  Trials reaching spatial contact: "
    f"{len(contacted_trials)}/{N_TRIALS}"
)
print(
    f"  Final orientation range: "
    f"{final_orientations.min():.3f} to "
    f"{final_orientations.max():.3f} rad"
)

print()
print(
    "NOTE: DNa02-derived steering and DNg100-derived forward "
    "locomotion act on the same embodied fly. The steering and "
    "forward gains are explicit simulation-level embodiment "
    "parameters, not biological calibrations."
)

print()
print("Seed 42 trajectory diagnostic:")
print(
    f"{'Step':>4s} | "
    f"{'Contact':>7s} | "
    f"{'Sugar':>5s} | "
    f"{'DNa02 L':>7s} | "
    f"{'DNa02 R':>7s} | "
    f"{'Turn':>7s} | "
    f"{'Orient':>7s} | "
    f"{'Move':>6s} | "
    f"{'X':>6s} | "
    f"{'Y':>6s} | "
    f"{'Dist':>6s}"
)

print("-" * 94)

trial_42 = next(
    trial for trial in trials
    if trial["seed"] == 42
)

for record in trial_42["history"]:
    print(
        f"{record['step']:4d} | "
        f"{str(record['contact_before']):>7s} | "
        f"{record['sugar_rate_hz']:5.0f} | "
        f"{record['dna02_left_spikes']:7d} | "
        f"{record['dna02_right_spikes']:7d} | "
        f"{record['angular_displacement']:7.3f} | "
        f"{record['orientation']:7.3f} | "
        f"{record['displacement']:6.3f} | "
        f"{record['x']:6.3f} | "
        f"{record['y']:6.3f} | "
        f"{record['distance_after']:6.3f}"
    )