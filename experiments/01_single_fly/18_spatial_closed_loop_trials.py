"""Experiment 18: repeated spatial closed-loop trials.

Tests whether the spatially closed sensorimotor transition established
in Experiment 17 is reproducible across stochastic neural simulations.

Each trial uses the same fixed experimental setup:

    current body position
        -> spatial sugar contact
        -> sugar-GRN sensory input
        -> persistent FlyWire FAFB v783 dynamics
        -> bilateral DNg100 activity
        -> forward-locomotion readout
        -> simulation-level locomotion actuator
        -> updated body position
        -> new spatial sugar contact

Within each trial, one FlyWire brain persists across all interaction
steps. Between trials, a fresh brain and FlyBody are created so that
trials are independent.

The geometry, BPN stimulation, sugar stimulation, contact radius,
interaction duration, and DNg100-to-body mapping are kept identical to
Experiment 17. Only the stochastic seed changes between trials.

The experiment does not require every trajectory to be identical.
It tests whether the qualitative closed-loop sequence is robust:

    locomotion before contact
        -> spatial contact
        -> sugar sensory recruitment
        -> reduced subsequent forward locomotion

No parameter is tuned between trials.

The contact radius and DNg100-to-displacement conversion remain
explicit simulation-level embodiment parameters and are not
biologically calibrated measurements.

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
print("=== EXPERIMENT 18: REPEATED SPATIAL CLOSED-LOOP TRIALS ===")
print(f"Trials:              {N_TRIALS}")
print(f"BPN stimulation:     {float(BPN_RATE / Hz):.0f} Hz")
print(f"Sugar stimulation:   {float(SUGAR_RATE / Hz):.0f} Hz")
print(f"Contact radius:      {CONTACT_RADIUS:.3f}")
print(f"Forward gain:        {FORWARD_GAIN}")

print()
print(
    f"{'Seed':>4s} | "
    f"{'Contact':>7s} | "
    f"{'First':>5s} | "
    f"{'Pre move':>8s} | "
    f"{'Post move':>9s} | "
    f"{'Reduction':>10s} | "
    f"{'Final dist':>10s}"
)

print("-" * 78)

reductions = []

for trial in trials:
    pre = trial["pre_contact_displacements"]
    post = trial["post_contact_displacements"]

    pre_mean = (
        float(pre.mean())
        if len(pre) > 0
        else float("nan")
    )

    post_mean = (
        float(post.mean())
        if len(post) > 0
        else float("nan")
    )

    if len(pre) > 0 and len(post) > 0 and pre_mean > 0:
        reduction = 100.0 * (
            1.0 - post_mean / pre_mean
        )
    else:
        reduction = float("nan")

    reductions.append(reduction)

    contacted = trial["first_contact_after"] is not None

    print(
        f"{trial['seed']:4d} | "
        f"{str(contacted):>7s} | "
        f"{str(trial['first_contact_after']):>5s} | "
        f"{pre_mean:8.3f} | "
        f"{post_mean:9.3f} | "
        f"{reduction:9.2f}% | "
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

trials_with_post_contact = [
    trial
    for trial in trials
    if len(trial["post_contact_displacements"]) > 0
]

valid_reductions = np.asarray([
    reduction
    for reduction in reductions
    if np.isfinite(reduction)
])

print()
print("Across-trial summary:")
print(
    f"  Trials reaching spatial contact: "
    f"{len(contacted_trials)}/{N_TRIALS}"
)
print(
    f"  Trials receiving contact-driven sugar input: "
    f"{len(trials_with_post_contact)}/{N_TRIALS}"
)

if len(valid_reductions) > 0:
    print(
        f"  Mean post-contact movement reduction: "
        f"{valid_reductions.mean():.2f}% +/- "
        f"{valid_reductions.std(ddof=1):.2f}%"
    )

print()
print(
    "NOTE: Experiment 18 reports stochastic robustness without "
    "requiring every trial to produce the same trajectory."
)