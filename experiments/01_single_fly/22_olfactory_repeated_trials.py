"""Experiment 21: spatial olfactory closed loop.

Tests whether spatially asymmetric odor input can influence the
trajectory of an embodied ProtoFly through the experimentally grounded
antennal Orco-like population, the whole-connectome FAFB v783 neural
substrate, and the DNa02 steering pathway.

The odor source provides no goal, desired direction, reward, or
navigation command. Left and right sensory stimulation is determined
only by the physical odor concentration sampled at the two antennal
positions.

Antenna geometry, odor-field shape, and concentration-to-rate mapping
are explicit simulation-level approximations and are not biologically
calibrated odor-plume or receptor models.

This experiment is an integration test of a pre-contact spatial
olfactory sensorimotor loop. It does not by itself demonstrate
odor-guided navigation.
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

from protofly.connectome.known_neurons import (
    BOLT_PROTOCEREBRAL_NEURONS,
    FORWARD_LOCOMOTION_DESCENDING_NEURONS,
    steering_descending_root_ids,
    antennal_orco_root_ids,
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
from protofly.agents.steering_readout import SteeringReadout
from protofly.agents.steering_actuator import SteeringActuator

from protofly.environment.spatial import (
    SpatialEnvironment,
    SpatialObject,
)
from protofly.sensory.olfaction import (
    antenna_positions,
    point_distance,
    odor_concentration,
    concentration_to_rate,
)


# ------------------------------------------------------------
# Frozen experiment parameters.
# ------------------------------------------------------------

# Walking-promoting input retained unchanged from Experiment 19.
BPN_RATE = 50 * Hz

# Closed-loop timing retained unchanged from Experiment 19.
STEP_DURATION = 50 * ms
N_STEPS = 20

# Existing simulation-level motor mappings from Experiment 19.
FORWARD_GAIN = 0.01
STEERING_GAIN = 0.01

# New explicit olfactory embodiment assumptions.
# These are not biologically calibrated anatomical or plume parameters.
ANTENNA_OFFSET = 0.10
ODOR_LENGTH_SCALE = 1.0

# Maximum sensory stimulation is anchored to the frozen Experiment 20
# Orco-like stimulation rate rather than selected from Experiment 21 behavior.
MAX_ORN_RATE_HZ = 100.0

TRIAL_SEEDS = range(42, 52)

BPN_ROOT_IDS = [
    root_id
    for data in BOLT_PROTOCEREBRAL_NEURONS.values()
    for side in ("left", "right")
    for root_id in data[side]
]


# ------------------------------------------------------------
# Load current FlyWire brain and annotations.
# ------------------------------------------------------------

print("Loading current FlyWire brain...")

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

annotations_df = pd.read_csv(
    "../flywire_annotations/supplemental_files/"
    "Supplemental_file1_neuron_annotations.tsv",
    sep="\t",
    low_memory=False,
)

neuron_ids = neurons_df["root_id"].tolist()


# ------------------------------------------------------------
# Resolve antennal Orco-like sensory populations.
# ------------------------------------------------------------

orco_ids = antennal_orco_root_ids(
    annotations_df,
    substrate_root_ids=neuron_ids,
)

left_orco_ids = orco_ids["left"]
right_orco_ids = orco_ids["right"]

print(f"Brain neurons:    {len(neuron_ids):,}")
print(f"Left Orco ORNs:   {len(left_orco_ids):,}")
print(f"Right Orco ORNs:  {len(right_orco_ids):,}")
print(
    f"Total Orco ORNs:  "
    f"{len(left_orco_ids) + len(right_orco_ids):,}"
)

def run_trial(trial_seed):


    # ------------------------------------------------------------
    # Build whole-connectome brain.
    # ------------------------------------------------------------

    neurons, synapses, index, circuit = build_flywire_network(
        neuron_ids,
        neurons_df,
        connections_df,
    )


    # ------------------------------------------------------------
    # Resolve antennal sensory indices.
    # ------------------------------------------------------------

    left_orco_indices = [
        index[root_id]
        for root_id in left_orco_ids
    ]

    right_orco_indices = [
        index[root_id]
        for root_id in right_orco_ids
    ]


    # ------------------------------------------------------------
    # Resolve walking and steering populations.
    # ------------------------------------------------------------

    bpn_indices = [
        index[root_id]
        for root_id in BPN_ROOT_IDS
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

    steering_ids = steering_descending_root_ids()

    dna02_left_index = index[steering_ids["DNa02_L"]]
    dna02_right_index = index[steering_ids["DNa02_R"]]

    print()
    print(f"Left ORN indices:   {len(left_orco_indices):,}")
    print(f"Right ORN indices:  {len(right_orco_indices):,}")
    print(f"BPN indices:        {len(bpn_indices):,}")
    print("DNg100 pair:        2/2")
    print("DNa02 pair:         2/2")
    print(f"Whole-brain edges:  {len(circuit):,}")


    # ------------------------------------------------------------
    # Lateralized olfactory sensory interface.
    # ------------------------------------------------------------

    left_stimulus = PoissonGroup(
        len(left_orco_indices),
        rates=0 * Hz,
    )

    right_stimulus = PoissonGroup(
        len(right_orco_indices),
        rates=0 * Hz,
    )

    left_sensory_synapses = Synapses(
        left_stimulus,
        neurons,
        model="w : volt",
        on_pre="v_post += w",
    )

    left_sensory_synapses.connect(
        i=list(range(len(left_orco_indices))),
        j=left_orco_indices,
    )

    right_sensory_synapses = Synapses(
        right_stimulus,
        neurons,
        model="w : volt",
        on_pre="v_post += w",
    )

    right_sensory_synapses.connect(
        i=list(range(len(right_orco_indices))),
        j=right_orco_indices,
    )

    # Frozen sensory-interface convention from Experiment 20.
    left_sensory_synapses.w = DEFAULT_PARAMS["w_syn"] * 250
    right_sensory_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

    for i in left_orco_indices:
        neurons.rfc[i] = 0 * ms

    for i in right_orco_indices:
        neurons.rfc[i] = 0 * ms


    # ------------------------------------------------------------
    # Walking-promoting neural input interface.
    # ------------------------------------------------------------

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

    # Frozen input-interface convention from Experiment 19.
    bpn_stimulus_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

    for i in bpn_indices:
        neurons.rfc[i] = 0 * ms


    # ------------------------------------------------------------
    # Record whole-brain spiking and assemble persistent network.
    # ------------------------------------------------------------

    spikes = SpikeMonitor(neurons)

    network = Network(
        neurons,
        synapses,
        bpn_stimulus,
        bpn_stimulus_synapses,
        left_stimulus,
        right_stimulus,
        left_sensory_synapses,
        right_sensory_synapses,
        spikes,
    )

    seed(trial_seed)

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

    odor_source = SpatialObject(
        kind="odor_source",
        x=6.0,
        y=5.0,
    )

    environment.add_object(odor_source)

    locomotion_readout = LocomotionReadout()
    locomotion_actuator = LocomotionActuator()

    steering_readout = SteeringReadout()
    steering_actuator = SteeringActuator(
        steering_gain=STEERING_GAIN
    )

    # ------------------------------------------------------------
    # Closed spatial olfactory sensorimotor loop.
    # ------------------------------------------------------------

    previous_counts = spikes.count[:].copy()

    step_duration_seconds = float(
        STEP_DURATION / (1000 * ms)
    )

    history = []


    for step in range(1, N_STEPS + 1):

        # --------------------------------------------------------
        # Sample the odor field at the current antennal positions.
        # --------------------------------------------------------

        left_antenna, right_antenna = antenna_positions(
            body,
            ANTENNA_OFFSET,
        )

        left_distance = point_distance(
            left_antenna,
            odor_source,
        )

        right_distance = point_distance(
            right_antenna,
            odor_source,
        )

        left_concentration = odor_concentration(
            left_distance,
            ODOR_LENGTH_SCALE,
        )

        right_concentration = odor_concentration(
            right_distance,
            ODOR_LENGTH_SCALE,
        )

        left_rate_hz = concentration_to_rate(
            left_concentration,
            MAX_ORN_RATE_HZ,
        )

        right_rate_hz = concentration_to_rate(
            right_concentration,
            MAX_ORN_RATE_HZ,
        )

        left_stimulus.rates = left_rate_hz * Hz
        right_stimulus.rates = right_rate_hz * Hz
        
        # --------------------------------------------------------
        # Advance the same persistent brain for one time step.
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

        dna02_left_spikes = int(
            step_counts[dna02_left_index]
        )

        dna02_right_spikes = int(
            step_counts[dna02_right_index]
        )

            # --------------------------------------------------------
        # Convert descending neural activity into body movement.
        # --------------------------------------------------------

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
        
        history.append(
            {
                "step": step,
                "x": body.x,
                "y": body.y,
                "orientation": body.orientation,
                "left_antenna_x": left_antenna[0],
                "left_antenna_y": left_antenna[1],
                "right_antenna_x": right_antenna[0],
                "right_antenna_y": right_antenna[1],
                "left_distance": left_distance,
                "right_distance": right_distance,
                "left_concentration": left_concentration,
                "right_concentration": right_concentration,
                "left_rate_hz": left_rate_hz,
                "right_rate_hz": right_rate_hz,
                "dna02_left_spikes": dna02_left_spikes,
                "dna02_right_spikes": dna02_right_spikes,
                "steering_drive": steering_drive,
                "angular_displacement": angular_displacement,
                "dng100_left_spikes": dng100_left_spikes,
                "dng100_right_spikes": dng100_right_spikes,
                "forward_drive": forward_drive,
                "displacement": displacement,
            }
        )

    # ------------------------------------------------------------
    # Report untouched pilot trajectory.
    # ------------------------------------------------------------

    history_df = pd.DataFrame(history)

    print()
    print("=== SPATIAL OLFACTORY CLOSED-LOOP PILOT ===")
    print()

    print(
        history_df[
            [
                "step",
                "x",
                "y",
                "orientation",
                "left_rate_hz",
                "right_rate_hz",
                "dna02_left_spikes",
                "dna02_right_spikes",
                "steering_drive",
                "forward_drive",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Initial source distance: "
        f"{((odor_source.x - 5.0) ** 2 + (odor_source.y - 4.0) ** 2) ** 0.5:.3f}"
    )

    final_distance = (
        (odor_source.x - body.x) ** 2
        + (odor_source.y - body.y) ** 2
    ) ** 0.5

    print(f"Final source distance:   {final_distance:.3f}")
    print(f"Final position:          ({body.x:.3f}, {body.y:.3f})")
    print(f"Final orientation:       {body.orientation:.3f} rad")
    return history_df

for trial_seed in TRIAL_SEEDS:
    print(f"\n{'=' * 60}")
    print(f"TRIAL SEED: {trial_seed}")
    print(f"{'=' * 60}")
    run_trial(trial_seed)