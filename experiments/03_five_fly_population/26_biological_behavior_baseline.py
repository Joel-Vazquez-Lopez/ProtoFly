"""Experiment 25: two independent connectome-based flies in one world.

Constructs two independent FlyWire FAFB v783 neural networks, each
with its own sensory inputs, spike monitor, motor readouts, and body.

Both flies inhabit the same spatial environment.

The initial experiment tests multi-agent simulation infrastructure,
not communication, social learning, or emergent collective behavior.

The existing neural and sensorimotor parameters are retained from
Experiment 21. No fly-to-fly signaling is introduced.
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

SEED = 42

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


# ------------------------------------------------------------
# Build whole-connectome brain.
# ------------------------------------------------------------

def build_fly_brain():
    return build_flywire_network(
        neuron_ids,
        neurons_df,
        connections_df,
    )

def build_fly_neural_system():
    neurons, synapses, index, circuit = build_fly_brain()

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

    return {
        "network": network,
        "neurons": neurons,
        "synapses": synapses,
        "spikes": spikes,
        "left_stimulus": left_stimulus,
        "right_stimulus": right_stimulus,
        "dng100_left_index": dng100_left_index,
        "dng100_right_index": dng100_right_index,
        "dna02_left_index": dna02_left_index,
        "dna02_right_index": dna02_right_index,
    }

seed(SEED)

fly_a_brain = build_fly_neural_system()
fly_b_brain = build_fly_neural_system()

print()
print("=== DESCENDING NEURON INPUT CONNECTIVITY ===")

target_ids = {
    "DNg100_L": FORWARD_LOCOMOTION_DESCENDING_NEURONS["DNg100"]["left"],
    "DNg100_R": FORWARD_LOCOMOTION_DESCENDING_NEURONS["DNg100"]["right"],
    **steering_descending_root_ids(),
}

for name, root_id in target_ids.items():
    inputs = connections_df[
        connections_df["post_root_id"] == root_id
    ]

    total_synapses = inputs["syn_count"].sum()

    print(
        f"{name}: "
        f"{len(inputs)} connection rows, "
        f"{total_synapses} input synapses"
    )

print()
print("=== DESCENDING NEURON INPUT SIGN BALANCE ===")

nt_sign = {
    "ACH": 1,
    "GABA": -1,
    "GLUT": -1,
    "DA": 1,
    "SER": 1,
    "OCT": 1,
}

nt_lookup = neurons_df.set_index("root_id")["nt_type"]

for name, root_id in target_ids.items():
    inputs = connections_df[
        connections_df["post_root_id"] == root_id
    ].copy()

    inputs["nt_type"] = inputs["pre_root_id"].map(nt_lookup)
    inputs["sign"] = inputs["nt_type"].map(nt_sign).fillna(0)

    excitatory = inputs.loc[
        inputs["sign"] == 1, "syn_count"
    ].sum()

    inhibitory = inputs.loc[
        inputs["sign"] == -1, "syn_count"
    ].sum()

    unknown = inputs.loc[
        inputs["sign"] == 0, "syn_count"
    ].sum()

    print(
        f"{name}: "
        f"positive={excitatory}, "
        f"negative={inhibitory}, "
        f"unknown={unknown}"
    )

assert fly_a_brain["neurons"] is not fly_b_brain["neurons"]
assert fly_a_brain["synapses"] is not fly_b_brain["synapses"]
assert fly_a_brain["network"] is not fly_b_brain["network"]
assert fly_a_brain["spikes"] is not fly_b_brain["spikes"]

print("Two independent neural systems constructed successfully.")

network = fly_a_brain["network"]
spikes = fly_a_brain["spikes"]

left_stimulus = fly_a_brain["left_stimulus"]
right_stimulus = fly_a_brain["right_stimulus"]

dng100_left_index = fly_a_brain["dng100_left_index"]
dng100_right_index = fly_a_brain["dng100_right_index"]
dna02_left_index = fly_a_brain["dna02_left_index"]
dna02_right_index = fly_a_brain["dna02_right_index"]

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

body_b = FlyBody(
    x=7.0,
    y=4.0,
    orientation=3.141592653589793,
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

locomotion_readout_b = LocomotionReadout()
locomotion_actuator_b = LocomotionActuator()

steering_readout_b = SteeringReadout()
steering_actuator_b = SteeringActuator(
    steering_gain=STEERING_GAIN
)

# ------------------------------------------------------------
# Closed spatial olfactory sensorimotor loop.
# ------------------------------------------------------------

previous_counts = spikes.count[:].copy()
previous_counts_b = fly_b_brain["spikes"].count[:].copy()

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

    left_antenna_b, right_antenna_b = antenna_positions(
        body_b,
        ANTENNA_OFFSET,
    )

    left_distance_b = point_distance(
        left_antenna_b,
        odor_source,
    )

    left_concentration_b = odor_concentration(
        left_distance_b,
        ODOR_LENGTH_SCALE,
    )

    left_rate_hz_b = concentration_to_rate(
        left_concentration_b,
        MAX_ORN_RATE_HZ,
    )

    fly_b_brain["left_stimulus"].rates = left_rate_hz_b * Hz

    right_distance_b = point_distance(
        right_antenna_b,
        odor_source,
    )

    right_concentration_b = odor_concentration(
        right_distance_b,
        ODOR_LENGTH_SCALE,
    )

    right_rate_hz_b = concentration_to_rate(
        right_concentration_b,
        MAX_ORN_RATE_HZ,
    )

    fly_b_brain["right_stimulus"].rates = right_rate_hz_b * Hz

    left_stimulus.rates = left_rate_hz * Hz
    right_stimulus.rates = right_rate_hz * Hz
    
    # --------------------------------------------------------
    # Advance the same persistent brain for one time step.
    # --------------------------------------------------------

    network.run(STEP_DURATION)
    fly_b_brain["network"].run(STEP_DURATION)

    current_counts = spikes.count[:].copy()

    step_counts = (
        current_counts - previous_counts
    )

    previous_counts = current_counts

    current_counts_b = fly_b_brain["spikes"].count[:].copy()
    step_counts_b = current_counts_b - previous_counts_b
    previous_counts_b = current_counts_b

    dng100_left_spikes_b = step_counts_b[fly_b_brain["dng100_left_index"]]
    dng100_right_spikes_b = step_counts_b[fly_b_brain["dng100_right_index"]]

    dna02_left_spikes_b = step_counts_b[fly_b_brain["dna02_left_index"]]
    dna02_right_spikes_b = step_counts_b[fly_b_brain["dna02_right_index"]]

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

    steering_readout_b.update(
        left_spikes=dna02_left_spikes_b,
        right_spikes=dna02_right_spikes_b,
        duration_seconds=step_duration_seconds,
    )

    steering_drive_b = steering_readout_b.steering_drive

    angular_displacement = steering_actuator.execute(
        readout=steering_readout,
        body=body,
        duration_seconds=step_duration_seconds,
    )

    angular_displacement_b = steering_actuator_b.execute(
        readout=steering_readout_b,
        body=body_b,
        duration_seconds=step_duration_seconds,
    )

    forward_drive = locomotion_readout.update(
        left_spikes=dng100_left_spikes,
        right_spikes=dng100_right_spikes,
        duration_seconds=step_duration_seconds,
    )

    forward_drive_b = locomotion_readout_b.update(
        left_spikes=int(dng100_left_spikes_b),
        right_spikes=int(dng100_right_spikes_b),
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

    displacement_b = (
        FORWARD_GAIN
        * forward_drive_b
        * step_duration_seconds
    )

    command_b = LocomotionCommand(
        command_type=LocomotionCommandType.FORWARD,
        magnitude=displacement_b,
    )

    locomotion_actuator_b.execute(
        command=command_b,
        body=body_b,
        environment=environment,
    )

    inter_fly_distance = np.hypot(
        body_b.x - body.x,
        body_b.y - body.y,
    )
    
    history.append(
        {
            "step": step,
            "inter_fly_distance": inter_fly_distance,
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

            # Fly B: independent behavioral measurements
            "x_b": body_b.x,
            "y_b": body_b.y,
            "orientation_b": body_b.orientation,
            "left_rate_hz_b": left_rate_hz_b,
            "right_rate_hz_b": right_rate_hz_b,
            "dna02_left_spikes_b": int(dna02_left_spikes_b),
            "dna02_right_spikes_b": int(dna02_right_spikes_b),
            "steering_drive_b": steering_drive_b,
            "angular_displacement_b": angular_displacement_b,
            "dng100_left_spikes_b": int(dng100_left_spikes_b),
            "dng100_right_spikes_b": int(dng100_right_spikes_b),
            "forward_drive_b": forward_drive_b,
            "displacement_b": displacement_b,
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

print()
print("=== FLY B FINAL STATE ===")
print(f"Final position:          ({body_b.x:.3f}, {body_b.y:.3f})")
print(f"Final orientation:       {body_b.orientation:.3f} rad")
print(f"Final forward drive:     {forward_drive_b:.3f}")
print(f"Final steering drive:    {steering_drive_b:.3f}")

print()
print("=== TWO-FLY BEHAVIORAL BASELINE ===")

for fly_name, suffix in [("Fly A", ""), ("Fly B", "_b")]:
    total_distance = history_df[f"displacement{suffix}"].sum()
    active_steps = (history_df[f"displacement{suffix}"] > 0).sum()
    total_turning = history_df[f"angular_displacement{suffix}"].abs().sum()

    print()
    print(f"{fly_name}:")
    print(f"  Total walking displacement: {total_distance:.4f}")
    print(f"  Active walking steps:       {active_steps}/{N_STEPS}")
    print(f"  Total absolute turning:     {total_turning:.4f} rad")
    print(f"  Mean forward drive:         {history_df[f'forward_drive{suffix}'].mean():.3f}")

print()
print(f"Initial inter-fly distance:  {np.hypot(7.0 - 5.0, 4.0 - 4.0):.4f}")
print(f"Final inter-fly distance:    {history_df['inter_fly_distance'].iloc[-1]:.4f}")
print(f"Minimum inter-fly distance:  {history_df['inter_fly_distance'].min():.4f}")

print()
print("=== MOTOR NEURON DIAGNOSTIC ===")

for fly_name, suffix in [("Fly A", ""), ("Fly B", "_b")]:
    print()
    print(f"{fly_name}:")

    for neuron in ("dng100", "dna02"):
        left = history_df[f"{neuron}_left_spikes{suffix}"]
        right = history_df[f"{neuron}_right_spikes{suffix}"]

        print(f"  {neuron.upper()}:")
        print(f"    Left total spikes:  {left.sum()}")
        print(f"    Right total spikes: {right.sum()}")
        print(f"    Active steps:       {((left + right) > 0).sum()}/{N_STEPS}")


print()
print("=== DNG100 TEMPORAL ACTIVITY ===")

print(
    history_df[
        [
            "step",
            "dng100_left_spikes",
            "dng100_right_spikes",
            "dng100_left_spikes_b",
            "dng100_right_spikes_b",
        ]
    ].to_string(index=False)
)