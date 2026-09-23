"""Experiment 01C: closed sensorimotor feeding loop.

Goal
----
Extend ProtoFly from a one-way sensory-to-motor experiment into its first
closed interaction loop:

    environment
        -> sensory input
        -> FlyWire brain
        -> feeding motor output
        -> action
        -> changed environment
        -> new sensory input

The FlyWire connectome and neural dynamics remain unchanged.

Important
---------
The mapping from neural motor activity to an external action is an
explicit simulation mechanism, not a claim that a particular firing-rate
threshold is the biological definition of feeding.

No learning, reward, or plasticity is introduced in this experiment.
"""

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

from protofly.environment.feeding import FeedingEnvironment
from protofly.agents.feeding_actuator import FeedingActuator
from protofly.connectome.known_neurons import feeding_motor_root_ids
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

# One interaction step corresponds to 1 second of neural simulation.
STEP_DURATION = 1000 * ms

# Experimental embodiment parameters.
# These are NOT biological measurements.
INITIAL_SUGAR = 3.0
SUGAR_PER_ACTION = 1.0

# We deliberately run one additional step after the sugar is exhausted
# so that the changed environment can feed back into sensory input.
N_STEPS = 4

SEED = 42


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


# ---------------------------------------------------------------------
# Resolve sensory and motor populations
# ---------------------------------------------------------------------

sugar_ids = [
    fly_id
    for fly_id in SUGAR_GRNS
    if fly_id in index
]

sugar_indices = [
    index[fly_id]
    for fly_id in sugar_ids
]

motor_ids = feeding_motor_root_ids()

missing_motor = {
    name: root_id
    for name, root_id in motor_ids.items()
    if root_id not in index
}

if missing_motor:
    raise RuntimeError(
        "Verified feeding motor neurons missing from substrate: "
        f"{missing_motor}"
    )

motor_indices = {
    name: index[root_id]
    for name, root_id in motor_ids.items()
}

print(f"Brain neurons:       {len(neurons_df):,}")
print(f"Sugar GRNs present:  {len(sugar_indices)}/{len(SUGAR_GRNS)}")
print(f"Motor neurons:       {len(motor_indices)}/{len(motor_ids)}")


# ---------------------------------------------------------------------
# Sensory interface
# ---------------------------------------------------------------------

stimulus = PoissonGroup(
    len(sugar_indices),
    rates=0 * Hz,
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


# ---------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    sensory_synapses,
    spikes,
)


# ---------------------------------------------------------------------
# Environment and actuator
# ---------------------------------------------------------------------

environment = FeedingEnvironment(
    sugar_present=True,
    sugar_amount=INITIAL_SUGAR,
)

actuator = FeedingActuator()

seed(SEED)


# ---------------------------------------------------------------------
# Closed sensorimotor loop
# ---------------------------------------------------------------------

print()
print("=== CLOSED SENSORIMOTOR LOOP ===")

previous_counts = spikes.count[:].copy()

for step in range(1, N_STEPS + 1):

    print()
    print(f"--- STEP {step} ---")

    sugar_before = environment.sugar_amount
    contact_before = environment.sugar_contact()

    # Environment -> sensory input
    if contact_before:
        stimulus.rates = SUGAR_RATE
    else:
        stimulus.rates = 0 * Hz

    # Advance the SAME brain. We do not restore/reset between steps.
    network.run(STEP_DURATION)

    current_counts = spikes.count[:].copy()

    # SpikeMonitor counts are cumulative, so isolate spikes generated
    # during this interaction step.
    step_counts = current_counts - previous_counts
    previous_counts = current_counts

    # Brain -> motor population
    motor_counts = {
        name: int(step_counts[idx])
        for name, idx in motor_indices.items()
    }

    actuator.update(motor_counts)

    # Motor state -> environmental action.
    #
    # This is an explicit embodiment rule:
    # any recruitment of the verified feeding motor population causes
    # one unit of sugar consumption.
    #
    # It is NOT asserted to be a biological feeding threshold.
    if actuator.motor_active and contact_before:
        consumed = environment.consume_sugar(
            SUGAR_PER_ACTION
        )
    else:
        consumed = 0.0

    active_neurons = int((step_counts > 0).sum())
    total_spikes = int(step_counts.sum())

    print(f"Sugar before:        {sugar_before:.1f}")
    print(f"Sugar contact:       {contact_before}")
    print(f"Sensory rate:        {float(stimulus.rates[0] / Hz):.0f} Hz")
    print(f"Active neurons:      {active_neurons:,}")
    print(f"Total spikes:        {total_spikes:,}")

    print("Motor output:")
    for motor_name in (
        "MN6_R",
        "MN6_L",
        "MN8_R",
        "MN8_L",
        "MN9_R",
        "MN9_L",
    ):
        print(
            f"  {motor_name:5s}: "
            f"{motor_counts[motor_name]:4d} spikes"
        )

    print(
        f"Motor population:    "
        f"{len(actuator.active_motor_neurons)}/6 active"
    )

    print(f"Sugar consumed:      {consumed:.1f}")
    print(f"Sugar after:         {environment.sugar_amount:.1f}")


print()
print("=== FINAL STATE ===")
print(f"Sugar remaining:     {environment.sugar_amount:.1f}")
print(f"Sugar contact:       {environment.sugar_contact()}")

print()
print(
    "NOTE: neural state was preserved continuously across interaction "
    "steps. The motor-to-consumption mapping is an explicit simulation "
    "mechanism, not a biological feeding threshold."
)