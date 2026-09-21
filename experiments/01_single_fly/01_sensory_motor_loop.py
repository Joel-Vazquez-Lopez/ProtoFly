"""Experiment 01: first ProtoFly sensory -> brain -> motor loop.

The environment controls only whether sugar is present.

Sugar contact stimulates the real FlyWire sugar-sensing GRNs.
Activity propagates through the current whole-brain connectome.
MN9 activity is read out as the feeding-related motor response.

No learning, reward, or predefined behavioural threshold is used.
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
from protofly.agents.feeding_action import FeedingAction, MN9
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


# ------------------------------------------------------------
# Identify biological sensory and motor neurons.
# ------------------------------------------------------------

sugar_ids = [
    fly_id
    for fly_id in SUGAR_GRNS
    if fly_id in index
]

sugar_indices = [
    index[fly_id]
    for fly_id in sugar_ids
]

if MN9 not in index:
    raise RuntimeError("MN9 is absent from the current FlyWire substrate.")

mn9_index = index[MN9]

print(f"Brain neurons:       {len(neurons_df):,}")
print(f"Sugar GRNs present:  {len(sugar_indices)}/{len(SUGAR_GRNS)}")
print(f"MN9 index:           {mn9_index}")


# ------------------------------------------------------------
# Sensory interface.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Motor observation.
# ------------------------------------------------------------

spikes = SpikeMonitor(neurons)

network = Network(
    neurons,
    synapses,
    stimulus,
    sensory_synapses,
    spikes,
)

network.store("initial")


# ------------------------------------------------------------
# Environment -> brain -> motor response
# ------------------------------------------------------------

def run_condition(name, sugar_present):
    network.restore("initial")
    seed(SEED)

    environment = FeedingEnvironment(
        sugar_present=sugar_present
    )

    if environment.sugar_contact():
        stimulus.rates = SUGAR_RATE
    else:
        stimulus.rates = 0 * Hz

    network.run(DURATION)

    counts = spikes.count[:]

    action = FeedingAction()
    action.update(
        spike_count=int(counts[mn9_index]),
        duration_seconds=float(DURATION / (1000 * ms)),
    )

    active_neurons = int((counts > 0).sum())
    total_spikes = int(counts.sum())

    print()
    print(f"=== {name} ===")
    print(f"Sugar present:  {environment.sugar_contact()}")
    print(f"Active neurons: {active_neurons:,}")
    print(f"Total spikes:   {total_spikes:,}")
    print(f"Motor output:   {action}")

    return action


no_sugar = run_condition(
    "NO SUGAR",
    sugar_present=False,
)

with_sugar = run_condition(
    "SUGAR CONTACT",
    sugar_present=True,
)


# ------------------------------------------------------------
# First closed-loop sanity check.
# ------------------------------------------------------------

print()
print("=== COMPARISON ===")
print(
    f"MN9 without sugar: {no_sugar.firing_rate:.2f} Hz"
)
print(
    f"MN9 with sugar:    {with_sugar.firing_rate:.2f} Hz"
)

if with_sugar.firing_rate > no_sugar.firing_rate:
    print(
        "PASS: changing the external environment produced "
        "a stronger MN9 motor response through the FlyWire brain."
    )
else:
    print(
        "FAIL: sugar did not increase the MN9 motor response."
    )