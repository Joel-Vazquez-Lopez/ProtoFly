"""Diagnostic: post-sugar whole-brain dynamics.

Question
--------
After sustained sugar-GRN stimulation is removed, does activity in the
current ProtoFly/FlyWire simulation:

1. decay rapidly,
2. show a transient rebound,
3. remain persistently active, or
4. exhibit another dynamical pattern?

This diagnostic contains no environment, actuator, reward, learning,
or behavioural interpretation.
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
WINDOW = 1000 * ms

# Three seconds with sugar followed by six seconds without it.
N_STIMULATED = 3
N_POST = 6

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

sugar_indices = [
    index[fly_id]
    for fly_id in SUGAR_GRNS
    if fly_id in index
]

motor_ids = feeding_motor_root_ids()

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

seed(SEED)

previous_counts = spikes.count[:].copy()


def run_window(label, rate):
    global previous_counts

    stimulus.rates = rate

    network.run(WINDOW)

    current_counts = spikes.count[:].copy()
    window_counts = current_counts - previous_counts
    previous_counts = current_counts

    active = int((window_counts > 0).sum())
    total = int(window_counts.sum())

    motor_counts = {
        name: int(window_counts[idx])
        for name, idx in motor_indices.items()
    }

    print()
    print(label)
    print(f"  input:          {float(rate / Hz):.0f} Hz")
    print(f"  active neurons: {active:,}")
    print(f"  total spikes:   {total:,}")

    print(
        "  motor spikes:   "
        + ", ".join(
            f"{name}={motor_counts[name]}"
            for name in (
                "MN6_R",
                "MN6_L",
                "MN8_R",
                "MN8_L",
                "MN9_R",
                "MN9_L",
            )
        )
    )


print()
print("=== STIMULATION ===")

for window in range(1, N_STIMULATED + 1):
    run_window(
        f"Stimulated second {window}",
        SUGAR_RATE,
    )


print()
print("=== INPUT REMOVED ===")

for window in range(1, N_POST + 1):
    run_window(
        f"Post-stimulus second {window}",
        0 * Hz,
    )


print()
print("Diagnostic complete.")