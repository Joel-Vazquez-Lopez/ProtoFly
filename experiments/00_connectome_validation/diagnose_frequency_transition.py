"""Diagnose the frequency-dependent activity transition in ProtoFly.

This experiment keeps the current FlyWire connectome, LIF dynamics,
and sugar stimulation fixed while changing only the assumed functional
sign of neurotransmitter classes.

This is a modelling diagnostic, not a biological experiment.
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

from protofly.connectome.network import build_flywire_network, NT_SIGN
from protofly.neural.lif import DEFAULT_PARAMS


SUGAR_REFERENCE = [
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

MN9 = 720575940660219265

FREQUENCIES = [60, 80, 100]
N_TRIALS = 5
DURATION = 1000 * ms


SIGN_CONDITIONS = {
    # Current ProtoFly assumption.
    "current": {
        "ACH": 1,
        "GABA": -1,
        "GLUT": -1,
        "DA": 1,
        "SER": 1,
        "OCT": 1,
    },

    # Diagnostic: change only glutamate from inhibitory to excitatory.
    "glut_excitatory": {
        "ACH": 1,
        "GABA": -1,
        "GLUT": 1,
        "DA": 1,
        "SER": 1,
        "OCT": 1,
    },

    # Diagnostic: remove glutamatergic outgoing influence entirely.
    "glut_blocked": {
        "ACH": 1,
        "GABA": -1,
        "GLUT": 0,
        "DA": 1,
        "SER": 1,
        "OCT": 1,
    },
}


print("Loading current FlyWire substrate...")

neurons_df = pd.read_csv(
    "data/raw/flywire/neurons.csv.gz"
)

connections_df = pd.read_csv(
    "data/raw/flywire/connections_princeton.csv.gz"
)

all_neuron_ids = neurons_df["root_id"].tolist()

results = []


for condition_name, sign_mapping in SIGN_CONDITIONS.items():

    print()
    print("=" * 70)
    print(f"SIGN CONDITION: {condition_name}")
    print("=" * 70)

    # build_flywire_network reads this module-level mapping.
    NT_SIGN.clear()
    NT_SIGN.update(sign_mapping)

    neurons, synapses, index, circuit = build_flywire_network(
        all_neuron_ids,
        neurons_df,
        connections_df,
    )

    sugar_ids = [
        fly_id
        for fly_id in SUGAR_REFERENCE
        if fly_id in index
    ]

    sugar_indices = [
        index[fly_id]
        for fly_id in sugar_ids
    ]

    if MN9 not in index:
        raise RuntimeError("MN9 is absent from the current substrate.")

    mn9_index = index[MN9]

    print(f"Sugar GRNs available: {len(sugar_indices)}/{len(SUGAR_REFERENCE)}")
    print(f"MN9 index: {mn9_index}")

    stimulus = PoissonGroup(
        len(sugar_indices),
        rates=0 * Hz,
    )

    input_synapses = Synapses(
        stimulus,
        neurons,
        model="w : volt",
        on_pre="v_post += w",
    )

    input_synapses.connect(
        i=list(range(len(sugar_indices))),
        j=sugar_indices,
    )

    input_synapses.w = DEFAULT_PARAMS["w_syn"] * 250

    for i in sugar_indices:
        neurons.rfc[i] = 0 * ms

    spikes = SpikeMonitor(neurons)

    network = Network(
        neurons,
        synapses,
        stimulus,
        input_synapses,
        spikes,
    )

    network.store("initial")

    for frequency in FREQUENCIES:

        print()
        print(f"{frequency} Hz")

        for trial in range(N_TRIALS):

            network.restore("initial")

            # Same seed schedule across sign conditions so that
            # comparisons use matched stochastic input conditions.
            trial_seed = 1000 + frequency * 100 + trial
            seed(trial_seed)

            stimulus.rates = frequency * Hz

            network.run(DURATION)

            counts = spikes.count[:]

            active_neurons = int((counts > 0).sum())
            total_spikes = int(counts.sum())
            mn9_spikes = int(counts[mn9_index])

            results.append(
                {
                    "condition": condition_name,
                    "frequency_hz": frequency,
                    "trial": trial,
                    "seed": trial_seed,
                    "active_neurons": active_neurons,
                    "total_spikes": total_spikes,
                    "mn9_spikes": mn9_spikes,
                }
            )

            print(
                f"trial {trial:02d}: "
                f"active={active_neurons:,} | "
                f"total={total_spikes:,} | "
                f"MN9={mn9_spikes}"
            )


results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby(["condition", "frequency_hz"])
    .agg(
        active_mean=("active_neurons", "mean"),
        active_std=("active_neurons", "std"),
        total_mean=("total_spikes", "mean"),
        total_std=("total_spikes", "std"),
        mn9_mean=("mn9_spikes", "mean"),
        mn9_std=("mn9_spikes", "std"),
    )
    .reset_index()
)


results_df.to_csv(
    "results/frequency_transition_diagnostic_trials.csv",
    index=False,
)

summary.to_csv(
    "results/frequency_transition_diagnostic_summary.csv",
    index=False,
)


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print()
print("Saved:")
print("results/frequency_transition_diagnostic_trials.csv")
print("results/frequency_transition_diagnostic_summary.csv")