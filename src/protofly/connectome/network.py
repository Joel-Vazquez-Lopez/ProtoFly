"""Build Brian2 networks from FlyWire connectome data."""

import pandas as pd
from brian2 import NeuronGroup, Synapses, mV

from protofly.neural.lif import (
    DEFAULT_PARAMS,
    EQUATIONS,
    THRESHOLD,
    RESET,
)


NT_SIGN = {
    "ACH": 1,
    "GABA": -1,
    "GLUT": -1,
    "DA": 1,
    "SER": 1,
    "OCT": 1,
}

def build_flywire_network(neuron_ids, neurons_df, connections_df):
    """Build a connectome-constrained Brian2 network."""

    # Stable ordering makes runs easier to reproduce.
    neuron_ids = sorted(neuron_ids)

    index = {
        root_id: i
        for i, root_id in enumerate(neuron_ids)
    }

    # --------------------------------------------------
    # Select all FlyWire connections inside the circuit
    # --------------------------------------------------

    circuit = connections_df[
        connections_df["pre_root_id"].isin(index)
        & connections_df["post_root_id"].isin(index)
    ].copy()

    # Combine neuropil-specific rows for the same pair.
    circuit = (
        circuit
        .groupby(
            ["pre_root_id", "post_root_id"],
            as_index=False,
            sort=False,
        )["syn_count"]
        .sum()
    )

    # --------------------------------------------------
    # Map FlyWire IDs -> Brian2 integer indices
    # --------------------------------------------------

    circuit["pre_index"] = (
        circuit["pre_root_id"]
        .map(index)
        .astype("int32")
    )

    circuit["post_index"] = (
        circuit["post_root_id"]
        .map(index)
        .astype("int32")
    )

    # --------------------------------------------------
    # Neurotransmitter sign of presynaptic neuron
    # --------------------------------------------------

    nt_lookup = (
        neurons_df
        .set_index("root_id")["nt_type"]
    )

    circuit["nt_type"] = (
        circuit["pre_root_id"]
        .map(nt_lookup)
    )

    circuit["sign"] = (
        circuit["nt_type"]
        .map(NT_SIGN)
        .fillna(0)
        .astype("int8")
    )

    # --------------------------------------------------
    # Create neurons
    # --------------------------------------------------

    group = NeuronGroup(
        len(neuron_ids),
        model=EQUATIONS,
        threshold=THRESHOLD,
        reset=RESET,
        refractory="rfc",
        method="linear",
        namespace=DEFAULT_PARAMS,
    )

    group.v = DEFAULT_PARAMS["v_0"]
    group.g = 0 * mV
    group.rfc = DEFAULT_PARAMS["t_rfc"]

    # --------------------------------------------------
    # Create synapses
    # --------------------------------------------------

    synapses = Synapses(
        group,
        group,
        model="w : volt",
        on_pre="g += w",
        delay=DEFAULT_PARAMS["t_dly"],
    )

    synapses.connect(
        i=circuit["pre_index"].to_numpy(),
        j=circuit["post_index"].to_numpy(),
    )

    weights = (
        circuit["sign"].to_numpy()
        * circuit["syn_count"].to_numpy()
    )

    synapses.w = weights * DEFAULT_PARAMS["w_syn"]

    return group, synapses, index, circuit