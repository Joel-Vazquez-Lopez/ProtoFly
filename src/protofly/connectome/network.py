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

    neuron_ids = list(neuron_ids)
    index = {root_id: i for i, root_id in enumerate(neuron_ids)}

    annotations = (
        neurons_df
        .set_index("root_id")
        .reindex(neuron_ids)
    )

    circuit = connections_df[
        connections_df["pre_root_id"].isin(index)
        & connections_df["post_root_id"].isin(index)
    ].copy()

    # Combine neuropil-specific rows belonging to the same directed pair.
    circuit = (
        circuit.groupby(["pre_root_id", "post_root_id"], as_index=False)
        ["syn_count"]
        .sum()
    )

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

    synapses = Synapses(
        group,
        group,
        model="w : volt",
        on_pre="g += w",
        delay=DEFAULT_PARAMS["t_dly"],
    )

    pre_indices = []
    post_indices = []
    weights = []

    for row in circuit.itertuples(index=False):
        pre_id = row.pre_root_id
        post_id = row.post_root_id

        nt_type = annotations.loc[pre_id, "nt_type"]
        sign = NT_SIGN.get(nt_type, 0)

        pre_indices.append(index[pre_id])
        post_indices.append(index[post_id])

        weights.append(
            sign * row.syn_count * DEFAULT_PARAMS["w_syn"]
        )

    synapses.connect(i=pre_indices, j=post_indices)
    synapses.w = weights

    return group, synapses, index, circuit