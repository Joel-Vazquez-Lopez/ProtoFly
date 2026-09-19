"""Utilities for extracting connected subnetworks from FlyWire."""

from collections import deque

import pandas as pd


def grow_downstream_subnetwork(
    seed_neuron,
    connections_df,
    max_neurons,
):
    """Grow a connected downstream subnetwork from one FlyWire neuron.

    Expansion follows directed FlyWire connections. Stronger outgoing
    connections are considered first.

    Parameters
    ----------
    seed_neuron
        FlyWire root ID from which expansion begins.
    connections_df
        FlyWire connection table.
    max_neurons
        Maximum number of unique neurons to include.

    Returns
    -------
    set
        FlyWire root IDs in the extracted subnetwork.
    """

    # Aggregate neuropil-specific rows into neuron-pair connections.
    pair_connections = (
        connections_df
        .groupby(
            ["pre_root_id", "post_root_id"],
            as_index=False,
            sort=False,
        )["syn_count"]
        .sum()
    )

    # Strong connections are explored first.
    pair_connections = pair_connections.sort_values(
        "syn_count",
        ascending=False,
    )

    # Build adjacency table:
    # presynaptic neuron -> downstream neuron IDs
    adjacency = {
        pre_id: group["post_root_id"].tolist()
        for pre_id, group in pair_connections.groupby(
            "pre_root_id",
            sort=False,
        )
    }

    selected = {seed_neuron}
    queue = deque([seed_neuron])

    while queue and len(selected) < max_neurons:
        current = queue.popleft()

        for target in adjacency.get(current, []):
            if target in selected:
                continue

            selected.add(target)
            queue.append(target)

            if len(selected) >= max_neurons:
                break

    return selected