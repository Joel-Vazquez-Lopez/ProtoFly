# Neural Dynamics Fidelity

## Purpose

ProtoFly uses the FlyWire connectome as its biological neural substrate and adds computational dynamics so that activity can propagate through the network.

The model must distinguish clearly between measured biological data, predicted biological annotations, and computational assumptions.

## Fidelity layers

| Component | Status | Source |
|---|---|---|
| Neuron identities | Observed | FlyWire FAFB |
| Connectivity | Observed | FlyWire FAFB |
| Synapse counts | Observed | FlyWire FAFB |
| Neuropil | Observed | FlyWire FAFB |
| Cell types | Annotated | FlyWire Codex |
| Neurotransmitter type | Predicted | FlyWire Codex |
| Neuron dynamics | Approximated | Shiu et al. whole-brain LIF model |
| Synaptic dynamics | Approximated | Shiu et al. whole-brain LIF model |
| Plasticity | None | Phase 0 |
| Learning | None | Phase 0 |

## LIF dynamics

ProtoFly initially follows the whole-brain leaky integrate-and-fire model used by Shiu et al.

Neuron dynamics:

    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau

Parameters:

| Parameter | Value |
|---|---:|
| Resting potential | -52 mV |
| Reset potential | -52 mV |
| Spike threshold | -45 mV |
| Membrane time constant | 20 ms |
| Synaptic time constant | 5 ms |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |
| Weight per synapse | 0.275 mV |

After a spike, membrane potential is reset to -52 mV and synaptic drive is reset to zero.

## Connection weights

For a directed connection from neuron i to neuron j:

    W_ij = sign_i * syn_count_ij * 0.275 mV

Initial neurotransmitter sign approximation follows the Shiu et al. model:

| Predicted neurotransmitter | Sign |
|---|---:|
| ACH | +1 |
| GABA | -1 |
| GLUT | -1 |
| DA | +1 |
| SER | +1 |
| OCT | +1 |

This sign is a modelling approximation and must not be interpreted as a universal statement about receptor-specific effects in the biological fly brain.

## External stimulation

Initial validation experiments use Poisson stimulation of selected neurons, following the published whole-brain LIF implementation.

Default reference stimulation:

    rate = 150 Hz

The stimulation mechanism is used only to inject activity into selected biological input populations. It does not constitute learning.

## Phase 0 constraint

There is no synaptic plasticity, reward learning, language representation, or communication mechanism in Experiment 00C.

The purpose of Experiment 00C is only to establish:

    FlyWire connectivity
            +
    published neural dynamics
            ↓
    reproducible propagation of neural activity

Learning mechanisms will be introduced only after the fixed-connectome dynamics have been independently validated.
