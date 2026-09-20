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

## Historical-substrate reproduction test

To distinguish implementation differences from connectome-materialization
differences, ProtoFly's neural dynamics were tested using the historical
connectivity substrate distributed with the Shiu et al. whole-brain
Drosophila model.

The test used:

- 127,400 neurons
- 14,687,178 directed connections
- 52,793,639 anatomical synapses
- all 21 right-hemisphere sugar-sensing GRNs
- 100 Hz Poisson stimulation
- 1 second per trial
- the signed connectivity weights from the original Shiu dataset
- ProtoFly's Brian2 LIF dynamics and stimulation implementation

### Results

| Model | MN9 response at 100 Hz |
|---|---:|
| Original Shiu implementation | 67.03 ± 6.71 spikes/s |
| ProtoFly implementation on Shiu substrate | 66.80 ± 5.73 spikes/s |

For the ProtoFly reproduction test, mean total network activity was
9,684.4 ± 307.7 spikes per trial. The original Shiu result was
9,635.8 ± 413.5 spikes per trial.

The MN9 mean therefore differed by only 0.23 spikes/s (~0.34%), while
mean total spike count differed by ~0.5%.

### Interpretation

ProtoFly reproduces the published computational model closely when both
implementations operate on the same historical connectome substrate.

This indicates that the substantially higher MN9 response observed with
ProtoFly's current FlyWire/Codex substrate (~128 spikes/s at 100 Hz)
should not be attributed simply to an error in the LIF or sensory-input
implementation.

The historical Shiu and current ProtoFly connectome representations differ
substantially:

| Property | Shiu historical substrate | Current ProtoFly substrate |
|---|---:|---:|
| Directed neuron pairs | 14,687,178 | 3,732,460 |
| Anatomical synapses | 52,793,639 | 50,666,648 |
| Shared directed pairs | 2,025,373 | 2,025,373 |

Only 54.26% of current ProtoFly directed pairs occur in the historical
Shiu substrate. However, synapse counts on shared pairs are strongly
correlated (Pearson r = 0.963).

Therefore, the current working interpretation is that the difference in
network response is primarily associated with differences in connectome
materialization and/or connectivity annotation rather than the basic
ProtoFly LIF implementation.

This does not establish that either substrate is a complete biological
representation of the living fly brain. It validates the computational
implementation against the published reference model under a matched
historical substrate.

## Neurotransmitter-sign sensitivity diagnostic

The current FlyWire/Codex substrate showed a sharp change in global
recruitment as sugar-sensory stimulation increased from 60 to 100 Hz.

Under the default ProtoFly sign convention:

| Frequency | Active neurons (mean ± SD) |
|---|---:|
| 60 Hz | 3460.2 ± 108.8 |
| 80 Hz | 2283.0 ± 1039.5 |
| 100 Hz | 812.8 ± 68.7 |

The unusually large variance at 80 Hz suggested that the model was
operating near a transition between high- and low-recruitment regimes.

To determine whether this behaviour depended on the assumed functional
sign of glutamatergic neurons, the experiment was repeated while keeping
the connectome, LIF dynamics, sensory neurons, stimulation protocol, and
random-seed schedule fixed.

Two diagnostic conditions were tested:

1. glutamatergic outgoing connections blocked;
2. glutamatergic outgoing connections treated as excitatory.

The frequency-dependent collapse disappeared in both controls.

| Sign condition | 60 Hz | 80 Hz | 100 Hz |
|---|---:|---:|---:|
| Default (GLUT inhibitory) | 3460 | 2283 | 813 |
| GLUT blocked | 6385 | 6422 | 6410 |
| GLUT excitatory | 15757 | 15767 | 15730 |

Treating glutamate as excitatory also produced very high global activity,
exceeding one million spikes per simulated second in these experiments.

### Interpretation

The 60–100 Hz transition is strongly dependent on the functional-sign
assignment of glutamatergic neurons.

This does not establish that the transition is a biological phenomenon.
The FlyWire connectome provides predicted neurotransmitter identities,
but neurotransmitter identity alone does not completely determine the
functional sign of every synapse. In particular, glutamatergic effects
depend on postsynaptic receptor context.

ProtoFly therefore retains glutamate as inhibitory for the Phase-0
whole-brain model. This convention follows the approximation used in the
published Shiu et al. whole-brain Drosophila model and is broadly
consistent with observations that glutamatergic signalling in the fly
brain is often inhibitory.

The alternative sensitivity conditions are diagnostics only and are not
candidate replacements for the default model.

Consequently, frequency-dependent global recruitment in the current
ProtoFly model should be interpreted as a property of the
connectome-constrained model under its stated neurotransmitter-sign
assumptions, rather than as established biological behaviour.

A future receptor-resolved or experimentally constrained effect model
could replace this approximation without changing the underlying
FlyWire connectome.
