"""Validate one ProtoFly LIF neuron using published-style stimulation."""

from brian2 import (
    NeuronGroup,
    PoissonInput,
    SpikeMonitor,
    Network,
    mV,
    ms,
    Hz,
)

from protofly.neural.lif import (
    DEFAULT_PARAMS,
    EQUATIONS,
    THRESHOLD,
    RESET,
)

params = DEFAULT_PARAMS

neuron = NeuronGroup(
    1,
    model=EQUATIONS,
    threshold=THRESHOLD,
    reset=RESET,
    refractory="rfc",
    method="linear",
    namespace=params,
)

neuron.v = params["v_0"]
neuron.g = 0 * mV
neuron.rfc = params["t_rfc"]

spikes = SpikeMonitor(neuron)

# Published-model stimulation:
# 150 Hz Poisson input, weight = w_syn * 250
stimulus = PoissonInput(
    neuron,
    "v",
    N=1,
    rate=150 * Hz,
    weight=params["w_syn"] * 250,
)

# Shiu et al. remove refractory period for stimulated targets.
neuron.rfc = 0 * ms

net = Network(neuron, spikes, stimulus)

net.run(100 * ms)

print("Single-neuron Poisson stimulation test")
print(f"Final voltage: {neuron.v[0] / mV:.2f} mV")
print(f"Spikes:        {spikes.num_spikes}")

if spikes.num_spikes > 0:
    print("\nPASS: ProtoFly's first neuron fired.")
else:
    print("\nFAIL: neuron did not fire.")
