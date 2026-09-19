"""Test activity propagation across one real FlyWire connection."""

from brian2 import (
    NeuronGroup,
    Synapses,
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

# Real FlyWire connection
PRE_ROOT_ID = 720575940621619627
POST_ROOT_ID = 720575940606756030

SYN_COUNT = 1482
SIGN = 1  # presynaptic neuron predicted ACH

params = DEFAULT_PARAMS

# Two neurons:
# index 0 = PRE_ROOT_ID
# index 1 = POST_ROOT_ID
neurons = NeuronGroup(
    2,
    model=EQUATIONS,
    threshold=THRESHOLD,
    reset=RESET,
    refractory="rfc",
    method="linear",
    namespace=params,
)

neurons.v = params["v_0"]
neurons.g = 0 * mV
neurons.rfc = params["t_rfc"]

# Real A -> B FlyWire connection
synapse = Synapses(
    neurons,
    neurons,
    model="w : volt",
    on_pre="g += w",
    delay=params["t_dly"],
)

synapse.connect(i=[0], j=[1])
synapse.w = SIGN * SYN_COUNT * params["w_syn"]

# Stimulate A only
stimulus = PoissonInput(
    neurons[0],
    "v",
    N=1,
    rate=150 * Hz,
    weight=params["w_syn"] * 250,
)

# Match published stimulation treatment for A.
neurons[0].rfc = 0 * ms

spikes = SpikeMonitor(neurons)

net = Network(neurons, synapse, stimulus, spikes)

# --------------------------------
# Condition 1: connection intact
# --------------------------------

net.store("initial")

net.run(100 * ms)

connected_pre = int(spikes.count[0])
connected_post = int(spikes.count[1])

# --------------------------------
# Condition 2: connection removed
# --------------------------------

net.restore("initial")

synapse.w = 0 * mV

net.run(100 * ms)

disconnected_pre = int(spikes.count[0])
disconnected_post = int(spikes.count[1])

print("Real FlyWire connection causal test")
print("===================================")

print(f"A: {PRE_ROOT_ID}")
print(f"B: {POST_ROOT_ID}")
print(f"Synapses: {SYN_COUNT}")
print(f"Original weight: {SIGN * SYN_COUNT * float(params['w_syn'] / mV):.2f} mV")

print("\nCONNECTED")
print(f"A spikes: {connected_pre}")
print(f"B spikes: {connected_post}")

print("\nCONNECTION REMOVED")
print(f"A spikes: {disconnected_pre}")
print(f"B spikes: {disconnected_post}")

if (
    connected_pre > 0
    and connected_post > 0
    and disconnected_pre > 0
    and disconnected_post == 0
):
    print("\nPASS: B's activity depends causally on the FlyWire A -> B connection.")
else:
    print("\nCHECK: causal-control result was not as expected.")