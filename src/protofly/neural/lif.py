"""Leaky integrate-and-fire dynamics used by ProtoFly.

Initial parameters reproduce the whole-brain Drosophila model
of Shiu et al. No learning or plasticity is implemented here.
"""

from brian2 import mV, ms


DEFAULT_PARAMS = {
    "v_0": -52 * mV,
    "v_rst": -52 * mV,
    "v_th": -45 * mV,
    "t_mbr": 20 * ms,
    "tau": 5 * ms,
    "t_rfc": 2.2 * ms,
    "t_dly": 1.8 * ms,
    "w_syn": 0.275 * mV,
}


EQUATIONS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
"""

THRESHOLD = "v > v_th"

RESET = """
v = v_rst
g = 0 * mV
"""
