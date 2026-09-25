"""Temporal steering dynamics for ProtoFly.

Experimental recordings in walking Drosophila show that bilateral
differences in DNa02 firing rate predict rotational velocity.

This module defines the interface between the biological DNa02 readout
and body rotational dynamics.

Important
---------
The numerical DNa02 -> rotational-velocity calibration is not yet
implemented.

Published experiments support an approximately linear bilateral
relationship over the measured range, together with temporal filtering
between DNa02 activity and steering behaviour. However, ProtoFly does
not currently contain the underlying experimental filter coefficients
or a sufficiently justified universal gain.

Therefore this module must not silently introduce an arbitrary
biological calibration.
"""


class SteeringDynamics:
    """Convert bilateral DNa02 activity into rotational dynamics.

    This class intentionally does not yet implement a numerical
    biological mapping. A calibrated implementation should be added
    only after the corresponding experimental relationship has been
    reconstructed from source data or otherwise justified explicitly.
    """

    def rotational_velocity(
        self,
        left_rate,
        right_rate,
    ):
        """Return predicted rotational velocity from DNa02 activity.

        Parameters
        ----------
        left_rate : float
            Left DNa02 firing rate in Hz.
        right_rate : float
            Right DNa02 firing rate in Hz.

        Returns
        -------
        float
            Rotational velocity.

        Raises
        ------
        NotImplementedError
            Until a biologically grounded numerical mapping has been
            established.
        """

        raise NotImplementedError(
            "DNa02 -> rotational-velocity calibration has not yet "
            "been established for ProtoFly."
        )