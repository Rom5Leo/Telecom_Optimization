"""Antenna-tilt optimization: network model and QUBO builders (one-hot and compact)."""

from telecomopt.antenna.tilt import (
    AntennaNetwork,
    antenna_tilt_qubo,
    decode_onehot,
    antenna_tilt_qubo_compact,
    decode_compact,
)

__all__ = [
    "AntennaNetwork",
    "antenna_tilt_qubo",
    "decode_onehot",
    "antenna_tilt_qubo_compact",
    "decode_compact",
]
