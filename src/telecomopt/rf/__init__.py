"""RF propagation models (Ericsson LTE)."""

from telecomopt.rf.propagation import (
    RFParams, DEFAULT_PARAMS, to_linear, to_db,
    path_loss_db, elevation_angle_deg, elevation_gain_db,
    path_gain_db, sinr, spectral_efficiency,
)

__all__ = [
    "RFParams", "DEFAULT_PARAMS", "to_linear", "to_db",
    "path_loss_db", "elevation_angle_deg", "elevation_gain_db",
    "path_gain_db", "sinr", "spectral_efficiency",
]
