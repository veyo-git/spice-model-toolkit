"""Shared fixtures for SPICE model toolkit tests."""

import pytest
import numpy as np
from src.device.mosfet import (
    MOSFETLevel1, MOSFETLevel3,
    MOSFETParamsLevel1, MOSFETParamsLevel3,
)


@pytest.fixture
def params_l1():
    """Standard Level-1 NMOS parameters."""
    return MOSFETParamsLevel1(
        W=10e-6, L=0.18e-6, VTH0=0.45, GAMMA=0.4,
        PHI=0.65, KP=200e-6, LAMBDA=0.05,
    )


@pytest.fixture
def model_l1(params_l1):
    """Level-1 MOSFET model instance."""
    return MOSFETLevel1(params_l1)


@pytest.fixture
def params_l3():
    """Standard Level-3 NMOS parameters."""
    return MOSFETParamsLevel3(
        W=10e-6, L=0.18e-6, VTH0=0.45, GAMMA=0.4,
        PHI=0.65, U0=400, THETA=0.05, VSAT=1e7,
        KAPPA=0.5, LAMBDA=0.05, ETA0=0.05, N0=1.5,
        RD=20, RS=20,
    )


@pytest.fixture
def model_l3(params_l3):
    """Level-3 MOSFET model instance."""
    return MOSFETLevel3(params_l3)


@pytest.fixture
def vgs_sweep():
    """Standard Vgs sweep array."""
    return np.linspace(-0.3, 2.5, 101)


@pytest.fixture
def vds_sweep():
    """Standard Vds sweep array."""
    return np.linspace(0.0, 2.5, 101)
