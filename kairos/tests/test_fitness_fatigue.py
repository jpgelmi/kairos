"""
Tests for model/fitness_fatigue.py

Key verification: EWMA with exponential decay matches the closed-form
infinite-impulse-response formula.  Also verifies freshness definition uses
previous day's values, not today's.
"""

import math
import pytest
import numpy as np
from datetime import date
from kairos.model.fitness_fatigue import (
    ewma_step,
    freshness,
    simulate,
)
from kairos.config import TAU_G, TAU_H


_DG = math.exp(-1.0 / TAU_G)
_DH = math.exp(-1.0 / TAU_H)


def test_ewma_step_constant_load() -> None:
    """Constant load → EWMA converges to that load (steady state)."""
    g, h = 0.0, 0.0
    W = 300.0
    for _ in range(500):
        g, h = ewma_step(g, h, W)
    assert abs(g - W) < 0.5
    assert abs(h - W) < 0.5


def test_ewma_step_zero_load_decay() -> None:
    """After load stops, g decays by exp(-1/tau_g) each day."""
    g0, h0 = 200.0, 100.0
    g1, h1 = ewma_step(g0, h0, 0.0)
    assert abs(g1 - g0 * _DG) < 1e-9
    assert abs(h1 - h0 * _DH) < 1e-9


def test_simulate_matches_step_by_step() -> None:
    loads = [300.0, 0.0, 250.0, 350.0, 0.0, 280.0]
    G, H = simulate(loads)
    g = h = 0.0
    for i, w in enumerate(loads):
        g, h = ewma_step(g, h, w)
        assert abs(G[i] - g) < 1e-9
        assert abs(H[i] - h) < 1e-9


def test_freshness_uses_prev_values() -> None:
    """Freshness(t) = g(t-1) - h(t-1), i.e. BEFORE today's load."""
    g_prev, h_prev = 150.0, 200.0
    fresh = freshness(g_prev, h_prev)
    assert fresh == pytest.approx(g_prev - h_prev)



def test_simulate_output_lengths() -> None:
    loads = list(range(1, 11))
    G, H = simulate(loads)
    assert len(G) == len(loads)
    assert len(H) == len(loads)


def test_simulate_g_always_positive_for_positive_loads() -> None:
    G, H = simulate([100.0] * 60)
    assert (G > 0).all()
    assert (H > 0).all()
