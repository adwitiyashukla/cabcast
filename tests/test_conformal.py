from __future__ import annotations

import numpy as np
import pytest

from urbanflow.models.conformal import (
    ConformalizedQuantile,
    MondrianConformalizedQuantile,
    SplitConformal,
    conformal_quantile,
)

ALPHA = 0.1


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(42)
    n = 12000
    mu = rng.uniform(1.0, 60.0, n)
    y = rng.poisson(mu).astype(float)
    lo = mu * 0.8
    hi = mu * 1.2
    half = n // 2
    return y, lo, hi, mu, slice(0, half), slice(half, n)


def test_finite_sample_quantile_is_conservative():
    scores = np.arange(100, dtype=float)
    q = conformal_quantile(scores, alpha=0.1)
    assert q >= np.quantile(scores, 0.9)


def test_cqr_achieves_nominal_coverage(sample):
    y, lo, hi, _, cal, tst = sample
    cq = ConformalizedQuantile(ALPHA).calibrate(y[cal], lo[cal], hi[cal])
    low, up = cq.interval(lo[tst], hi[tst])
    assert np.mean((y[tst] >= low) & (y[tst] <= up)) == pytest.approx(1 - ALPHA, abs=0.02)


def test_split_conformal_achieves_nominal_coverage(sample):
    y, _, _, mu, cal, tst = sample
    sc = SplitConformal(ALPHA).calibrate(y[cal], mu[cal])
    low, up = sc.interval(mu[tst])
    assert np.mean((y[tst] >= low) & (y[tst] <= up)) == pytest.approx(1 - ALPHA, abs=0.02)


def test_mondrian_improves_conditional_coverage(sample):
    y, lo, hi, mu, cal, tst = sample
    cq = ConformalizedQuantile(ALPHA).calibrate(y[cal], lo[cal], hi[cal])
    mc = MondrianConformalizedQuantile(ALPHA, 6).calibrate(y[cal], lo[cal], hi[cal], mu[cal])

    g_lo, g_hi = cq.interval(lo[tst], hi[tst])
    m_lo, m_hi = mc.interval(lo[tst], hi[tst], mu[tst])

    bins = np.digitize(mu[tst], np.quantile(mu[tst], [0.2, 0.4, 0.6, 0.8]))
    spread = []
    for low, up in ((g_lo, g_hi), (m_lo, m_hi)):
        cov = [np.mean((y[tst][bins == b] >= low[bins == b]) & (y[tst][bins == b] <= up[bins == b])) for b in range(5)]
        spread.append(max(cov) - min(cov))
    assert spread[1] <= spread[0]


def test_intervals_are_ordered_and_non_negative(sample):
    y, lo, hi, mu, cal, tst = sample
    mc = MondrianConformalizedQuantile(ALPHA, 6).calibrate(y[cal], lo[cal], hi[cal], mu[cal])
    low, up = mc.interval(lo[tst], hi[tst], mu[tst])
    assert (up >= low).all()
    assert (low >= 0).all()


def test_tighter_alpha_widens_the_interval(sample):
    y, lo, hi, _, cal, tst = sample
    wide = ConformalizedQuantile(0.01).calibrate(y[cal], lo[cal], hi[cal])
    narrow = ConformalizedQuantile(0.20).calibrate(y[cal], lo[cal], hi[cal])
    assert wide.qhat_ > narrow.qhat_
