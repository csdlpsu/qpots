import math

import pytest
import torch

from qpots.utils.tc_utils import corr_and_total_correlation
from qpots.utils.utils import corr_and_total_correlation as legacy_total_correlation


@pytest.mark.parametrize(
    "implementation", [corr_and_total_correlation, legacy_total_correlation]
)
def test_independent_tasks_have_zero_total_correlation(implementation):
    correlation, total = implementation(torch.eye(3, dtype=torch.float64))

    assert torch.allclose(correlation, torch.eye(3, dtype=torch.float64))
    assert total.item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "implementation", [corr_and_total_correlation, legacy_total_correlation]
)
def test_correlated_tasks_have_positive_total_correlation(implementation):
    covariance = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)

    _, total = implementation(covariance, jitter=0.0)

    assert total.item() == pytest.approx(-0.5 * math.log(0.75))


def test_total_correlation_preserves_batch_dimensions():
    covariance = torch.stack(
        (
            torch.eye(2, dtype=torch.float64),
            torch.tensor([[1.0, 0.25], [0.25, 1.0]], dtype=torch.float64),
        )
    )

    correlation, total = corr_and_total_correlation(covariance)

    assert correlation.shape == (2, 2, 2)
    assert total.shape == (2,)
    assert total[0].item() == pytest.approx(0.0, abs=1e-12)
    assert total[1].item() > 0.0
