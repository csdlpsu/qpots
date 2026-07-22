from importlib.metadata import version

import qpots
from qpots.acquisition import Acquisition
from qpots.function import EvaluationResult, Function
from qpots.model_object import ModelObject
from qpots.runner import IterationResult, OptimizationResult, QPOTSConfig, QPOTSRunner
from qpots.tsemo_runner import TSEMORunner


def test_supported_interfaces_are_exported_from_package_root():
    expected_exports = {
        "Acquisition": Acquisition,
        "EvaluationResult": EvaluationResult,
        "Function": Function,
        "IterationResult": IterationResult,
        "ModelObject": ModelObject,
        "OptimizationResult": OptimizationResult,
        "QPOTSConfig": QPOTSConfig,
        "QPOTSRunner": QPOTSRunner,
        "TSEMORunner": TSEMORunner,
    }

    for name, interface in expected_exports.items():
        assert getattr(qpots, name) is interface
        assert name in qpots.__all__


def test_package_version_comes_from_installed_metadata():
    assert qpots.__version__ == version("qpots")


def test_public_export_list_has_no_duplicates():
    assert len(qpots.__all__) == len(set(qpots.__all__))
