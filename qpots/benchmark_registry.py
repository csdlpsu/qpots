"""Registry of reusable BoTorch benchmark functions exposed by qPOTS."""

from __future__ import annotations

from collections.abc import Callable

from botorch.test_functions.multi_objective import (
    BNH,
    C2DTLZ2,
    CONSTR,
    DH1,
    DH2,
    DH3,
    DH4,
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ7,
    GMM,
    MW7,
    OSY,
    SRN,
    ZDT1,
    ZDT2,
    ZDT3,
    BraninCurrin,
    CarSideImpact,
    ConstrainedBraninCurrin,
    DiscBrake,
    Penicillin,
    ToyRobust,
    VehicleSafety,
    WeldedBeam,
)
from botorch.test_functions.synthetic import Branin

BenchmarkFactory = Callable[[int, int], object]


BENCHMARK_REGISTRY: dict[str, BenchmarkFactory] = {
    "branincurrin": lambda dim, nobj: BraninCurrin(negate=True),
    "dtlz1": lambda dim, nobj: DTLZ1(dim=dim, num_objectives=nobj, negate=True),
    "dtlz2": lambda dim, nobj: DTLZ2(dim=dim, num_objectives=nobj, negate=True),
    "c2dtlz2": lambda dim, nobj: C2DTLZ2(dim=dim, num_objectives=nobj, negate=True),
    "dtlz3": lambda dim, nobj: DTLZ3(dim=dim, num_objectives=nobj, negate=False),
    "dtlz4": lambda dim, nobj: DTLZ4(dim=dim, num_objectives=nobj, negate=False),
    "dtlz5": lambda dim, nobj: DTLZ5(dim=dim, num_objectives=nobj, negate=False),
    "dtlz7": lambda dim, nobj: DTLZ7(dim=dim, num_objectives=nobj, negate=True),
    "dh1": lambda dim, nobj: DH1(dim=dim, negate=True),
    "dh2": lambda dim, nobj: DH2(dim=dim, negate=True),
    "dh3": lambda dim, nobj: DH3(dim=dim, negate=True),
    "dh4": lambda dim, nobj: DH4(dim=dim, negate=True),
    "gmm": lambda dim, nobj: GMM(num_objectives=nobj, negate=True),
    "penicillin": lambda dim, nobj: Penicillin(negate=True),
    "vehicle": lambda dim, nobj: VehicleSafety(negate=True),
    "carside": lambda dim, nobj: CarSideImpact(negate=True),
    "zdt1": lambda dim, nobj: ZDT1(dim=dim, num_objectives=nobj, negate=False),
    "zdt2": lambda dim, nobj: ZDT2(dim=dim, num_objectives=nobj, negate=False),
    "zdt3": lambda dim, nobj: ZDT3(dim=dim, num_objectives=nobj, negate=True),
    "constrainedbc": lambda dim, nobj: ConstrainedBraninCurrin(negate=True),
    "discbrake": lambda dim, nobj: DiscBrake(negate=True),
    "mw7": lambda dim, nobj: MW7(dim=dim, negate=True),
    "osy": lambda dim, nobj: OSY(negate=True),
    "weldedbeam": lambda dim, nobj: WeldedBeam(negate=True),
    "branin": lambda dim, nobj: Branin(negate=True),
    "toyrobust": lambda dim, nobj: ToyRobust(negate=False),
    "srn": lambda dim, nobj: SRN(negate=False),
    "bnh": lambda dim, nobj: BNH(negate=False),
    "constr": lambda dim, nobj: CONSTR(negate=True),
}


def available_benchmarks() -> tuple[str, ...]:
    """Return the registered benchmark names in alphabetical order."""
    return tuple(sorted(BENCHMARK_REGISTRY))


def create_benchmark(name: str, *, dim: int, nobj: int) -> object:
    """Create a registered benchmark by its case-insensitive name."""
    normalized_name = name.lower()
    try:
        factory = BENCHMARK_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown test function '{normalized_name}'. Check the available functions."
        ) from exc
    return factory(dim, nobj)
