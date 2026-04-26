#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from botorch.test_functions.multi_fidelity import (  # noqa: F401
        AugmentedBranin,
        AugmentedHartmann,
        AugmentedRosenbrock,
    )
    from botorch.test_functions.multi_objective import (  # noqa: F401
        BNH,
        BraninCurrin,
        C2DTLZ2,
        CarSideImpact,
        CONSTR,
        ConstrainedBraninCurrin,
        DiscBrake,
        DTLZ1,
        DTLZ2,
        DTLZ3,
        DTLZ4,
        DTLZ5,
        DTLZ7,
        GMM,
        MW7,
        OSY,
        Penicillin,
        SRN,
        ToyRobust,
        VehicleSafety,
        WeldedBeam,
        ZDT1,
        ZDT2,
        ZDT3,
    )
    from botorch.test_functions.multi_objective_multi_fidelity import (  # noqa: F401
        MOMFBraninCurrin,
        MOMFPark,
    )
    from botorch.test_functions.synthetic import (  # noqa: F401
        Ackley,
        Beale,
        Branin,
        Bukin,
        Cosine8,
        DixonPrice,
        DropWave,
        EggHolder,
        Griewank,
        Hartmann,
        HolderTable,
        Levy,
        Michalewicz,
        Powell,
        PressureVessel,
        Rastrigin,
        Rosenbrock,
        Shekel,
        SixHumpCamel,
        SpeedReducer,
        StyblinskiTang,
        SyntheticTestFunction,
        TensionCompressionString,
        ThreeHumpCamel,
        WeldedBeamSO,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AugmentedBranin": (".multi_fidelity", "AugmentedBranin"),
    "AugmentedHartmann": (".multi_fidelity", "AugmentedHartmann"),
    "AugmentedRosenbrock": (".multi_fidelity", "AugmentedRosenbrock"),
    "BNH": (".multi_objective", "BNH"),
    "BraninCurrin": (".multi_objective", "BraninCurrin"),
    "C2DTLZ2": (".multi_objective", "C2DTLZ2"),
    "CarSideImpact": (".multi_objective", "CarSideImpact"),
    "CONSTR": (".multi_objective", "CONSTR"),
    "ConstrainedBraninCurrin": (".multi_objective", "ConstrainedBraninCurrin"),
    "DiscBrake": (".multi_objective", "DiscBrake"),
    "DTLZ1": (".multi_objective", "DTLZ1"),
    "DTLZ2": (".multi_objective", "DTLZ2"),
    "DTLZ3": (".multi_objective", "DTLZ3"),
    "DTLZ4": (".multi_objective", "DTLZ4"),
    "DTLZ5": (".multi_objective", "DTLZ5"),
    "DTLZ7": (".multi_objective", "DTLZ7"),
    "GMM": (".multi_objective", "GMM"),
    "MW7": (".multi_objective", "MW7"),
    "OSY": (".multi_objective", "OSY"),
    "Penicillin": (".multi_objective", "Penicillin"),
    "SRN": (".multi_objective", "SRN"),
    "ToyRobust": (".multi_objective", "ToyRobust"),
    "VehicleSafety": (".multi_objective", "VehicleSafety"),
    "WeldedBeam": (".multi_objective", "WeldedBeam"),
    "ZDT1": (".multi_objective", "ZDT1"),
    "ZDT2": (".multi_objective", "ZDT2"),
    "ZDT3": (".multi_objective", "ZDT3"),
    "MOMFBraninCurrin": (".multi_objective_multi_fidelity", "MOMFBraninCurrin"),
    "MOMFPark": (".multi_objective_multi_fidelity", "MOMFPark"),
    "Ackley": (".synthetic", "Ackley"),
    "Beale": (".synthetic", "Beale"),
    "Branin": (".synthetic", "Branin"),
    "Bukin": (".synthetic", "Bukin"),
    "Cosine8": (".synthetic", "Cosine8"),
    "DixonPrice": (".synthetic", "DixonPrice"),
    "DropWave": (".synthetic", "DropWave"),
    "EggHolder": (".synthetic", "EggHolder"),
    "Griewank": (".synthetic", "Griewank"),
    "Hartmann": (".synthetic", "Hartmann"),
    "HolderTable": (".synthetic", "HolderTable"),
    "Levy": (".synthetic", "Levy"),
    "Michalewicz": (".synthetic", "Michalewicz"),
    "Powell": (".synthetic", "Powell"),
    "PressureVessel": (".synthetic", "PressureVessel"),
    "Rastrigin": (".synthetic", "Rastrigin"),
    "Rosenbrock": (".synthetic", "Rosenbrock"),
    "Shekel": (".synthetic", "Shekel"),
    "SixHumpCamel": (".synthetic", "SixHumpCamel"),
    "SpeedReducer": (".synthetic", "SpeedReducer"),
    "StyblinskiTang": (".synthetic", "StyblinskiTang"),
    "SyntheticTestFunction": (".synthetic", "SyntheticTestFunction"),
    "TensionCompressionString": (".synthetic", "TensionCompressionString"),
    "ThreeHumpCamel": (".synthetic", "ThreeHumpCamel"),
    "WeldedBeamSO": (".synthetic", "WeldedBeamSO"),
}

__all__ = list(_LAZY_IMPORTS.keys())


if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:  # pragma: no cover
        if name in _LAZY_IMPORTS:
            rel_module, attr = _LAZY_IMPORTS[name]
            module = importlib.import_module(rel_module, __name__)
            value = getattr(module, attr)
            globals()[name] = value
            return value
        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__() -> list[str]:  # pragma: no cover
        return __all__
