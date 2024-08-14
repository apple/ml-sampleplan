import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

LOT_SIZE = 1000
NUM_REPETITIONS = 1000

PATH_PLOTS = Path(__file__).parent.parent / "plots"
PATH_RESULTS = Path(__file__).parent.parent / "results"
PATH_DATA = Path(__file__).parent.parent / "data"

PATH_RESULTS_ASNS = PATH_RESULTS / "asn"


# https://stackoverflow.com/a/49677241
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class QaConfig:
    name: str
    p1: float
    p2: float
    alpha: float
    beta: float
    confidence_interval_half_width: float
    asn_xlim: float


class Approach(Enum):
    CONFIDENCE_INTERVAL_HYPERGEOMETRIC = auto()
    SINGLE_SAMPLING_HYPERGEOMETRIC = auto()
    DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL = auto()
    DOUBLE_SAMPLING_HYPERGEOMETRIC_CURTAILED = auto()
    SPRT_HYPERGEOMETRIC_FULL = auto()
    SPRT_HYPERGEOMETRIC_CURTAILED = auto()

    CONFIDENCE_INTERVAL_BINOMIAL = auto()
    SINGLE_SAMPLING_BINOMIAL = auto()
    DOUBLE_SAMPLING_BINOMIAL_FULL = auto()
    DOUBLE_SAMPLING_BINOMIAL_CURTAILED = auto()
    SPRT_BINOMIAL_CURTAILED = auto()
    SPRT_BINOMIAL_FULL = auto()

    @property
    def distribution(self):
        if "BINOMIAL" in self.name:
            return "binomial"
        elif "HYPERGEOMETRIC" in self.name:
            return "hypergeometric"
        raise RuntimeError("Unknown distribution")

    @property
    def kind(self):
        if "SINGLE_SAMPLING" in self.name:
            return "Single Sampling"
        elif "DOUBLE_SAMPLING" in self.name and "FULL" in self.name:
            return "Double Sampling Full"
        elif "DOUBLE_SAMPLING" in self.name and "CURTAILED" in self.name:
            return "Double Sampling Curtailed"
        elif "SPRT" in self.name and "CURTAILED" in self.name:
            return "Sequential Sampling Curtailed"
        elif "SPRT" in self.name and "FULL" in self.name:
            return "Sequential Sampling Full"
        elif "CONFIDENCE" in self.name:
            return "ci"
        raise RuntimeError("Unknown kind")


CONFIG_STRICT = QaConfig(
    name="strict", p1=0.01, p2=0.03, alpha=0.01, beta=0.05, confidence_interval_half_width=0.01, asn_xlim=0.075
)
CONFIG_RELAXED = QaConfig(
    name="relaxed", p1=0.02, p2=0.05, alpha=0.05, beta=0.1, confidence_interval_half_width=0.02, asn_xlim=0.15
)

QA_CONFIGS = [CONFIG_STRICT, CONFIG_RELAXED]
