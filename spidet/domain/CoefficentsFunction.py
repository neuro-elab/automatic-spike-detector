from dataclasses import dataclass

from spidet.domain.DetectionFunction import DetectionFunction


@dataclass
class CoefficientsFunction(DetectionFunction):
    codes_for_spikes: bool
