from dataclasses import dataclass

from spidet.domain.ActivationFunction import ActivationFunction


@dataclass
class CoefficientsFunction(ActivationFunction):
    codes_for_spikes: bool
