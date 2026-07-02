from .instantiation import instantiate_callbacks, instantiate_loggers
from .data import (
    broadcast, 
    gumbel_sample,
    convert_to_numpy, 
    convert_to_torch, 
    continuous_to_discrete,
    CoupleDataset,
    RepeatedDataset,
    SampledCoupleDataset,
    optimize_coupling
)
from .visualization import fig2img
