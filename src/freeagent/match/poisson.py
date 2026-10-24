"""Independent Poisson baseline using the same identifiable parameterisation."""
from .dixon_coles import DixonColes
class PoissonModel(DixonColes):
    def __init__(self, decay_days: float = 365.):
        super().__init__(decay_days=decay_days, correction=False)
