# Cell, Run, ShockFront classes

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Variable:
    """Represents a physical variable with metadata"""
    name: str
    symbol: str  # LaTeX
    units_cgs: str
    units_norm: str
    compute_func: callable
    dependencies: list[str]
    log_scale: bool = True
    
    def get_label(self, normalized: bool = False) -> str:
        units = self.units_norm if normalized else self.units_cgs
        return f"{self.symbol} {units}" if units else self.symbol

class Cell:
    """Represents a computational cell with history"""
    def __init__(self, cell_id: int, initial_data: dict):
        self.id = cell_id
        self.history = []  # List of timestep data
        self._cache = {}
        
    def add_timestep(self, time: float, data: dict):
        self.history.append({'time': time, **data})
        self._cache.clear()  # Invalidate cache
        
    def get_variable(self, var_name: str, env, 
                     times: Optional[np.ndarray] = None):
        """Compute derived variable with caching"""
        cache_key = (var_name, id(times))
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        var = VARIABLE_REGISTRY[var_name]
        result = var.compute_func(self, env, times)
        self._cache[cache_key] = result
        return result

class ShockFront:
    """Represents a shock front (RS or FS)"""
    def __init__(self, name: str, env):
        self.name = name  # 'RS' or 'FS'
        self.env = env
        self.cells = []
        self.trajectory = []  # (time, radius, properties)
        
    def add_cell(self, cell: Cell, time: float):
        self.cells.append((time, cell))
        
    def fit_evolution(self, variable: str) -> 'FitResult':
        """Fit power-law evolution of variable"""
        from .analysis.fitting import PowerLawFitter
        fitter = PowerLawFitter()
        return fitter.fit(self, variable)

class Run:
    """Represents a full simulation run"""
    def __init__(self, key: str):
        self.key = key
        self.env = MyEnv(key)
        self.cells = {}
        self.shocks = {'RS': ShockFront('RS', self.env),
                      'FS': ShockFront('FS', self.env)}
        
    @classmethod
    def load(cls, key: str, itmin: int = 0, itmax: Optional[int] = None):
        """Factory method to load from disk"""
        run = cls(key)
        run._load_data(itmin, itmax)
        return run
        
    def plot(self, variable: str, **kwargs):
        """Dispatch to appropriate plotter"""
        from .plotting import get_plotter
        plotter = get_plotter(variable)
        return plotter.plot(self, **kwargs)