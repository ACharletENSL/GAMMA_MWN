# Base plotting classes

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class BasePlotter(ABC):
    """Base class for all plotters"""
    
    def __init__(self, variable: Variable):
        self.variable = variable
        
    @abstractmethod
    def plot(self, data, ax=None, **kwargs):
        """Create the plot"""
        pass
    
    def _setup_axis(self, ax, xscale='linear', yscale='log', 
                   xlabel='', normalized=False):
        """Common axis setup"""
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.variable.get_label(normalized))
        
        return ax

class TimeSeriesPlotter(BasePlotter):
    """Plot variable evolution over time"""
    
    def plot(self, run: Run, shock: str = 'RS', 
             xscale='t0', normalized=True, ax=None, **kwargs):
        
        ax = self._setup_axis(ax, xlabel=self._get_xlabel(xscale))
        
        shock_obj = run.shocks[shock]
        times, values = shock_obj.get_time_series(self.variable.name)
        
        if normalized:
            values = self._normalize(values, run.env)
        
        x = self._transform_time(times, xscale, run.env)
        ax.plot(x, values, color=SHOCK_COLORS[shock], **kwargs)
        
        return ax
    
    def _get_xlabel(self, xscale):
        labels = {
            't0': r'$t/t_0$',
            'T': r'$\bar{T}$',
            'r': r'$r/R_0$'
        }
        return labels.get(xscale, 't')
    
    @staticmethod
    def _normalize(values, env):
        # Implementation depends on variable
        pass

# Factory function
def get_plotter(variable: str) -> BasePlotter:
    """Get appropriate plotter for variable"""
    var = VARIABLE_REGISTRY[variable]
    
    # Could be more sophisticated
    if variable in ['nu_m', 'Lth', 'lfac']:
        return TimeSeriesPlotter(var)
    elif variable in ['rho', 'p']:
        return SnapshotPlotter(var)
    else:
        return TimeSeriesPlotter(var)  # Default