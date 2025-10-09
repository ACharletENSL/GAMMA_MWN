# Power-law fits

from dataclasses import dataclass
from scipy.optimize import curve_fit

@dataclass
class FitResult:
    """Results from power-law fit"""
    params: dict
    covariance: np.ndarray
    residuals: np.ndarray
    variable: str
    shock: str
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate fitted function"""
        return self._func(x, **self.params)

class PowerLawFitter:
    """Fits power-law segments to data"""
    
    def fit(self, shock: ShockFront, variable: str, 
            model: str = 'hybrid') -> FitResult:
        """
        Fit evolution of variable
        
        Models:
        - 'planar': X * r^m
        - 'spherical': X_sph
        - 'hybrid': smooth transition between planar & spherical
        """
        r, values = shock.get_radial_profile(variable)
        
        if model == 'hybrid':
            return self._fit_hybrid(r, values, variable)
        elif model == 'planar':
            return self._fit_planar(r, values, variable)
        else:
            return self._fit_spherical(r, values, variable)
    
    def _fit_hybrid(self, r, y, variable):
        """Fit with smooth transition"""
        # Determine initial slope
        i_sample = 20
        m0 = self._logslope(r[0], y[0], r[i_sample], y[i_sample])
        
        # Define fitting function
        def fitfunc(r, X_pl, X_sph, m, s):
            if m < 0:
                return ((X_pl * r**m)**s + X_sph**s)**(1/s)
            else:
                return ((X_pl * r**m)**-s + X_sph**-s)**(-1/s)
        
        # Bounds
        bounds = self._get_bounds(variable, m0)
        
        # Fit
        popt, pcov = curve_fit(fitfunc, r, y, bounds=bounds)
        
        return FitResult(
            params={'X_pl': popt[0], 'X_sph': popt[1], 
                   'm': popt[2], 's': popt[3]},
            covariance=pcov,
            residuals=y - fitfunc(r, *popt),
            variable=variable,
            shock=shock.name
        )
    
    @staticmethod
    def _logslope(x1, y1, x2, y2):
        return (np.log10(y2) - np.log10(y1)) / (np.log10(x2) - np.log10(x1))