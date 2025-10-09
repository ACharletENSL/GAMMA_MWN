# Registry pattern for variables

class VariableRegistry:
    def __init__(self):
        self._variables = {}
        
    def register(self, var: Variable):
        self._variables[var.name] = var
        
    def get(self, name: str) -> Variable:
        return self._variables[name]
    
    def __getitem__(self, name: str) -> Variable:
        return self.get(name)

# Global registry
VARIABLE_REGISTRY = VariableRegistry()

# Register variables (can be in separate file)
VARIABLE_REGISTRY.register(Variable(
    name='lfac',
    symbol=r'$\Gamma$',
    units_cgs='',
    units_norm='',
    compute_func=lambda cell, env, times: derive_Lorentz(
        cell.get_raw('vx', times)
    ),
    dependencies=['vx'],
    log_scale=True
))

VARIABLE_REGISTRY.register(Variable(
    name='nu_m',
    symbol=r'$\nu_m$',
    units_cgs=' (Hz)',
    units_norm=r'$/\nu_0$',
    compute_func=derive_nu_m_func,
    dependencies=['rho', 'vx', 'p'],
    log_scale=True
))