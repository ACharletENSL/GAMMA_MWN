# Pipeline design

class AnalysisPipeline:
    """Complete analysis pipeline"""
    
    def __init__(self, run_key: str):
        self.run = Run.load(run_key)
        self.results = {}
        
    def extract_cells(self):
        """Step 1: Extract cell histories"""
        from ..io.cell_extraction import CellExtractor
        extractor = CellExtractor(self.run)
        self.run.cells = extractor.extract()
        return self
    
    def fit_evolution(self, variables: list[str]):
        """Step 2: Fit power-law evolution"""
        fitter = PowerLawFitter()
        self.results['fits'] = {}
        
        for shock_name in ['RS', 'FS']:
            shock = self.run.shocks[shock_name]
            self.results['fits'][shock_name] = {}
            
            for var in variables:
                self.results['fits'][shock_name][var] = \
                    fitter.fit(shock, var)
        return self
    
    def compute_radiation(self, nu_obs, T_obs):
        """Step 3: Calculate radiation"""
        from ..physics.radiation import RadiationCalculator
        calc = RadiationCalculator(self.run.env)
        self.results['radiation'] = calc.compute(
            self.run, nu_obs, T_obs
        )
        return self
    
    def save(self, output_dir: Path):
        """Save all results"""
        # Save fits, radiation, etc.
        pass
    
    def plot_all(self, output_dir: Path):
        """Generate all standard plots"""
        plots = [
            ('lfac', {}),
            ('ShSt', {}),
            ('nu_m', {'xscale': 'r'}),
            # etc.
        ]
        
        for var, kwargs in plots:
            fig = self.run.plot(var, **kwargs)
            fig.savefig(output_dir / f'{var}.png')