from pathlib import Path
from .parameters import Parameters
from .utils import load_pickle
from .build import Builder, load_node
from .simulator import Simulator
from .logging import logger

class Model:
    """Main model class
    """
    def __init__(self):
        self._parameters = Parameters()
        self._builder = Builder(self.parameters)
        self._simulator = Simulator(self.builder)

    @property
    def parameters(self)-> Parameters:
        """Model parameters

        Returns
        -------
        Parameters
            Model parameters
        """
        return self._parameters
    @property
    def builder(self)-> Builder:
        """Model builder

        Returns
        -------
        Builder
            Model builder
        """
        return self._builder
    @property
    def simulator(self)-> Simulator:
        """Model simulator

        Returns
        -------
        Simulator
            Model simulator
        """
        return self._simulator
    def _load_nodes(self):
        for node_path in self.simulator._nodes_path.iterdir():
            node = load_node(node_path)
            self.simulator.put_node_to_grid(node)
        return

    def _load_1D_results(self):
        logger.debug(f"Loading model from {self.parameters.out_path}")
        self._load_nodes()
        return
    
    def load(self,path:Path|str):
        """Load model and override current model

        Parameters
        ----------
        path : Path | str
            path to model data
        """
        if isinstance(path,str):
            path = Path(path)
        self.parameters.out_path = path
        self.parameters = load_pickle(self.simulator._parameters_path)
        try:
            self.builder.grid = load_pickle(self.simulator._grid_path)
        except:
            pass
        self.parameters.out_path = path
        self._load_1D_results()
        return




