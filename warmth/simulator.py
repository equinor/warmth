from multiprocessing import  get_context
import concurrent.futures
from progress.bar import Bar
from pathlib import Path
import math
import time
import numpy as np
from warmth.parameters import Parameters
from warmth.postprocessing import Results_interpolator
from warmth.utils import load_pickle
from .logging import logger
from .forward_modelling import Forward_model
from .build import Builder, single_node,load_node

PARAMETER_FILE = 'parameters.pickle'
NODES_DIR = 'nodes'
GRID_FILE ='grid.pickle'
LOCATION_FILE='locations.npy'



class _nodeWorker:
    def __init__(self, args) -> None:
        self.parameter_path:Path = args[0]
        self.node_path:Path = args[1]
        self.node=load_node(self.node_path)
        self.parameters:Parameters = load_pickle(self.parameter_path)
        pass

    def _pad_sediments(self):
        # pad sed array with bottom (zero-sized?) sediments
        while (self.node.sed.shape[0] < len(self.node.sediments_inputs)-1):
            mm = [np.amax(self.node.sed[:, :, i])
                  for i in range(self.node.sed.shape[2])]
            self.node.sed = np.concatenate(
                [self.node.sed, np.tile(mm, (1, 2, 1))], axis=0)
        return


    def _save_results(self) -> Path:
        filename = self.node._name+"_results"
        filepath = self.parameters.output_path / NODES_DIR/filename
        self.node._dump(filepath)
        return filepath

    def run(self) -> Path:
        try:
            fw = Forward_model(self.parameters, self.node)
            if self.node._full_simulation:
                fw.simulate_single_node()
                self.node.max_time = self.node._depth_out.shape[1]
                self.node.compute_derived_arrays()
            else:
                fw._sedimentation()
            self.node = fw.current_node
            self._pad_sediments()
            self.node.simulated_at = time.time()
            self.node.node_path = self.node_path
            filepath = self._save_results()
            # Delete input node
            self.node_path.unlink(missing_ok=True)
        except Exception as e:
            self.node.error = e
            filepath = self._save_results()
            logger.error(self.node.error)
        return filepath


def runWorker(args):
    worker = _nodeWorker(args)
    result_path = worker.run()
    return result_path

class Simulator:
    """Solving model
    """

    def __init__(self, builder: Builder) -> None:
        """Utilities for simulating nodes

        Parameters
        ----------
        builder : Builder
            model builder
        """
        self._builder = builder
        self.hLith_calibration = False
        self.hc_calibrating = False
        self.forward_modelling = Forward_model(self._builder.parameters,
                                               None)
        self.process = 2
        self.cpu=self.process
        self.simulate_every = 1
        self._builder.parameters.output_path:Path=self._builder.parameters.output_path
        pass

    @property
    def _nodes_path(self):
        return self._builder.parameters.output_path / NODES_DIR

    @property
    def _parameters_path(self):
        return self._builder.parameters.output_path / PARAMETER_FILE
    @property
    def _grid_path(self):
        return self._builder.parameters.output_path / GRID_FILE



    def dump_input_nodes(self, node: single_node):
        filename = node._name + '.pickle'
        p = self._nodes_path / filename

        node._dump(p)
        return p

    def dump_input_data(self, use_mpi=False):
        p = []
        parameter_data_path = self._parameters_path
        if use_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD        
            if (comm.rank==0):
                self._builder.parameters.dump(self._parameters_path)
                if isinstance(self._builder.grid,type(None)) is False:
                    self._builder.grid.dump(self._grid_path)                
                from mpi4py.futures import MPIPoolExecutor
                with MPIPoolExecutor(max_workers=10) as th:
                    futures = [th.submit(self.dump_input_nodes,  i)
                            for i in self._builder.iter_node() if i is not False]
                    for future in concurrent.futures.as_completed(futures):
                        p.append([parameter_data_path, future.result()])
        else:
            self._builder.parameters.dump(self._parameters_path)
            if isinstance(self._builder.grid,type(None)) is False:
                self._builder.grid.dump(self._grid_path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as th:
                futures = [th.submit(self.dump_input_nodes,  i)
                        for i in self._builder.iter_node() if i is not False]
                for future in concurrent.futures.as_completed(futures):
                    p.append([parameter_data_path, future.result()])
        return p

    def setup_directory(self, purge=False):
        if self._builder.parameters.output_path.exists():
            if purge:
                from shutil import rmtree
                rmtree(self._builder.parameters.output_path)
            else:
                raise Exception(
                    f'Output directory {self._builder.parameters.output_path} already exist. Use purge=True to delete existing data')
        self._nodes_path.mkdir(parents=True, exist_ok=True)
        return

    def run(self, save=False,purge=False,parallel=True,use_mpi=True):
        if parallel:
            self._parellel_run(save,purge,use_mpi=use_mpi)
        else:
            if self.simulate_every != 1:
                logger.warning("Serial simulation will run full simulation on all nodes")
            for i in self._builder.iter_node():
                self.forward_modelling.current_node=i
                self.forward_modelling.simulate_single_node()
        return

    def _filter_full_sim(self)->int:
        count=0
        minimum_node_per_axis=5
        if self.simulate_every < 1:
            raise Exception("Invalid input")
        short_axis_count = self._builder.grid.num_nodes_x if self._builder.grid.num_nodes_x <self._builder.grid.num_nodes_y else self._builder.grid.num_nodes_y
        if short_axis_count/self.simulate_every <minimum_node_per_axis:
            self.simulate_every = math.floor(short_axis_count/minimum_node_per_axis)
            logger.warning(f"Simulating every {self.simulate_every} node to make sure each axis has minimum {minimum_node_per_axis} nodes")
        if self.simulate_every ==1:
            pass
        else:
            for index in self._builder.grid.indexing_arr:
                #for rows
                if (index[0] % self.simulate_every > 0):
                    pass
                else:
                    if isinstance(self._builder.nodes[index[0]][index[1]],bool) is False:
                        self._builder.nodes[index[0]][index[1]]._full_simulation = False
                        count+=1
                #for cols
                if (index[1] % self.simulate_every > 0):
                    pass
                else:
                    if isinstance(self._builder.nodes[index[0]][index[1]],bool) is False:
                        self._builder.nodes[index[0]][index[1]]._full_simulation = False
                        count+=1

        if count >0:
            logger.info(f"Setting {count} nodes to partial simulation")
        return count


    def _parellel_run(self, save, purge, use_mpi=False):
        filtered = self._filter_full_sim()
        if use_mpi:
            from mpi4py.futures import MPIPoolExecutor
            from mpi4py import MPI
            comm = MPI.COMM_WORLD        
            if (comm.rank==0):
                self.setup_directory(purge)
            p = self.dump_input_data(use_mpi=use_mpi)
            with MPIPoolExecutor(max_workers=20) as executor:
                results = [executor.submit(runWorker, i) for i in p]
                with Bar('Processing...',check_tty=False, max=len(p)) as bar:
                    for future in concurrent.futures.as_completed(results):
                        bar.next()
                        try:
                            path_result = future.result()
                            n= load_node(path_result) # numerical model error should still resovle
                            if save==False:
                                path_result.unlink()
                            self.put_node_to_grid(n)
                        except Exception as e:
                            logger.error(e)
        else:
            self.setup_directory(purge)
            p = self.dump_input_data(use_mpi=use_mpi)
            with concurrent.futures.ProcessPoolExecutor(mp_context=get_context('spawn')) as executor:
                results = [executor.submit(runWorker, i) for i in p]
                with Bar('Processing...',check_tty=False, max=len(p)) as bar:
                    for future in concurrent.futures.as_completed(results):
                        bar.next()
                        try:
                            path_result = future.result()
                            n= load_node(path_result) # numerical model error should still resovle
                            if save==False:
                                path_result.unlink()
                            self.put_node_to_grid(n)
                        except Exception as e:
                            logger.error(e)
        # pick up node with no results (failed)
        #if comm.rank==0:                            
        for node_path in self._nodes_path.iterdir():
            str_f = str(node_path)
            if str_f.endswith(".pickle"):
                n=load_node(node_path)
                if save==False:
                    node_path.unlink()
                self.put_node_to_grid(n)
                logger.warning(f"No result file for node X:{n.X}, Y:{n.Y}")
        if save==False:
            from shutil import rmtree
            rmtree(self._builder.parameters.output_path)
        # if filtered >0:
        #     logger.info(f"Interpolating results back to {filtered} partial simulated nodes")
        #     interp_res= Results_interpolator(self._builder)
        #     interp_res.run()
        return
    def put_node_to_grid(self,node:single_node):
        node.clear_unused_data()
        self._builder.nodes[node.indexer[0]][node.indexer[1]]=node
        return


