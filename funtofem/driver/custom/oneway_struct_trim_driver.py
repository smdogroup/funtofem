
__all__ = ["OnewayStructTrimDriver"]

from ..oneway_struct_driver import OnewayStructDriver
import numpy as np
from mpi4py import MPI

class OnewayStructTrimDriver(OnewayStructDriver):

    """
    goal of this class is to run oneway-coupled sizing
    while also trimming the wing using a set of loads obtained 
    when pull up might not be satisfied..

    aero / struct loads are scaled up as an AOA variable is scaled up
    for all uncoupled scenarios (have scenario.coupled = False)

    the user should setup a load factor based composite function
    such as lift - load_factor * weight = 0
    where load_factor is user-specified for that scenario
    """

    def __init__(
        self,
        solvers,
        model,
        initial_trim_dict:dict,
        transfer_settings=None,
        nprocs=None,
        fun3d_dir=None,
        external_shape=False,
        timing_file=None,
    ):
        
        # create base class OnewayStructDriver
        super(OnewayStructTrimDriver,self).__init__(
            solvers,
            model,
            transfer_settings,
            nprocs,
            fun3d_dir,
            external_shape,
            timing_file,
        )
        
        # get data from scenario initial trim dict
        # assumed to hold initial values for (not case sensitive)
        # and lift is C_L normalized by area and qinf
        # {scenario_name : {'cl' : cl_0, 'AOA' : AOA_0}}
        # only required to put scenarios which are uncoupled here
        # with scenario.coupled = False boolean

        self.initial_trim_dict = initial_trim_dict

        # save initial struct loads vectors for each scenario
        self._orig_struct_loads = {}
        self.uncoupled_scenarios = [scenario for scenario in model.scenarios if not(scenario.coupled)]
        for scenario in self.uncoupled_scenarios:
            self._orig_struct_loads[scenario.name] = {}
            for body in model.bodies:
                struct_loads = body.struct_loads[scenario.id]
                self._orig_struct_loads[scenario.name][body.name] = struct_loads * 1.0

    @classmethod
    def prime_loads_from_file(
        cls,
        filename,
        solvers,
        initial_trim_dict,
        model,
        nprocs,
        transfer_settings,
        external_shape=False,
        init_transfer=False,
        timing_file=None,
    ):
        # same as base class prime_loads_from_file but with extra input argument
        # aka initial_trim_dict
        comm = solvers.comm
        world_rank = comm.Get_rank()
        if world_rank < nprocs:
            color = 1
        else:
            color = MPI.UNDEFINED
        tacs_comm = comm.Split(color, world_rank)

        # initialize transfer settings
        comm_manager = solvers.comm_manager

        # read in the loads from the file
        loads_data = model._read_aero_loads(comm, filename)

        # initialize the transfer scheme then distribute aero loads
        for body in model.bodies:
            body.initialize_transfer(
                comm=comm,
                struct_comm=tacs_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=transfer_settings,
            )
            for scenario in model.scenarios:
                body.initialize_variables(scenario)
                assert scenario.steady
            body._distribute_aero_loads(loads_data, steady=True)

        tacs_driver = cls(
            solvers,
            model,
            initial_trim_dict,
            nprocs=nprocs,
            external_shape=external_shape,
            timing_file=timing_file,
        )
        if init_transfer:
            tacs_driver._transfer_fixed_aero_loads()

        return tacs_driver

    def solve_forward(self):

        # scale up the loads by new AOA vs previous AOA
        # note this only works for steady-state case
        for scenario in self.uncoupled_scenarios:
            orig_AOA = self.initial_trim_dict[scenario.name]['AOA']
            new_AOA = scenario.get_variable('AOA').value.real
            for body in self.model.bodies:
                orig_struct_loads = self._orig_struct_loads[scenario.name][body.name]
                body.struct_loads[scenario.id][:] = (orig_struct_loads * new_AOA / orig_AOA)[:]

        # now do super class solve_forward which will include
        # transferring fixed aero loads to the new struct loads and then linear static solve
        super(OnewayStructTrimDriver,self).solve_forward()

        # compute new lift values, for function name cl
        for scenario in self.uncoupled_scenarios:
            orig_cl = self.initial_trim_dict[scenario.name]['cl']
            orig_AOA = self.initial_trim_dict[scenario.name]['AOA']
            new_AOA = scenario.get_variable('AOA').value.real

            for func in scenario.functions:
                if func.name == 'cl':
                    func.value = orig_cl * new_AOA / orig_AOA

        # composite functions are evaluated in the OptimizationManager FYI and will also be updated after this..

    def solve_adjoint(self):

        # do super class solve_adjoint (same adjoint solve as before)
        # since modified f_A is output of adjoint solve and not coupled...
        # so doesn't matter really
        super(OnewayStructTrimDriver,self).solve_adjoint()

    def _solve_steady_adjoint(self, scenario, bodies):
        super()._solve_steady_adjoint(scenario, bodies)

        # get additional derivative terms for custom
        self._get_custom_derivatives(scenario)

    def _solve_unsteady_adjoint(self, scenario, bodies):
        super()._solve_unsteady_adjoint(scenario, bodies)

        # get additional derivative terms for custom
        self._get_custom_derivatives(scenario)

    def _get_custom_derivatives(self, scenario):
        """get custom trim derivatives, this is used in the """

        orig_cl = self.initial_trim_dict[scenario.name]['cl']
        orig_AOA = self.initial_trim_dict[scenario.name]['AOA']
        aoa_var = scenario.get_variable('AOA')

        # since mass not adjoint function only iterate over these guys
        adjoint_functions = [func for func in scenario.functions if func.adjoint]
        for ifunc,func in enumerate(adjoint_functions):
            if func.name == 'cl':
                func.derivatives[aoa_var] = orig_cl / orig_AOA 
                continue

            # account for changing loads terms in AOA
            AOA_deriv = 0.0
            for body in self.model.bodies:
                struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                func_fs_ajp = struct_loads_ajp[:,ifunc]
                orig_struct_loads = self._orig_struct_loads[scenario.name][body.name]
                free_struct_loads = orig_struct_loads * 1.0

                # this didn't change anything
                # # temp - try to zero BCs at ext force locations?
                # structural = self.solvers.structural
                # assembler = structural.assembler
                # ext_force = structural.ext_force
                # ext_force_array = ext_force.getArray()
                # ndof = assembler.getVarsPerNode()
                # for i in range(3): 
                #     # set only 0,1,2 into ext_force, then we will apply BCs to zero out dirichlet BCs
                #     ext_force_array[i::ndof] = free_struct_loads[i::3]
                # assembler.setBCs(ext_force) # zero out forces at dirichlet BCs (since have no effect on structure)
                # for i in range(3):
                #     free_struct_loads[i::3] = ext_force_array[i::ndof]         

                AOA_deriv += np.dot(func_fs_ajp, free_struct_loads / orig_AOA)
            
            # add across all processors then reduce
            global_derivative = self.comm.reduce(AOA_deriv, op=MPI.SUM, root=0)
            global_derivative = self.comm.bcast(global_derivative)

            func.derivatives[aoa_var] = global_derivative