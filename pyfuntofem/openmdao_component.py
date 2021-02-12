
from __future__ import print_function
from openmdao.api import ExplicitComponent

class FuntofemComponent(ExplicitComponent):
    """
    OpenMDAO component that wraps pyfuntofem

    """

    def initialize(self):
        self.options.declare('driver')

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)
        self.driver = self.options['driver']
        self.model  = self.driver.model

        f2f_var_list  = self.model.get_variables()
        f2f_func_list = self.model.get_functions()

        self.var_list = []
        for var in f2f_var_list:
            if var.scenario is not None:
                om_name = 'scenario'+str(var.scenario)+'_'+var.name
            if var.body is not None:
                om_name = 'body'+str(var.body)+'_'+var.name
            self.add_input(om_name, var.value)
            self.var_list.append(om_name)

        self.add_output('f', shape=len(f2f_func_list))

    def compute(self, inputs, outputs):
        f2f_var_list = self.model.get_variables()
        for ivar, var in enumerate(f2f_var_list):
            var.value = inputs[self.var_list[ivar]]
            if self.comm.Get_rank()==0:
                print('F2F Variable:', self.var_list[ivar], var.value)
        self.driver.solve_forward()
        funcs = self.model.get_functions()

        for i in range(len(funcs)):
            outputs['f'][i] = funcs[i].value
            if self.comm.Get_rank()==0:
                print('F2F Functions:', funcs[i].name, funcs[i].value)
        self.new_forward = True

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if self.comm.Get_rank()==0:
            print('compute_jacvec_product')
        if 'f' in d_outputs:
            if self.new_forward:
                f2f_var_list = self.model.get_variables()
                for ivar, var in enumerate(f2f_var_list):
                    var.value = inputs[self.var_list[ivar]]
                self.driver.solve_adjoint()
                self.grad = self.model.get_function_gradients()
                self.new_forward = False
                if self.comm.Get_rank() ==0:
                    funcs = self.model.get_functions()
                    for i, func in enumerate(funcs):
                        print('FUNCTION: ' + funcs[i].name + " = ", funcs[i].value)
                        for j, var in enumerate(f2f_var_list):
                                print(' var ' + var.name, self.grad[i][j])

            for ivar, var in enumerate(self.model.get_variables()):
                if self.var_list[ivar] in d_inputs:
                    for i in range(d_outputs['f'].size):
                        if mode == 'fwd':
                            d_outputs['f'][i] += self.grad[i][ivar] * d_inputs[self.var_list[ivar]]
                        elif mode == 'rev':
                            d_inputs[self.var_list[ivar]] += self.grad[i][ivar] * d_outputs['f'][i]
                            if self.comm.Get_rank()==0:
                                print('di',i,'d', self.var_list[ivar], self.grad[i][ivar], d_outputs['f'][i])
