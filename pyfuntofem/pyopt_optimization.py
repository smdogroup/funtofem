#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

class PyOptOptimization(object):
    def __init__(self,comm,eval_forward,eval_gradient,number_of_steps=1e9,read_history=True,unscale_design_variables=None):
        """
        A class to save PyOpt optimization histories. It can also be used for problems with FUN3D that only can take one
        a few passes through the solvers at a time to avoid memory leaks stopping the run. The history is saved to the
        disk. If history files exist, They will be read in for all previous steps so that the optimizer has the full
        history even if it only takes one step this time.

        Parameters
        ----------
        comm: MPI communicator
            MPI communicator for the optimization problem
        eval_forward: function
            function that evaluates the forward analysis. The format of eval_forward matches what PyOpt expects:
            obj, con, fail = eval_forward(x)
        eval_gradient:function
            function that evaluates the gradients. The format of eval_forward matches what PyOpt expects:
            g, A, fail = eval_gradient(x,obj,con)
        number_of_steps: int
            This class will feed the optimizer any steps from the history then will allow the optimizer to take a
            maximum number of number_of_steps new adjoint steps. This class then quit(). The default value of 1e9 will
            essentially let the optimization run without stopping it.
        read_history: bool
            If false, existing history files will be ignored and overwritten, i.e., the optimization problem will
            restart.
        unscale_design_variables: function
            Function that takes the scaled design variables from the optimizer and returns the dimensional dv's.
            If this function is provided, the unscaled variables will be saved to the history.
            If no unscale_design_variable function is provided, the scaled design variables will be saved.
            x_unscaled = unscale_design_variables(x)
        """

        self.comm = comm

        self.number_of_steps = number_of_steps
        self.forward_step = 0
        self.adjoint_step = 0

        # Get the histories if they exist
        if read_history and self.comm.Get_rank()==0:
            try:
                obj_hist          = np.load('obj_hist.npy')
                con_hist          = np.load('con_hist.npy')
                dv_hist           = np.load('dv_hist.npy')
                fail_hist         = np.load('fail_hist.npy')
                obj_grad_hist     = np.load('obj_grad_hist.npy')
                con_grad_hist     = np.load('con_grad_hist.npy')
                forward_hist_step = obj_hist.shape[1]
                adjoint_hist_step = obj_grad_hist.shape[2]
            except:
                obj_hist          = None
                con_hist          = None
                dv_hist           = None
                fail_hist         = None
                obj_grad_hist     = None
                con_grad_hist     = None
                forward_hist_step = 0
                adjoint_hist_step = 0
        else:
            obj_hist          = None
            con_hist          = None
            dv_hist           = None
            fail_hist         = None
            obj_grad_hist     = None
            con_grad_hist     = None
            forward_hist_step = 0
            adjoint_hist_step = 0

        self.dv_hist           = comm.bcast(dv_hist,root=0)
        self.obj_hist          = comm.bcast(obj_hist,root=0)
        self.con_hist          = comm.bcast(con_hist,root=0)
        self.fail_hist         = comm.bcast(fail_hist,root=0)
        self.obj_grad_hist     = comm.bcast(obj_grad_hist,root=0)
        self.con_grad_hist     = comm.bcast(con_grad_hist,root=0)
        self.forward_hist_step = comm.bcast(forward_hist_step,root=0)
        self.adjoint_hist_step = comm.bcast(adjoint_hist_step,root=0)
        if self.comm.Get_rank()==0 and self.obj_hist is None:
            print("PyOpt Optimization: Couldn't/didn't read history files. Starting from the beginning")


        # hold onto the forward and gradient evaluation functions
        self.eval_forward = eval_forward
        self.eval_gradient = eval_gradient

        if unscale_design_variables is None:
            self.unscale_design_variables = self._noscaling_of_design_variables
        else:
            self.unscale_design_variables = unscale_design_variables

    def eval_obj_con(self,x):
        """
        Wrapper for the function evaluation that saves/reads history data.
        If taking a new step, evaluate the functions and save to the disk.
        If not taking a new step, read the functions' values from the disk.
        This is the function to give the PyOpt optimizer as the function evaluation.

        Parameters
        ----------
        x: numpy array or list
            design variable list

        Returns
        -------
        obj: numpy array
            objective value(s)
        con: numpy array
            constraint value(s)
        fail: int
            whether or not the simulation failed
        """

        fail = 0
        if self.adjoint_step>=self.adjoint_hist_step+self.number_of_steps:
            if self.comm.Get_rank()==0:
                print("PyOpt Optimization: Reached the requested number of adjoint steps... Stopping")
            quit()

        elif self.forward_step >= self.forward_hist_step:
            obj, con, fail = self.eval_forward(x)

            # Write the history of the objective and constraint evaluations
            if np.size(obj)==1:
                obj_array = np.ones(1) * obj
            else:
                obj_array = np.array(obj)
            obj_array=np.expand_dims(obj_array,axis=1)

            if self.obj_hist is None:
                self.obj_hist = obj_array
            else:
                self.obj_hist = np.concatenate((self.obj_hist,obj_array),axis=1)

            con_array = np.expand_dims(np.array(con),axis=1)
            if self.con_hist is None:
                self.con_hist = con_array
            else:
                self.con_hist = np.concatenate((self.con_hist,con_array),axis=1)

            fail_array = np.ones(1,dtype=int) * fail
            if self.fail_hist is None:
                self.fail_hist = fail_array
            else:
                self.fail_hist = np.concatenate((self.fail_hist,fail_array),axis=0)

            dv_array = np.expand_dims(np.array(self.unscale_design_variables(x)),axis=1)

            if self.dv_hist is None:
                self.dv_hist = dv_array
            else:
                self.dv_hist = np.concatenate((self.dv_hist,dv_array),axis=1)

            if self.comm.Get_rank()==0:
                np.save('obj_hist.npy',self.obj_hist)
                np.save('con_hist.npy',self.con_hist)
                np.save('fail_hist.npy',self.fail_hist)
                np.save('dv_hist.npy'  ,self.dv_hist)

        else:
            if self.comm.Get_rank()==0:
                print("PyOpt Optimization: Reading from history, forward step",self.forward_step)
            obj = self.obj_hist[:,self.forward_step]
            con = self.con_hist[:,self.forward_step]
            fail= self.fail_hist[self.forward_step]

        self.forward_step += 1

        return obj, con, fail

    def eval_obj_con_grad(self,x,obj,con):
        """
        Wrapper for the function gradient evaluation that saves/reads history data.
        If taking a new step, evaluate the function gradient and save to the disk.
        If not taking a new step, read the function gradient values from the disk.
        This is the function to give the PyOpt optimizer as the gradient evaluation.

        Parameters
        ----------
        x: numpy array or list
            design variable list
        obj: numpy array
            objective value(s)
        con: numpy array
            constraint value(s)

        Returns
        -------
        g: numpy array
            objective gradient(s)
        a: numpy array
            constraint gradient(s)
        fail: int
            whether or not the simulation failed

        """

        fail = 0
        if self.adjoint_step >= self.adjoint_hist_step:
            g, a, fail = self.eval_gradient(x,obj,con)

            # Write the history of the objective and constraint gradients
            obj_grad_array = np.expand_dims(g,axis=2)
            if self.obj_grad_hist is None:
                self.obj_grad_hist = obj_grad_array
            else:
                self.obj_grad_hist = np.concatenate((self.obj_grad_hist,obj_grad_array),axis=2)

            con_grad_array = np.expand_dims(a,axis=2)
            if self.con_grad_hist is None:
                self.con_grad_hist = con_grad_array
            else:
                self.con_grad_hist = np.concatenate((self.con_grad_hist,con_grad_array),axis=2)

            if self.comm.Get_rank()==0:
                np.save('obj_grad_hist.npy',self.obj_grad_hist)
                np.save('con_grad_hist.npy',self.con_grad_hist)

        else:
            if self.comm.Get_rank()==0:
                print("PyOpt Optimization: Reading from history, adjoint step",self.adjoint_step)
            g = self.obj_grad_hist[:,:,self.adjoint_step]
            a = self.con_grad_hist[:,:,self.adjoint_step]

        self.adjoint_step += 1

        return g, a, fail

    def _noscaling_of_design_variables(self,x):
        """
        Dummy function for design variable unscaling for the case where an unscaling function isn't provided

        Parameters
        ----------
        x: list or np array
            design variables

        Returns
        -------
        x: list or np array
            design variables

        """
        return x
