__all__ = [
    "TestResult",
    "CoordinateDerivativeTester",
]

import numpy as np
import os


class TestResult:
    def __init__(
        self,
        name,
        func_names,
        complex_TD,
        adjoint_TD,
        rel_error=None,
        comm=None,
        method="complex_step",
        i_funcs=None,
        m_funcs=None,
        f_funcs=None,
        var_names=None,
        epsilon=None,
    ):
        """
        Class to store test results from complex step method
        """
        self.name = name
        self.func_names = func_names  # list of function names
        self.var_names = var_names
        self.complex_TD = complex_TD
        self.adjoint_TD = adjoint_TD
        self.method = method
        self.i_funcs = i_funcs
        self.m_funcs = m_funcs
        self.f_funcs = f_funcs
        self.epsilon = epsilon
        if rel_error is None:
            rel_error = []
            for i, _ in enumerate(self.complex_TD):
                rel_error.append(
                    TestResult.relative_error(complex_TD[i], adjoint_TD[i])
                )
        self.rel_error = rel_error
        self.comm = comm

        self.nfuncs = len(func_names)

    def set_name(self, new_name):
        self.name = new_name
        return self

    @property
    def root_proc(self) -> bool:
        return self.comm is None or self.comm.rank == 0

    def write(self, file_hdl):
        """
        write the test result out to a file handle
        """
        if self.root_proc:
            file_hdl.write(f"Test: {self.name}\n")
            if self.epsilon is not None:
                file_hdl.write(f"\tStep size: {self.epsilon}\n")
            if self.var_names is not None:
                file_hdl.write(f"\tVariables = {self.var_names}\n")
            if isinstance(self.func_names, list):
                for ifunc in range(self.nfuncs):
                    file_hdl.write(f"\tFunction {self.func_names[ifunc]}\n")
                    if self.i_funcs is not None:
                        if self.f_funcs is not None:  # if both defined write this
                            file_hdl.write(
                                f"\t\tinitial value = {self.i_funcs[ifunc]}\n"
                            )
                            if self.m_funcs is not None:
                                file_hdl.write(
                                    f"\t\tmid value = {self.m_funcs[ifunc]}\n"
                                )
                            file_hdl.write(f"\t\tfinal value = {self.f_funcs[ifunc]}\n")
                        else:
                            file_hdl.write(f"\t\tvalue = {self.i_funcs[ifunc]}\n")
                    file_hdl.write(f"\t\t{self.method} TD = {self.complex_TD[ifunc]}\n")
                    file_hdl.write(f"\t\tAdjoint TD = {self.adjoint_TD[ifunc]}\n")
                    file_hdl.write(f"\t\tRelative error = {self.rel_error[ifunc]}\n")
                file_hdl.flush()
            else:
                file_hdl.write(f"\tFunction {self.func_names}")
                if self.i_funcs is not None:
                    if self.f_funcs is not None:  # if both defined write this
                        file_hdl.write(f"\t\tinitial value = {self.i_funcs[ifunc]}\n")
                        file_hdl.write(f"\t\tfinal value = {self.f_funcs[ifunc]}\n")
                    else:
                        file_hdl.write(f"\t\tvalue = {self.i_funcs[ifunc]}\n")
                file_hdl.write(f"\t{self.method} TD = {self.complex_TD}\n")
                file_hdl.write(f"\tAdjoint TD = {self.adjoint_TD}\n")
                file_hdl.write(f"\tRelative error = {self.rel_error}\n")
                file_hdl.flush()
            file_hdl.close()
        return self

    def report(self):
        if self.root_proc:
            print(f"Test Result - {self.name}")
            print("\tFunctions = ", self.func_names)
            print(f"\t{self.method}  = ", self.complex_TD)
            print("\tAdjoint TD      = ", self.adjoint_TD)
            print("\tRelative error        = ", self.rel_error)
        return self

    @classmethod
    def relative_error(cls, truth, pred):
        if truth == 0.0 and pred == 0.0:
            print("Warning the derivative test is indeterminate!")
            return 0.0
        elif truth == 0.0 and pred != 0.0:
            return 1.0  # arbitrary 100% error provided to fail test avoiding /0
        elif abs(truth) <= 1e-8 and abs(pred) < 1e-8:
            print("Warning the derivative test has very small derivatives!")
            return pred - truth  # use absolute error if too small a derivative
        else:
            return (pred - truth) / truth

    @classmethod
    def design_sweep(
        cls,
        model,
        driver,
        nsweep=10,
        eps=1e-4,
        base_folder=None,
        csv_file_prefix="design-sweep",
        include_derivatives=True,
    ):
        """
        perform a design sweep on a model and driver to determine the range of function values
        """

        # open up the csv files
        if driver.comm.rank == 0:
            for ifunc, func in enumerate(model.get_functions()):
                filename = csv_file_prefix + f"_{func.name}.csv"
                if base_folder is None:
                    csv_file = filename
                else:
                    csv_file = os.path.join(base_folder, csv_file)
                hdl = open(csv_file, "w")
                hdl.write(f",alpha,func_val,adjoint\n")
                hdl.close()

        nfunctions = len(model.get_functions())
        nvariables = len(model.get_variables())

        # generate random contravariant tensor in input space x(s)
        if nvariables > 1:
            dxds = np.random.rand(nvariables)
        else:
            dxds = np.array([1.0])

        # store initial variable values
        orig_vars = [var.value * 1.0 for var in model.get_variables()]

        # perform the design sweep
        alphas = np.linspace(-eps / 2.0, eps / 2.0, nsweep)

        func_vals_dict = {func.name: [] for func in model.get_functions()}
        if include_derivatives:
            adj_derivs_dict = {func.name: [] for func in model.get_functions()}
            FD_derivs_dict = {func.name: [] for func in model.get_functions()}

        for ialpha, alpha in enumerate(alphas):

            # change the variables
            for ivar, var in enumerate(model.get_variables()):
                var.value = orig_vars[ivar] + alpha * dxds[ivar]

            # compute forward analysise f(x) and df/dx with adjoint
            driver.solve_forward()
            for func in model.get_functions():
                func_vals_dict[func.name] += [func.value.real]

            if include_derivatives:
                driver.solve_adjoint()
                model.evaluate_composite_functions()
                gradients = model.get_function_gradients(all=True)

                adjoint_TD = np.zeros((nfunctions))
                for ifunc, func in enumerate(model.get_functions()):
                    for ivar in range(nvariables):
                        adjoint_TD[ifunc] += gradients[ifunc][ivar].real * dxds[ivar]
                    adj_derivs_dict[func.name] += [adjoint_TD[ifunc]]

            # write out these derivatives to csv files for each function
            for ifunc, func in enumerate(model.get_functions()):
                df_dict = {
                    "alpha": [alphas[ialpha]],
                    "func-val": func_vals_dict[func.name][ialpha],
                }
                if include_derivatives:
                    df_dict["adjoint"] = adj_derivs_dict[func.name][ialpha]
                    # df_dict["finite_diff"] = FD_derivs_dict[func.name][ialpha]
                if driver.comm.rank == 0:
                    filename = csv_file_prefix + f"_{func.name}.csv"
                    if base_folder is None:
                        csv_file = filename
                    else:
                        csv_file = os.path.join(base_folder, csv_file)
                    hdl = open(csv_file, "a")
                    hdl.write(
                        f"{ialpha},{alphas[ialpha]},{func.value.real},{adj_derivs_dict[func.name][ialpha]}\n"
                    )
                    hdl.close()

        # close the csv file then write it again with FD too
        if driver.comm.rank == 0:
            for ifunc, func in enumerate(model.get_functions()):
                if base_folder is None:
                    csv_file = filename
                else:
                    csv_file = os.path.join(base_folder, csv_file)
                hdl = open(csv_file, "w")
                hdl.write(f",alpha,func_val,adjoint,FD\n")
                hdl.close()
        # then rewrite each one with all the alphas and finite diff derivatives
        for ialpha, alpha in enumerate(alphas):
            # update the finite difference dict
            if include_derivatives:
                dalpha = alphas[1] - alphas[0]
                for ifunc, func in enumerate(model.get_functions()):
                    if ialpha == 0:
                        # forward difference
                        FD_deriv = (
                            func_vals_dict[func.name][ialpha + 1]
                            - func_vals_dict[func.name][ialpha]
                        )
                        FD_deriv /= dalpha
                    elif ialpha == nsweep - 1:
                        # backward difference
                        FD_deriv = (
                            func_vals_dict[func.name][ialpha]
                            - func_vals_dict[func.name][ialpha - 1]
                        )
                        FD_deriv /= dalpha
                    else:
                        # central difference
                        FD_deriv = (
                            func_vals_dict[func.name][ialpha + 1]
                            - func_vals_dict[func.name][ialpha - 1]
                        )
                        FD_deriv /= 2.0 * dalpha

                    FD_derivs_dict[func.name] += [FD_deriv]

        for ifunc, func in enumerate(model.get_functions()):
            df_dict = {
                "alpha": list(alphas),
                "func-val": func_vals_dict[func.name],
            }
            if include_derivatives:
                df_dict["adjoint"] = adj_derivs_dict[func.name]
                df_dict["finite_diff"] = FD_derivs_dict[func.name]
            filename = csv_file_prefix + f"_{func.name}.csv"
            if driver.comm.rank == 0:
                filename = csv_file_prefix + f"_{func.name}.csv"
                if base_folder is None:
                    csv_file = filename
                else:
                    csv_file = os.path.join(base_folder, csv_file)
                hdl = open(csv_file, "a")
                for ialpha, alpha in enumerate(alphas):
                    hdl.write(
                        f",{alphas[ialpha]},{func_vals_dict[func.name][ialpha]},{adj_derivs_dict[func.name][ialpha]},{FD_derivs_dict[func.name][ialpha]}\n"
                    )
                hdl.close()
        return

    @classmethod
    def complex_step(cls, test_name, model, driver, status_file, epsilon=1e-30):
        """
        perform complex step test on a model and driver for multiple functions & variables
        used for fun3d+tacs coupled derivative tests only...
        """

        # determine the number of functions and variables
        nfunctions = len(model.get_functions(all=True))
        nvariables = len(model.get_variables())
        func_names = [func.full_name for func in model.get_functions(all=True)]

        # generate random contravariant tensor, an input space curve tangent dx/ds for design vars
        dxds = np.random.rand(nvariables)

        # solve the adjoint
        if driver.solvers.uses_fun3d:
            if driver.comm.rank == 0:
                print("Check before real primal adjoint for 'uses fun3d': True")
            driver.solvers.make_flow_real()
        driver.solve_forward()
        driver.solve_adjoint()
        model.evaluate_composite_functions()
        gradients = model.get_function_gradients(all=True)

        # compute the adjoint total derivative df/ds = df/dx * dx/ds
        adjoint_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            for ivar in range(nvariables):
                adjoint_TD[ifunc] += gradients[ifunc][ivar].real * dxds[ivar]

        # perform complex step method
        print(f"uses fun3d = {driver.solvers.uses_fun3d}", flush=True)
        if (
            driver.solvers.uses_fun3d
        ):  # NOTE: This is likely preventing make_flow_complex()
            print(f"make flow complex call", flush=True)
            driver.solvers.make_flow_complex()
        variables = model.get_variables()

        # perturb the design vars by x_pert = x + 1j * h * dx/ds
        for ivar in range(nvariables):
            variables[ivar].value += 1j * epsilon * dxds[ivar]

        # run the complex step method
        driver.solve_forward()
        model.evaluate_composite_functions()
        functions = model.get_functions(all=True)

        # compute the complex step total derivative df/ds = Im{f(x+ih * dx/ds)}/h for each func
        complex_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            complex_TD[ifunc] += functions[ifunc].value.imag / epsilon

        # compute rel error between adjoint & complex step for each function
        rel_error = [
            TestResult.relative_error(
                truth=complex_TD[ifunc], pred=adjoint_TD[ifunc]
            ).real
            for ifunc in range(nfunctions)
        ]

        # make test results object and write it to file
        file_hdl = open(status_file, "a") if driver.comm.rank == 0 else None
        cls(
            test_name,
            func_names,
            complex_TD,
            adjoint_TD,
            rel_error,
            comm=driver.comm,
            var_names=[var.name for var in model.get_variables()],
            i_funcs=[func.value.real for func in functions],
            f_funcs=None,
            epsilon=epsilon,
            method="complex_step",
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        return max_rel_error

    @classmethod
    def finite_difference(
        cls,
        test_name,
        model,
        driver,
        status_file,
        epsilon=1e-5,
        central_diff=True,
        both_adjoint=False,  # have to call adjoint in both times for certain drivers
    ):
        """
        perform finite difference test on a model and driver for multiple functions & variables
        """
        nfunctions = len(model.get_functions(all=True))
        nvariables = len(model.get_variables())
        func_names = [func.full_name for func in model.get_functions(all=True)]

        # generate random contravariant tensor in input space x(s)
        if nvariables > 1:
            dxds = np.random.rand(nvariables)
        else:
            dxds = np.array([1.0])

        # central difference approximation
        variables = model.get_variables()
        # compute forward analysise f(x) and df/dx with adjoint
        driver.solve_forward()
        driver.solve_adjoint()
        model.evaluate_composite_functions()
        gradients = model.get_function_gradients(all=True)
        m_functions = [func.value.real for func in model.get_functions(all=True)]

        # compute adjoint total derivative df/dx
        adjoint_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            for ivar in range(nvariables):
                adjoint_TD[ifunc] += gradients[ifunc][ivar].real * dxds[ivar]

        # compute f(x-h)
        if central_diff:
            for ivar in range(nvariables):
                variables[ivar].value -= epsilon * dxds[ivar]
            driver.solve_forward()
            if both_adjoint:
                driver.solve_adjoint()
            model.evaluate_composite_functions()
            i_functions = [func.value.real for func in model.get_functions(all=True)]
        else:
            i_functions = [None for func in model.get_functions()]

        # compute f(x+h)
        alpha = 2 if central_diff else 1
        for ivar in range(nvariables):
            variables[ivar].value += alpha * epsilon * dxds[ivar]
        driver.solve_forward()
        if both_adjoint:
            driver.solve_adjoint()
        model.evaluate_composite_functions()
        f_functions = [func.value.real for func in model.get_functions(all=True)]

        finite_diff_TD = [
            (
                (f_functions[ifunc] - i_functions[ifunc]) / 2 / epsilon
                if central_diff
                else (f_functions[ifunc] - m_functions[ifunc]) / epsilon
            )
            for ifunc in range(nfunctions)
        ]

        # compute relative error
        rel_error = [
            TestResult.relative_error(
                truth=finite_diff_TD[ifunc], pred=adjoint_TD[ifunc]
            ).real
            for ifunc in range(nfunctions)
        ]

        # make test results object and write to file
        file_hdl = open(status_file, "a") if driver.comm.rank == 0 else None
        cls(
            test_name,
            func_names,
            finite_diff_TD,
            adjoint_TD,
            rel_error,
            comm=driver.comm,
            var_names=[var.name for var in model.get_variables()],
            i_funcs=i_functions,
            m_funcs=m_functions,
            f_funcs=f_functions,
            epsilon=epsilon,
            method="central_diff" if central_diff else "finite_diff",
        ).write(file_hdl).report()
        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))
        return max_rel_error

    @classmethod
    def derivative_test(
        cls, test_name, model, driver, status_file, complex_mode=True, epsilon=None
    ):
        """
        call either finite diff or complex step test depending on real mode of funtofem + TACS
        """
        if complex_mode:
            if epsilon is None:
                epsilon = 1e-30
            return cls.complex_step(
                test_name,
                model,
                driver,
                status_file,
                epsilon=epsilon,
            )
        else:
            if epsilon is None:
                epsilon = 1e-5
            return cls.finite_difference(
                test_name,
                model,
                driver,
                status_file,
                epsilon=epsilon,
            )


class CoordinateDerivativeTester:
    """
    Perform a complex step test over the coordinate derivatives of a driver
    """

    def __init__(self, driver):
        self.driver = driver
        self.comm = self.driver.comm

    @property
    def flow_solver(self):
        return self.driver.solvers.flow

    @property
    def aero_X(self):
        """aero coordinate derivatives in FUN3D"""
        return self.flow_solver.aero_X

    @property
    def struct_solver(self):
        return self.driver.solvers.structural

    @property
    def struct_X(self):
        """structure coordinates in TACS"""
        return self.struct_solver.struct_X.getArray()

    @property
    def model(self):
        return self.driver.model

    def test_struct_coordinates(
        self,
        test_name,
        status_file,
        body=None,
        scenario=None,
        epsilon=1e-30,
        complex_mode=True,
    ):
        """test the structure coordinate derivatives struct_X with complex step"""
        # assumes only one body
        if body is None:
            body = self.model.bodies[0]
        if scenario is None:  # test doesn't work for multiscenario yet
            scenario = self.model.scenarios[0]

        # random covariant tensor to aggregate derivative error among one or more functions
        # compile full struct shape term
        nf = len(self.model.get_functions())
        func_names = [func.full_name for func in self.model.get_functions()]

        # random contravariant tensor d(struct_X)/ds for testing struct shape
        dstructX_ds = np.random.rand(body.struct_X.shape[0])
        dstructX_ds_row = np.expand_dims(dstructX_ds, axis=0)

        if self.driver.solvers.uses_fun3d:
            self.driver.solvers.make_flow_real()

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # add coordinate derivatives among scenarios
        dfdxS0 = body.get_struct_coordinate_derivatives(scenario)
        dfdx_adjoint = dstructX_ds_row @ dfdxS0
        dfdx_adjoint = list(np.reshape(dfdx_adjoint, newshape=(nf)))
        adjoint_derivs = [dfdx_adjoint[i].real for i in range(nf)]

        if complex_mode:
            """Complex step to compute coordinate total derivatives"""
            # perturb the coordinate derivatives
            if self.driver.solvers.uses_fun3d:
                self.driver.solvers.make_flow_complex()
            body.struct_X += 1j * epsilon * dstructX_ds
            self.driver.solve_forward()

            truth_derivs = np.array(
                [func.value.imag / epsilon for func in self.model.get_functions()]
            )

        else:  # central finite difference
            # f(x;xA-h)
            body.struct_X -= epsilon * dstructX_ds
            self.driver.solve_forward()
            i_functions = [func.value.real for func in self.model.get_functions()]

            # f(x;xA+h)
            body.struct_X += 2 * epsilon * dstructX_ds
            self.driver.solve_forward()
            f_functions = [func.value.real for func in self.model.get_functions()]

            truth_derivs = np.array(
                [
                    (f_functions[i] - i_functions[i]) / 2 / epsilon
                    for i in range(len(self.model.get_functions()))
                ]
            )

        rel_error = [
            TestResult.relative_error(truth_derivs[i], adjoint_derivs[i])
            for i in range(nf)
        ]

        # make test results object and write it to file
        file_hdl = open(status_file, "a") if self.comm.rank == 0 else None
        TestResult(
            test_name,
            func_names,
            truth_derivs,
            adjoint_derivs,
            rel_error,
            method="complex_step" if complex_mode else "finite_diff",
            epsilon=epsilon,
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        return max_rel_error

    def test_aero_coordinates(
        self,
        test_name,
        status_file,
        scenario=None,
        body=None,
        epsilon=1e-30,
        complex_mode=True,
    ):
        """test the structure coordinate derivatives struct_X with complex step"""
        # assumes only one body
        if body is None:
            body = self.model.bodies[0]
        if scenario is None:  # test doesn't work for multiscenario yet
            scenario = self.model.scenarios[0]

        # random covariant tensor to aggregate derivative error among one or more functions
        # compile full struct shape term
        nf = len(self.model.get_functions())
        func_names = [func.full_name for func in self.model.get_functions()]

        # random contravariant tensor d(aero_X)/ds for testing aero shape
        daeroX_ds = np.random.rand(body.aero_X.shape[0])
        daeroX_ds_row = np.expand_dims(daeroX_ds, axis=0)

        if self.driver.solvers.uses_fun3d:
            self.driver.solvers.make_flow_real()

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # add coordinate derivatives among scenarios
        dfdxA0 = body.get_aero_coordinate_derivatives(scenario)
        dfdx_adjoint = daeroX_ds_row @ dfdxA0
        dfdx_adjoint = list(np.reshape(dfdx_adjoint, newshape=(nf)))
        adjoint_derivs = [dfdx_adjoint[i].real for i in range(nf)]

        if complex_mode:
            """Complex step to compute coordinate total derivatives"""
            # perturb the coordinate derivatives
            if self.driver.solvers.uses_fun3d:
                self.driver.solvers.make_flow_complex()
            body.aero_X += 1j * epsilon * daeroX_ds
            self.driver.solve_forward()

            truth_derivs = np.array(
                [func.value.imag / epsilon for func in self.model.get_functions()]
            )

        else:  # central finite difference
            # f(x;xA-h)
            body.aero_X -= epsilon * daeroX_ds
            self.driver.solve_forward()
            i_functions = [func.value.real for func in self.model.get_functions()]

            # f(x;xA+h)
            body.aero_X += 2 * epsilon * daeroX_ds
            self.driver.solve_forward()
            f_functions = [func.value.real for func in self.model.get_functions()]

            truth_derivs = np.array(
                [
                    (f_functions[i] - i_functions[i]) / 2 / epsilon
                    for i in range(len(self.model.get_functions()))
                ]
            )

        rel_error = [
            TestResult.relative_error(truth_derivs[i], adjoint_derivs[i])
            for i in range(nf)
        ]

        # make test results object and write it to file
        file_hdl = open(status_file, "a") if self.comm.rank == 0 else None
        TestResult(
            test_name,
            func_names,
            truth_derivs,
            adjoint_derivs,
            rel_error,
            comm=self.comm,
            method="complex_step" if complex_mode else "finite_diff",
            epsilon=epsilon,
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        return max_rel_error
