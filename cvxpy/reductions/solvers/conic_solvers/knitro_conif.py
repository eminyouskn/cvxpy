"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, PowCone3D
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver, dims_to_solver_dict
from cvxpy.utilities.citations import CITATION_DICT


def kn_isinf(x) -> bool:
    """Check if x is -inf or inf."""
    if x <= -np.inf or x >= np.inf:
        return True
    if x <= float("-inf") or x >= float("inf"):
        return True

    import knitro as kn

    if x <= -kn.KN_INFINITY or x >= kn.KN_INFINITY:
        return True
    return False


def kn_rm_inf(arr) -> tuple[list[int], list[float]]:
    """Convert -inf to -kn.KN_INFINITY and inf to kn.KN_INFINITY."""
    idx, a = [], []
    for i, v in enumerate(arr):
        if not kn_isinf(v):
            idx.append(i)
            a.append(v)
    return idx, a


class KNITRO(ConicSolver):
    """
    Conic interface for the Knitro solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, PowCone3D]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"
    Y_INIT_KEY = "y_init"

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = [X_INIT_KEY, Y_INIT_KEY]

    # Map of Knitro status to CVXPY status.
    STATUS_MAP = {
        0: s.OPTIMAL,
        -100: s.OPTIMAL_INACCURATE,
        -101: s.USER_LIMIT,
        -102: s.USER_LIMIT,
        -103: s.USER_LIMIT,
        -200: s.INFEASIBLE,
        -201: s.INFEASIBLE,
        -202: s.INFEASIBLE,
        -203: s.INFEASIBLE,
        -204: s.INFEASIBLE,
        -205: s.INFEASIBLE,
        -300: s.UNBOUNDED,
        -301: s.UNBOUNDED,
        -400: s.USER_LIMIT,
        -401: s.USER_LIMIT,
        -402: s.USER_LIMIT,
        -403: s.USER_LIMIT,
        -404: s.USER_LIMIT,
        -405: s.USER_LIMIT,
        -406: s.USER_LIMIT,
        -410: s.USER_LIMIT,
        -411: s.USER_LIMIT,
        -412: s.USER_LIMIT,
        -413: s.USER_LIMIT,
        -415: s.USER_LIMIT,
        -416: s.USER_LIMIT,
        -500: s.SOLVER_ERROR,
        -501: s.SOLVER_ERROR,
        -502: s.SOLVER_ERROR,
        -503: s.SOLVER_ERROR,
        -504: s.SOLVER_ERROR,
        -505: s.SOLVER_ERROR,
        -506: s.SOLVER_ERROR,
        -507: s.SOLVER_ERROR,
        -508: s.SOLVER_ERROR,
        -509: s.SOLVER_ERROR,
        -510: s.SOLVER_ERROR,
        -511: s.SOLVER_ERROR,
        -512: s.SOLVER_ERROR,
        -513: s.SOLVER_ERROR,
        -514: s.SOLVER_ERROR,
        -515: s.SOLVER_ERROR,
        -516: s.SOLVER_ERROR,
        -517: s.SOLVER_ERROR,
        -518: s.SOLVER_ERROR,
        -519: s.SOLVER_ERROR,
        -520: s.SOLVER_ERROR,
        -521: s.SOLVER_ERROR,
        -522: s.SOLVER_ERROR,
        -523: s.SOLVER_ERROR,
        -524: s.SOLVER_ERROR,
        -525: s.SOLVER_ERROR,
        -526: s.SOLVER_ERROR,
        -527: s.SOLVER_ERROR,
        -528: s.SOLVER_ERROR,
        -529: s.SOLVER_ERROR,
        -530: s.SOLVER_ERROR,
        -531: s.SOLVER_ERROR,
        -532: s.SOLVER_ERROR,
        -600: s.SOLVER_ERROR,
    }  # MEM_LIMIT

    def name(self):
        """The name of the solver."""
        return s.KNITRO

    def import_solver(self) -> None:
        """Imports the solver."""
        import knitro

        knitro

    def accepts(self, problem) -> bool:
        return super(KNITRO, self).accepts(problem)

    def apply(self, problem: ParamConeProg):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(KNITRO, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data["is_mip"] = data[s.BOOL_IDX] or data[s.INT_IDX]
        return data, inv_data

    def invert(self, results, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        import knitro as kn

        kc = results[KNITRO.CONTEXT_KEY]
        status_kn, obj_kn, x_kn, y_kn = kn.KN_get_solution(kc)
        num_iters = kn.KN_get_number_iters(kc)
        solve_time = kn.KN_get_solve_time_real(kc)
        attr = {
            s.SOLVE_TIME: solve_time,
            s.NUM_ITERS: num_iters,
            s.EXTRA_STATS: kc,
        }
        status = self.STATUS_MAP.get(status_kn, s.SOLVER_ERROR)
        if status == s.UNBOUNDED:
            return Solution(status, -np.inf, {}, {}, attr)

        if (status not in s.SOLUTION_PRESENT) or (x_kn is None):
            return failure_solution(status, attr)

        obj = obj_kn + inverse_data[s.OFFSET]
        x = np.array(x_kn)
        primal_vars = {inverse_data[KNITRO.VAR_ID]: x}

        dual_vars = None
        is_mip = bool(inverse_data.get("is_mip", False))
        y_kn = kn.KN_get_con_dual_values(kc)
        if y_kn is not None and not is_mip:
            y = np.array(y_kn)
            dims = dims_to_solver_dict(inverse_data[s.DIMS] or {})
            n_eqs = int(dims[s.EQ_DIM])
            eq_dual_vars = utilities.get_dual_values(
                y[:n_eqs], utilities.extract_dual_value, inverse_data[KNITRO.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                y[n_eqs:],
                utilities.extract_dual_value,
                inverse_data[KNITRO.NEQ_CONSTR],
            )
            dual_vars = {**eq_dual_vars, **ineq_dual_vars}

        # Free the Knitro context.
        kn.KN_free(kc)

        return Solution(status, obj, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        The result of the call to the knitro solver.
        """
        import knitro as kn

        P = data.get(s.P)
        c = data.get(s.C)
        b = data.get(s.B)
        A = data.get(s.A)
        dims = dims_to_solver_dict(data.get(s.DIMS) or {})
        lb = data.get(s.LOWER_BOUNDS)
        ub = data.get(s.UPPER_BOUNDS)

        try:
            kc = kn.KN_new()
        except Exception:
            return {s.STATUS: s.SOLVER_ERROR}

        if not verbose:
            # Disable Knitro output.
            kn.KN_set_int_param(kc, kn.KN_PARAM_OUTLEV, kn.KN_OUTLEV_NONE)

        n_vars = int(c.shape[0])

        # Add n variables to the problem.
        kn.KN_add_vars(kc, n_vars)

        # Set the lower and upper bounds on the variables.
        if lb is not None:
            var_idxs, lb = kn_rm_inf(lb)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs, xLoBnds=lb)
        if ub is not None:
            var_idxs, ub = kn_rm_inf(ub)
            kn.KN_set_var_upbnds(kc, indexVars=var_idxs, xUpBnds=ub)

        # Set the variable types.
        # - default: KN_VARTYPE_CONTINUOUS.
        # - binray: KN_VARTYPE_BINARY.
        # - integer: KN_VARTYPE_INTEGER.
        var_types = [kn.KN_VARTYPE_CONTINUOUS] * n_vars
        if s.BOOL_IDX in data:
            for j in data[s.BOOL_IDX]:
                var_types[j] = kn.KN_VARTYPE_BINARY
        if s.INT_IDX in data:
            for j in data[s.INT_IDX]:
                var_types[j] = kn.KN_VARTYPE_INTEGER
        kn.KN_set_var_types(kc, xTypes=var_types)

        # Set the initial values of the primal variables.
        if KNITRO.X_INIT_KEY in solver_opts:
            var_idxs, vals = solver_opts[KNITRO.X_INIT_KEY]
            kn.KN_set_var_primal_init_values(kc, indexVars=var_idxs, xInitVals=vals)

        # Add constraints to the problem.
        n_eqs = int(dims[s.EQ_DIM])
        n_ineqs = int(dims[s.LEQ_DIM])
        socs = list(dims.get(s.SOC_DIM, []))
        n_socs = len(socs)
        n_cons = n_eqs + n_ineqs + n_socs + sum(socs)

        if n_cons > 0:
            kn.KN_add_cons(kc, n_cons)

        # Set the linear equality and inequality constraints.
        if n_eqs > 0:
            con_idxs = [i for i in range(n_eqs)]
            kn.KN_set_con_eqbnds(kc, indexCons=con_idxs, cEqBnds=b[:n_eqs])
        if n_ineqs > 0:
            con_idxs = [i for i in range(n_eqs, n_eqs + n_ineqs)]
            kn.KN_set_con_upbnds(kc, indexCons=con_idxs, cUpBnds=b[n_eqs : n_eqs + n_ineqs])
        if n_eqs + n_ineqs > 0:
            D = sp.coo_matrix(A[: n_eqs + n_ineqs, :])
            con_idxs, var_idxs, coefs = D.row, D.col, D.data
            kn.KN_add_con_linear_struct(kc, indexCons=con_idxs, indexVars=var_idxs, coefs=coefs)

        # Set the SOC constraints.
        if n_socs > 0:
            m = n_eqs + n_ineqs
            for k in range(n_socs):
                soc_var_idxs = kn.KN_add_vars(kc, socs[k])
                kn.KN_set_var_lobnds(kc, indexVars=soc_var_idxs[0], xLoBnds=0.0)
                kn.KN_add_con_quadratic_struct(
                    kc,
                    indexCons=m + k + socs[k],
                    indexVars1=soc_var_idxs,
                    indexVars2=soc_var_idxs,
                    coefs=[1.0] + [-1.0] * (socs[k] - 1),
                )
                kn.KN_set_con_lobnds(kc, indexCons=m + k + socs[k], cLoBnds=0.0)

                D = sp.coo_matrix(A[m : m + socs[k], :])
                e = b[m : m + socs[k]]
                con_idxs = [m + k + i for i in range(socs[k])]
                kn.KN_set_con_eqbnds(kc, indexCons=con_idxs, cEqBnds=e)
                kn.KN_add_con_linear_struct(
                    kc,
                    indexCons=con_idxs,
                    indexVars=soc_var_idxs,
                    coefs=[1.0] * socs[k],
                )
                con_idxs, var_idxs, coefs = m + k + D.row, D.col, D.data
                kn.KN_add_con_linear_struct(
                    kc,
                    indexCons=con_idxs,
                    indexVars=var_idxs,
                    coefs=coefs,
                )
                m += socs[k]

        # Set the initial values of the dual variables.
        if KNITRO.Y_INIT_KEY in solver_opts:
            con_idxs, vals = solver_opts[KNITRO.Y_INIT_KEY]
            kn.KN_set_con_dual_init_values(kc, indexCons=con_idxs, yInitVals=vals)

        # Set the linear part of the objective function.
        if c is not None:
            var_idxs = [i for i in range(n_vars)]
            kn.KN_add_obj_linear_struct(kc, indexVars=var_idxs, coefs=c)

        # Set the quadratic part of the objective function.
        if P is not None and P.nnz != 0:
            Q = sp.coo_matrix(0.5 * P)
            var1_idxs, var2_idxs, coefs = Q.row, Q.col, Q.data
            kn.KN_add_obj_quadratic_struct(
                kc, indexVars1=var1_idxs, indexVars2=var2_idxs, coefs=coefs
            )

        # Set the sense of the objective function.
        kn.KN_set_obj_goal(kc, kn.KN_OBJGOAL_MINIMIZE)

        # Set the values of the parameters.
        for key, val in solver_opts.items():
            if key in KNITRO.INTERFACE_ARGS:
                continue
            param_type = kn.KN_get_param_type(kc, key)
            if param_type == kn.KN_PARAMTYPE_INTEGER:
                kn.KN_set_int_param(kc, key, val)
            elif param_type == kn.KN_PARAMTYPE_FLOAT:
                kn.KN_set_double_param(kc, key, val)
            elif param_type == kn.KN_PARAMTYPE_STRING:
                kn.KN_set_char_param(kc, key, val)

        # Optimize the problem.
        results = {}
        try:
            kn.KN_solve(kc)
        except Exception:  # Error in the solution
            results[s.STATUS] = s.SOLVER_ERROR

        results[KNITRO.CONTEXT_KEY] = kc

        # Cache the Knitro context.
        if solver_cache is not None:
            solver_cache[self.name()] = kc

        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT[s.KNITRO]
