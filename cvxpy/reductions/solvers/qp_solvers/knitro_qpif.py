"""
Copyright, the CVXPY authors

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
from scipy import sparse

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


def kn_coo(*mats: sparse.coo_matrix) -> tuple[list[int], list[int], list[float]]:
    con_idxs = []
    var_idxs = []
    coefs = []
    shape = 0
    for mat in mats:
        con_idxs.extend(shape + mat.row)
        var_idxs.extend(mat.col)
        coefs.extend(mat.data)
        shape += mat.shape[0]
    return con_idxs, var_idxs, coefs

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


class KNITRO(QpSolver):
    """QP interface for the Knitro solver"""

    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True

    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = [X_INIT_KEY]

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
        -301: s.INFEASIBLE_OR_UNBOUNDED,
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
        return s.KNITRO

    def import_solver(self) -> None:
        import knitro

        knitro

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        """
        return super(KNITRO, self).apply(problem)

    def invert(self, results, inverse_data):
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
            return Solution(
                status, -np.inf, {}, {}, attr
            )

        if (status not in s.SOLUTION_PRESENT) or (x_kn is None):
            return failure_solution(status, attr)

        obj = obj_kn + inverse_data[s.OFFSET]
        x = np.array(x_kn)
        primal_vars = {KNITRO.VAR_ID: x}

        dual_vars = None
        if y_kn is not None:
            y = np.array(y_kn)
            dual_vars = {KNITRO.DUAL_VAR_ID: y}

        # Free the Knitro context.
        kn.KN_free(kc)

        return Solution(status, obj, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

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
        solver_cache: None
            None

        Returns
        -------
        The result of the call to the knitro solver.
        """
        import knitro as kn

        P = data[s.P]
        q = data[s.Q]
        A = data[s.A].tocoo()
        b = data[s.B]
        F = data[s.F].tocoo()
        g = data[s.G]
        n_vars = int(data["n_var"])
        lb = data[s.LOWER_BOUNDS]
        ub = data[s.UPPER_BOUNDS]

        try:
            kc = kn.KN_new()
        except Exception: # Error in the Knitro.
            return {s.STATUS: s.SOLVER_ERROR}

        if not verbose:
            # Disable Knitro output.
            kn.KN_set_int_param(kc, kn.KN_PARAM_OUTLEV, kn.KN_OUTLEV_NONE)

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
            for ind in data[s.BOOL_IDX]:
                var_types[ind] = kn.KN_VARTYPE_BINARY
        if s.INT_IDX in data:
            for ind in data[s.INT_IDX]:
                var_types[ind] = kn.KN_VARTYPE_INTEGER
        kn.KN_set_var_types(kc, xTypes=var_types)

        if solver_opts:
            if KNITRO.X_INIT_KEY in solver_opts:
                var_idxs, vals = solver_opts[KNITRO.X_INIT_KEY]
                kn.KN_set_var_primal_init_values(
                    kc, indexVars=var_idxs, xInitVals=vals
                )

        # Get the number of equality and inequality constraints.
        n_eq, n_ineq = A.shape[0], F.shape[0]

        # Add the constraints to the problem.
        if n_eq + n_ineq > 0:
            kn.KN_add_cons(kc, n_eq + n_ineq)

        # Add the equality bounds.
        if n_eq > 0:
            con_idxs = [i for i in range(n_eq)]
            kn.KN_set_con_eqbnds(
                kc, indexCons=con_idxs, cEqBnds=b
            )

        # Add the inequality bounds.
        if n_ineq > 0:
            con_idxs = [i for i in range(n_eq, n_eq + n_ineq)]
            kn.KN_set_con_upbnds(
                kc, indexCons=con_idxs, cUpBnds=g
            )

        # Set the constraint coefficients.
        if n_eq + n_ineq > 0:
            con_idxs, var_idxs, coefs = kn_coo(A, F)
            kn.KN_add_con_linear_struct(
                kc, indexCons=con_idxs, indexVars=var_idxs, coefs=coefs
            )

        # Set the objective function.

        # Set the linear part of the objective function.
        if q is not None:
            var_idxs = [i for i in range(n_vars)]
            kn.KN_add_obj_linear_struct(
                kc, indexVars=var_idxs, coefs=q
            )

        # Set the quadratic part of the objective function.
        if P is not None and P.nnz != 0:
            var1_idxs, var2_idxs, coefs = kn_coo(P.tocoo())
            coefs = 0.5 * np.array(coefs)
            kn.KN_add_obj_quadratic_struct(
                kc, indexVars1=var1_idxs, indexVars2=var2_idxs, coefs=coefs
            )

        # Set the sense of the objective function.
        kn.KN_set_obj_goal(kc, kn.KN_OBJGOAL_MINIMIZE)

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
        return CITATION_DICT["KNITRO"]
