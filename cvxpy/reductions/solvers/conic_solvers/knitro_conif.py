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

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone
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


class Dims:
    def __init__(self, dims: dict):
        self.eq_dim = int(dims.get(s.EQ_DIM, 0))
        self.leq_dim = int(dims.get(s.LEQ_DIM, 0))
        self.soc_dims = [int(d) for d in dims.get(s.SOC_DIM, [])]
        self.n_exps = int(dims.get(s.EXP_DIM, 0))
        self.n_socs = len(self.soc_dims)
        self.n_cones = self.n_socs + self.n_exps
        self.soc_dim = sum(self.soc_dims)
        self.exp_dim = 3 * self.n_exps
        self.cone_dim = self.soc_dim + self.exp_dim
        self.eqs = np.arange(self.eq_dim)
        self.leqs = self.eq_dim + np.arange(self.leq_dim)
        self.cone_eqs = self.eq_dim + self.leq_dim + np.arange(self.cone_dim)
        self.cones = self.eq_dim + self.leq_dim + self.cone_dim + np.arange(self.n_cones)
        self.socs = np.insert(np.cumsum(self.soc_dims), 0, 0)

    def get_dual(self, y):
        y = np.array(y)
        idx = np.concatenate(
            (
                self.eqs,
                self.leqs,
                self.cone_eqs[: self.soc_dim],
                self.cones[: self.n_socs],
                self.cone_eqs[self.soc_dim : self.soc_dim + self.exp_dim],
                self.cones[self.n_socs : self.n_socs + self.n_exps],
            )
        )
        return np.array(y[idx])


class ExpCallbackParams:
    def __init__(self, n_cones, var_idxs, con_idxs):
        self.n_cones = n_cones
        self.var_idxs = var_idxs
        self.con_idxs = con_idxs


class KNITRO(ConicSolver):
    """
    Conic interface for the Knitro solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"
    Y_INIT_KEY = "y_init"
    N_VARS_KEY = "n_vars"

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = [X_INIT_KEY, Y_INIT_KEY]

    EXP_CONE_ORDER = [0, 1, 2]
    EXP_DOUBLE_LIMIT = 705.0

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

        if KNITRO.CONTEXT_KEY not in results:
            return failure_solution(s.SOLVER_ERROR)

        kc = results[KNITRO.CONTEXT_KEY]
        num_iters = kn.KN_get_number_iters(kc)
        solve_time = kn.KN_get_solve_time_real(kc)
        attr = {
            s.SOLVE_TIME: solve_time,
            s.NUM_ITERS: num_iters,
            s.EXTRA_STATS: kc,
        }

        if s.STATUS in results and results[s.STATUS] == s.SOLVER_ERROR:
            solution = failure_solution(s.SOLVER_ERROR, attr)
        else:
            status_kn, obj_kn, x_kn, y_kn = kn.KN_get_solution(kc)
            status = self.STATUS_MAP.get(status_kn, s.SOLVER_ERROR)

            if status == s.UNBOUNDED:
                solution = Solution(status, -np.inf, {}, {}, attr)
            elif (status not in s.SOLUTION_PRESENT) or (x_kn is None):
                solution = failure_solution(status, attr)
            else:
                n_vars = int(results[KNITRO.N_VARS_KEY])

                obj = obj_kn + inverse_data[s.OFFSET]
                x_kn = x_kn[:n_vars]
                x = np.array(x_kn)
                primal_vars = {inverse_data[KNITRO.VAR_ID]: x}

                dual_vars = None
                is_mip = bool(inverse_data.get("is_mip", False))
                y_kn = list(kn.KN_get_con_dual_values(kc))
                if y_kn is not None and not is_mip:
                    dims = Dims(dims_to_solver_dict(inverse_data[s.DIMS] or {}))
                    y = dims.get_dual(y_kn)
                    eq_dual_vars = utilities.get_dual_values(
                        y[: dims.eq_dim],
                        utilities.extract_dual_value,
                        inverse_data[KNITRO.EQ_CONSTR],
                    )
                    ineq_dual_vars = utilities.get_dual_values(
                        y[dims.eq_dim :],
                        utilities.extract_dual_value,
                        inverse_data[KNITRO.NEQ_CONSTR],
                    )
                    dual_vars = {**eq_dual_vars, **ineq_dual_vars}
                solution = Solution(status, obj, primal_vars, dual_vars, attr)
        # Free the Knitro context.
        kn.KN_free(kc)
        return solution

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
        lb = data.get(s.LOWER_BOUNDS)
        ub = data.get(s.UPPER_BOUNDS)
        dims = Dims(dims_to_solver_dict(data.get(s.DIMS) or {}))

        results = {}
        try:
            kc = kn.KN_new()
        except Exception:
            results[s.STATUS] = s.SOLVER_ERROR
            return results

        results[KNITRO.CONTEXT_KEY] = kc

        if not verbose:
            # Disable Knitro output.
            kn.KN_set_int_param(kc, kn.KN_PARAM_OUTLEV, kn.KN_OUTLEV_NONE)

        n_vars = int(c.shape[0])
        results[KNITRO.N_VARS_KEY] = n_vars

        # Add n variables to the problem.
        kn.KN_add_vars(kc, n_vars)

        # Set the lower and upper bounds on the variables.
        if lb is not None:
            idxs, lb = kn_rm_inf(lb)
            kn.KN_set_var_lobnds(kc, indexVars=idxs, xLoBnds=lb)
        if ub is not None:
            idxs, ub = kn_rm_inf(ub)
            kn.KN_set_var_upbnds(kc, indexVars=idxs, xUpBnds=ub)

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
            idxs, vals = solver_opts[KNITRO.X_INIT_KEY]
            kn.KN_set_var_primal_init_values(kc, indexVars=idxs, xInitVals=vals)

        # Add constraints to the problem.
        n_cons = int(A.shape[0]) if A is not None else 0
        if n_cons > 0:
            kn.KN_add_cons(kc, n_cons)

        if dims.cone_dim > 0:
            kn.KN_add_vars(kc, dims.cone_dim)
        if dims.n_cones > 0:
            kn.KN_add_cons(kc, dims.n_cones)

        D = sp.coo_matrix(A)
        if D.nnz != 0:
            con_idxs, var_idxs, coefs = D.row, D.col, D.data
            kn.KN_add_con_linear_struct(kc, indexCons=con_idxs, indexVars=var_idxs, coefs=coefs)

        con_idxs = np.concatenate((dims.eqs, dims.cone_eqs))
        e = b[con_idxs]
        kn.KN_set_con_eqbnds(kc, indexCons=con_idxs, cEqBnds=e)
        con_idxs = dims.leqs
        e = b[con_idxs]
        kn.KN_set_con_upbnds(kc, indexCons=con_idxs, cUpBnds=e)
        con_idxs = dims.cone_eqs
        coefs = np.ones(dims.cone_dim)
        var_idxs = n_vars + np.arange(dims.cone_dim)
        kn.KN_add_con_linear_struct(kc, indexCons=con_idxs, indexVars=var_idxs, coefs=coefs)

        con_idxs = dims.cones[: dims.n_socs]
        for k in range(dims.n_socs):
            var_idxs = n_vars + dims.socs[k] + np.arange(dims.soc_dims[k])
            con_idx = con_idxs[k]
            coefs = np.ones(dims.soc_dims[k])
            coefs[0] *= -1.0
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[0], xLoBnds=0.0)
            kn.KN_add_con_quadratic_struct(
                kc, indexCons=con_idx, indexVars1=var_idxs, indexVars2=var_idxs, coefs=coefs
            )
            kn.KN_set_con_upbnds(kc, indexCons=con_idx, cUpBnds=0.0)

        if dims.n_exps > 0:
            con_idxs = dims.cones[dims.n_socs : dims.n_socs + dims.n_exps]
            var_idxs = n_vars + dims.soc_dim + np.arange(dims.exp_dim)
            n_cones = dims.n_exps

            exp_cb_params = ExpCallbackParams(
                n_cones=n_cones,
                var_idxs=var_idxs[::3],
                con_idxs=con_idxs,
            )
            bnds = np.zeros(n_cones)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[2::3], xLoBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[1::3], xLoBnds=bnds)
            kn.KN_set_con_upbnds(kc, indexCons=con_idxs, cUpBnds=bnds)

            def exp_cb_eval(
                kc,
                cb,
                req: kn.KN_eval_request,
                res: kn.KN_eval_result,
                params: ExpCallbackParams,
            ):
                if req.type != kn.KN_RC_EVALFC:
                    return -1
                v = req.x
                for i in range(params.n_cones):
                    ptr = params.var_idxs[i]
                    x = v[ptr]
                    y = v[ptr + 1]
                    z = v[ptr + 2]
                    if np.isclose(y, 0.0) or (x / y) > KNITRO.EXP_DOUBLE_LIMIT:
                        if x <= 0.0:
                            res.c[i] = -1.0
                        else:
                            res.c[i] = 0.0
                    else:
                        res.c[i] = y * np.exp(x / y) - z
                return 0

            def exp_cb_grad(
                kc,
                cb: kn.CB_context,
                req: kn.KN_eval_request,
                res: kn.KN_eval_result,
                params: ExpCallbackParams,
            ):
                if req.type != kn.KN_RC_EVALGA:
                    return -1
                v = req.x
                for i in range(params.n_cones):
                    ptr = params.var_idxs[i]
                    x = v[ptr]
                    y = v[ptr + 1]
                    if np.isclose(y, 0.0) or (x / y) > KNITRO.EXP_DOUBLE_LIMIT:
                        res.jac[3 * i] = kn.KN_INFINITY
                        res.jac[3 * i + 1] = kn.KN_INFINITY
                        res.jac[3 * i + 2] = kn.KN_INFINITY
                    else:
                        res.jac[3 * i] = np.exp(x / y)
                        res.jac[3 * i + 1] = (1 - (x / y)) * np.exp(x / y)
                        res.jac[3 * i + 2] = -1.0
                return 0

            def exp_cb_hess(
                kc,
                cb: kn.CB_context,
                req: kn.KN_eval_request,
                res: kn.KN_eval_result,
                params: ExpCallbackParams,
            ):
                if req.type != kn.KN_RC_EVALH and req.type != kn.KN_RC_EVALH_NO_F:
                    return -1
                v = req.x
                u = req.lambda_
                for i in range(params.n_cones):
                    var_ptr = params.var_idxs[i]
                    con_ptr = params.con_idxs[i]
                    x = v[var_ptr]
                    y = v[var_ptr + 1]
                    if np.isclose(y, 0.0) or (x / y) > KNITRO.EXP_DOUBLE_LIMIT:
                        res.hess[3 * i] = kn.KN_INFINITY
                        res.hess[3 * i + 1] = kn.KN_INFINITY
                        res.hess[3 * i + 2] = kn.KN_INFINITY
                    else:
                        res.hess[3 * i] = (1 / y) * np.exp(x / y) * u[con_ptr]
                        res.hess[3 * i + 1] = -(x / y**2) * np.exp(x / y) * u[con_ptr]
                        res.hess[3 * i + 2] = (x**2 / y**3) * np.exp(x / y) * u[con_ptr]
                return 0

            cb = kn.KN_add_eval_callback(kc, indexCons=con_idxs, funcCallback=exp_cb_eval)
            kn.KN_set_cb_grad(
                kc,
                cb,
                jacIndexCons=np.repeat(con_idxs, 3),
                jacIndexVars=var_idxs,
                gradCallback=exp_cb_grad,
            )
            var_idxs = np.repeat(exp_cb_params.var_idxs, 3)
            var1_idxs = var_idxs + np.tile(np.array([0, 0, 1]), dims.n_exps)
            var2_idxs = var_idxs + np.tile(np.array([0, 1, 1]), dims.n_exps)
            kn.KN_set_cb_hess(
                kc,
                cb,
                hessIndexVars1=var1_idxs,
                hessIndexVars2=var1_idxs,
                hessCallback=exp_cb_hess,
            )
            kn.KN_set_cb_user_params(kc, cb, exp_cb_params)

        # Set the initial values of the dual variables.
        if KNITRO.Y_INIT_KEY in solver_opts:
            idxs, vals = solver_opts[KNITRO.Y_INIT_KEY]
            kn.KN_set_con_dual_init_values(kc, indexCons=idxs, yInitVals=vals)

        # Set the linear part of the objective function.
        if c is not None:
            var_idxs = np.arange(n_vars)
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
            param_id = kn.KN_get_param_id(kc, key)
            param_type = kn.KN_get_param_type(kc, param_id)
            fn = kn.KN_set_char_param
            if param_type == kn.KN_PARAMTYPE_INTEGER:
                fn = kn.KN_set_int_param
            elif param_type == kn.KN_PARAMTYPE_FLOAT:
                fn = kn.KN_set_double_param
            fn(kc, param_id, val)

        # Optimize the problem.
        try:
            kn.KN_solve(kc)
        except Exception:  # Error in the solution
            results[s.STATUS] = s.SOLVER_ERROR

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
