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
from cvxpy.constraints import SOC, ExpCone, PowCone3D
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
        self.n_eqs = int(dims.get(s.EQ_DIM, 0))
        self.n_ineqs = int(dims.get(s.LEQ_DIM, 0))
        self.socs = [int(d) for d in dims.get(s.SOC_DIM, [])]
        self.n_exps = int(dims.get(s.EXP_DIM, 0))
        self.pow3ds = dims.get("p", [])
        self.psds = dims.get(s.PSD_DIM, [])
        self.n_pow3d = len(self.pow3ds)
        self.n_socs = len(self.socs)
        self.n_psds = len(self.psds)
        self.n_cones = self.n_socs + self.n_exps + self.n_pow3d + self.n_psds
        self.n_soc_vars = sum(self.socs)
        self.n_exp_vars = 3 * self.n_exps
        self.n_pow3d_vars = 3 * self.n_pow3d
        self.n_psd_vars = sum(d * (d + 1) // 2 for d in self.psds)
        self.n_cone_vars = self.n_soc_vars + self.n_exp_vars + self.n_pow3d_vars + self.n_psd_vars


class CB:
    # Knitro callback
    def __init__(self, f, grad=None, hess=None):
        self.f = f
        self.grad = grad
        self.hess = hess


class ECCP:
    # Exponential cone callback parameters.
    def __init__(self, n, x, c):
        self.n = n
        self.x = x
        self.c = c

class P3dCCP:
    # Power cone 3D callback parameters.
    def __init__(self, n, x, c, a):
        self.n = n
        self.x = x
        self.c = c
        self.a = a

def build_exp_cb() -> CB:
    import knitro as kn

    def f(
        _,
        cb: kn.CB_context,
        req: kn.KN_eval_request,
        res: kn.KN_eval_result,
        params: ECCP,
    ):
        if req.type != kn.KN_RC_EVALFC:
            return -1
        v = req.x
        for k in range(params.n):
            j = params.x[k]
            x, y, z = v[j : j + 3]
            if np.isclose(y, 0.0) or (x / y) > KNITRO.EXP_DOUBLE_LIMIT:
                if x <= 0.0:
                    res.c[k] = -1.0
                else:
                    res.c[k] = 0.0
            else:
                res.c[k] = y * np.exp(x / y) - z
        return 0

    def grad(
        _,
        cb: kn.CB_context,
        req: kn.KN_eval_request,
        res: kn.KN_eval_result,
        params: ECCP,
    ):
        if req.type != kn.KN_RC_EVALGA:
            return -1
        v = req.x
        for k in range(params.n):
            j = params.x[k]
            x, y = v[j], v[j + 1]
            if np.isclose(y, 0.0) or (x / y) > KNITRO.EXP_DOUBLE_LIMIT:
                res.jac[3 * k] = kn.KN_INFINITY
                res.jac[3 * k + 1] = kn.KN_INFINITY
                res.jac[3 * k + 2] = kn.KN_INFINITY
            else:
                res.jac[3 * k] = np.exp(x / y)
                res.jac[3 * k + 1] = (1 - (x / y)) * np.exp(x / y)
                res.jac[3 * k + 2] = -1.0
        return 0

    def hess(
        _,
        cb: kn.CB_context,
        req: kn.KN_eval_request,
        res: kn.KN_eval_result,
        params: ECCP,
    ):
        if req.type != kn.KN_RC_EVALH and req.type != kn.KN_RC_EVALH_NO_F:
            return -1
        v = req.x
        u = req.lambda_
        for k in range(params.n):
            j = params.x[k]
            i = params.c[k]
            x, y = v[j], v[j + 1]
            if np.isclose(y, 0.0) or (x / y) > KNITRO.EXP_DOUBLE_LIMIT:
                res.hess[3 * k] = kn.KN_INFINITY
                res.hess[3 * k + 1] = kn.KN_INFINITY
                res.hess[3 * k + 2] = kn.KN_INFINITY
            else:
                res.hess[3 * k] = (1 / y) * np.exp(x / y) * u[i]
                res.hess[3 * k + 1] = -(x / y**2) * np.exp(x / y) * u[i]
                res.hess[3 * k + 2] = (x**2 / y**3) * np.exp(x / y) * u[i]
        return 0

    return CB(f=f, grad=grad, hess=hess)


def build_pow3d_cb() -> CB:
    import knitro as kn

    def f(
        _,
        cb: kn.CB_context,
        req: kn.KN_eval_request,
        res: kn.KN_eval_result,
        params: P3dCCP,
    ):
        if req.type != kn.KN_RC_EVALFC:
            return -1
        v = req.x
        for k in range(params.n):
            j = params.x[k]
            x, y, z = v[j : j + 3]
            a = params.a[k]
            res.c[k] = np.power(x, a) * np.power(y, 1-a) - np.abs(z)
    
        return 0

    def grad(
        _,
        cb: kn.CB_context,
        req: kn.KN_eval_request,
        res: kn.KN_eval_result,
        params: P3dCCP,
    ):
        if req.type != kn.KN_RC_EVALGA:
            return -1
        v = req.x
        for k in range(params.n):
            j = params.x[k]
            x, y, z = v[j : j + 3]
            a = params.a[k]
            res.jac[3 * k] = a * np.power(x, a - 1) * np.power(y, 1 - a)
            res.jac[3 * k + 1] = (1 - a) * np.power(x, a) * np.power(y, -a) 
            res.jac[3 * k + 2] = -np.sign(z)
        return 0

    def hess(
        _,
        cb: kn.CB_context,
        req: kn.KN_eval_request,
        res: kn.KN_eval_result,
        params: P3dCCP,
    ):
        if req.type != kn.KN_RC_EVALH and req.type != kn.KN_RC_EVALH_NO_F:
            return -1
        v = req.x
        u = req.lambda_
        for k in range(params.n):
            j = params.x[k]
            i = params.c[k]
            x, y = v[j : j + 2]
            a = params.a[k]
            b = a * (1 - a)
            res.hess[3 * k] = -b * np.power(x, a - 2) * np.power(y, 1 - a) * u[i]
            res.hess[3 * k + 1] = b * np.power(x, a - 1) * np.power(y, -a) * u[i]
            res.hess[3 * k + 2] = -b * np.power(x, a) * np.power(y, -a - 1) * u[i]
        return 0
    return CB(f=f, grad=grad, hess=hess)

class KNITRO(ConicSolver):
    """
    Conic interface for the Knitro solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"
    Y_INIT_KEY = "y_init"
    N_VARS_KEY = "n_vars"
    N_CONS_KEY = "n_cons"

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
                n_cons = int(results[KNITRO.N_CONS_KEY])

                obj = obj_kn + inverse_data[s.OFFSET]
                x_kn = x_kn[:n_vars]
                x = np.array(x_kn)
                primal_vars = {inverse_data[KNITRO.VAR_ID]: x}

                dual_vars = None
                is_mip = bool(inverse_data.get("is_mip", False))
                y_kn = kn.KN_get_con_dual_values(kc)
                if y_kn is not None and not is_mip:
                    dims = dims_to_solver_dict(inverse_data[s.DIMS] or {})
                    y_kn = y_kn[:n_cons]
                    n_eqs = int(dims.get(s.EQ_DIM, 0))
                    y = np.array(y_kn)
                    eq_dual_vars = utilities.get_dual_values(
                        y[:n_eqs],
                        utilities.extract_dual_value,
                        inverse_data[KNITRO.EQ_CONSTR],
                    )
                    ineq_dual_vars = utilities.get_dual_values(
                        y[n_eqs:],
                        utilities.extract_dual_value,
                        inverse_data[KNITRO.NEQ_CONSTR],
                    )
                    dual_vars = {**eq_dual_vars, **ineq_dual_vars}
                solution = Solution(status, obj, primal_vars, dual_vars, attr)
        # Free the Knitro context.
        print(solution)
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
        results[KNITRO.N_CONS_KEY] = n_cons
        if n_cons > 0:
            kn.KN_add_cons(kc, n_cons)

        if dims.n_cone_vars > 0:
            kn.KN_add_vars(kc, dims.n_cone_vars)
        if dims.n_cones > 0:
            kn.KN_add_cons(kc, dims.n_cones)

        D = sp.coo_matrix(A)
        if D.nnz != 0:
            con_idxs, var_idxs, coefs = D.row, D.col, D.data
            kn.KN_add_con_linear_struct(kc, indexCons=con_idxs, indexVars=var_idxs, coefs=coefs)

        con_idxs = np.arange(n_cons)
        kn.KN_set_con_eqbnds(kc, indexCons=con_idxs[: dims.n_eqs], cEqBnds=b[: dims.n_eqs])
        kn.KN_set_con_upbnds(
            kc,
            indexCons=con_idxs[dims.n_eqs : -dims.n_cone_vars],
            cUpBnds=b[dims.n_eqs : -dims.n_cone_vars],
        )
        kn.KN_set_con_eqbnds(
            kc,
            indexCons=con_idxs[-dims.n_cone_vars :],
            cEqBnds=b[-dims.n_cone_vars :],
        )

        var_idxs = n_vars + np.arange(dims.n_cone_vars)
        coefs = np.ones_like(var_idxs, dtype=float)
        kn.KN_add_con_linear_struct(
            kc,
            indexCons=con_idxs[-dims.n_cone_vars :],
            indexVars=var_idxs,
            coefs=coefs,
        )

        var_offset = n_vars
        con_offset = n_cons
        for k in range(dims.n_socs):
            var_idxs = var_offset + np.arange(dims.socs[k])
            con_idx = con_offset + k
            coefs = np.ones_like(var_idxs, dtype=float)
            coefs[0] *= -1.0
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[0], xLoBnds=0.0)
            kn.KN_add_con_quadratic_struct(
                kc,
                indexCons=con_idx,
                indexVars1=var_idxs,
                indexVars2=var_idxs,
                coefs=coefs,
            )
            kn.KN_set_con_upbnds(kc, indexCons=con_idx, cUpBnds=0.0)
            var_offset += dims.socs[k]
        con_offset += dims.n_socs

        if dims.n_exps > 0:
            con_idxs = con_offset + np.arange(dims.n_exps)
            var_idxs = var_offset + np.arange(dims.n_exp_vars)
            bnds = np.zeros_like(con_idxs, dtype=float)
            kn.KN_set_con_upbnds(kc, indexCons=con_idxs, cUpBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[1::3], xLoBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[2::3], xLoBnds=bnds)

            cb = build_exp_cb()

            kb = kn.KN_add_eval_callback(kc, indexCons=con_idxs, funcCallback=cb.f)
            jac_con_idxs = np.repeat(con_idxs, 3)
            jac_var_idxs = var_idxs
            kn.KN_set_cb_grad(
                kc,
                kb,
                jacIndexCons=jac_con_idxs,
                jacIndexVars=jac_var_idxs,
                gradCallback=cb.grad,
            )
            hess_var_idxs = np.repeat(var_idxs[0::3], 3)
            hess_var1_idxs = hess_var_idxs + np.tile(np.array([0, 0, 1]), dims.n_exps)
            hess_var2_idxs = hess_var_idxs + np.tile(np.array([0, 1, 1]), dims.n_exps)
            kn.KN_set_cb_hess(
                kc,
                kb,
                hessIndexVars1=hess_var1_idxs,
                hessIndexVars2=hess_var2_idxs,
                hessCallback=cb.hess,
            )
            params = ECCP(n=dims.n_exps, x=var_idxs[0::3], c=con_idxs)
            kn.KN_set_cb_user_params(kc, kb, params)
        con_offset += dims.n_exps
        var_offset += dims.n_exp_vars

        if dims.n_pow3d > 0:
            con_idxs = con_offset + np.arange(dims.n_pow3d)
            var_idxs = var_offset + np.arange(dims.n_pow3d_vars)
            bnds = np.zeros_like(con_idxs, dtype=float)
            kn.KN_set_con_lobnds(kc, indexCons=con_idxs, cLoBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[0::3], xLoBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=var_idxs[1::3], xLoBnds=bnds)

            cb = build_pow3d_cb()

            kb = kn.KN_add_eval_callback(kc, indexCons=con_idxs, funcCallback=cb.f)
            jac_con_idxs = np.repeat(con_idxs, 3)
            jac_var_idxs = var_idxs
            kn.KN_set_cb_grad(
                kc,
                kb,
                jacIndexCons=jac_con_idxs,
                jacIndexVars=jac_var_idxs,
                gradCallback=cb.grad,
            )
            hess_var_idxs = np.repeat(var_idxs[0::3], 3)
            hess_var1_idxs = hess_var_idxs + np.tile(np.array([0, 0, 1]), dims.n_pow3d)
            hess_var2_idxs = hess_var_idxs + np.tile(np.array([0, 1, 1]), dims.n_pow3d)
            kn.KN_set_cb_hess(
                kc,
                kb,
                hessIndexVars1=hess_var1_idxs,
                hessIndexVars2=hess_var2_idxs,
                hessCallback=cb.hess,
            )
            params = P3dCCP(n=dims.n_pow3d, x=var_idxs[0::3], c=con_idxs, a=dims.pow3ds)
            kn.KN_set_cb_user_params(kc, kb, params)
        con_offset += dims.n_pow3d
        var_offset += dims.n_pow3d_vars

        if dims.n_psds > 0:
            pass

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
