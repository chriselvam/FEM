"""
optimiser
=========

Defines the Optimiser class.
"""

import logging
from time import perf_counter as timer
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from aso.logging import format_array_for_logging
from aso.optimisation_problem import OptimisationProblem
from aso.optimisation_result import OptimisationResult

logger = logging.getLogger(__name__)


class Optimiser:
    """
    Contains various optimisation algorithms to solve an `OptimisationProblem`.

    Attributes
    ----------
    problem : OptimisationProblem
        The optimisation problem to be solved.
    x : numpy.ndarray
        Current design variable values.
    n : int
        Number of design variables.
    lm : numpy.ndarray
        Current Lagrange multipliers.
    """

    def __init__(
        self,
        problem: OptimisationProblem,
        x: NDArray,
        lm: NDArray | None = None,
    ) -> None:
        """Initialize an `Optimiser` instance.

        Parameters
        ----------
        problem : OptimisationProblem
            Optimisation problem to solve.
        x : numpy.ndarray
            Initial design variables.
        lm : numpy.ndarray, optional
            Initial Lagrange multipliers.

        Notes
        -----
        The given array of design variables will be modified in place.
        Hence, the optimiser does currently not reuturn the optimised
        design variables but only the number of outer-loop iterations.
        This behavior may change in future versions.
        """
        self.problem = problem
        self.x = x
        self.n = x.size

        # Check and, if necessary, initialise the Lagrange multipliers:
        if lm is None:
            self.lm = np.zeros(problem.m + problem.me)
        elif lm.size != problem.m + problem.me:
            raise ValueError(
                "The number of Lagrange multipliers must match the number of constraints."
            )
        else:
            self.lm = lm

    def optimise(
        self,
        algorithm: Literal[
            "SQP",
            "MMA",
            "STEEPEST_DESCENT",
            "CONJUGATE_GRADIENTS",
        ] = "SQP",
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        Distinguish constrained and unconstrained optimization problems
        and call an appropriate optimisation function.

        Parameters
        ----------
        algorithm : str, default: "SQP"
            Algorithm to use.
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect (intermediate) optimization results.

        Returns
        -------
        iteration : int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If `algorithm` is unknown or not suitable for constrained
            optimisation.
        """

        start = timer()

        if self.problem.constrained:
            match algorithm:
                case "SQP":
                    iteration = self.sqp_constrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case "MMA":
                    iteration = self.mma()
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for constrained optimisation."
                    )
        else:
            match algorithm:
                case "STEEPEST_DESCENT":
                    iteration = self.steepest_descent(
                        iteration_limit=iteration_limit,
                    )
                case "CONJUGATE_GRADIENTS":
                    iteration = self.conjugate_gradients(
                        iteration_limit=iteration_limit,
                    )
                case "SQP":
                    iteration = self.sqp_unconstrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for unconstrained optimisation."
                    )

        end = timer()
        elapsed_ms = round((end - start) * 1000, 3)

        if iteration == -1:
            logger.info(
                f"Algorithm {algorithm} failed to converge in {elapsed_ms} ms after {iteration} "
                f"iterations. Consider using another algorithm or increasing the iteration limit.",
            )
        else:
            logger.info(
                f"Algorithm {algorithm} converged in {elapsed_ms} ms after {iteration} "
                f"iterations. Optimised design variables: {format_array_for_logging(self.x)}",
            )

        return iteration

    def steepest_descent(
        self,
        iteration_limit: int = 1000,
    ) -> int:
        """Steepest-descent algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer loop iterations.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def conjugate_gradients(
        self,
        iteration_limit: int = 1000,
        beta_formula: Literal[
            "FLETCHER-REEVES",
            "POLAK-RIBIERE",
            "HESTENES-STIEFEL",
            "DAI-YUAN",
        ] = "FLETCHER-REEVES",
    ) -> int:
        """Conjugate-gradient algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        beta_formula : str, : optional
            Heuristic formula for computing the conjugation factor beta.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_unconstrained(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """SQP algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : str, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_constrained(
        self,
        iteration_limit: int = 1000,
        working_set: list[int] | None = None,
        working_set_size: int | None = None,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        SQP algorithm with an active-set strategy for constrained
        optimisation.

        Parameters `m_w` and `working_set` are currently ignored.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        working_set : list of int, optional
            Initial working set.
        working_set_size : int, optional
            Size of the working set (ignored if `working_set` is provided).
        callback : callable, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If the size of the working set is too large or too small.

        References
        ----------
        .. [1] K. Schittkowski, "An Active Set Strategy for Solving Optimization Problems with up to 200,000,000 Nonlinear Constraints." Accessed: May 25, 2025. [Online]. Available: https://klaus-schittkowski.de/SC_NLPQLB.pdf
        """
        ...

    def mma(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        MMA algorithm for constrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect intermediate results.
        """
        ...

    def converged(
        self,
        gradient: NDArray,
        constraints: NDArray | None = None,
        gradient_tol: float = 1e-5,
        constraint_tol: float = 1e-5,
        complementarity_tol: float = 1e-5,
    ) -> bool:
        """
        Check convergence according to the first-order necessary (KKT)
        conditions assuming LICQ.

        See, for example, Theorem 12.1 in [1]_.

        Parameters
        ----------
        gradient : numpy.ndarray
            Current gradient of the Lagrange function with respect to
            the design variables.
        constraints : numpy.ndarray, optional
            Current constraint values.
        gradient_tol : float, default: 1e-5
            Tolerance applied to each component of the gradient.
        constraint_tol : float, default: 1e-5
            Tolerance applied to each constraint.
        complementarity_tol : float, default: 1e-5
            Tolerance applied to each complementarity condition.

        References
        ----------
        .. [1] J. Nocedal and S. J. Wright, Numerical Optimization. Springer New York, 2006. doi: https://doi.org/10.1007/978-0-387-40065-5.
        """
        ...

    def line_search(
        self,
        direction: NDArray,
        alpha_ini: float = 1,
        alpha_min: float = 1e-6,
        alpha_max: float = 1,
        algorithm: Literal[
            "WOLFE",
            "STRONG_WOLFE",
            "GOLDSTEIN-PRICE",
        ] = "STRONG_WOLFE",
        m1: float = 0.01,
        m2: float = 0.90,
        callback: Callable[[OptimisationResult], Any] | None = None,
        callback_iteration: int | None = None,
    ) -> float:
        """
        Perform a line search and returns an approximately optimal step size.

        Parameters
        ----------
        direction : numpy.ndarray
            Search direction.
        alpha_ini : float
            Initial step size.
        alpha_min : float, optional
            Minimum step size.
        alpha_max : float
            Maximum step size.
        algorithm : str, optional
            Line search algorithm to use.
        m1 : float, optional
            Parameter for the sufficient decrease condition.
        m2 : float, optional
            Parameter for the curvature condition.
        callback : callable, optional
            Callback function for collecting intermediate results.
        callback_iteration : int, optional
            Iteration number for the callback function.

        Returns
        -------
        float
            Approximately optimal step size.
        """
        ...
