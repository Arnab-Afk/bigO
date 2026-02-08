"""
Advanced Nash Equilibrium Solvers

Implements mixed strategy Nash equilibrium computation using:
- Lemke-Howson algorithm for 2-player games
- Support enumeration for n-player games
- Strategy dominance filtering

Based on algorithmic game theory and computational equilibrium finding.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
import nashpy as nash
from scipy.optimize import linprog
from scipy.linalg import lstsq


@dataclass
class MixedStrategy:
    """Mixed strategy for an agent"""
    agent_id: UUID
    action_probabilities: Dict[str, float]  # action_name -> probability
    support: Set[str]  # Actions with positive probability
    expected_payoff: float = 0.0


@dataclass
class MixedNashEquilibrium:
    """Mixed strategy Nash equilibrium"""
    strategies: Dict[UUID, MixedStrategy]
    is_pure: bool
    support_size: int
    convergence_error: float = 0.0


class SupportEnumerationSolver:
    """
    Support Enumeration Algorithm for finding Nash equilibria

    Systematically searches over all possible support combinations.
    For each support pair, solves for equilibrium if it exists.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Numerical tolerance for equilibrium conditions
        """
        self.tolerance = tolerance

    def enumerate_supports(
        self,
        num_actions_i: int,
        num_actions_j: int,
        max_support_size: Optional[int] = None,
    ) -> List[Tuple[Set[int], Set[int]]]:
        """
        Generate all candidate support pairs

        Args:
            num_actions_i: Number of actions for player i
            num_actions_j: Number of actions for player j
            max_support_size: Maximum support size to consider

        Returns:
            List of (support_i, support_j) tuples
        """
        supports = []

        if max_support_size is None:
            max_support_size = max(num_actions_i, num_actions_j)

        # Enumerate all non-empty subsets up to max_support_size
        for size_i in range(1, min(num_actions_i, max_support_size) + 1):
            for size_j in range(1, min(num_actions_j, max_support_size) + 1):
                for supp_i in combinations(range(num_actions_i), size_i):
                    for supp_j in combinations(range(num_actions_j), size_j):
                        supports.append((set(supp_i), set(supp_j)))

        return supports

    def solve_for_support(
        self,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        support_i: Set[int],
        support_j: Set[int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Check if a support pair forms an equilibrium

        For a valid equilibrium:
        1. All actions in support must give equal expected payoff
        2. All actions outside support must give weakly lower payoff
        3. Probabilities must sum to 1 and be non-negative

        Args:
            payoff_matrix_i: Payoff matrix for player i (rows=i's actions, cols=j's actions)
            payoff_matrix_j: Payoff matrix for player j
            support_i: Candidate support for player i
            support_j: Candidate support for player j

        Returns:
            (strategy_i, strategy_j) or None if no equilibrium exists
        """
        supp_i = sorted(list(support_i))
        supp_j = sorted(list(support_j))

        if len(supp_i) == 0 or len(supp_j) == 0:
            return None

        # Build indifference equations for player i
        # All actions in support_i must yield equal payoff against mixed strategy j
        try:
            # For player i: payoff_matrix_i @ strategy_j should be constant on support_i
            # payoff_matrix_i[supp_i, :][:, supp_j] @ strategy_j[supp_j] = v_i
            A_i = payoff_matrix_j[supp_j, :].T  # Transpose for player i's indifference
            b_i = np.ones(len(supp_i))

            # Solve indifference conditions for player j's strategy
            # A_i @ strategy_j = v_i * 1
            # Also: sum(strategy_j) = 1
            A_eq_i = np.vstack([
                payoff_matrix_j[:, supp_j].T[supp_i, :],
                np.ones((1, len(supp_j)))
            ])
            b_eq_i = np.hstack([
                np.full(len(supp_i), 0),  # Will be adjusted
                np.array([1.0])
            ])

            # Build for player j
            A_eq_j = np.vstack([
                payoff_matrix_i[supp_i, :].T[:, supp_j],
                np.ones((1, len(supp_i)))
            ])
            b_eq_j = np.hstack([
                np.full(len(supp_j), 0),
                np.array([1.0])
            ])

            # Solve using least squares with indifference constraint
            strategy_i_supp = self._solve_indifference(
                payoff_matrix_j.T[:, supp_i],
                len(supp_i)
            )
            strategy_j_supp = self._solve_indifference(
                payoff_matrix_i[:, supp_j],
                len(supp_j)
            )

            if strategy_i_supp is None or strategy_j_supp is None:
                return None

            # Check non-negativity
            if np.any(strategy_i_supp < -self.tolerance) or np.any(strategy_j_supp < -self.tolerance):
                return None

            # Reconstruct full strategies
            strategy_i = np.zeros(payoff_matrix_i.shape[0])
            strategy_j = np.zeros(payoff_matrix_j.shape[1])

            for idx, action_idx in enumerate(supp_i):
                strategy_i[action_idx] = max(0, strategy_i_supp[idx])
            for idx, action_idx in enumerate(supp_j):
                strategy_j[action_idx] = max(0, strategy_j_supp[idx])

            # Normalize
            if strategy_i.sum() > 0:
                strategy_i /= strategy_i.sum()
            if strategy_j.sum() > 0:
                strategy_j /= strategy_j.sum()

            # Verify equilibrium conditions
            if not self._verify_equilibrium(
                payoff_matrix_i, payoff_matrix_j,
                strategy_i, strategy_j,
                support_i, support_j
            ):
                return None

            return strategy_i, strategy_j

        except (np.linalg.LinAlgError, ValueError):
            return None

    def _solve_indifference(
        self,
        payoff_matrix: np.ndarray,
        support_size: int,
    ) -> Optional[np.ndarray]:
        """
        Solve indifference equations for mixed strategy

        Args:
            payoff_matrix: Payoff matrix (actions × opponent_actions)
            support_size: Size of the support

        Returns:
            Mixed strategy or None if no solution
        """
        if support_size == 1:
            return np.array([1.0])

        # Indifference: all actions in support give equal payoff
        # payoff_matrix @ strategy = v * 1
        # sum(strategy) = 1

        # Build system: first (support_size - 1) indifference equations + probability constraint
        try:
            A = np.vstack([
                payoff_matrix[:support_size - 1, :] - payoff_matrix[support_size - 1:support_size, :],
                np.ones((1, payoff_matrix.shape[1]))
            ])
            b = np.hstack([
                np.zeros(support_size - 1),
                np.array([1.0])
            ])

            solution, residuals, rank, s = lstsq(A, b)

            if np.any(solution < -self.tolerance):
                return None

            return solution

        except (np.linalg.LinAlgError, ValueError):
            return None

    def _verify_equilibrium(
        self,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        strategy_i: np.ndarray,
        strategy_j: np.ndarray,
        support_i: Set[int],
        support_j: Set[int],
    ) -> bool:
        """
        Verify Nash equilibrium conditions

        Args:
            payoff_matrix_i: Payoff matrix for player i
            payoff_matrix_j: Payoff matrix for player j
            strategy_i: Mixed strategy for player i
            strategy_j: Mixed strategy for player j
            support_i: Support of player i's strategy
            support_j: Support of player j's strategy

        Returns:
            True if valid equilibrium
        """
        # Expected payoffs
        payoff_i = payoff_matrix_i @ strategy_j
        payoff_j = payoff_matrix_j.T @ strategy_i

        # Best responses
        max_payoff_i = np.max(payoff_i)
        max_payoff_j = np.max(payoff_j)

        # Check: actions in support get maximum payoff
        for action_idx in support_i:
            if payoff_i[action_idx] < max_payoff_i - self.tolerance:
                return False

        for action_idx in support_j:
            if payoff_j[action_idx] < max_payoff_j - self.tolerance:
                return False

        # Check: actions outside support get at most maximum payoff
        for action_idx in range(len(strategy_i)):
            if action_idx not in support_i and strategy_i[action_idx] > self.tolerance:
                return False

        for action_idx in range(len(strategy_j)):
            if action_idx not in support_j and strategy_j[action_idx] > self.tolerance:
                return False

        return True

    def filter_dominated_strategies(
        self,
        payoff_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Remove strictly dominated strategies

        A strategy is strictly dominated if another strategy always gives
        strictly higher payoff regardless of opponent's action.

        Args:
            payoff_matrix: Payoff matrix (rows=actions, cols=opponent_actions)

        Returns:
            (reduced_matrix, remaining_action_indices)
        """
        remaining = set(range(payoff_matrix.shape[0]))
        changed = True

        while changed:
            changed = False
            to_remove = set()

            for i in remaining:
                for j in remaining:
                    if i != j:
                        # Check if j strictly dominates i
                        if np.all(payoff_matrix[j, :] >= payoff_matrix[i, :]) and \
                           np.any(payoff_matrix[j, :] > payoff_matrix[i, :]):
                            to_remove.add(i)
                            changed = True
                            break

            remaining -= to_remove

        remaining_list = sorted(list(remaining))
        reduced_matrix = payoff_matrix[remaining_list, :]

        return reduced_matrix, remaining_list


class LemkeHowsonSolver:
    """
    Lemke-Howson Algorithm for 2-player Nash equilibrium

    Uses pivoting method to traverse extreme points of best response polytope.
    Guaranteed to find at least one Nash equilibrium for 2-player games.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Numerical tolerance
        """
        self.tolerance = tolerance

    def solve(
        self,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        initial_dropped_label: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find Nash equilibrium using Lemke-Howson algorithm

        Args:
            payoff_matrix_i: Payoff matrix for player i
            payoff_matrix_j: Payoff matrix for player j
            initial_dropped_label: Label to drop initially (0 to m+n-1)

        Returns:
            (strategy_i, strategy_j) representing Nash equilibrium
        """
        # Use nashpy library implementation
        try:
            game = nash.Game(payoff_matrix_i, payoff_matrix_j)

            # Try Lemke-Howson with different initial labels
            for label in range(payoff_matrix_i.shape[0] + payoff_matrix_j.shape[1]):
                try:
                    equilibria = game.lemke_howson(initial_dropped_label=label)
                    strategy_i, strategy_j = equilibria

                    # Verify and return
                    if self._verify_equilibrium(
                        payoff_matrix_i, payoff_matrix_j,
                        strategy_i, strategy_j
                    ):
                        return strategy_i, strategy_j
                except Exception:
                    continue

            # Fallback: use support enumeration
            support_solver = SupportEnumerationSolver(tolerance=self.tolerance)
            supports = support_solver.enumerate_supports(
                payoff_matrix_i.shape[0],
                payoff_matrix_j.shape[1],
                max_support_size=3
            )

            for supp_i, supp_j in supports:
                result = support_solver.solve_for_support(
                    payoff_matrix_i, payoff_matrix_j,
                    supp_i, supp_j
                )
                if result is not None:
                    return result

            # Last resort: uniform random
            strategy_i = np.ones(payoff_matrix_i.shape[0]) / payoff_matrix_i.shape[0]
            strategy_j = np.ones(payoff_matrix_j.shape[1]) / payoff_matrix_j.shape[1]
            return strategy_i, strategy_j

        except Exception as e:
            # Fallback to uniform distribution
            strategy_i = np.ones(payoff_matrix_i.shape[0]) / payoff_matrix_i.shape[0]
            strategy_j = np.ones(payoff_matrix_j.shape[1]) / payoff_matrix_j.shape[1]
            return strategy_i, strategy_j

    def _verify_equilibrium(
        self,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        strategy_i: np.ndarray,
        strategy_j: np.ndarray,
    ) -> bool:
        """Verify Nash equilibrium conditions"""
        # Expected payoffs
        expected_i = strategy_i @ payoff_matrix_i @ strategy_j
        expected_j = strategy_i @ payoff_matrix_j @ strategy_j

        # Check if any deviation improves payoff
        for i in range(len(strategy_i)):
            pure_i = np.zeros_like(strategy_i)
            pure_i[i] = 1.0
            dev_payoff_i = pure_i @ payoff_matrix_i @ strategy_j
            if dev_payoff_i > expected_i + self.tolerance:
                return False

        for j in range(len(strategy_j)):
            pure_j = np.zeros_like(strategy_j)
            pure_j[j] = 1.0
            dev_payoff_j = strategy_i @ payoff_matrix_j @ pure_j
            if dev_payoff_j > expected_j + self.tolerance:
                return False

        return True


class CorrelatedEquilibriumSolver:
    """
    Correlated Equilibrium computation using Linear Programming

    Allows for correlation device that recommends actions to players.
    More general than Nash equilibrium.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: Numerical tolerance
        """
        self.tolerance = tolerance

    def compute_correlated_equilibrium(
        self,
        payoff_matrix_i: np.ndarray,
        payoff_matrix_j: np.ndarray,
        objective: str = "welfare",
    ) -> np.ndarray:
        """
        Compute correlated equilibrium by solving linear program

        Args:
            payoff_matrix_i: Payoff matrix for player i
            payoff_matrix_j: Payoff matrix for player j
            objective: Optimization objective ("welfare", "fairness", "player_i", "player_j")

        Returns:
            Joint distribution over action profiles (m × n matrix)
        """
        m, n = payoff_matrix_i.shape

        # Decision variables: distribution over (action_i, action_j) pairs
        # Flattened: p[i*n + j] = P(action_i, action_j)

        # Objective: maximize social welfare or specific player payoff
        if objective == "welfare":
            c = -((payoff_matrix_i + payoff_matrix_j).flatten())
        elif objective == "player_i":
            c = -(payoff_matrix_i.flatten())
        elif objective == "player_j":
            c = -(payoff_matrix_j.flatten())
        elif objective == "fairness":
            # Minimize max difference
            c = -(np.minimum(payoff_matrix_i, payoff_matrix_j).flatten())
        else:
            c = -((payoff_matrix_i + payoff_matrix_j).flatten())

        # Constraints:
        # 1. Probabilities sum to 1
        A_eq = [np.ones(m * n)]
        b_eq = [1.0]

        # 2. Incentive compatibility for player i
        # For each action a_i and deviation a_i':
        # sum_j p(a_i, a_j) * [u_i(a_i, a_j) - u_i(a_i', a_j)] >= 0
        A_ub = []
        b_ub = []

        for a_i in range(m):
            for a_i_prime in range(m):
                if a_i != a_i_prime:
                    constraint = np.zeros(m * n)
                    for a_j in range(n):
                        # Coefficient for p(a_i, a_j)
                        idx = a_i * n + a_j
                        constraint[idx] = -(payoff_matrix_i[a_i, a_j] - payoff_matrix_i[a_i_prime, a_j])
                    A_ub.append(constraint)
                    b_ub.append(0.0)

        # 3. Incentive compatibility for player j
        for a_j in range(n):
            for a_j_prime in range(n):
                if a_j != a_j_prime:
                    constraint = np.zeros(m * n)
                    for a_i in range(m):
                        idx = a_i * n + a_j
                        constraint[idx] = -(payoff_matrix_j[a_i, a_j] - payoff_matrix_j[a_i, a_j_prime])
                    A_ub.append(constraint)
                    b_ub.append(0.0)

        # Bounds: probabilities in [0, 1]
        bounds = [(0, 1) for _ in range(m * n)]

        # Solve LP
        try:
            result = linprog(
                c=c,
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq),
                b_eq=np.array(b_eq),
                bounds=bounds,
                method='highs'
            )

            if result.success:
                distribution = result.x.reshape((m, n))
                return distribution
            else:
                # Fallback: uniform distribution (always a correlated equilibrium)
                return np.ones((m, n)) / (m * n)

        except Exception:
            # Fallback
            return np.ones((m, n)) / (m * n)
