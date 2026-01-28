# --- Test Functions ---
import numpy as np
from differential_evolution import differential_evolution


def sphere_function(x):
    """Simple convex function. Min at 0."""
    return np.sum(x**2)


def rosenbrock_function(x):
    """Non-convex function. Min at (1, 1, ..., 1)."""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rastrigin_function(x):
    """Multimodal function (many local optima). Min at 0."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# --- Runner ---


def run_tests():
    print("🧪 Running Independent Differential Evolution Tests...\n")

    # TEST 1: Sphere Function (5 Dimensions)
    print("--- Test 1: Sphere Function (5D) ---")
    bounds = [(-5.0, 5.0)] * 5
    best_sol, best_val, _ = differential_evolution(
        sphere_function, bounds, pop_size=50, max_generations=200
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    assert best_val < 1e-3, "Failed to optimize Sphere function!"
    print("✅ Passed\n")

    # TEST 2: Rosenbrock Function (2 Dimensions)
    print("--- Test 2: Rosenbrock Function (2D) ---")
    bounds = [(-2.0, 2.0)] * 2
    best_sol, best_val, _ = differential_evolution(
        rosenbrock_function, bounds, pop_size=100, max_generations=1000
    )
    print(f"Result: {best_val:.6f} at {best_sol} (Expected 0.0 at [1. 1.])")
    assert best_val < 1e-2, "Failed to optimize Rosenbrock function!"
    print("✅ Passed\n")

    # TEST 3: Rastrigin Function (5 Dimensions - Hard!)
    print("--- Test 3: Rastrigin Function (5D) ---")
    bounds = [(-5.12, 5.12)] * 5
    best_sol, best_val, _ = differential_evolution(
        rastrigin_function, bounds, pop_size=100, max_generations=1000
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    assert best_val < 1e-1, "Failed to optimize Rastrigin function!"
    print("✅ Passed\n")


if __name__ == "__main__":
    run_tests()
