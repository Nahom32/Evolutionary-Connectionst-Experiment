from particle_swarm_optimization import particle_swarm_optimization
import numpy as np
# --- Test Functions ---


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
    print("🐝 Running Independent Particle Swarm Optimization Tests...\n")

    # TEST 1: Sphere Function (5 Dimensions)
    # Simple bowl shape, particles should slide to bottom easily.
    print("--- Test 1: Sphere Function (5D) ---")
    bounds = [(-5.0, 5.0)] * 5
    best_sol, best_val, _ = particle_swarm_optimization(
        sphere_function, bounds, num_particles=30, max_iter=100
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    assert best_val < 1e-3, "Failed to optimize Sphere function!"
    print("✅ Passed\n")

    # TEST 2: Rosenbrock Function (2 Dimensions)
    # A curved valley. Particles often struggle to find the exact bottom.
    print("--- Test 2: Rosenbrock Function (2D) ---")
    bounds = [(-2.0, 2.0)] * 2
    best_sol, best_val, _ = particle_swarm_optimization(
        rosenbrock_function, bounds, num_particles=50, max_iter=200
    )
    print(f"Result: {best_val:.6f} at {best_sol} (Expected 0.0 at [1. 1.])")
    assert best_val < 1e-2, "Failed to optimize Rosenbrock function!"
    print("✅ Passed\n")

    # TEST 3: Rastrigin Function (5 Dimensions - Hard!)
    # Many holes (local minima). Particles must have enough velocity to jump out of them.
    print("--- Test 3: Rastrigin Function (5D) ---")
    bounds = [(-5.12, 5.12)] * 5
    best_sol, best_val, _ = particle_swarm_optimization(
        rastrigin_function,
        bounds,
        num_particles=100,
        max_iter=500,
        w=0.7,  # Higher inertia helps exploration
    )
    print(f"Result: {best_val:.6f} (Expected ~0.0)")
    # Note: PSO sometimes gets stuck in local optima for Rastrigin if params aren't tuned perfect.
    # We use a looser assertion here compared to DE.
    assert best_val < 1.0, "Failed to optimize Rastrigin function!"
    print("✅ Passed\n")


if __name__ == "__main__":
    run_tests()
