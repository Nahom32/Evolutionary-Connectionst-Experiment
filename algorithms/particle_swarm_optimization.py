import numpy as np


def particle_swarm_optimization(
    objective_func,
    bounds,
    num_particles=30,
    max_iter=100,
    w=0.5,
    c1=1.5,
    c2=1.5,
    tol=1e-6,
):
    """
    Standard Particle Swarm Optimization (PSO) for global minimization.

    Args:
        objective_func (callable): The function to minimize.
        bounds (list of tuples): List of (min, max) for each dimension.
        num_particles (int): Number of particles in the swarm.
        max_iter (int): Maximum number of iterations.
        w (float): Inertia weight. Controls how much the particle keeps its previous velocity.
        c1 (float): Cognitive parameter. Pulls particle toward its own best position.
        c2 (float): Social parameter. Pulls particle toward the swarm's best position.
        tol (float): Tolerance for convergence stopping.

    Returns:
        g_best_pos (np.array): The best solution found globally.
        g_best_val (float): The value of the objective function at the best solution.
        history (list): List of best fitness values per generation.
    """
    # 1. Initialization
    bounds = np.array(bounds)
    dim = len(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]

    # Initialize positions randomly within bounds
    # Shape: (num_particles, dim)
    positions = min_b + (max_b - min_b) * np.random.rand(num_particles, dim)

    # Initialize velocities (start with zero or small random values)
    velocities = np.zeros((num_particles, dim))

    # Initialize Personal Best (p_best)
    p_best_pos = positions.copy()
    p_best_val = np.array([objective_func(p) for p in positions])

    # Initialize Global Best (g_best)
    g_best_idx = np.argmin(p_best_val)
    g_best_pos = p_best_pos[g_best_idx].copy()
    g_best_val = p_best_val[g_best_idx]

    history = [g_best_val]

    # 2. Main Loop
    for i in range(max_iter):
        # Generate random coefficients for this iteration (r1, r2)
        # Shape: (num_particles, dim)
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)

        # --- Update Velocity ---
        # v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        cognitive_component = c1 * r1 * (p_best_pos - positions)
        social_component = c2 * r2 * (g_best_pos - positions)

        velocities = (w * velocities) + cognitive_component + social_component

        # --- Update Position ---
        positions = positions + velocities

        # Boundary Handling: Clip positions to stay valid
        positions = np.clip(positions, min_b, max_b)

        # --- Evaluate Fitness ---
        current_vals = np.array([objective_func(p) for p in positions])

        # --- Update Personal Bests ---
        # Identify particles that found a new personal best
        improved_indices = current_vals < p_best_val
        p_best_pos[improved_indices] = positions[improved_indices]
        p_best_val[improved_indices] = current_vals[improved_indices]

        # --- Update Global Best ---
        min_val_idx = np.argmin(current_vals)
        if current_vals[min_val_idx] < g_best_val:
            g_best_val = current_vals[min_val_idx]
            g_best_pos = positions[min_val_idx].copy()

        history.append(g_best_val)

        # Convergence Check (optional)
        if g_best_val < tol:
            # Depending on problem, you might want to stop early if target reached
            # break
            pass

    return g_best_pos, g_best_val, history
