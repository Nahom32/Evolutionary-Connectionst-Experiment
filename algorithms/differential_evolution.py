import numpy as np


def differential_evolution(
    objective_func,
    bounds,
    pop_size=50,
    mutation_factor=0.8,
    crossover_prob=0.7,
    max_generations=1000,
    tol=1e-6,
):
    """
    Standard Differential Evolution (DE/rand/1/bin) for global minimization.

    Args:
        objective_func (callable): The function to minimize. Takes a 1D numpy array, returns a float.
        bounds (list of tuples): List of (min, max) for each dimension. E.g., [(-5, 5), (-5, 5)].
        pop_size (int): Number of individuals in the population.
        mutation_factor (float): Differential weight (F). Typically between 0.5 and 1.0.
        crossover_prob (float): Crossover probability (CR). Between 0.0 and 1.0.
        max_generations (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (stops if variance of population fitness is low).

    Returns:
        best_vector (np.array): The best solution found.
        best_fitness (float): The value of the objective function at the best solution.
        history (list): List of best fitness values per generation (for plotting).
    """
    # 1. Initialization
    bounds = np.array(bounds)
    dim = len(bounds)

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    population = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)

    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])

    # Track best solution
    best_idx = np.argmin(fitness)
    best_vector = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    history = [best_fitness]

    # 2. Main Loop
    for gen in range(max_generations):
        # Create a new population for the next generation
        new_population = np.copy(population)

        for i in range(pop_size):
            # --- Mutation (DE/rand/1) ---
            # Select 3 distinct random indices distinct from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            x_a = population[a]
            x_b = population[b]
            x_c = population[c]

            # Create mutant vector: v = a + F * (b - c)
            mutant = x_a + mutation_factor * (x_b - x_c)

            # Clip mutant to bounds (optional but recommended)
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover (Binomial) ---
            # Create trial vector by mixing target (i) and mutant
            cross_points = np.random.rand(dim) < crossover_prob

            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial = np.where(cross_points, mutant, population[i])

            # --- Selection ---
            # Greedy selection: if trial is better, keep it
            f_trial = objective_func(trial)

            if f_trial < fitness[i]:
                new_population[i] = trial
                fitness[i] = f_trial

                # Update global best if needed
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_vector = trial.copy()

        population = new_population
        history.append(best_fitness)

        # Convergence Check (Variance of fitness is small)
        if np.std(fitness) < tol:
            break

    return best_vector, best_fitness, history
