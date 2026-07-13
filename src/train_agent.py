import gymnasium as gym
import numpy
from policy import policy_action

def mutate_parameters(vector, prob=0.09, scale=0.3):
    mask = numpy.random.random(vector.shape[0]) < prob
    if numpy.any(mask):
        vector[mask] += numpy.random.randn(numpy.count_nonzero(mask)) * scale
    return vector

def blend(parent_a, parent_b):
    ratio = numpy.random.random()
    return (1 - ratio) * parent_b + ratio * parent_a

def compute_fitness(individual, runs=15):
    total_score = 0.0
    for _ in range(runs):
        env = gym.make("LunarLander-v3")
        state, _ = env.reset()
        temp = False
        score = 0.0
        while not temp:
            action = policy_action(individual, state)
            step_out = env.step(action)
            state = step_out[0]
            score += step_out[1]
            temp = step_out[2] or step_out[3]
        total_score += score
        env.close()

    return total_score/runs

def evolve(population_size=150, itterations=3000, best_ratio=0.25):
    dim = 212
    best_count = int(population_size * best_ratio)
    best_score = -numpy.inf
    best_vector = None
    population = numpy.random.randn(population_size, dim)
    for gen in range(itterations):
        fitness_vals = numpy.array([compute_fitness(p) for p in population])
        order = numpy.argsort(fitness_vals)[::-1]
        best = population[order[:best_count]]
        top_score = fitness_vals[order[0]]
        if top_score > best_score:
            best_score = top_score
            best_vector = population[order[0]].copy()
        print(f"Itteration {gen+1} | Best = {best_score:.2f}")
        next_population = best.tolist()
        while len(next_population) < population_size:
            idx = numpy.random.choice(best_count, 2, replace=False)
            parent1, parent2 = best[idx[0]], best[idx[1]]
            child = blend(parent1, parent2)
            child = mutate_parameters(child)
            next_population.append(child)
        population = numpy.array(next_population)
    return best_vector

if __name__ == "__main__":
    result = evolve()
    file_out = "best_policy.npy"
    numpy.save(file_out, result)
    print("Saved:", file_out)