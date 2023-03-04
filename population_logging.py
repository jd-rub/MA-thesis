import numpy as np
import matplotlib.pyplot as plt

from population import Population

class PopulationLogger:
    def __init__(self) -> None:
        self.logged_steps = []
        self.mean_fitness = []
        self.mean_fitness_best_records = []
        self.elitist_fitness = []
    
    def log_population(self, pop:Population, step:int) -> None:
        self.logged_steps.append(step)
        self.mean_fitness.append(np.mean([individual.fitness for individual in pop.individuals]))
        self.mean_fitness_best_records.append(np.mean([record.fitness for i, record in enumerate(pop.archive.values())]))
        self.elitist_fitness.append(pop.get_best_individual().fitness)

    def plot_log(self):
        raise NotImplementedError