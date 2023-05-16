import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .jaccard import jaccard_error, class_mode
from .population import Population

class PopulationLogger:
    """Logs population fitness for each step.
    """
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

class CombinedLogger():
    """Given a set of annotations, logs the jaccard approximation
    error over the annotated onsets, as well as population fitness.
    """
    def __init__(self, annotations, logging_interval:int = 1) -> None:
        self.annotations = annotations
        self.logging_interval = logging_interval
        self.logged_errors = []
        self.logged_fitnesses = []

    def log_errors(self, pop):
        j_i = jaccard_error(population=pop, annotations=self.annotations, mode=class_mode.INSTRUMENTS)
        j_p = jaccard_error(population=pop, annotations=self.annotations, mode=class_mode.PITCHES)
        j_ip = jaccard_error(population=pop, annotations=self.annotations, mode=class_mode.COMBINED)
        self.logged_errors.append((j_i, j_p, j_ip))

    def log_fitness(self, pop):
        self.logged_fitnesses.append(np.mean([record.fitness for record in pop.archive.values()]))
        
    def log_population(self, pop, step):
        if step % self.logging_interval == 0:
            self.log_errors(pop)
            self.log_fitness(pop)

    def to_csv(self, filename:str):
        df = pd.DataFrame(self.logged_errors, columns=["j_i", "j_p", "j_ip"])
        df["fitness"] = self.logged_fitnesses
        df.to_csv(filename, index=False, header=True)
        