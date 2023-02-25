from individual import BaseIndividual
import numpy as np

class Population:
    def __init__(self) -> None:
        self.individuals = [] # List of BaseIndividuals
        self.best_collections_per_onset = {} # Dict of onset: SampleCollection
    
    def calc_best_fitnesses(self):
        for individual in self.individual:
            for onset in individual.onset_locations:
                if self.best_collections_per_onset.get(onset, np.inf) > individual.fitness_by_onset[onset]:
                    self.best_collections_per_onset[onset] = individual.fitness_by_onset[onset]

    def to_feature_vector(self):
        raise NotImplementedError()