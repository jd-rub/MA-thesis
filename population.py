from individual import BaseIndividual, SampleCollection
import numpy as np
import bisect

class Population:
    def __init__(self) -> None:
        self.individuals = [] # List of BaseIndividuals, Sorted by fitness values
        self.best_collections_per_onset = {} # Dict of onset: SampleCollection
    
    def __str__(self) -> str:
        return "\n".join([str(individual) for individual in self.individuals])

    def calc_best_fitnesses(self):
        for individual in self.individuals:
            for collection in individual.sample_collections.values():
                if collection.fitness < self.best_collections_per_onset.get(collection.onset, SampleCollection()).fitness:
                    self.best_collections_per_onset[collection.onset] = collection

    def sort_individuals_by_fitness(self):
        self.individuals.sort(key=lambda item: item.fitness)

    def insert_individual(self, individual:BaseIndividual):
        bisect.insort_left(self.individuals, individual, key=lambda item: item.fitness)

    def remove_worst(self, n:int):
        self.individuals = self.individuals[:-n]

    def get_best_individual(self):
        return self.individuals[0]

    def to_feature_vector(self):
        raise NotImplementedError()