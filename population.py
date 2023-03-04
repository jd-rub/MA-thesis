from individual import BaseIndividual
import bisect
import pickle
from typing import Union
import numpy as np

class Population:
    def __init__(self) -> None:
        self.individuals = [] # List of BaseIndividuals, Sorted by fitness values
        self.archive = {} # Dict of onset: SampleCollection # TODO: Refactor and expand archive to hold records for each instrument
    
    def __str__(self) -> str:
        return "\n".join([str(individual) for individual in self.individuals])

    def init_archive(self, onsets:Union[list[int], np.ndarray[int]]) -> None:
        """Initializes the archive of best individuals per onset.

        Parameters
        ----------
        onsets : list[int]
            list of onsets from the target piece
        """
        for individual in self.individuals:
            for i, onset in enumerate(onsets):
                if individual.fitness_per_onset[i] < self.archive.get(onset, BaseIndividual()).fitness:
                    self.archive[onset] = ArchiveRecord(onset=onset, fitness=individual.fitness_per_onset[i], individual=individual)

    def sort_individuals_by_fitness(self):
        """Sorts the list of individuals by fitness. Meant to only be done upon initialization. 
        """
        self.individuals.sort(key=lambda item: item.fitness)

    def insert_individual(self, individual:BaseIndividual):
        """Inserts an individual into the population. 
        Insertion is done into the self.individuals list, preserving fitness order.

        Parameters
        ----------
        individual : BaseIndividual
            An individual containing one or more samples and calculated fitness value.
        """
        # Insert individual
        bisect.insort_left(self.individuals, individual, key=lambda item: item.fitness)
        # Update record of best onset approximations
        for i, onset in enumerate(self.archive):
            if individual.fitness_per_onset[i] < self.archive[onset].fitness:
                self.archive[onset].individual = individual
                self.archive[onset].fitness = individual.fitness_per_onset[i]

    def remove_worst(self, n:int):
        """Removes the worst n individuals from the population.
        This method assumes that self.individuals is already sorted by fitness.

        Parameters
        ----------
        n : int
            Number of individuals to remove from the population.
        """
        self.individuals = self.individuals[:-n]

    def get_best_individual(self) -> BaseIndividual:
        """Returns the individual with highest fitness.

        Returns
        -------
        BaseIndividual
            Individual with the highest fitness
        """
        return self.individuals[0]
    
    def save_as_file(self, filename:str):
        """Saves the population to a pickled file.

        Parameters
        ----------
        filename : str
            Desired name of the file.
        """
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)
        
    @classmethod 
    def from_file(cls, filename:str):
        """Loads a population from a pickled file.

        Parameters
        ----------
        filename : str
            Name of the population file.

        Returns
        -------
        Population
            The loaded population contained in the given file.
        """
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

class ArchiveRecord():
    def __init__(self, onset:int, fitness:float=None, individual:BaseIndividual=None) -> None:
        self.onset = onset
        self.fitness = fitness
        self.individual = individual