from sample_library import SampleLibrary
from individual import BaseIndividual
from mutations import Mutator
from fitness import fitness, multi_onset_fitness_cached
from population import Population
from population_logging import PopulationLogger
from target import Target
from typing import Union
import numpy as np
from tqdm import tqdm

MAX_SAMPLES_PER_ONSET = 5
STOPPING_FITNESS = 0.001

def base_algorithm_1plus1_single_onset(target_y:Union[np.ndarray, list], max_steps:int, sample_lib:SampleLibrary):
    mutator = Mutator(sample_lib)
    target = Target(target_y)

    best_individual = BaseIndividual.create_random_individual(sample_lib=sample_lib, phi=1)    
    best_individual.fitness = fitness(target.y, best_individual.to_mixdown())

    for n in (pbar := tqdm(range(max_steps))):
        candidate_individual = mutator.mutate_individual(BaseIndividual.from_copy(best_individual))
        candidate_individual.fitness = fitness(target.y, candidate_individual.to_mixdown())

        if candidate_individual.fitness < best_individual.fitness:
            best_individual = candidate_individual
            pbar.set_postfix_str('\t' * 100 + f"Best individual: {str(best_individual)} with fitness {best_individual.fitness}")

        if best_individual.fitness < STOPPING_FITNESS:
            break
    
    return best_individual

def approximate_piece(target_y:Union[np.ndarray, list], max_steps:int, sample_lib:SampleLibrary, popsize:int, n_offspring:int, onset_frac:float, mutator:Mutator=None, logger:PopulationLogger=None) -> Population:
    """Evolutionary approximation of a polyphonic musical piece

    Parameters
    ----------
    target_y : Union[np.ndarray, list]
        signal of the target musical piece, as imported by librosa
    max_steps : int
        Maximum number of iterations (generations) before termination
    sample_lib : SampleLibrary
        Library of samples which define the algorithm's search space
    popsize : int
        Size of the population (µ)
    n_offspring : int
        Number of offspring (λ) per generation 
    onset_frac : float
        Fraction of approximated onsets (φ) per individual
    mutator : Mutator
        Pre-initialized Mutator object that supports the mutate_individual(BaseIndividual) method
    logger: PopulationLogger
        Logging object, if desired. Can be None to omit logging

    Returns
    -------
    Population
        The full population of individual approximations after max_steps of iterations
    """
    # Initialization
    if mutator is None:
        mutator = Mutator(sample_lib) # Applies mutations and handles stft updates
    target = Target(target_y)

    # Create initial population
    population = Population()
    population.individuals = [BaseIndividual.create_random_individual(sample_lib=sample_lib, phi=onset_frac) for _ in tqdm(range(popsize), desc="Initializing Population")]
    for individual in tqdm(population.individuals, desc="Calculating initial fitness"):
        # Calc initial fitness
        individual.fitness_per_onset = multi_onset_fitness_cached(target, individual)
        individual.calc_phi_fitness()
    population.init_archive(target.onsets) # Initial record of best approximations of each onset
    population.sort_individuals_by_fitness() # Sort population for easier management

    # Evolutionary Loop
    for step in (pbar := tqdm(range(max_steps))):
        # Create lambda offspring
        parents = np.random.choice(population.individuals, size=n_offspring)
        offspring = [mutator.mutate_individual(BaseIndividual.from_copy(individual)) for individual in parents]

        # Evaluate fitness of offspring
        for individual in offspring:
            individual.fitness_per_onset = multi_onset_fitness_cached(target, individual)
            individual.calc_phi_fitness()
            # Insert individual into population
            population.insert_individual(individual)
        
        # Remove lambda worst individuals
        population.remove_worst(n_offspring)

        # Update progress bar
        pbar.set_postfix_str(f"Best individual: {str(population.get_best_individual())}")
        if logger:
            logger.log_population(population, step)

    # Return final population
    return population