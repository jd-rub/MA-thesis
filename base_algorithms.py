from sample_library import SampleLibrary
from individual import BaseIndividual, SampleCollection
from mutations import Mutator
from fitness import fitness, multi_onset_fitness_cached
from population import Population
from target import Target
from typing import Union
import numpy as np
from tqdm import tqdm
import bisect

MAX_SAMPLES_PER_ONSET = 5
STOPPING_FITNESS = 0.001

def base_algorithm_1plus1_single_onset(target_y:Union[np.ndarray, list], max_steps:int, sample_lib:SampleLibrary):
    mutator = Mutator(sample_lib)

    best_individual = BaseIndividual()
    best_individual.onset_locations.add(0)
    best_individual.sample_collections[0] = SampleCollection()
    for _ in range(np.random.choice(list(range(MAX_SAMPLES_PER_ONSET)), p=[0.1, 0.3, 0.3, 0.2, 0.1]) + 1):
        random_sample = sample_lib.get_random_sample_uniform()
        best_individual.sample_collections[0].samples.append(random_sample)
    
    best_individual.fitness = fitness(target_y, best_individual.sample_collections[0].to_mixdown())

    for n in (pbar := tqdm(range(max_steps))):
        candidate_individual = BaseIndividual()
        candidate_individual.sample_collections[0] = mutator.mutate_sample_collection(best_individual.sample_collections[0])
        candidate_individual.fitness = fitness(target_y, candidate_individual.sample_collections[0].to_mixdown())

        if candidate_individual.fitness < best_individual.fitness:
            best_individual = candidate_individual
            pbar.set_postfix_str('\t' * 100 + f"Best individual: {str(best_individual)} with fitness {best_individual.fitness}")

        if best_individual.fitness < STOPPING_FITNESS:
            break
    
    return best_individual

def base_algorithm_1plus1_multi_offset(target_y:Union[np.ndarray, list], max_steps:int, sample_lib:SampleLibrary, onset_frac:int = 0.1):
    mutator = Mutator(sample_lib)

    # onsets = librosa.onset.onset_detect(y=target_y, units="samples")
    
    target = Target(target_y)

    best_individual = create_multi_onset_individual(target.onsets, onset_frac, sample_lib)
    #best_individual.fitness = multi_onset_fitness(target_y, best_individual, onsets)
    best_individual.fitness = multi_onset_fitness_cached(target=target, individual=best_individual)

    for n in (pbar := tqdm(range(max_steps))):
        candidate_individual = mutator.mutate_individual(BaseIndividual.from_copy(best_individual))
        # candidate_individual.fitness = multi_onset_fitness(target_y, candidate_individual, onsets)
        candidate_individual.fitness = multi_onset_fitness_cached(target=target, individual=candidate_individual)

        if candidate_individual.fitness < best_individual.fitness:
            best_individual = candidate_individual
            pbar.set_postfix_str('\t' * 100 + f"Best individual: {str(best_individual)} with fitness {best_individual.fitness}")

        if best_individual.fitness < STOPPING_FITNESS:
            break
    
    return best_individual

def create_multi_onset_individual(onsets:np.ndarray, onset_frac:float, sample_lib:SampleLibrary):
    onset_subset_idx = np.random.choice(a=len(onsets), size=int(np.ceil(len(onsets)*onset_frac)), replace=False)
    onset_subset = onsets[onset_subset_idx]
    individual = BaseIndividual(onset_subset)
    for onset in onset_subset:
        collection = individual.sample_collections[onset]
        for _ in range(np.random.choice(list(range(MAX_SAMPLES_PER_ONSET)), p=[0.1, 0.3, 0.3, 0.2, 0.1]) + 1):
            random_sample = sample_lib.get_random_sample_uniform()
            collection.samples.append(random_sample)
    return individual

def approximate_piece(target_y:Union[np.ndarray, list], max_steps:int, sample_lib:SampleLibrary, popsize:int, n_offspring:int, onset_frac:float) -> Population:
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

    Returns
    -------
    Population
        The full population of individual approximations after max_steps of iterations
    """
    # Initialization
    mutator = Mutator(sample_lib) # Applies mutations and handles stft updates
    target = Target(target_y)

    # Create initial population
    population = Population()
    population.individuals = [create_multi_onset_individual(target.onsets, onset_frac, sample_lib) for _ in tqdm(range(popsize), desc="Initializing Population")]
    for individual in tqdm(population.individuals, desc="Calculating initial fitness"):
        # Calc initial fitness
        individual.fitness = multi_onset_fitness_cached(target, individual)
    population.calc_best_fitnesses() # Initial record of best approximations of each onset
    population.sort_individuals_by_fitness() # Sort population for easier management

    # Evolutionary Loop
    for step in (pbar := tqdm(range(max_steps))):
        # Create lambda offspring
        parents = np.random.choice(population.individuals, size=n_offspring)
        offspring = [mutator.mutate_individual(BaseIndividual.from_copy(individual)) for individual in parents]
        
        # Evaluate fitness of offspring
        for individual in offspring:
            individual.fitness = multi_onset_fitness_cached(target, individual)
            # Insert individual into population
            population.insert_individual(individual)
        
        # Remove lambda worst individuals
        population.remove_worst(n_offspring)

        # Update progress bar
        pbar.set_postfix_str('\t' * 100 + f"Best individual: {str(population.get_best_individual())}")


    # Return final population
    return population