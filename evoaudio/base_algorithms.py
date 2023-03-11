from typing import Union
import numpy as np
from tqdm import tqdm

from .sample_library import SampleLibrary
from .individual import BaseIndividual
from .mutations import Mutator
from .fitness import multi_onset_fitness_cached
from .population import Population
from .population_logging import PopulationLogger
from .target import Target

MAX_SAMPLES_PER_ONSET = 5
STOPPING_FITNESS = 0.001

def approximate_piece(target_y:Union[np.ndarray, list], max_steps:int, sample_lib:SampleLibrary, popsize:int, n_offspring:int, onset_frac:float, mutator:Mutator=None, logger:PopulationLogger=None, onsets:Union[np.ndarray, list]=None, verbose:bool=True) -> Population:
    """Evolutionary approximation of a polyphonic musical piece

    Parameters
    ----------
    target_y : Union[np.ndarray, list]
        Signal of the target musical piece, as imported by librosa
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
    onsets: Union[np.ndarray, list], Optional
        Positions of onsets (in samples) within the target piece. 
        If not provided, they will be estimated by librosa.onset.onset_detect.
    verbose: bool
        If True, will print a progress bar and additional information to console during each step

    Returns
    -------
    Population
        The full population of individual approximations after max_steps of iterations
    """
    # Initialization
    if mutator is None:
        mutator = Mutator(sample_lib) # Applies mutations and handles stft updates
    target = Target(target_y, onsets)

    # Create initial population
    population = Population()
    population.individuals = [BaseIndividual.create_random_individual(sample_lib=sample_lib, phi=onset_frac) for _ in tqdm(range(popsize), desc="Initializing Population", disable=(not verbose))]
    for individual in tqdm(population.individuals, desc="Calculating initial fitness", disable=(not verbose)):
        # Calc initial fitness
        individual.fitness_per_onset = multi_onset_fitness_cached(target, individual)
        individual.calc_phi_fitness()
    population.init_archive(target.onsets) # Initial record of best approximations of each onset
    population.sort_individuals_by_fitness() # Sort population for easier management

    # Evolutionary Loop
    for step in (pbar := tqdm(range(max_steps), disable=(not verbose))):
        _step(population, target, n_offspring, mutator, logger, step)
        if verbose:
            # Update progress bar
            pbar.set_postfix_str(f"Best individual: {str(population.get_best_individual())}")

    # Return final population
    return population

def _step(population:Population, target:Target, n_offspring:int, mutator:Mutator=None, logger:PopulationLogger=None, step:int=None):
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

    if logger:
        logger.log_population(population, step)
