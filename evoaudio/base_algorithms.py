from typing import Any, Callable, Union

import numpy as np
from tqdm import tqdm

from .sample_library import SampleLibrary
from .individual import BaseIndividual
from .mutations import Mutator
from .fitness import multi_onset_fitness_cached
from .population import Population
from .population_logging import PopulationLogger
from .target import Target

#def approximate_piece_per_onset(target_y:Union[np.ndarray, list], max_steps:int, 
#                                sample_lib:SampleLibrary, popsize:int, n_offspring:int, 
#                                zeta:float=None, early_stopping_fitness:float=None, 
#                                population:Population=None, mutator:Mutator=None, 
#                                logger:PopulationLogger=None, onsets:Union[np.ndarray, list]=None, 
#                                verbose:bool=True
#                                ) -> Population:
#    # WARNING: THIS TAKES FOREVER RIGHT NOW, DO NOT USE!
#    pop = Population()
#    for onset in onsets:
#        result = approximate_piece(target_y=target_y, max_steps=max_steps, sample_lib=sample_lib, popsize=popsize, n_offspring=n_offspring, zeta=zeta, onset_frac=1, early_stopping_fitness=early_stopping_fitness, population=population, mutator=mutator, logger=logger, onsets=[onset], verbose=verbose)
#        pop.merge_populations(result)
#    return pop

def approximate_piece(target_y:Union[np.ndarray, list], max_steps:int, 
                      sample_lib:SampleLibrary, popsize:int, n_offspring:int, 
                      onset_frac:float, zeta:float=None, early_stopping_fitness:float=None, 
                      population:Population=None, mutator:Mutator=None, logger:PopulationLogger=None, 
                      onsets:Union[np.ndarray, list]=None, verbose:bool=True, callback:Callable[[Population, int], Any]=None
                      ) -> Population:
    """Evolutionary approximation of a polyphonic musical piece.

    Parameters
    ----------
    target_y : Union[np.ndarray, list]
        Signal of the target musical piece, as imported by librosa.
    max_steps : int
        Maximum number of iterations (generations) before termination.
    sample_lib : SampleLibrary
        Library of samples which define the algorithm's search space.
    popsize : int
        Size of the population (µ).
    n_offspring : int
        Number of offspring (λ) per generation.
    onset_frac : float
        Fraction of approximated onsets (φ) per individual.
    zeta : float, optional
        Optional parameter for step size adaptation.
    early_stopping_fitness : float, optional
        Algorithm will terminate early if this value is provided and the best individual.
        achieves a fitness below this threshold.
    population : Population, optional
        Pre-initialized population object.
    mutator : Mutator, optional
        Pre-initialized Mutator object that supports the mutate_individual(BaseIndividual) method.
    logger : PopulationLogger, optional
        Logging object, if desired. Can be None to omit logging.
    onsets : Union[np.ndarray, list], optional
        Positions of onsets (in samples) within the target piece. 
        If not provided, they will be estimated by librosa.onset.onset_detect.
    verbose : bool, optional
        If True, will print a progress bar and additional information to console during each step.
    callback : Callable, optional
        Callback function that receives a population and the current step as input.

    Returns
    -------
    Population
        The full population of individual approximations after max_steps of iterations.
    """
    # Initialization
    if mutator is None:
        mutator = Mutator(sample_lib) # Applies mutations and handles stft updates
    target = Target(target_y, onsets)

    # Create initial population
    if population is None:
        population = _init_population(sample_lib=sample_lib, target=target, onset_frac=onset_frac, popsize=popsize, verbose=verbose)

    # Evolutionary Loop
    for step in (pbar := tqdm(range(max_steps), disable=(not verbose))):
        done = _step(population=population, target=target, n_offspring=n_offspring, mutator=mutator, zeta=zeta, early_stopping_fitness=early_stopping_fitness, logger=logger, step=step)
        if verbose:
            # Update progress bar
            pbar.set_postfix_str(f"Best individual: {str(population.get_best_individual())}")
        if callback is not None:
            callback(population, step)
        # Early stopping
        if done:
            break

    # Return final population
    return population

def _init_population(sample_lib:SampleLibrary, target:Target, onset_frac:float, popsize:int, verbose:bool) -> Population:
    # Create initial population
    population = Population()
    population.individuals = [BaseIndividual.create_random_individual(sample_lib=sample_lib, phi=onset_frac) for _ in tqdm(range(popsize), desc="Initializing Population", disable=(not verbose))]
    for individual in tqdm(population.individuals, desc="Calculating initial fitness", disable=(not verbose)):
        # Calc initial fitness
        individual.fitness_per_onset = multi_onset_fitness_cached(target, individual)
        individual.calc_phi_fitness()
    population.init_archive(target.onsets) # Initial record of best approximations of each onset
    population.sort_individuals_by_fitness() # Sort population for easier management
    return population


def _step(population:Population, target:Target, n_offspring:int, mutator:Mutator=None, zeta:float=None, early_stopping_fitness:float=None, logger:PopulationLogger=None, step:int=None):
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

    # Step size adaptation
    if zeta is not None:
        mutator.step_size_control(zeta)

    if logger is not None:
        logger.log_population(population, step)
    
    # Early stopping
    if (early_stopping_fitness is not None 
        and population.get_best_individual().fitness <= early_stopping_fitness):
        return True
    else:
        return False
