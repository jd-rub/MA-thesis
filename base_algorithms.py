from sample_library import SampleLibrary
from individual import BaseIndividual, SampleCollection
from mutations import Mutator
from fitness import fitness, multi_onset_fitness, multi_onset_fitness_cached
from target import Target
from typing import Union
import numpy as np
from tqdm import tqdm

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