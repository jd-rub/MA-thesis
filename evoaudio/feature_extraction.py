import numpy as np
from tqdm import tqdm

from .pitch import Pitch
from .population import Population
from .sample_library import SampleLibrary

def extract_features_for_window(pop: Population, lib: SampleLibrary, window_start: int, window_end: int) -> np.ndarray:
    ## Grab the onsets in those windows, and the associated best records from the population
    relevant_collections = [collection for collection in pop.archive.values() if collection.onset >= window_start and collection.onset < window_end]

    ## For each window separately, calculate the maximum, minimum, and mean fitnesses for each occasion of
    ## An instrument
    instrument_occurrences_fitness = dict()
    
    ## A pitch
    pitch_occurrences_fitness = dict()
    for record in relevant_collections:
        collection = record.individual
        for sample in collection.samples:
            if sample.instrument in instrument_occurrences_fitness:
                instrument_occurrences_fitness[sample.instrument].append(collection.fitness)
            else:
                instrument_occurrences_fitness[sample.instrument] = [collection.fitness]
            if sample.pitch in pitch_occurrences_fitness:
                pitch_occurrences_fitness[sample.pitch].append(collection.fitness)
            else:
                pitch_occurrences_fitness[sample.pitch] = [collection.fitness]
    instrument_min = {instrument: np.min(instrument_occurrences_fitness[instrument]) for instrument in instrument_occurrences_fitness}
    instrument_max = {instrument: np.max(instrument_occurrences_fitness[instrument]) for instrument in instrument_occurrences_fitness}
    instrument_mean = {instrument: np.mean(instrument_occurrences_fitness[instrument]) for instrument in instrument_occurrences_fitness}

    pitch_min = {pitch: np.min(pitch_occurrences_fitness[pitch]) for pitch in pitch_occurrences_fitness}
    pitch_max = {pitch: np.max(pitch_occurrences_fitness[pitch]) for pitch in pitch_occurrences_fitness}
    pitch_mean = {pitch: np.mean(pitch_occurrences_fitness[pitch]) for pitch in pitch_occurrences_fitness}

    ## Finally, give each instrument a rank from 1 to n_instruments, based on their mean distances (smallest = rank 1, highest = rank n_instruments)
    # instrument_sort = np.argsort([instrument_mean[instrument] for instrument in instrument_mean])
    instrument_sort = {k: v for k, v in sorted(instrument_mean.items(), key=lambda item: item[1])}
    instrument_ranks = {instrument: i + 1 for i, instrument in enumerate(instrument_sort)}
    
    # Create feature vector
    instrument_features = []
    for instr_name in lib.instruments:
        if instr_name in instrument_ranks:
            instrument_features.append([instrument_min[instr_name], instrument_mean[instr_name], instrument_max[instr_name], instrument_ranks[instr_name]])
        else:
            instrument_features.append([100, 100, 100, 51])
    pitch_features = []
    for pitch in [p.value for p in Pitch][1:]:
        if pitch in pitch_min:
            pitch_features.append([pitch_min[pitch], pitch_mean[pitch], pitch_max[pitch]])
        else:
            pitch_features.append([100, 100, 100])
    flat_instr_features = np.array(instrument_features).flatten()
    flat_pitch_features = np.array(pitch_features).flatten()
    features = np.concatenate((flat_instr_features, flat_pitch_features))
    return features

def extract_features_for_windows(pop:Population, lib:SampleLibrary, window_lengths:list, n_total_samples:int, sr:int):
    window_features = []
    for window_length in window_lengths:
        end_offset = window_length * sr
        last_possible_sample = n_total_samples - end_offset
        window_start = np.random.randint(low=0, high=last_possible_sample)
        window_end = window_start + end_offset
        window_features.append(extract_features_for_window(pop, lib, window_start, window_end))
    return np.concatenate(window_features)