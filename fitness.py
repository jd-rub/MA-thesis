import numpy as np
import librosa
from individual import BaseIndividual, SampleCollection
from target import Target

def cosh_distance(stft_x, stft_y):
    a = np.average(abs(stft_x), axis=1)
    b = np.average(abs(stft_y), axis=1)
    return np.sum(a/b - np.log(a/b) + b/a - np.log(b/a) - 2) / len(a)

def cosh_distance_no_abs(abs_stft_x, abs_stft_y):
    a = np.average(abs_stft_x, axis=1)
    b = np.average(abs_stft_y, axis=1)
    return np.sum(a/b - np.log(a/b) + b/a - np.log(b/a) - 2) / len(a)

def fitness(x, y) -> float:
    stft_x = librosa.stft(x)
    stft_y = librosa.stft(y)
    return cosh_distance(stft_x, stft_y)

def fitness_cached(sample: SampleCollection, target_stft: np.ndarray):
    if sample.stft is None:
        sample.calc_stft()
    #return cosh_distance(sample.stft, target_stft)
    return cosh_distance_no_abs(sample.abs_stft, target_stft)

def multi_onset_fitness_cached(target:Target, individual:BaseIndividual):
    for onset in individual.onset_locations:
        # NOTE: Here we are NOT cutting the sample 
        # to the same size as the target snippet
        collection = individual.sample_collections[onset]
        if collection.recalc_fitness:
            target_stft = target.abs_stft_per_snippet[onset]
            individual.fitness_by_onset[onset] = fitness_cached(collection, target_stft)
            collection.recalc_fitness = False
            collection.stft = None # Memory optimization
    return np.mean(list(individual.fitness_by_onset.values()))

def multi_onset_fitness(y:np.ndarray, individual:BaseIndividual, onsets:np.ndarray) -> np.ndarray:
    for onset in individual.onset_locations:
        # Get location of next onset
        onset_idx = np.argwhere(onsets == onset)[0].item()
        next_onset = None
        if len(onsets) > onset_idx + 1:
            next_onset = onsets[onset_idx + 1]
        else:
            # if we got the last onset, then go from there until the end of y
            next_onset = len(y) - 1
        snippet_duration = next_onset - onset

        # Slice x and y to the same length
        sliced_x = individual.sample_collections[onset].to_mixdown()[:snippet_duration]
        sliced_y = y[onset:next_onset]

        snippet_fitness = fitness(sliced_x, sliced_y)
        individual.fitness_by_onset[onset] = snippet_fitness
    return np.mean(list(individual.fitness_by_onset.values()))