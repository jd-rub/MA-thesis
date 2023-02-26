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
    sample_abs_stft = sample.calc_abs_stft()
    return cosh_distance_no_abs(sample_abs_stft, target_stft)

def multi_onset_fitness_cached(target:Target, individual:BaseIndividual) -> np.ndarray:
    for onset in individual.onset_locations:
        # NOTE: Here we are NOT cutting the sample 
        # to the same size as the target snippet
        collection = individual.sample_collections[onset]
        if collection.recalc_fitness:
            target_stft = target.abs_stft_per_snippet[onset]
            collection.fitness = fitness_cached(collection, target_stft)
            collection.recalc_fitness = False
    return np.mean([collection.fitness for collection in individual.sample_collections.values()])