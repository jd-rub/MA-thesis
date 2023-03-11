import numpy as np
import librosa

from .individual import BaseIndividual
from .target import Target

def cosh_distance(stft_x, stft_y):
    a = np.average(abs(stft_x), axis=1)
    b = np.average(abs(stft_y), axis=1)
    return np.sum(a/b - np.log(a/b) + b/a - np.log(b/a) - 2) / len(a)

def cosh_distance_no_abs(abs_stft_x, abs_stft_y) -> float:
    """Calculates the cosh-distance between two magnitude spectra x and y.

    Parameters
    ----------
    abs_stft_x : np.ndarray[float]
        Absolute stft values (magnitude spectrum) for x
    abs_stft_y : np.ndarray[float]
        Absolute stft values (magnitude spectrum) for y

    Returns
    -------
    float
        Cosh distance between the provided spectra.
    """
    a = np.average(abs_stft_x, axis=1)
    b = np.average(abs_stft_y, axis=1)
    return np.sum(a/b - np.log(a/b) + b/a - np.log(b/a) - 2) / len(a)

def fitness(x, y) -> float:
    stft_x = librosa.stft(x)
    stft_y = librosa.stft(y)
    return cosh_distance(stft_x, stft_y)

def fitness_cached(sample: BaseIndividual, target_stft: np.ndarray) -> float:
    """Calculates fitness between a sample and a single onset.

    Parameters
    ----------
    sample : SampleCollection
        Candidate sample mix.
    target_stft : np.ndarray
        Absolute stft values (magnitudes) of the target audio.

    Returns
    -------
    float
        Fitness for the candidate sample.
    """
    if sample.abs_stft is None:
        sample.calc_abs_stft()
    return cosh_distance_no_abs(sample.abs_stft, target_stft)

def multi_onset_fitness_cached(target:Target, individual:BaseIndividual) -> np.ndarray:
    """Returns a vector of fitness values. One for each onset in target.

    Parameters
    ----------
    target : Target
        Target piece that is being approximated.
    individual : SampleCollection
        Candidate individual.

    Returns
    -------
    np.ndarray
        Vector of fitness values for each onset.
    """
    if individual.recalc_fitness:
        fitnesses = []
        for onset in target.onsets:
        # NOTE: Here we are NOT cutting the sample 
        # to the same size as the target snippet
            target_abs_stft = target.abs_stft_per_snippet[onset]
            fitness = fitness_cached(individual, target_abs_stft)
            fitnesses.append(fitness)
            
        return np.array(fitnesses)
    else:
        return individual.fitness_per_onset