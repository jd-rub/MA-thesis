from copy import copy

import numpy as np
import librosa

from .sample_library import SampleLibrary
from .base_sample import BaseSample

INITIAL_N_SAMPLES_P = [0.1, 0.3, 0.3, 0.2, 0.1]

class BaseIndividual:
    samples: list[BaseSample]
    phi: float
    fitness_per_onset: list[float]
    fitness: float
    recalc_fitness: bool
    abs_stft: np.ndarray

    def __init__(self, phi:float=0.1):
        self.samples = [] # List of samples in the collection
        self.phi = phi # Fraction of onsets that form the basis of overall fitness for this individual 
        self.fitness_per_onset = [] # Vector of fitnesses per onset
        self.fitness = np.inf # Mean fitness to top φ% of approximated onsets
        self.recalc_fitness = True # True if sample has been modified but fitness has yet to be recalculated
        self.abs_stft = None # Absolute stft values for fitness calculation
    
    def __str__(self):
        s = f"Fitness: {self.fitness} | " + ", ".join(str(x) for x in self.samples)
        return s

    # def calc_abs_stft(self) -> None:
    #     """Calculates the absolute stft values of the sample mix.
    #     """
    #     stft = librosa.stft(self.to_mixdown())
    #     self.abs_stft = np.abs(stft)
    #     self.recalc_fitness = True

    def calc_abs_stft(self) -> None:
        # Version with 1-second snippets (Ginsel et. al 2022)
        """Calculates the absolute stft values of the sample mix.
        """
        stft = librosa.stft(self.to_mixdown()[:22050])
        self.abs_stft = np.abs(stft)
        self.recalc_fitness = True

    def calc_phi_fitness(self) -> None:
        """Calculates the fitness of the len(samples)*phi best onsets.
        """
        n_onsets = int(np.ceil(len(self.fitness_per_onset) * self.phi)) # Number of onsets to include
        partition_idx = np.argpartition(self.fitness_per_onset, n_onsets-1) # Indices of the included onsets
        top_fitnesses = self.fitness_per_onset[partition_idx[:n_onsets]]
        self.fitness = np.mean(top_fitnesses) 
        self.recalc_fitness = False
        self.abs_stft = None # Memory optimization

    def to_mixdown(self) -> np.ndarray:
        """Creates a mix of the samples contained in the collection.

        Returns
        -------
        np.ndarray
            Mix of the samples.
        """
        # Resize by expanding all samples to the same length
        max_length = np.max([len(sample.y) for sample in self.samples])
        ys_equal_length = [np.pad(sample.y, (0, max_length - len(sample.y))) for sample in self.samples]
        return np.sum(ys_equal_length, axis=0)

    @classmethod
    def from_copy(cls, obj):
        """Efficiently creates a copy of the given Individual.

        Parameters
        ----------
        obj : BaseIndividual
            Individual that shall be copied.

        Returns
        -------
        BaseIndividual
            Equivalent copy of the Individual that can be modified without modifying the original.
        """
        instance = cls()
        instance.samples = [copy(sample) for sample in obj.samples] # Copies only the reference to a sample for better performance.
        instance.phi = obj.phi
        instance.fitness_per_onset = [fitness for fitness in obj.fitness_per_onset]
        instance.recalc_fitness = obj.recalc_fitness
        instance.fitness = obj.fitness
        instance.abs_stft = obj.abs_stft
        return instance

    @classmethod
    def create_random_individual(cls, sample_lib:SampleLibrary, max_samples:int=5, sample_num_p:list[float]=INITIAL_N_SAMPLES_P, phi:float=0.1):
        """Creates an individual from a sample library and given parameters.

        Parameters
        ----------
        sample_lib : SampleLibrary
            SampleLibrary object containing the instrument and pitch information,
            as well as the samples themselves.
        max_samples : int, optional
            Maximum number of samples in an individual upon initialization, by default 5.
        sample_num_p : list[float], optional
            List probabilities of the number of samples from 1 to max_samples, by default [0.1, 0.3, 0.3, 0.2, 0.1].
        phi : float, optional
            Fraction of onsets that affect fitness calculation, by default 0.1.

        Returns
        -------
        BaseIndividual
            Initialized BaseIndividual containing samples from the provided SampleLibrary.
        """
        individual = cls(phi=phi)
        for _ in range(np.random.choice(list(range(max_samples)), p=sample_num_p) + 1):
            individual.samples.append(sample_lib.get_random_sample_uniform())
        return individual
