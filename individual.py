import numpy as np
import librosa
from sample_library import SampleLibrary
from copy import copy

INITIAL_N_SAMPLES_P = [0.1, 0.3, 0.3, 0.2, 0.1]

class SampleCollection:
    def __init__(self, onset:int = 0):
        self.onset = onset # Position of this sample's onset in the approximated piece
        self.samples = [] # List of samples in the collection
        self.recalc_fitness = True # True if sample has been modified but fitness has yet to be recalculated
        self.fitness = np.inf
    
    def __str__(self):
        s = f"Onset: {self.onset} | " + ", ".join(str(x) for x in self.samples) + f" | Fitness: {self.fitness}"
        return s

    def calc_abs_stft(self):
        stft = librosa.stft(self.to_mixdown())
        abs_stft = np.abs(stft)
        self.recalc_fitness = True
        return abs_stft

    def to_mixdown(self):
        # Resize by expanding all samples to the same length
        max_length = np.max([len(sample.y) for sample in self.samples])
        ys_equal_length = [np.pad(sample.y, (0, max_length - len(sample.y))) for sample in self.samples]
        return np.sum(ys_equal_length, axis=0)

    @classmethod
    def from_copy(cls, obj):
        instance = cls()
        instance.onset = obj.onset
        instance.samples = [copy(sample) for sample in obj.samples]
        instance.recalc_fitness = obj.recalc_fitness
        instance.fitness = obj.fitness
        return instance

class BaseIndividual:
    def __init__(self, onset_locations = set()):
        self.onset_locations = onset_locations # Set of integer indices of the onset locations that are approximated
        self.sample_collections = {onset: SampleCollection(onset) for onset in onset_locations} # Dict of int:SampleCollection, keys should equal onset_locations 
        self.fitness = np.inf
    
    def __str__(self):
        s = "[" + "], [".join(str(self.sample_collections[x]) for x in self.sample_collections) + "], Fitness: " + str(self.fitness) +"]"
        return s
    
    @classmethod
    def from_copy(cls, obj):
        instance = cls()
        instance.onset_locations = obj.onset_locations
        instance.sample_collections = {onset: SampleCollection.from_copy(obj.sample_collections[onset]) for onset in obj.onset_locations}
        instance.fitness = obj.fitness
        return instance
    
    @classmethod
    def create_multi_onset_individual(cls, onsets:np.ndarray, onset_frac:float, sample_lib:SampleLibrary, max_samples_per_onset:int = 5):
        onset_subset_idx = np.random.choice(a=len(onsets), size=int(np.ceil(len(onsets)*onset_frac)), replace=False)
        onset_subset = onsets[onset_subset_idx]
        individual = cls(onset_subset)
        for onset in onset_subset:
            collection = individual.sample_collections[onset]
            for _ in range(np.random.choice(list(range(max_samples_per_onset)), p=INITIAL_N_SAMPLES_P) + 1):
                random_sample = sample_lib.get_random_sample_uniform()
                collection.samples.append(random_sample)
        return individual