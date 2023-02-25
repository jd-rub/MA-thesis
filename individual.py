import numpy as np
import librosa
from copy import copy

class SampleCollection:
    def __init__(self):
        self.samples = [] # List of samples in the collection
        self.stft = None # STFT of the mixed-down samples
        self.recalc_fitness = True # If modified, this must be true until fitness is recalculated
    
    def __str__(self):
        s = ", ".join(str(x) for x in self.samples)
        return s

    def calc_stft(self):
        self.stft = librosa.stft(self.to_mixdown())
        self.abs_stft = np.abs(self.stft)
        self.recalc_fitness = True

    def to_mixdown(self):
        # Resize by expanding all samples to the same length
        max_length = np.max([len(sample.y) for sample in self.samples])
        ys_equal_length = [np.pad(sample.y, (0, max_length - len(sample.y))) for sample in self.samples]
        return np.sum(ys_equal_length, axis=0)

    @classmethod
    def from_copy(cls, obj):
        instance = cls()
        instance.samples = [copy(sample) for sample in obj.samples]
        if obj.stft is not None:
            instance.stft = np.copy(obj.stft)
        else:
            instance.stft = None
        instance.recalc_fitness = obj.recalc_fitness
        return instance

class BaseIndividual:
    def __init__(self, onset_locations = set()):
        self.onset_locations = onset_locations # Set of integer indices of the onset locations that are approximated
        self.sample_collections = {onset: SampleCollection() for onset in onset_locations} # Dict of int:SampleCollection, keys should equal onset_locations 
        self.fitness = np.inf
        self.fitness_by_onset = {onset: np.inf for onset in onset_locations}
    
    def __str__(self):
        s = "[" + "], [".join(str(self.sample_collections[x]) for x in self.sample_collections) + "]"
        return s
    @classmethod
    def from_copy(cls, obj):
        instance = cls()
        instance.onset_locations = obj.onset_locations
        instance.sample_collections = {onset: SampleCollection.from_copy(obj.sample_collections[onset]) for onset in obj.onset_locations}
        instance.fitness = obj.fitness
        instance.fitness_by_onset = {onset: obj.fitness_by_onset[onset] for onset in obj.onset_locations}
        return instance