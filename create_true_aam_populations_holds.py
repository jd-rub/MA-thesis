from glob import glob
from enum import Enum
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import librosa
import numpy as np

from evoaudio.fitness import fitness_cached
from evoaudio.individual import BaseIndividual
from evoaudio.base_sample import BaseSample, FlatSample
from evoaudio.population import ArchiveRecord, Population
from evoaudio.target import Target
from evoaudio.sample_library import SampleLibrary
from parsing.arff_parsing import parse_arff

N_PROCESSES = 5
N_SEARCHES = 100

class ModdedIndividual(BaseIndividual):
    def __init__(self, phi:float = 0.1, hold_delay:int = 5512):
        super().__init__(phi)
        self.holds = [False for _ in self.samples] # Bool, if True makes the note a hold
        self.hold_delay = hold_delay
    def add_sample(self, sample:BaseSample, hold:bool):
        self.samples.append(sample)
        self.holds.append(hold)
    def to_mixdown(self) -> np.ndarray:
        ys_holding = []
        for i in range(len(self.samples)):
            if self.holds[i]:
                ys_holding.append(self.samples[i].y[self.hold_delay:])
            else:
                ys_holding.append(self.samples[i].y)
        # Resize by expanding all samples to the same length
        max_length = np.max([len(y) for y in ys_holding])
        ys_equal_length = [np.pad(y, (0, max_length - len(y))) for y in ys_holding]
        return np.sum(ys_equal_length, axis=0)

def get_all_styles(instrument, pitch, sample_lib:SampleLibrary) -> list[str]:
    instr_info = sample_lib.get_instrument_info(instrument)
    valid_styles = [style for style in instr_info.pitches if pitch in instr_info.pitches[style]]
    return valid_styles

from itertools import product
def annotation_to_best_individual(annotation, target:Target, onset:int, sample_lib:SampleLibrary):
    # Make a set of all instruments and a list of instruments+pitches in the annotation
    # Get all styles for these instruments+pitches (dict: {instr: set(styles)})
    # Unify styles for each instrument with set intersection
    # Hold a list of pointer-like indices for each instrument, pointing at a style
    # Loop over this list of pointers, always increasing the last one until it reached len(styles_per_instr)
    # Then increment the pointer before that and repeat until done (all pointers == len(styles_per_instr))

    # Alternative with product:
    # For each instrument create a list of tuples (instrument, style, [pitches])
    instrument_tuples = dict()
    for tup in annotation:
        instrument = tup[0]
        pitch = int(tup[1].replace("+", ""))
        valid_styles = get_all_styles(instrument, pitch, sample_lib)
        if instrument in instrument_tuples:
            for style in valid_styles:
                found_style = False
                for instr_tup in instrument_tuples[instrument]:
                    if instr_tup[1] == style:
                        instr_tup[2].append(pitch)
                        found_style = True
                        break
                if not found_style and len(instrument_tuples[instrument][0][2]) == 1:
                    instrument_tuples[instrument].append((instrument, style, [tup[1]]))
        else:
            instrument_tuples[instrument] = []
            for style in valid_styles:
                instrument_tuples[instrument].append((instrument, style, [tup[1]]))

    # Calculate the product across these lists
    all_combinations = list(product(*list(instrument_tuples.values())))
    combinations = [all_combinations[i] for i in np.random.choice(a=len(all_combinations), size=min(len(all_combinations), N_SEARCHES), replace=False)]

    # Randomly choose N for random search
    individuals = []
    fitnesses = []
    # Then parse the tuples into samples
    for combination in combinations:
        ind = ModdedIndividual()
        for instrument, style, pitches_str in combination:
            for pitch_str in pitches_str:
                hold = "+" not in str(pitch_str)
                pitch = int(str(pitch_str).replace("+", ""))
                sample = sample_lib.get_sample(instrument, style, pitch)
                ind.add_sample(sample, hold)
        individuals.append(ind)
        fitness = fitness_cached(ind, target.abs_stft_per_snippet[onset])
        ind.abs_stft = None
        fitnesses.append(fitness)
        # Flatten individual for memory conservation
        ind.samples = [FlatSample(sample.instrument, sample.style, sample.pitch) for sample in ind.samples]

    best_idx = np.argmin(np.array(fitnesses))
    individuals[best_idx].fitness = fitnesses[best_idx]
    return individuals[best_idx]

def save_best_population(name:str, sample_lib:SampleLibrary, annotations:dict, target_mix:list, target_sr=22050, onsets=None):
    print(f"{name} starting.")
    pop = Population()
    if onsets is None:
        onsets = [int(round(float(annotation_time) * target_sr)) for annotation_time in annotations]
    target = Target(y=target_mix, onsets=onsets)
    update_threshold = 0.25
    for i, annotation in enumerate(list(annotations.values())[:-1]):
        try:
            onset = onsets[i]
            ind = annotation_to_best_individual(annotation, target, onset, sample_lib)
            pop.archive[onset] = ArchiveRecord(onset=onset, fitness=ind.fitness, individual=ind)
        except ValueError:
            print(f"Error with Song {name} on onset time {list(annotations.keys())[i]}, no valid individuals found for {str(annotation)}.")
        if i / len(annotations) >= update_threshold:
            print(f"{name}: {update_threshold * 100} % done.")
            update_threshold += 0.25
    pop.save_as_file(f"true_pop_{name}.pkl")
    print(f"{name} done.")

class LibraryManager(BaseManager):
    pass

def create_sample_set():
    # Create sample set
    mixes = {file.split('_mix.mp3')[0][-4:]: librosa.load(file) for file in glob("./audio/tiny_aam/audio-mixes-mp3/*.mp3")}
    annotations = {file.split('_onsets.arff')[0][-4:]: parse_arff(file) for file in glob("./audio/tiny_aam/annotations/*onsets.arff")}
    return annotations, mixes
    
if __name__ == "__main__":
    LibraryManager.register('SampleLibrary', SampleLibrary)
    with LibraryManager() as manager:
        shared_lib = manager.SampleLibrary()
    
        annotations, mixes = create_sample_set()
        processes = []

        for annotation_name in annotations:
            processes.append(Process(target=save_best_population, args=(annotation_name, shared_lib, annotations[annotation_name], mixes[annotation_name][0], mixes[annotation_name][1])))
        for i in range(0, len(processes), N_PROCESSES):
            processes = [Process(target=save_best_population, args=(
                annotation_name, 
                shared_lib, 
                annotations[annotation_name], 
                mixes[annotation_name][0], 
                mixes[annotation_name][1]
                )) for annotation_name in list(annotations.keys())[i:(i+N_PROCESSES)]]
            print(len(processes))
            for process in processes:
                process.start()
            for process in processes:
                process.join()