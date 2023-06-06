from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from glob import glob
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np

from evoaudio.base_algorithms import approximate_piece
from evoaudio.base_sample import BaseSample
from evoaudio.fitness import fitness_cached
from evoaudio.individual import BaseIndividual
from evoaudio.mutations import Mutator
from evoaudio.population import Population, ArchiveRecord
from evoaudio.sample_library import SampleLibrary
from evoaudio.target import Target
from evoaudio.jaccard import calc_and_save_jaccard, calc_jaccard_for_chord_approximation

RESULT_CSV = "./experiments/ground_truth_scan.csv"

N_RUNS = 100
MAX_STEPS = 100
POPSIZE = 10
N_OFFSPRING = 1
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 10
ZETA = 0.9954
PITCH_SHIFT_STD = 15
MAX_PROCESSES = 10

class LibraryManager(BaseManager):
    pass

class Logger():
    def __init__(self, annotation) -> None:
        self.annotation = annotation
        self.logged_errors = []
        self.logged_fitnesses = []
    def log_errors(self, pop):
        errors = calc_jaccard_for_chord_approximation(pop=pop, annotation=self.annotation)
        self.logged_errors.append(errors)
    def log_fitness(self, pop):
        self.logged_fitnesses.append(pop.archive[0].fitness)
    def log_population(self, pop, step):
        self.log_errors(pop)
        self.log_fitness(pop)
    
def create_sample_set(sample_lib:SampleLibrary):
    individuals = [BaseIndividual.create_random_individual(sample_lib=sample_lib, phi=1.0) for _ in range(N_RUNS)]
    # Ensure pitches are below a threshold (c6) so that 
    # the shift doesn't raise pitch too high
    for individual in individuals:
        for i, sample in enumerate(individual.samples):
            if sample.pitch > 84: # c6
                new_pitch = sample.pitch - 12
                instrument, style = sample_lib.get_random_instrument_for_pitch(new_pitch)
                individual.samples[i] = sample_lib.get_sample(instrument=instrument, style=style, pitch=new_pitch)
    annotations = [[(sample.instrument, str(sample.pitch.value)) for sample in ind.samples] for ind in individuals]
    return individuals, annotations

def run_experiment(annotations:list[tuple[str, str]], target_individuals:list[BaseIndividual], sample_lib:SampleLibrary, errors:list, fitnesses:list, proc_id:int):
    results = []
    for i, annotation in enumerate(annotations):
        logger = Logger(annotation)
        mutator = Mutator(sample_library=sample_lib, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND) 
        result = approximate_piece(
            target_y=target_individuals[i].to_mixdown(), max_steps=MAX_STEPS, 
            sample_lib=sample_lib, popsize=POPSIZE, 
            n_offspring=N_OFFSPRING, onset_frac=1, 
            zeta=ZETA, early_stopping_fitness=0.0001, 
            mutator=mutator, onsets=[0], 
            verbose=proc_id==0, logger=logger)
        result.archive[0] = ArchiveRecord(0, result.get_best_individual().fitness, result.get_best_individual())
        results.append(result)
        errors.append(logger.logged_errors)
        fitnesses.append(logger.logged_fitnesses)

if __name__ == "__main__":
        LibraryManager.register('SampleLibrary', SampleLibrary)
        with LibraryManager() as manager:
            shared_lib = manager.SampleLibrary()
            # Create sample set
            true_individuals, annotations = create_sample_set(shared_lib)
            manager_2 = Manager()
            errors = manager_2.list()
            fitnesses = manager_2.list()
            # Run experiments
            ind_per_process = int(len(true_individuals) / MAX_PROCESSES)
            processes = [Process(target=run_experiment, 
                                args=(annotations[proc_id*ind_per_process:(proc_id+1)*ind_per_process], 
                                    true_individuals[proc_id*ind_per_process:(proc_id+1)*ind_per_process], 
                                    shared_lib, errors, fitnesses, proc_id)) for proc_id in range(MAX_PROCESSES)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            # Save errors and fitnesses
            calc_and_save_jaccard(filename=RESULT_CSV, errors=[run_errors[-1] for run_errors in errors], params={
                "POPSIZE": POPSIZE, "N_OFFSPRING": N_OFFSPRING, "MAX_STEPS": MAX_STEPS, 
                "ALPHA": ALPHA, "BETA": BETA, "L_BOUND": L_BOUND, "U_BOUND": U_BOUND,
                "ZETA": ZETA, "PITCH_SHIFT_STD": PITCH_SHIFT_STD, "N_RUNS": N_RUNS
            })
            with open(f"./experiments/ground_truth_scan/preprocess_thresh_errors.pkl", "wb") as fp:
                pickle.dump(list(errors), fp)
            with open(f"./experiments/ground_truth_scan/preprocess_thresh_fitnesses.pkl", "wb") as fp:
                pickle.dump(list(fitnesses), fp)
