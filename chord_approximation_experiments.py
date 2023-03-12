from tqdm import tqdm
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

import numpy as np

from evoaudio.sample_library import SampleLibrary
from evoaudio.base_algorithms import approximate_piece
from evoaudio.population import Population, ArchiveRecord
from evoaudio.mutations import Mutator
from evoaudio.pitch import Pitch
from evoaudio.individual import BaseIndividual
from evoaudio.fitness import fitness
from jaccard import calc_jaccard_for_chord_approximation

POPSIZE = 10
N_OFFSPRING = 1
MAX_STEPS = 1000
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 20
ZETA = 0.9954
N_RUNS = 20
MAX_PROCESSES = 10

class LibraryManager(BaseManager):
    pass

class ListManager(BaseManager):
    pass

def create_sample_set(sample_lib):
    # Create sample set
    target_chords = [
        [("Trumpet", Pitch.c4)],
        [("Violin", Pitch.c4)],
        [("Piano", Pitch.c4)],
        [("Trumpet", Pitch.c4), ("Trombone", Pitch.e4)],
        [("Violin", Pitch.c4), ("Viola", Pitch.e4)],
        [("Piano", Pitch.c4), ("Piano", Pitch.e4)],
        [("Trumpet", Pitch.c4), ("Trombone", Pitch.e4), ("Tuba", Pitch.c3), ("Trumpet", Pitch.g4)],
        [("Violin", Pitch.c4), ("Viola", Pitch.e4), ("Cello", Pitch.c3), ("Violin", Pitch.g4)],
        [("Piano", Pitch.c4), ("Piano", Pitch.e4), ("Piano", Pitch.g4), ("Piano", Pitch.c3)],
        [("Cello", Pitch.c3), ("Trumpet", Pitch.c4), ("Trumpet", Pitch.e4), ("Piano", Pitch.g4), ("Piano", Pitch.c5)]    
    ]
    samples = [[sample_lib.get_sample(instrument=note[0], pitch=note[1]) for note in chord] for chord in target_chords ]
    target_individuals = [BaseIndividual() for i in range(len(samples))]
    for i, target in enumerate(target_individuals):
        target.samples = samples[i]
    target_mixes = [individual.to_mixdown() for individual in target_individuals]
    return target_chords, target_mixes

def get_valid_sample(sample_lib, instrument):
    try:
        return sample_lib.get_sample(instrument=instrument, pitch=sample_lib.get_random_pitch_for_instrument_uniform(instrument, sample_lib.get_random_style_for_instrument(instrument)))
    except:
        return get_valid_sample(sample_lib, instrument)

def run_experiment(target_chords, target_mixes, sample_lib, errors):
    results = []
    # Initialize populations with a-priori knowledge
    populations = [Population() for _ in range(len(target_chords))]
    for i, pop in enumerate(populations):
        for j in range(POPSIZE):
            individual = BaseIndividual()
            for note in target_chords[i]:
                individual.samples.append(get_valid_sample(sample_lib, note[0]))
            individual.fitness = fitness(target_mixes[i], individual.to_mixdown())
            individual.fitness_per_onset.append(individual.fitness)
            pop.insert_individual(individual)
    # Only allow the mutate_pitch mutation
    mutator = Mutator(sample_library=sample_lib, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND, choose_mutation_p=[0, 0, 1]) 

    for i, pop in enumerate(populations):
        result = approximate_piece(target_y=target_mixes[i], max_steps=MAX_STEPS, sample_lib=sample_lib, popsize=POPSIZE, n_offspring=N_OFFSPRING, onset_frac=1, zeta=ZETA, early_stopping_fitness=0.0001, population=pop, mutator=mutator, onsets=[0], verbose=False)
        result.archive[0] = ArchiveRecord(0, result.get_best_individual().fitness, result.get_best_individual())
        results.append(result)
        errors.append(calc_jaccard_for_chord_approximation(result, target_chords[i]))

if __name__ == "__main__":
    manager = Manager()
    errors = manager.list()

    LibraryManager.register('SampleLibrary', SampleLibrary)
    with LibraryManager() as manager:
        shared_lib = manager.SampleLibrary()
        target_chords, target_mixes = create_sample_set(shared_lib)

        finished_runs = 0
        while finished_runs < N_RUNS:
            n_processes = min((N_RUNS - finished_runs), MAX_PROCESSES)

            processes = [Process(target=run_experiment, args=(target_chords, target_mixes, shared_lib, errors)) for _ in range(n_processes)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            finished_runs += n_processes
            print(f"Finished {finished_runs} runs.")
    
    j_i = np.mean([tup[0] for tup in errors])
    j_p = np.mean([tup[1] for tup in errors])
    j_ip = np.mean([tup[2] for tup in errors])

    j_i_std = np.std([tup[0] for tup in errors])
    j_p_std = np.std([tup[1] for tup in errors])
    j_ip_std = np.std([tup[2] for tup in errors])

    print("Chord approximation results for parameters:")
    print(f"""POPSIZE = {POPSIZE},
N_OFFSPRING = {N_OFFSPRING},
MAX_STEPS = {MAX_STEPS},
ALPHA = {ALPHA},
BETA = {BETA},
L_BOUND = {L_BOUND},
U_BOUND = {U_BOUND},
ZETA = {ZETA},
N_RUNS = {N_RUNS}
------------------
Errors:""")
    print(f"Mean: j_i={j_i}, j_p={j_p}, j_ip={j_ip}")
    print(f"Std: j_i={j_i_std}, j_p={j_p_std}, j_ip={j_ip_std}")