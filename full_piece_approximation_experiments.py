from csv import DictWriter
from glob import glob
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import os

import numpy as np
import librosa
from parsing.arff_parsing import parse_arff

from evoaudio.sample_library import SampleLibrary
from evoaudio.base_algorithms import approximate_piece
from evoaudio.population import Population, ArchiveRecord
from evoaudio.mutations import Mutator
from evoaudio.pitch import Pitch
from evoaudio.individual import BaseIndividual
from evoaudio.fitness import fitness
from jaccard import calc_jaccard_for_chord_approximation

RESULT_FOLDER = "./experiments/tiny_aam/"

POPSIZE = 300
N_OFFSPRING = 1
MAX_STEPS = 3000
ONSET_FRAC = 0.05
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 20
ZETA = 0.9954
PITCH_SHIFT_STD = 15
N_RUNS = 20
MAX_PROCESSES = 10

PARAM_STR = f"{POPSIZE}_{N_OFFSPRING}_{MAX_STEPS}_{ONSET_FRAC}_{ALPHA}_{BETA}_{L_BOUND}_{U_BOUND}_{ZETA}_{PITCH_SHIFT_STD}_{N_RUNS}_1sec"

class LibraryManager(BaseManager):
    pass

def create_sample_set():
    # Create sample set
    mixes = {file.split('_mix.mp3')[0][-4:]: librosa.load(file) for file in glob("./audio/tiny_aam/audio-mixes-mp3/*.mp3")}
    annotations = {file.split('_onsets.arff')[0][-4:]: parse_arff(file) for file in glob("./audio/tiny_aam/annotations/*onsets.arff")}
    return annotations, mixes

def get_valid_sample(sample_lib, instrument, pitch):
    # Retrying until valid style is found for pitch
    try:
        return sample_lib.get_sample(instrument=instrument, pitch=pitch)
    except:
        return get_valid_sample(sample_lib, instrument, pitch)

def run_experiment(annotations, target_mixes, sample_lib:SampleLibrary, run_id, proc_id):
    results = {}
    os.makedirs(RESULT_FOLDER + PARAM_STR + "/" + f"{run_id + proc_id}", exist_ok=True)

    for i, name in enumerate(annotations):
        target_mix, target_sr = target_mixes[name]
        annotation = annotations[name]
        onsets = [int(round(float(onset_time) * target_sr)) for onset_time in annotation.keys()]
        result = approximate_piece(target_y=target_mix, max_steps=MAX_STEPS, sample_lib=sample_lib, popsize=POPSIZE, n_offspring=N_OFFSPRING, onset_frac=ONSET_FRAC, zeta=ZETA, onsets=onsets, verbose=proc_id==0)
        result.save_as_file(RESULT_FOLDER + PARAM_STR + "/" + f"{run_id + proc_id}/" + name + ".pkl")
        results[name] = result
    
# def result_to_csv(j_i, j_i_std, j_p, j_p_std, j_ip, j_ip_std):
#     field_names = ["POPSIZE", "N_OFFSPRING", "MAX_STEPS", "ONSET_FRAC", "ALPHA", "BETA",
#                     "L_BOUND", "U_BOUND", "ZETA", "PITCH_SHIFT_STD", "N_RUNS",
#                     "j_i", "j_i_std", "j_p", "j_p_std", "j_ip", "j_ip_std"]
#     row = {"POPSIZE": POPSIZE, "N_OFFSPRING": N_OFFSPRING, "MAX_STEPS": MAX_STEPS, "ONSET_FRAC": ONSET_FRAC,
#     "ALPHA": ALPHA, "BETA": BETA, "L_BOUND": L_BOUND, "U_BOUND": U_BOUND,
#     "ZETA": ZETA, "PITCH_SHIFT_STD": PITCH_SHIFT_STD, "N_RUNS": N_RUNS,
#     "j_i": j_i, "j_i_std": j_i_std, "j_p": j_p, "j_p_std": j_p_std, 
#     "j_ip": j_ip, "j_ip_std": j_ip_std}

#     with open(RESULT_CSV, 'a') as f:
#         writer = DictWriter(f, fieldnames=field_names)
#         writer.writerow(row)
#         f.close()

if __name__ == "__main__":
    # manager = Manager()
    # errors = manager.list()

    LibraryManager.register('SampleLibrary', SampleLibrary)
    with LibraryManager() as manager:
        shared_lib = manager.SampleLibrary()
        annotations, target_mixes = create_sample_set()
        
        finished_runs = 0
        while finished_runs < N_RUNS:
            n_processes = min((N_RUNS - finished_runs), MAX_PROCESSES)
            run_id = finished_runs
            processes = [Process(target=run_experiment, args=(annotations, target_mixes, shared_lib, run_id, proc_id)) for proc_id in range(n_processes)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            finished_runs += n_processes
            print(f"Finished {finished_runs} runs.")
    
    # j_i = np.mean([tup[0] for tup in errors])
    # j_p = np.mean([tup[1] for tup in errors])
    # j_ip = np.mean([tup[2] for tup in errors])

    # j_i_std = np.std([tup[0] for tup in errors])
    # j_p_std = np.std([tup[1] for tup in errors])
    # j_ip_std = np.std([tup[2] for tup in errors])

#     print("Chord approximation results for parameters:")
#     print(f"""POPSIZE = {POPSIZE},
# N_OFFSPRING = {N_OFFSPRING},
# MAX_STEPS = {MAX_STEPS},
# ONSET_FRAC = {ONSET_FRAC},
# ALPHA = {ALPHA},
# BETA = {BETA},
# L_BOUND = {L_BOUND},
# U_BOUND = {U_BOUND},
# ZETA = {ZETA},
# PITCH_SHIFT_STD = {PITCH_SHIFT_STD},
# N_RUNS = {N_RUNS}
# ------------------
# Errors:""")
#     print(f"Mean: j_i={j_i}, j_p={j_p}, j_ip={j_ip}")
#     print(f"Std: j_i={j_i_std}, j_p={j_p_std}, j_ip={j_ip_std}")
#     result_to_csv(j_i=j_i, j_i_std=j_i_std, j_p=j_p, j_p_std=j_p_std, j_ip=j_ip, j_ip_std=j_ip_std)