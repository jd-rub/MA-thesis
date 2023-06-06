from csv import DictWriter
from glob import glob
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import os
import pickle

import numpy as np
import librosa
from parsing.arff_parsing import parse_arff

from evoaudio.sample_library import SampleLibrary
from evoaudio.base_algorithms import approximate_piece
from evoaudio.population import Population, ArchiveRecord
from evoaudio.population_logging import PopulationLogger
from evoaudio.mutations import Mutator
from evoaudio.pitch import Pitch
from evoaudio.individual import BaseIndividual
from evoaudio.fitness import fitness
from evoaudio.jaccard import class_mode, jaccard_error

RESULT_FOLDER = "./experiments/1517_artists/"

POPSIZE = 300
N_OFFSPRING = 1
MAX_STEPS = 0
ONSET_FRAC = 0.05
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 20
ZETA = 0.9954
PITCH_SHIFT_STD = 15
MAX_PROCESSES = 10
SNAPSHOT_GEN = 500

PARAM_STR = f"{POPSIZE}_{N_OFFSPRING}_{MAX_STEPS}_{ONSET_FRAC}_{ALPHA}_{BETA}_{L_BOUND}_{U_BOUND}_{ZETA}_{PITCH_SHIFT_STD}_1sec"

class LibraryManager(BaseManager):
    pass

def remove_existing(soundfiles):
    existing = glob(RESULT_FOLDER + PARAM_STR + "/" + "*.pkl")
    existing_names = [os.path.basename(file).split(".")[0] for file in existing]
    not_existing = []
    for file in soundfiles:
        name = os.path.basename(os.path.dirname(file)) + "-" + os.path.basename(file).split(".")[0]
        if name not in existing_names:
            not_existing.append(file)
    return not_existing

def run_experiment(filenames, sample_lib:SampleLibrary, proc_id):
    for file in filenames:
        try:
            run_name = os.path.basename(os.path.dirname(file)) + "-" + os.path.basename(file).split(".")[0]

            def saving_callback(pop:Population, step:int):
                if step == SNAPSHOT_GEN:
                    pop.save_as_file(RESULT_FOLDER + PARAM_STR + "/" + f"{run_name}-500gens.pkl")
                    pop._expand(sample_lib=sample_lib)

            target_mix, target_sr = librosa.load(file)
            logger = PopulationLogger()
            result = approximate_piece(target_y=target_mix, 
                                    max_steps=MAX_STEPS, sample_lib=sample_lib, 
                                    popsize=POPSIZE, n_offspring=N_OFFSPRING, 
                                    onset_frac=ONSET_FRAC, zeta=ZETA, 
                                    logger=logger, verbose=proc_id==0, callback=saving_callback)
            result.save_as_file(RESULT_FOLDER + PARAM_STR + "/" + f"{run_name}-init.pkl")
            with open(RESULT_FOLDER + PARAM_STR + "/" + f"{run_name}.logger.pkl", "wb") as fp:
                pickle.dump(logger, fp)
        except Exception as e:
            print("Exception with file: " + file)
            print(e)
            print("Skipping...")
    
if __name__ == "__main__":
    os.makedirs(RESULT_FOLDER + PARAM_STR + "/", exist_ok=True)
    all_soundfiles = glob("./audio/1517-Artists/**/*.mp3", recursive=True)
    soundfiles = remove_existing(all_soundfiles)
    LibraryManager.register('SampleLibrary', SampleLibrary)
    with LibraryManager() as manager:
        shared_lib = manager.SampleLibrary()
        files_per_proc = len(soundfiles) // MAX_PROCESSES
        processes = [Process(target=run_experiment, args=(
            soundfiles[i*files_per_proc:(i+1)*files_per_proc], 
            shared_lib, i)) for i in range(MAX_PROCESSES)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    