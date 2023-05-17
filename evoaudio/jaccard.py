from csv import DictWriter
from enum import Enum
from glob import glob
import os

import numpy as np
import pandas as pd

from evoaudio.individual import BaseIndividual
from evoaudio.population import Population
from parsing.arff_parsing import parse_arff

class class_mode(Enum):
    INSTRUMENTS = 0
    PITCHES = 1
    COMBINED = 2

def extract_instruments(individual:BaseIndividual):
    seen_instruments = []
    for sample in individual.samples:
        if sample.instrument not in seen_instruments:
            seen_instruments.append(sample.instrument)
    return seen_instruments

def extract_pitches(individual:BaseIndividual):
    seen_pitches = []
    for sample in individual.samples:
        if sample.pitch not in seen_pitches:
            seen_pitches.append(sample.pitch)
    return seen_pitches

def extract_samples(individual:BaseIndividual):
    seen_samples = []
    for sample in individual.samples:
        if (sample.instrument, sample.pitch) not in seen_samples:
            seen_samples.append((sample.instrument, str(sample.pitch.value)))
    return seen_samples

def jaccard_error(population:Population, annotations:dict, mode:class_mode) -> float:
    """Iteratively calculates the jaccard error for each onset, then returns the mean.

    Parameters
    ----------
    population : Population
        candidate population.
    annotations : dict
        extracted annotations in style {onset: [(instrument1, pitch1), (instrument2, pitch2), ...]}
    mode : class_mode
        whether to calculate the error for instrument, pitch or combined approximation

    Returns
    -------
    float
        mean jaccard error across all onsets.
    """
    time_onsets = list(annotations.keys())
    jaccard_errors_per_onset = []
    for i, onset in enumerate(population.archive):
        individual = population.archive[onset].individual
        time_onset = time_onsets[i]
        match mode:
            case class_mode.INSTRUMENTS:
                extracted_features = extract_instruments(individual)
                annotated_features = [annotation[0] for annotation in annotations[time_onset]]
            case class_mode.PITCHES:
                extracted_features = extract_pitches(individual)
                annotated_features = [int(annotation[1].replace("+", "")) for annotation in annotations[time_onset]]
            case class_mode.COMBINED:
                extracted_features = extract_samples(individual)
                annotated_features = annotations[time_onset]
        intersection = [instrument for instrument in extracted_features if instrument in annotated_features]
        union = list(set(extracted_features) | set(annotated_features))
        false_positives = set(extracted_features).symmetric_difference(intersection)
        false_negatives = set(annotated_features).symmetric_difference(intersection)
        jaccard_errors_per_onset.append((len(false_positives) + len(false_negatives)) / len(union))

    return np.mean(jaccard_errors_per_onset)

def calc_jaccard_for_piece_approximation(experiment_name:str, save_to_csv:bool=False
) -> pd.Series:
    j_i = []
    j_p = []
    j_ip = []

    experiment_name = 'tiny_aam_5000_005_300_1'
    popfiles = glob(f'./experiments/{experiment_name}/*.pkl')
    run_names = []

    for popfile in popfiles:
        run_name = os.path.basename(popfile).split("_")[0]
        run_names.append(run_name)
        pop = Population.from_file(popfile)
        annotations = parse_arff(f'./audio/tiny_aam/annotations/{run_name}_onsets.arff')
        for time_onset in annotations:
            [int(annotation[1].replace("+", "")) for annotation in annotations[time_onset]]
        j_i.append(jaccard_error(pop, annotations, class_mode.INSTRUMENTS))
        j_p.append(jaccard_error(pop, annotations, class_mode.PITCHES))
        j_ip.append(jaccard_error(pop, annotations, class_mode.COMBINED))

    df = pd.DataFrame({'j_i': j_i, 'j_p': j_p, 'j_ip': j_ip}, index=run_names)

    if save_to_csv: df.to_csv(f'./experiments/{experiment_name}_results.csv')

    return df.mean()

def calc_jaccard_for_chord_approximation(pop:Population, annotation:list[tuple]
) -> tuple[float, float, float]:
    j_i = jaccard_error(pop, {0: annotation}, class_mode.INSTRUMENTS)
    j_p = jaccard_error(pop, {0: annotation}, class_mode.PITCHES)
    j_ip = jaccard_error(pop, {0: annotation}, class_mode.COMBINED)

    return j_i, j_p, j_ip

def calc_and_save_jaccard(filename, errors, params:dict):
    # Calculate error statistics
    j_i = np.mean([tup[0] for tup in errors])
    j_p = np.mean([tup[1] for tup in errors])
    j_ip = np.mean([tup[2] for tup in errors])

    j_i_std = np.std([tup[0] for tup in errors])
    j_p_std = np.std([tup[1] for tup in errors])
    j_ip_std = np.std([tup[2] for tup in errors])

    j_i_per_run = []
    j_p_per_run = []
    j_ip_per_run = []
    n = 10

    for i in range(0, len(errors), n):
        j_i_per_run.append(np.mean([tup[0] for tup in errors[i:i+n]]))
        j_p_per_run.append(np.mean([tup[1] for tup in errors[i:i+n]]))
        j_ip_per_run.append(np.mean([tup[2] for tup in errors[i:i+n]]))

    j_i_median = np.median(j_i_per_run)
    j_p_median = np.median(j_p_per_run)
    j_ip_median = np.median(j_ip_per_run)

    j_i_min = np.min(j_i_per_run)
    j_p_min = np.min(j_p_per_run)
    j_ip_min = np.min(j_ip_per_run)

    j_i_max = np.max(j_i_per_run)
    j_p_max = np.max(j_p_per_run)
    j_ip_max = np.max(j_ip_per_run)

    # Save to .csv
    field_names = list(params.keys()) + [ "j_i", "j_i_std", "j_i_median", "j_i_min", "j_i_max", 
                    "j_p", "j_p_std", "j_p_median", "j_p_min", "j_p_max", 
                    "j_ip", "j_ip_std", "j_ip_median", "j_ip_min", "j_ip_max"]

    row = params | {
        "j_i": j_i, "j_i_std": j_i_std, "j_i_median": j_i_median, "j_i_min": j_i_min, "j_i_max": j_i_max,
        "j_p": j_p, "j_p_std": j_p_std, "j_p_median": j_p_median, "j_p_min": j_p_min, "j_p_max": j_p_max,
        "j_ip": j_ip, "j_ip_std": j_ip_std, "j_ip_median": j_ip_median, "j_ip_min": j_ip_min, "j_ip_max": j_ip_max}

    print("Chord approximation results for parameters:")
    print(f"""{params}
------------------
Errors:""")
    print(f"Mean: j_i={j_i}, j_p={j_p}, j_ip={j_ip}")
    print(f"Std: j_i={j_i_std}, j_p={j_p_std}, j_ip={j_ip_std}")

    with open(filename, 'a') as f:
        writer = DictWriter(f, fieldnames=field_names)
        writer.writerow(row)
        f.close()

def jaccard_results_to_csv(filename, errors, popsize, n_offspring, max_steps, 
    alpha, beta, l_bound, u_bound,
    zeta, pitch_shift_std, n_runs):
    # Calculate error statistics
    j_i = np.mean([tup[0] for tup in errors])
    j_p = np.mean([tup[1] for tup in errors])
    j_ip = np.mean([tup[2] for tup in errors])

    j_i_std = np.std([tup[0] for tup in errors])
    j_p_std = np.std([tup[1] for tup in errors])
    j_ip_std = np.std([tup[2] for tup in errors])

    j_i_per_run = []
    j_p_per_run = []
    j_ip_per_run = []
    n = 10

    for i in range(0, len(errors), n):
        j_i_per_run.append(np.mean([tup[0] for tup in errors[i:i+n]]))
        j_p_per_run.append(np.mean([tup[1] for tup in errors[i:i+n]]))
        j_ip_per_run.append(np.mean([tup[2] for tup in errors[i:i+n]]))

    j_i_median = np.median(j_i_per_run)
    j_p_median = np.median(j_p_per_run)
    j_ip_median = np.median(j_ip_per_run)

    j_i_min = np.min(j_i_per_run)
    j_p_min = np.min(j_p_per_run)
    j_ip_min = np.min(j_ip_per_run)

    j_i_max = np.max(j_i_per_run)
    j_p_max = np.max(j_p_per_run)
    j_ip_max = np.max(j_ip_per_run)

    # Save to .csv
    field_names = ["POPSIZE", "N_OFFSPRING", "MAX_STEPS", "ALPHA", "BETA",
                    "L_BOUND", "U_BOUND", "ZETA", "PITCH_SHIFT_STD", "N_RUNS",
                    "j_i", "j_i_std", "j_i_median", "j_i_min", "j_i_max", 
                    "j_p", "j_p_std", "j_p_median", "j_p_min", "j_p_max", 
                    "j_ip", "j_ip_std", "j_ip_median", "j_ip_min", "j_ip_max",]

    row = {"POPSIZE": popsize, "N_OFFSPRING": n_offspring, "MAX_STEPS": max_steps, 
    "ALPHA": alpha, "BETA": beta, "L_BOUND": l_bound, "U_BOUND": u_bound,
    "ZETA": zeta, "PITCH_SHIFT_STD": pitch_shift_std, "N_RUNS": n_runs,
    "j_i": j_i, "j_i_std": j_i_std, "j_i_median": j_i_median, "j_i_min": j_i_min, "j_i_max": j_i_max,
    "j_p": j_p, "j_p_std": j_p_std, "j_p_median": j_p_median, "j_p_min": j_p_min, "j_p_max": j_p_max,
    "j_ip": j_ip, "j_ip_std": j_ip_std, "j_ip_median": j_ip_median, "j_ip_min": j_ip_min, "j_ip_max": j_ip_max}

    print("Chord approximation results for parameters:")
    print(f"""POPSIZE = {popsize},
N_OFFSPRING = {n_offspring},
MAX_STEPS = {max_steps},
ALPHA = {alpha},
BETA = {beta},
L_BOUND = {l_bound},
U_BOUND = {u_bound},
ZETA = {zeta},
PITCH_SHIFT_STD = {pitch_shift_std},
N_RUNS = {n_runs}
------------------
Errors:""")
    print(f"Mean: j_i={j_i}, j_p={j_p}, j_ip={j_ip}")
    print(f"Std: j_i={j_i_std}, j_p={j_p_std}, j_ip={j_ip_std}")

    with open(filename, 'a') as f:
        writer = DictWriter(f, fieldnames=field_names)
        writer.writerow(row)
        f.close()