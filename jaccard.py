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
            seen_samples.append((sample.instrument, sample.pitch))
    return seen_samples

def jaccard_error(population:Population, annotations:dict, mode:class_mode):
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
                annotated_features = [annotation[1] for annotation in annotations[time_onset]]
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