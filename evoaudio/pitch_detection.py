import librosa
import numpy as np

def extract_pitch_probabilities(y, sr=22050, n_bins=88, bins_per_octave=12, fmin=librosa.note_to_hz("a0"), k_pitches=3):
    """Estimates the pitches in the given signal y and turns
    these estimations into a probability distribution.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    sr : int, optional
        Sample rate, by default 22050
    n_bins : int, optional
        Number of bins to return, 
        Passed to librosa.cqt, by default 88
    bins_per_octave : int, optional
        Passed to librosa.cqt, by default 12
    fmin : float, optional
        Minimum frequency for librosa.cqt, by default librosa.note_to_hz("a0")
    k_pitches : int, optional
        Number of pitches considered for probability calculation,
        as ranked by summed magnitude.

    Returns
    -------
    np.ndarray
        A probability distribution across the 88 pitches we are working with.
    """
    cqt = librosa.cqt(y=y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin)
    abs_cqt = np.abs(cqt)
    clipped_cqt = np.clip(abs_cqt, a_min=0, a_max=None)
    cqt_sum = np.sum(clipped_cqt, axis=1)
    part_idx = np.argpartition(cqt_sum, kth=-k_pitches)[-k_pitches]
    part_val = cqt_sum[part_idx]
    cqt_sum[cqt_sum < (part_val - 0.0001)] = 0
    sum_norm = cqt_sum / np.sum(cqt_sum)
    if np.isnan(sum_norm).any() or not np.isclose(np.sum(sum_norm), 1):
        return [1/len(sum_norm) for _ in range(len(sum_norm))]
    else:
        return sum_norm