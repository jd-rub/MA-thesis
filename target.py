import librosa
import numpy as np

class Target():
    def __init__(self, y) -> None:
        self.y = y
        self.onsets = librosa.onset.onset_detect(y=y, units="samples")
        self.stft_per_snippet = self.calc_stft_for_snippets(y, self.onsets) # Dict with onset: stft pairs
        self.abs_stft_per_snippet = {onset: np.abs(self.stft_per_snippet[onset]) for onset in self.onsets}
    
    def calc_stft_for_snippets(self, y, onsets):
        stft_per_snippet = dict()
        for i, onset in enumerate(onsets):
            if i + 1 < len(onsets):
                next_onset = onsets[i+1]
                snippet = y[onset:next_onset]
                stft_per_snippet[onset] = librosa.stft(snippet)
            else:
                # final onset until end of piece
                snippet = y[onset:]
                stft_per_snippet[onset] = librosa.stft(snippet)
        return stft_per_snippet
