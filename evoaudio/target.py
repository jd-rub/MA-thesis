import librosa
import numpy as np

class Target():
    def __init__(self, y, onsets=None) -> None:
        self.y = y
        if onsets is None:
            self.onsets = librosa.onset.onset_detect(y=y, units="samples")
        else:
            self.onsets = onsets
        # self.stft_per_snippet = self.calc_stft_for_snippets() # Dict with onset: stft pairs
        self.stft_per_snippet = self.calc_stft_for_snippets_1sec() # Dict with onset: stft pairs
        self.abs_stft_per_snippet = {onset: np.abs(self.stft_per_snippet[onset]) for onset in self.onsets}
    
    def calc_stft_for_snippets(self):
        stft_per_snippet = dict()
        for i, onset in enumerate(self.onsets):
            if i + 1 < len(self.onsets):
                next_onset = self.onsets[i+1]
                snippet = self.y[onset:next_onset]
                stft_per_snippet[onset] = librosa.stft(snippet)
            else:
                # final onset until end of piece
                snippet = self.y[onset:]
                stft_per_snippet[onset] = librosa.stft(snippet)
        return stft_per_snippet

    def calc_stft_for_snippets_1sec(self):
        # Version with 1-second long snippets (Ginsel et. al. 2022)
        stft_per_snippet = dict()
        for onset in self.onsets:
            outset = onset + 22050
            if outset > len(self.y):
                outset = len(self.y) - 1
            stft_per_snippet[onset] = librosa.stft(self.y[onset:outset])
        return stft_per_snippet
