import librosa
import numpy as np

class Target():
    def __init__(self, y, onsets=None, calc_stft=True) -> None:
        self.y = y
        if onsets is None:
            # self.onsets = librosa.onset.onset_detect(y=y, units="samples")
            self.onsets = self.detect_onsets()
        else:
            self.onsets = onsets
        # self.stft_per_snippet = self.calc_stft_for_snippets() # Dict with onset: stft pairs
        if calc_stft:
            self.stft_per_snippet = self.calc_stft_for_snippets_adaptive() # Dict with onset: stft pairs
            self.abs_stft_per_snippet = {onset: np.abs(self.stft_per_snippet[onset]) for onset in self.onsets}
        else:  
            self.stft_per_snippet = dict()
            self.abs_stft_per_snippet = dict()

    def detect_onsets(self):
        y = librosa.resample(y=self.y, orig_sr=22050, target_sr=11025)
        onset_frames = librosa.onset.onset_detect(y=y, units='frames')
        oenv = librosa.onset.onset_strength(y=y)
        backtracked_onset_frames = librosa.onset.onset_backtrack(onset_frames, oenv)
        return librosa.frames_to_samples(backtracked_onset_frames)

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

    def calc_stft_for_snippets_adaptive(self):
        # Version with snippets of length min(onset_n-1 - onset_n, 1, len(song) - onset_n)
        stft_per_snippet = dict()
        for i, onset in enumerate(self.onsets[:-1]):
            outset = int(min(self.onsets[i+1], onset + 22050))
            stft_per_snippet[onset] = librosa.stft(self.y[onset:outset])
        # Final onset
        final_onset = self.onsets[-1]
        stft_per_snippet[final_onset] = librosa.stft(self.y[final_onset:(final_onset+int(min(len(self.y) - final_onset, final_onset + 22050)))])
        return stft_per_snippet