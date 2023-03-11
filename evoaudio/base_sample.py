from typing import Union

from .pitch import Pitch, DrumHit

class BaseSample:
    def __init__(self, instrument, style, pitch:Union[Pitch, DrumHit], y, sr):
        self.instrument = instrument
        self.style = style
        self.pitch = pitch
        self.y = y
        self.sr = sr
    
    def __str__(self):
        return f"({self.instrument}, {self.style}, {self.pitch.name})"

    def to_audio(self):
        return self.y

class FlatSample(BaseSample):
    def __init__(self, instrument, style, pitch:Union[Pitch, DrumHit], y=None, sr=None):
        super().__init__(instrument, style, pitch, y = None, sr = None)
    
    def expand(self, sample_lib) -> BaseSample:
        return sample_lib.get_sample(self.instrument, self.style, self.pitch)