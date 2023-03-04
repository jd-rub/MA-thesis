from pitch import Pitch
class BaseSample:
    def __init__(self, instrument, style, pitch:Pitch, y, sr):
        self.instrument = instrument
        self.style = style
        self.pitch = pitch
        self.y = y
        self.sr = sr
    
    def __str__(self):
        return f"({self.instrument}, {self.style}, {self.pitch})"

    def to_audio(self):
        return self.y
