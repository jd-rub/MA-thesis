from .pitch import Pitch

class InstrumentInfo:
    name: str
    styles: set[str]
    pitches: dict[str, list[Pitch]]
    min_pitches: dict[str, Pitch]
    max_pitches: dict[str, Pitch]
    
    def __init__(self, name:str, styles:set, pitches:dict):
        self.name = name # Instrument name string
        self.styles = styles # Set of known instrument styles
        self.pitches = pitches # Dict of pairs {style: {B2, A2, Ais2, ...}}
        self.min_pitches = dict() # Lowest possible pitch per style
        self.max_pitches = dict() # Highest possible pitch per style
    
    def calc_min_max_pitches(self):
        for style in self.styles:
                self.min_pitches[style] = min(self.pitches[style])
                self.max_pitches[style] = max(self.pitches[style])

    def __str__(self) -> str:
        return f"Instrument Name: {self.name}, Instrument Styles: {self.styles}, Known Pitches: {self.pitches}"