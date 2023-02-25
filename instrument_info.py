class InstrumentInfo:
    def __init__(self, name:str, styles:set, pitches:dict, is_single_style:bool):
        self.name = name # Instrument name string
        self.styles = styles # Set of known instrument styles, or {"base"} if only single style exists
        self.pitches = pitches # Dict of pairs {style: {B2, A2, Ais2, ...}}
        self.is_single_style = is_single_style # If len(styles) = 1
    
    def __str__(self) -> str:
        return f"Instrument Name: {self.name}, Instrument Styles: {self.styles}, Known Pitches: {self.pitches}"