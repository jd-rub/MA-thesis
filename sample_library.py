import librosa
from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np
from base_sample import BaseSample
from instrument_info import InstrumentInfo
from pitch import Pitch

# TODO: Set instruments, styles, and pitches in enum-style
class SampleLibrary:
    def __init__(self, path='./audio/StructuredSamples/'):
        self.instruments = set() # Holds InstrumentInfo objects of the known instruments
        self.known_instruments_by_pitch = dict() # Holds lists of valid instruments+styles for a given pitch
        self.samples = dict()
        self.load_samples_multithreaded(path=path, n_threads=8)
        pass

    def load_samples_multithreaded(self, path:str, n_threads:int) -> None:
        """Loads the samples contained in the the subfolders of path.

        Parameters
        ----------
        path : str
            Path to the sample library. 
        n_threads : int
            Number of threads with which to load the files for better performance.
        """
        wav_files = glob(path + "**/*.wav", recursive=True)
        wav_files = [file for file in wav_files if "Drums" not in file]
        thread_map(self.load_file, wav_files, max_workers=n_threads, chunksize=len(wav_files)//n_threads//10, desc="Loading samples")
    
    def load_file(self, path:str) -> None:
        """Loads a single sample file at path.

        Parameters
        ----------
        path : str
            Path to the sample file.
        """
        y, sr = librosa.load(path)

        # Get instrument name, style and note from file path
        instrument_name, tail = path.split(
            "StructuredSamples\\")[1].split("\\", maxsplit=1)
        style, tail = tail.split("\\")[-2:]
        pitch = tail.split("_")[-1].split(".")[0].lower()

        sample = BaseSample(instrument=instrument_name, style=style, pitch=pitch, y=y, sr=sr)

        new_instrument = True
        for known_instrument in self.instruments:
            if known_instrument.name == instrument_name:
                # Handle style
                if style not in known_instrument.styles:
                    known_instrument.styles.add(style)
                    known_instrument.pitches[style] = {pitch}
                else:
                    known_instrument.pitches[style].add(pitch)
                # Handle pitch by instrument mapping
                if pitch in self.known_instruments_by_pitch:
                    self.known_instruments_by_pitch[pitch].add((instrument_name, style))
                else:
                    self.known_instruments_by_pitch[pitch] = {(instrument_name, style)}
                new_instrument = False

        if new_instrument:
            self.instruments.add(InstrumentInfo(name=instrument_name, styles={style}, pitches={style:{pitch}}, is_single_style=False))
            self.known_instruments_by_pitch[pitch] = {(instrument_name, style)}

        # Save in sample dictionary
        if instrument_name not in self.samples:
            self.samples[instrument_name] = {style: {pitch: sample}}
        else:
            if style not in self.samples[instrument_name]:
                self.samples[instrument_name][style] = {pitch: sample}
            else:
                self.samples[instrument_name][style][pitch] = sample

    def load_samples_single_thread(self, path:str) -> None:
        """Loads all samples in the subfolders of path in a single thread.

        Parameters
        ----------
        path : str
            Path to the sample library.
        """
        wav_files = glob(path + "**/*.wav", recursive=True)
        instrument_names = set()
        samples = dict()

        for file in tqdm(wav_files):
            y = librosa.load(file)

            # Get instrument name, style and note from file path
            instrument_name, tail = file.split(
                "ForMixing\\")[1].split("\\", maxsplit=1)
            instrument_names.add(instrument_name)
            style, tail = tail.split("\\")[-2:]
            note = tail.split("_")[-1].split(".")[0]

            # Save in sample dictionary
            if instrument_name not in samples:
                samples[instrument_name] = {style: {note: y}}
            else:
                if style not in samples[instrument_name]:
                    samples[instrument_name][style] = {note: y}
                else:
                    samples[instrument_name][style][note] = y
        self.instruments = instrument_names
        self.samples = samples

    def get_sample(self, instrument:str, style:str, pitch:str) -> BaseSample:
        """Returns the audio of a sample.

        Parameters
        ----------
        instrument : str
            Name of the instrument
        style : str
            Name of the instrument style
        pitch : str
            Pitch of the sample

        Returns
        -------
        BaseSample
            Sample object from the library

        Raises
        ------
        KeyError
            If the desired sample is not contained in the library.
        """
        try: 
            return self.samples[instrument][style][pitch]
        except:
            print(f"Could not find sample: {instrument}, {style}, {pitch}")
            raise KeyError()

    def get_random_sample_uniform(self) -> BaseSample:
        """Gets a uniform random sample from the library. 
        Note: Instrument, style and pitch are drawn sequentially, 
        meaning that instruments or styles with more samples 
        are not more likely to be drawn than others.

        Returns
        -------
        BaseSample
            Uniformly drawn random sample from the library.
        """
        instrument = np.random.choice(list(self.instruments))
        style = np.random.choice(list(instrument.styles))
        pitch = np.random.choice(list(instrument.pitches[style]))
        return self.get_sample(instrument.name, style, pitch)

    def get_random_instrument_for_pitch(self, pitch:str) -> str:
        """Helper function to draw a random instrument that is valid for the provided pitch.

        Parameters
        ----------
        pitch : str
            Desired pitch that the instrument must be valid for.

        Returns
        -------
        str
            Name of the drawn instrument.
        """
        idx = np.random.choice(len(self.known_instruments_by_pitch[pitch]))
        return list(self.known_instruments_by_pitch[pitch])[idx]

    def get_random_style_for_instrument(self, instrument:str) -> str:
        """Helper function to draw a random style for a provided instrument.

        Parameters
        ----------
        instrument : str
            Desired instrument that the style must be valid for.

        Returns
        -------
        str
            Name of the style that was drawn.

        Raises
        ------
        ValueError
            If instrument was not found in the library.
        """
        instr_info = None
        for info in self.instruments:
            if instrument == info.name:
                instr_info = info
        if instr_info:
            # Instrument found, return random style
            return np.random.choice(list(instr_info.styles))
        else:
            raise ValueError(f"Instrument '{instrument}' not found in sample library.")

    def get_random_pitch_for_instrument_uniform(self, instrument:str, style:str) -> str: 
        """Helper function to draw a uniform random pitch for a given instrument and style.

        Parameters
        ----------
        instrument : str
            Desired instrument that the pitch must be valid for.
        style : str
            Desired style that the pitch must be valid for.

        Returns
        -------
        str
            Name of the pitch that was drawn.
            
        Raises
        ------
        ValueError
            If instrument is not known to the library, or the style is not valid for the instrument.
        """
        instr_info = None
        for info in self.instruments:
            if instrument == info.name:
                instr_info = info
        if instr_info is None:
            raise ValueError(f"Instrument '{instrument}' not found in sample library.")
        if style in instr_info.pitches:
            return np.random.choice(list(instr_info.pitches[style]))
        else:
            raise ValueError(f"Style '{style}' not valid for instrument {instrument}.")
    
    def get_shifted_pitch(self, instrument:str, style:str, old_pitch:str, shift_by:int) -> str:
        # Get old pitch from pitch table
        # Get shifted pitch from pitch table
        # Handle shift out of bounds for instrument (Clip to instrument range)
        # return new pitch
        return