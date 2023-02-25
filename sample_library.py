import librosa
from glob import glob
from tqdm import tqdm
import threading
import numpy as np
from base_sample import BaseSample
from instrument_info import InstrumentInfo

# TODO: Set instruments, styles, and pitches in enum-style
class SampleLibrary:
    def __init__(self, path='./audio/EvoMix-NormalizedSamples/Normalized/Archive/SingleInstrumentSamples/'):
        self.instruments = set() # Holds InstrumentInfo objects of the known instruments
        self.notes = set()
        self.known_instruments_by_pitch = dict() # Holds lists of valid instruments+styles for a given pitch
        self.samples = dict()
        self.load_samples_multithreaded(path=path, n_threads=8)
        pass

    def load_samples_multithreaded(self, path, n_threads):
        wav_files = glob(path + "**/*.wav", recursive=True)
        
        threads = [SampleLibrary._LoadingThread(i, f"Thread-{i}", wav_files[(i*round(len(wav_files)/n_threads)):(i+1)*round(len(wav_files)/n_threads)], self) for i in range(n_threads)]
        for thread in threads:
            thread.start()

        for t in threads:
            t.join()

    
    def load_file(self, path):
        y, sr = librosa.load(path)

        # Get instrument name, style and note from file path
        instrument_name, tail = path.split(
            "ForMixing\\")[1].split("\\", maxsplit=1)
        style, tail = tail.split("\\")[-2:]
        pitch = tail.split("_")[-1].split(".")[0].lower()
        self.notes.add(pitch)

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
            # TODO: single-style detection 
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

    def load_samples_single_thread(self, path):
        wav_files = glob(path + "**/*.wav", recursive=True)
        instrument_names = set()
        notes = set()
        samples = dict()

        for file in tqdm(wav_files):
            y = librosa.load(file)

            # Get instrument name, style and note from file path
            instrument_name, tail = file.split(
                "ForMixing\\")[1].split("\\", maxsplit=1)
            instrument_names.add(instrument_name)
            style, tail = tail.split("\\")[-2:]
            note = tail.split("_")[-1].split(".")[0]
            notes.add(note)

            # Save in sample dictionary
            if instrument_name not in samples:
                samples[instrument_name] = {style: {note: y}}
            else:
                if style not in samples[instrument_name]:
                    samples[instrument_name][style] = {note: y}
                else:
                    samples[instrument_name][style][note] = y
        self.instruments = instrument_names
        self.notes = notes
        self.samples = samples

    def get_sample(self, instrument:str, style:str, pitch:str):
        try: 
            return self.samples[instrument][style][pitch]
        except:
            print(f"Could not find sample: {instrument}, {style}, {pitch}")
            raise KeyError()

    def get_random_sample_uniform(self):
        instrument = np.random.choice(list(self.instruments))
        style = np.random.choice(list(instrument.styles))
        pitch = np.random.choice(list(instrument.pitches[style]))
        return self.get_sample(instrument.name, style, pitch)

    def get_random_instrument_for_pitch(self, pitch:str):
        idx = np.random.choice(len(self.known_instruments_by_pitch[pitch]))
        return list(self.known_instruments_by_pitch[pitch])[idx]

    def get_random_style_for_instrument(self, instrument:str):
        instr_info = None
        for info in self.instruments:
            if instrument == info.name:
                instr_info = info
        if instr_info:
            # Instrument found, return random style
            return np.random.choice(list(instr_info.styles))
        else:
            raise ValueError(f"Instrument '{instrument}' not found in sample library.")

    def get_random_pitch_for_instrument(self, instrument:str, style:str): #TODO: check if all styles have the same pitches available
        instr_info = None
        for info in self.instruments:
            if instrument == info.name:
                instr_info = info
        if instr_info:
            return np.random.choice(list(instr_info.pitches[style]))

    class _LoadingThread(threading.Thread):
        def __init__(self, threadID, name, filepaths, library):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.filepaths = filepaths
            self.library = library
        
        def run(self):
            print("Starting " + self.name)
            for i, filepath in enumerate(self.filepaths):
                self.library.load_file(filepath)
                if i % 50 == 0:
                    print(f"{self.threadID}: Progress: {i}")
