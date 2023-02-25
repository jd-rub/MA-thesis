import numpy as np
from base_sample import BaseSample
from individual import SampleCollection, BaseIndividual
from sample_library import SampleLibrary

SAMPLE_NUMBER_INCREASE_P = [1, 0.8, 0.4, 0.1, 0] # for 1, 2, 3, 4 or 5 samples currently present
ALPHA = 6
BETA = 3
L_BOUND = 1
U_BOUND = 10

class Mutator:
    def __init__(self, sample_library:SampleLibrary, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND, sample_number_increase_p:list[float]=SAMPLE_NUMBER_INCREASE_P):
        self.sample_library = sample_library
        self.alpha = alpha
        self.beta = beta
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.sample_number_increase_p = sample_number_increase_p

    def mutate_sample_collection(self, sample_collection:SampleCollection) -> SampleCollection:
        # Decide which mutation to apply
        mutation = np.random.choice([self.mutate_n_samples, self.mutate_instrument, self.mutate_pitch], p=[0.2, 0.4, 0.4])
        # Apply mutation
        mutated_sample = mutation(sample_collection)
        # Recalc stft for sample
        mutated_sample.calc_stft()
        # Return mutated genotype
        return mutated_sample

    def mutate_individual(self, individual:BaseIndividual) -> BaseIndividual:
        # Draw number of mutation
        n_mutations = int(np.clip(np.floor(np.random.normal(loc=0, scale=1) * self.alpha + self.beta), a_min=self.l_bound, a_max=self.u_bound))

        for _ in range(n_mutations):
            # Pick a random sample collection
            onset = np.random.choice(individual.onset_locations)
            collection = individual.sample_collections[onset]
            individual.sample_collections[onset] = self.mutate_sample_collection(collection)
        
        return individual

    def mutate_n_samples(self, sample_collection:SampleCollection) -> SampleCollection:
        pre_mutation_n_samples = len(sample_collection.samples)
        increase_probability = self.sample_number_increase_p[pre_mutation_n_samples - 1]
        # Increase or decrease number of samples
        rnd = np.random.random()
        if rnd < increase_probability:
            # Add a sample
            new_sample = self.sample_library.get_random_sample_uniform()
            sample_collection.samples.append(new_sample)
        else:
            # Remove a sample
            idx = np.random.choice(pre_mutation_n_samples)
            sample_collection.samples.pop(idx)
        return sample_collection

    def mutate_instrument(self, sample_collection:SampleCollection) -> SampleCollection:
        pre_mutation_n_samples = len(sample_collection.samples)

        # Choose one instrument
        change_idx = np.random.choice(pre_mutation_n_samples)
        pitch = sample_collection.samples[change_idx].pitch

        # Randomly change instrument and style uniformly
        new_instrument, new_style = self.sample_library.get_random_instrument_for_pitch(pitch)
        #new_style = self.sample_library.get_random_style_for_instrument(new_instrument)

        # See if old pitch exists for new instrument
        new_sample = self.sample_library.get_sample(new_instrument, new_style, pitch)
        if new_sample:
            sample_collection.samples[change_idx] = new_sample
        else:
            # Something went wrong
            raise RuntimeError("A sample for the requested instrument, style and pitch does not exist.")

        return sample_collection

    def mutate_pitch(self, sample_collection:SampleCollection) -> SampleCollection:
        pre_mutation_n_samples = len(sample_collection.samples)

        # Choose one instrument
        change_idx = np.random.choice(pre_mutation_n_samples)
        chosen_sample = sample_collection.samples[change_idx]
        
        # Choose a new pitch
        new_pitch = self.sample_library.get_random_pitch_for_instrument(chosen_sample.instrument, chosen_sample.style)
        sample_collection.samples[change_idx].pitch = new_pitch

        return sample_collection