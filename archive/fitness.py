import numpy as np

from archive import Filterable
from archive.filterable import FilterableList


class Fitness(Filterable):
    """Holds the fitness values from an individual from a simulation.

    The fitnesses are stored in 2 N-length arrays, where N is the number of individuals sent in the generation.

    """

    def __init__(self, fitness, is_sender, is_fittest=False, genome=None, run=None, generation=None, species=None,
                 subspecies=None):
        super().__init__(run, generation, species, subspecies)

        self.fitness = fitness
        self.is_sender = is_sender
        self.is_fittest = is_fittest
        self.genome = None

    @property
    def is_receiver(self):
        return not self.is_sender

    @property
    def has_genome(self):
        return self.genome is not None


class FitnessList(FilterableList):

    @property
    def fittest(self):
        return self.filter(fittest=True)

    def filter(self, role=None, fittest: bool = None, genome: bool = None, run=None, generation=None, species=None,
               subspecies=None):
        sublist = self.data

        if role is not None:
            if role == 'sender':
                sublist = filter(lambda d: d.is_sender, self.data)
            elif role == 'receiver':
                sublist = filter(lambda d: d.is_receiver, self.data)
            else:
                raise ValueError('Role must be \'sender\' or \'receiver\'. Not {}.'.format(role))

            return self.__class__(sublist).filter(fittest=fittest, genome=genome,
                                           run=run, generation=generation, species=species, subspecies=subspecies)

        if fittest is not None:
            sublist = filter(lambda d: d.is_fittest, self.data)
            return self.__class__(sublist).filter(role=role, genome=genome,
                                           run=run, generation=generation, species=species, subspecies=subspecies)

        if genome is not None:
            sublist = filter(lambda d: d.has_genome, self.data)
            return self.__class__(sublist).filter(role=role, fittest=fittest,
                                           run=run, generation=generation, species=species, subspecies=subspecies)

        return super().filter(run=run, generation=generation, species=species, subspecies=subspecies)

    @property
    def senders(self):
        return self.filter(role='sender')

    @property
    def receivers(self):
        return self.filter(role='receiver')

    @property
    def fitness(self):
        """Gets fitnesses as a numpy array.

        :return:
        """
        f = np.array([d.fitness for d in self.data])
        return f

    @property
    def genome(self):
        """Get genomes.

        :return:
        """
        with_genomes = self.filter(genome=True)
        g = [d.genome for d in self.data]
        return g

    @staticmethod
    def from_statistics_reporters(senders, receivers, run: int, species: int):
        fitnesses = []

        for role, is_sender in [(senders, True), (receivers, False)]:
            for generation in range(len(role.generation_statistics)):
                for subspecies in role.generation_statistics[generation]:
                    for individual in role.generation_statistics[generation][subspecies]:
                        fitness = role.generation_statistics[generation][subspecies][individual]

                        genome = None
                        is_fittest = False
                        if role.most_fit_genomes[generation].key == individual:
                            genome = role.most_fit_genomes[generation]
                            is_fittest = True

                        f = Fitness(fitness, is_sender, is_fittest=is_fittest, genome=genome, run=run,
                                    generation=generation, species=species, subspecies=subspecies)
                        fitnesses.append(f)

        return FitnessList(fitnesses)