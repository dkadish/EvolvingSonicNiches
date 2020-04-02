from collections import UserList
import pandas as pd

class Filterable:

    def __init__(self, run=None, generation=None, species=None, subspecies=None):

        self.run = run
        self.generation = generation
        self.species = species
        self.subspecies = subspecies


class FilterableList(UserList):

    def generation(self, g):
        """Returns a FilterableList with only items in the specified generation(s)

        :return: FilterableList with items from generation g
        """
        return self.filter(generation=g)

    def species(self, s):
        """Returns a FilterableList with only items in the specified species.

        Species here refers to experimental species (for example, if there are two distinct
        species communicating within a soundscape).

        :return: FilterableList with items from species s
        """
        return self.filter(species=s)

    def subspecies(self, s):
        """Returns a FilterableList with only items in the specified subspecies

        Subspecies refers to the NEAT concept of species, in which an evolutionary population
        is subdivided to maintain diversity. Here, NEAT species are called subspecies.

        :return: FilterableList with items from subspecies s
        """
        return self.filter(subspecies=s)

    def run(self, r):
        """Returns a FilterableList with only items from the specified run(s)

        :return: FilterableList with items from run r
        """
        return self.filter(run=r)

    def filter(self, run=None, generation=None, species=None, subspecies=None):
        """A general filter for paring the list down by multiple parameters

        :param run: int or list of runs to select
        :param generation: int or list of generations to select
        :param species: int or list of species to select
        :param subspecies: int or list of subspecies to select
        :return:
        """
        sublist = self.data

        if run is not None:
            sublist = filter(lambda i: i.run in listify(run), sublist)

        if generation is not None:
            sublist = filter(lambda i: i.generation in listify(generation), sublist)

        if species is not None:
            sublist = filter(lambda i: i.species in listify(species), sublist)

        if subspecies is not None:
            sublist = filter(lambda i: i.subspecies in listify(subspecies), sublist)

        return self.__class__(sublist)

    @property
    def count(self):
        return len(self.data)

    @property
    def runs(self):
        return list(set(map(lambda i: i.run, self.data)))

    @property
    def generations(self):
        return list(set(map(lambda i: i.generation, self.data)))

    @property
    def next_run(self):
        if self.runs:
            return self.runs[-1] + 1
        else:
            return 0


def listify(v):
    if type(v) is int:
        return [v]
    return v