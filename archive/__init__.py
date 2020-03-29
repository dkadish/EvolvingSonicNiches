import joblib

from archive.filterable import Filterable
from archive.messages import MessageList


class Spectrum(Filterable):

    def __init__(self, run=None, generation=None, species=None, subspecies=None):
        super().__init__(run, generation, species, subspecies)

class Archive:

    def __init__(self, messages=[], stats={}, configs={}, **kwargs) -> None:
        self.messages = MessageList(messages)

    def add_run(self, message_list, run_id=None):
        # if run_id is None:
        #     run_id = self.messages.next_run
        #
        # ml = MessageList.from_message_archive(message_archive, run=run_id)

        self.messages.extend(message_list)

    def save(self, filename):
        joblib.dump(self, filename)

    def __add__(self, other):
        raise NotImplementedError('Not Implemented yet.')

    @staticmethod
    def load(filename):
        return joblib.load(filename)

    @staticmethod
    def createArchive(message_archive):
        pass

