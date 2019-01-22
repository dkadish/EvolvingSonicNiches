from multiprocessing import Array, Lock, Queue
from ctypes import Structure, c_bool

class Message(Structure):
    _fields_ = [('a', c_bool), ('b', c_bool), ('c', c_bool)]

class TwoPopulation:

    def __init__(self):
        self.lock = Lock()
        self.messages = Array(Message, [], lock=self.lock)
        self.encoded_messages = Queue()
        self.decoded_messages = Queue()


    def generate_messages(self):
        pass

    def eval_encoders(self):
        pass

    def eval_decoders(self):
        pass