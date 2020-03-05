from enum import Enum


class MessageType(Enum):
    FINISHED = 1
    GENERATION = 2
    MESSAGE = 3


class Message:
    def __init__(self, species_id: int, m_type: MessageType):
        self.species_id = species_id
        self.type = m_type
        self.message = None

    @classmethod
    def Finished(cls, species_id=None):
        return Message(species_id, MessageType.FINISHED)

    @classmethod
    def Generation(cls, species_id):
        message = Message(species_id, MessageType.GENERATION)
        message.message = None  #generation_number
        return message

    @classmethod
    def Encoded(cls, species_id, genome_id, original, encoded, received):
        message = Message(species_id, MessageType.MESSAGE)
        message.message = {'genome_id': genome_id,
                           'original': original,
                           'encoded': encoded,
                           'received': received}
        return message
