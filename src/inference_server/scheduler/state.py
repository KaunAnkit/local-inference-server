from enum import Enum

class RequestState(Enum):
    PREFILL = 1
    DECODING = 2
    WAITING_FOR_BLOCK = 3
    FINISHED = 4