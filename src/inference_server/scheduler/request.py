from dataclasses import dataclass, field
from inference_server.scheduler.state import RequestState
@dataclass
class Request:

    id: int
    prompt: str
    max_new_tokens: int

    encoded_input: list[int] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)

    past_key_values : any = None

    finished: bool = False

    temperature : int = 0.5

    top_k : int = 50
    top_p : int = 0.9

    penalty : int = 1

    count : int = 0

    block_table : list[int] = field(default_factory = list) 

    '''When defining a dataclass, default_factory is 
    essential for mutable types like lists, 
    dictionaries, or sets. ''' 

    state : RequestState = RequestState.PREFILL