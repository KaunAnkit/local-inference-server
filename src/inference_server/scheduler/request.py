from dataclasses import dataclass, field

@dataclass
class Request:
    prompt: str
    max_new_tokens: int

    encoded_input: list[int] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)

    past_key_values : Any = None

    finished: bool = False

    temperature : int

    top_k : int
    top_p : int

    penalty : int

    count : int = 0

    block_table : list[int] = field(default_factory = list) 

    '''When defining a dataclass, default_factory is 
    essential for mutable types like lists, 
    dictionaries, or sets. ''' 

    state : RequestState = RequestState.PREFILL