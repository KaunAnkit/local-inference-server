from  inference_server.scheduler.request import Request
from inference_server.scheduler.state import RequestState


class Generator:

    def __init__(self,tokenizer,model,sampler):

        self.tokenizer = tokenizer
        self.model = model
        self.sampler = sampler
        self.block_size = 16

    def generate(
            self,
            prompt: str,
            max_new_tokens: int,
            temperature: float = 0.7,
            top_k: int | None = 50,
            top_p: float | None = 0.9,
            penalty: float = 1.0,
        ):

        request = Request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            penalty=penalty,
        )

        self.initialize(request)

        while not request.finished:

            if request.state == RequestState.PREFILL:
                yield self.prefill(request)

            elif request.state == RequestState.DECODING:
                yield self.decode(request)



    def initialize(self,request : Request):

        request.encoded_input = self.tokenizer.encode(request.prompt)

        request.generated_ids = []

        request.past_key_values = None

        request.finished = False

    def step(self,request : Request):

        data = request.encoded_input

        cache = request.past_key_values

        if cache is None:

            logits,cache = self.model.forward(data)

        else:
            last_token = request.encoded_input[-1]


            logits,cache = self.model.forward([last_token],cache)

        request.past_key_values = cache

        next_token = self.sampler.sample(
                logits,
                request.generated_ids,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                penalty=request.penalty,
            )

        request.encoded_input.append(next_token)
        request.generated_ids.append(next_token)

        request.count += 1

        if next_token == self.tokenizer.eos_token_id or request.max_new_tokens <= request.count:
            request.finished = True
            return ""

        return self.tokenizer.decode([next_token])

    def prepare_prefill_input(self, request):

        return request.encoded_input

    def prepare_decode_input(self, request):

        return [request.encoded_input[-1]]

    
    def prefill(self,request):
        
        inputs  = self.prepare_prefill_input(request)

        logits,cache = self.model.forward(inputs)


        request.state = RequestState.DECODING
        
        return self.process_output(request, logits, cache)

    def decode(self, request):
        inputs = self.prepare_decode_input(request)

        logits, cache = self.model.forward(
            inputs,
            request.past_key_values
        )

        return self.process_output(request, logits, cache)

    def process_output(self,request : Request, logits, cache):
        
        request.past_key_values = cache


        logits = logits[-1]

    


        next_token = self.sampler.sample(
            logits,
            request.generated_ids,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            penalty=request.penalty,
        )



        request.encoded_input.append(next_token)
        request.generated_ids.append(next_token)

        request.count += 1

        if next_token == self.tokenizer.eos_token_id or request.count >= request.max_new_tokens:
            request.finished = True
            request.state = RequestState.FINISHED
            return None

        
        return self.tokenizer.decode([next_token])

    def need_new_block(self,request):

        tokens_needed = len(request.generated_ids) + 1

        expected_blocks = (tokens_needed + self.block_size - 1) // self.block_size

        allocated_blocks = len(request.block_table)

        return expected_blocks > allocated_blocks

        
    def decode_batch(self, requests):

        tokens = {}

        for request in requests:

            token = self.decode(request)
        
            if token:
                tokens[request.id] = token

        return tokens