from  inference_server.scheduler.request import Request


class Generator:

    def __init__(self,tokenizer,model,sampler):

        self.tokenizer = tokenizer
        self.model = model
        self.sampler = sampler

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

            yield self.step(request)



    def initialize(self,request : Request):

        request.encoded_input = self.tokenizer.encode(request.prompt)

        request.generated_id = []

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


    def prepare_input(self, request: Request):

        if request.past_key_values is None:
            return request.encoded_input

        return [request.encoded_input[-1]]
    
    def process_output(self,request : Request, logits, cache):
        
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