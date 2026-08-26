
from inference_server.generation.generator import Generator
from inference_server.model.HFModel import HFModel

class Scheduler:

    def __init__(self):
        self.requests = []
        self.finished = []

    def add(self, request):
        self.requests.append(request)


    def remove(self, request):
        self.requests.remove(request)

    def has_requests(self):

        return len(self.requests) > 0

    def step(self, generator : Generator, model : HFModel):

        prefill_requests = []
        decode_requests = []

        if not self.requests:
            return {}
        tokens = {}

        for request in self.requests:

            if request.past_key_values is None:

                prefill_requests.append(request)

            else:
                decode_requests.append(request)


        for request in prefill_requests:

            inputs = generator.prepare_input(request)
            logits, cache = model.forward(
                    inputs,
                    request.past_key_values
                )

            token = generator.process_output(request,logits,cache)

            if token:
                tokens[request.id] = token

            if request.finished:
                self.finished.append(request)

        batch_inputs = []

        for request in decode_requests:

            batch_inputs.append(
                generator.prepare_input(request)
            )

        
        for request in self.finished:
            self.remove(request)

        self.finished.clear()
        
        return tokens

        
                