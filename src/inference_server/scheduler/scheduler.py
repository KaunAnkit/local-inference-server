
from inference_server.generation.generator import Generator

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

    def step(self, generator : Generator):

        if not self.requests:
            return {}
        tokens = {}

        for request in self.requests:

            token = generator.step(request)

            if token:
                tokens[request.id] = token

            if request.finished:
                self.finished.append(request)
        
        for request in self.finished:
            self.remove(request)

        self.finished.clear()
        
        return tokens
        
                